#!/usr/bin/env python3
import os
import time
import math
import csv
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Base paths (absolute) to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")

################################################################################
# Utility: indicators (EMA, RSI) without TA-Lib
################################################################################

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window//2)).mean()

################################################################################
# Exchange setup
################################################################################

def make_exchange(testnet: bool):
    proxies = None
    # Optional corporate proxy support
    proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
        **({"proxies": proxies} if proxies else {}),
    })
    if testnet:
        exchange.set_sandbox_mode(True)
        # Override API URLs for Binance spot testnet
        exchange.urls["api"]["public"] = "https://testnet.binance.vision/api"
        exchange.urls["api"]["private"] = "https://testnet.binance.vision/api"
    return exchange

def set_binance_host(exchange: ccxt.binance, base_host: str):
    """Override Binance REST endpoints to a different base host."""
    base = f"https://{base_host}"
    if "api" in exchange.urls:
        exchange.urls["api"]["public"] = base + "/api"
        exchange.urls["api"]["private"] = base + "/api"
        # Common additional keys used by ccxt
        for key in ["v1", "v3"]:
            if key in exchange.urls["api"]:
                exchange.urls["api"][key] = base + "/api"
        if "sapi" in exchange.urls["api"]:
            exchange.urls["api"]["sapi"] = base + "/sapi"
        if "wapi" in exchange.urls["api"]:
            exchange.urls["api"]["wapi"] = base + "/wapi"

################################################################################
# Helpers
################################################################################

def load_config():
    load_dotenv()
    cfg = {
        "BASE_QUOTE": os.getenv("BASE_QUOTE", "USDT"),
        "UNIVERSE_SIZE": int(os.getenv("UNIVERSE_SIZE", "12")),
        "MAX_OPEN_POSITIONS": int(os.getenv("MAX_OPEN_POSITIONS", "5")),
        "TIMEFRAME": os.getenv("TIMEFRAME", "5m"),
        "RISK_PER_TRADE": float(os.getenv("RISK_PER_TRADE", "0.01")),
        "STOP_LOSS_PCT": float(os.getenv("STOP_LOSS_PCT", "0.03")),
        "TAKE_PROFIT_PCT": float(os.getenv("TAKE_PROFIT_PCT", "0.06")),
        "MIN_RSI": int(os.getenv("MIN_RSI", "55")),
        "MAX_RSI": int(os.getenv("MAX_RSI", "80")),
        "LOOKBACK_BARS": int(os.getenv("LOOKBACK_BARS", "50")),
        "COOLDOWN_MINUTES": int(os.getenv("COOLDOWN_MINUTES", "60")),
        "MIN_24H_QUOTE_VOLUME": float(os.getenv("MIN_24H_QUOTE_VOLUME", "20000000")),
        "PAPER_TRADING": os.getenv("PAPER_TRADING", "true").lower() == "true",
        "TESTNET": os.getenv("TESTNET", "true").lower() == "true",
    "VERBOSE": os.getenv("VERBOSE", "false").lower() == "true",
    }
    return cfg

def filter_symbols(tickers, base_quote="USDT", min_quote_vol=20_000_000):
    symbols = []
    for sym, data in tickers.items():
        if not sym.endswith(f"/{base_quote}"):
            continue
        # Exclude leveraged tokens and weird pairs
        if any(x in sym for x in ["UP/", "DOWN/", "BULL/", "BEAR/", "LDOT", "PAXG"]):
            continue
        quote_volume = data.get("quoteVolume") or 0
        if quote_volume and quote_volume >= min_quote_vol:
            symbols.append((sym, quote_volume))
    symbols.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in symbols]

def fetch_ohlcv_df(exchange, symbol, timeframe, limit=200):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def recent_high(series: pd.Series, lookback: int) -> float:
    return series.iloc[-lookback:].max()

def now_utc():
    return datetime.now(timezone.utc)

def floor_to_step(qty, step):
    if step is None or step == 0:
        return qty
    return math.floor(qty / step) * step

def round_price(price, tick_size):
    if tick_size <= 0:
        return price
    precision = max(0, int(round(-math.log10(tick_size), 0)))
    return round(price, precision)

################################################################################
# Structure and patterns
################################################################################

def find_pivots(series: pd.Series, left: int = 3, right: int = 3):
    highs = []
    lows = []
    for i in range(left, len(series) - right):
        window = series[i-left:i+right+1]
        if series[i] == window.max():
            highs.append(i)
        if series[i] == window.min():
            lows.append(i)
    return highs, lows

def structure_trend(df: pd.DataFrame, lookback: int = 120) -> str:
    # Simple HH/HL vs LH/LL check on recent pivots
    scope = df.tail(lookback)
    highs_idx, lows_idx = find_pivots(scope['high'].reset_index(drop=True)), find_pivots(scope['low'].reset_index(drop=True))
    # highs_idx and lows_idx are tuples; unpack
    h_idx = highs_idx[0] if isinstance(highs_idx, tuple) else highs_idx
    l_idx = lows_idx[1] if isinstance(lows_idx, tuple) else lows_idx
    # Fallback: slope of SMA200
    if len(h_idx) < 2 or len(l_idx) < 2:
        ma200 = sma(scope['close'], 200)
        if len(ma200.dropna()) >= 5:
            s = (ma200.iloc[-1] - ma200.iloc[-5])
            return 'up' if s > 0 else 'down' if s < 0 else 'sideways'
        return 'sideways'
    # Get last two pivot highs/lows values
    close_vals = scope['close'].reset_index(drop=True)
    last_highs = [scope['high'].iloc[h] for h in h_idx[-3:]]
    last_lows = [scope['low'].iloc[l] for l in l_idx[-3:]]
    hh = last_highs[-1] > last_highs[-2] if len(last_highs) >= 2 else False
    hl = last_lows[-1] > last_lows[-2] if len(last_lows) >= 2 else False
    lh = last_highs[-1] < last_highs[-2] if len(last_highs) >= 2 else False
    ll = last_lows[-1] < last_lows[-2] if len(last_lows) >= 2 else False
    if hh and hl:
        return 'up'
    if lh and ll:
        return 'down'
    # fallback to SMA slope
    ma200 = sma(scope['close'], 200)
    if len(ma200.dropna()) >= 5:
        s = (ma200.iloc[-1] - ma200.iloc[-5])
        return 'up' if s > 0 else 'down' if s < 0 else 'sideways'
    return 'sideways'

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    return (cur['close'] > cur['open']) and (prev['close'] < prev['open']) and (cur['close'] >= prev['open']) and (cur['open'] <= prev['close'])

def is_hammer(df: pd.DataFrame) -> bool:
    if len(df) < 1:
        return False
    c = df.iloc[-1]
    body = abs(c['close'] - c['open'])
    lower_wick = min(c['open'], c['close']) - c['low']
    upper_wick = c['high'] - max(c['open'], c['close'])
    return lower_wick > body * 2 and upper_wick < body

def near_level(price: float, level: float, tolerance: float = 0.005) -> bool:
    # within 0.5% by default
    return abs(price - level) / max(1e-9, level) <= tolerance

def key_levels(df: pd.DataFrame, pivots: int = 5) -> Dict[str, List[float]]:
    h_idx, l_idx = find_pivots(df['high']), find_pivots(df['low'])
    # unpack
    highs = h_idx[0] if isinstance(h_idx, tuple) else h_idx
    lows = l_idx[1] if isinstance(l_idx, tuple) else l_idx
    levels_high = sorted({float(df['high'].iloc[i]) for i in highs[-pivots:]}, reverse=True)
    levels_low = sorted({float(df['low'].iloc[i]) for i in lows[-pivots:]})
    return {'resistance': levels_high, 'support': levels_low}

################################################################################
# Mock data for testing/demo when APIs are unavailable
################################################################################

def generate_mock_ohlcv(symbol: str, timeframe: str, limit: int = 350) -> pd.DataFrame:
    """Generate realistic mock OHLCV data for demonstration purposes."""
    import random
    import numpy as np
    
    # Base price for different symbols
    base_prices = {
        'BTC/USDT': 45000,
        'ETH/USDT': 2500,
        'BNB/USDT': 300,
        'ADA/USDT': 0.5,
        'SOL/USDT': 100
    }
    
    base_price = base_prices.get(symbol, 1000)
    
    # Generate timestamps
    now = datetime.now(timezone.utc)
    tf_minutes = {'1w': 10080, '1d': 1440, '4h': 240, '1h': 60, '15m': 15, '5m': 5}
    interval_minutes = tf_minutes.get(timeframe, 5)
    
    timestamps = []
    data = []
    current_price = base_price
    
    for i in range(limit):
        ts = now - timedelta(minutes=(limit - i - 1) * interval_minutes)
        timestamp_ms = int(ts.timestamp() * 1000)
        
        # Add some trend and volatility - smaller changes
        daily_change = random.uniform(-0.005, 0.005)  # +/- 0.5% per candle max
        current_price = max(base_price * 0.7, min(base_price * 1.5, current_price * (1 + daily_change)))
        
        # OHLCV for this candle
        high = current_price * random.uniform(1.0, 1.005)
        low = current_price * random.uniform(0.995, 1.0)
        open_price = data[-1][4] if data else current_price  # Previous close or current
        close_price = current_price
        volume = random.uniform(1000, 50000)
        
        data.append([timestamp_ms, open_price, high, low, close_price, volume])
    
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

################################################################################
# Trinity analysis
################################################################################

def trinity_analyze(symbol: str,
                    exchange=None,
                    macro_tfs: List[str] = None,
                    bias_tfs: List[str] = None,
                    exec_tfs: List[str] = None,
                    ma_periods = (21, 50, 200),
                    rsi_period: int = 14,
                    use_mock_data: bool = False) -> Dict[str, Any]:
    macro_tfs = macro_tfs or ['1w', '1d']
    bias_tfs = bias_tfs or ['4h', '1h']
    exec_tfs = exec_tfs or ['15m', '5m']

    # Create exchange if not given and not using mock data
    if exchange is None and not use_mock_data:
        cfg = load_config()
        try:
            exchange = make_exchange(cfg['TESTNET'])
            exchange.load_markets()
        except Exception:
            try:
                exchange = make_exchange(False)
                exchange.load_markets()
            except Exception as e:
                # If all API connections fail, switch to mock data mode
                use_mock_data = True
                exchange = None

    errors: List[str] = []
    
    def fetch_df(tf: str) -> pd.DataFrame:
        limit = 400 if tf in ('1w','1d') else 350
        
        if use_mock_data or exchange is None:
            try:
                return generate_mock_ohlcv(symbol, tf, limit)
            except Exception as e:
                errors.append(f"{tf}: Mock data generation failed: {e}")
                return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
        
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            errors.append(f"{tf}: {e}")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    def analyze_tf(df: pd.DataFrame, tf: str) -> Dict[str, Any]:
        a21, a50, a200 = ma_periods
        out: Dict[str, Any] = {'timeframe': tf}
        if len(df) < max(a200+5, rsi_period+5):
            out['insufficient'] = True
            return out
        df = df.copy()
        df['sma21'] = sma(df['close'], a21)
        df['sma50'] = sma(df['close'], a50)
        df['sma200'] = sma(df['close'], a200)
        df['rsi'] = rsi(df['close'], rsi_period)
        last = df.iloc[-1]
        trend = structure_trend(df)
        levels = key_levels(df)
        price = float(last['close'])
        out.update({
            'price': price,
            'trend': trend,
            'above200': bool(price > float(last['sma200'] or price)),
            'above50': bool(price > float(last['sma50'] or price)),
            'above21': bool(price > float(last['sma21'] or price)),
            'rsi': float(last['rsi']) if pd.notnull(last['rsi']) else None,
            'support': [float(x) for x in levels['support'][-3:]],
            'resistance': [float(x) for x in levels['resistance'][-3:]],
        })
        # Nearby support/resistance hint
        near_sup = any(near_level(price, lvl) for lvl in levels['support'][-3:])
        near_res = any(near_level(price, lvl) for lvl in levels['resistance'][-3:])
        out['near_support'] = bool(near_sup)
        out['near_resistance'] = bool(near_res)
        # Candles (execution only meaningful on lower tfs but we compute regardless)
        out['bullish_engulfing'] = bool(is_bullish_engulfing(df))
        out['hammer'] = bool(is_hammer(df))
        return out

    result: Dict[str, Any] = {
        'symbol': symbol,
        'macro': [], 'bias': [], 'execution': [],
        'confluence': {},
        'mock_data': use_mock_data  # Flag to indicate if mock data was used
    }
    # Analyze all tfs
    for tf in macro_tfs:
        result['macro'].append(analyze_tf(fetch_df(tf), tf))
    for tf in bias_tfs:
        result['bias'].append(analyze_tf(fetch_df(tf), tf))
    for tf in exec_tfs:
        result['execution'].append(analyze_tf(fetch_df(tf), tf))

    # Confluence logic
    notes: List[str] = []
    macro_up = sum(1 for m in result['macro'] if m.get('trend') == 'up' and m.get('above200')) >= 1
    macro_down = sum(1 for m in result['macro'] if m.get('trend') == 'down' and not m.get('above200')) >= 1
    direction = 'none'
    if macro_up:
        direction = 'long'
        notes.append('Macro uptrend and above 200SMA')
    elif macro_down:
        direction = 'short'
        notes.append('Macro downtrend and below 200SMA')

    # Bias: look for pullback near support with RSI 40-55 when macro long; inverse for short
    bias_ok = False
    if direction == 'long':
        for b in result['bias']:
            if b.get('near_support') and b.get('rsi') is not None and 35 <= b['rsi'] <= 55:
                bias_ok = True; notes.append(f"Bias {b['timeframe']} near support with RSI reset")
                break
    elif direction == 'short':
        for b in result['bias']:
            if b.get('near_resistance') and b.get('rsi') is not None and 45 <= b['rsi'] <= 65:
                bias_ok = True; notes.append(f"Bias {b['timeframe']} near resistance with RSI cool-off")
                break

    # Execution: candle confirmation on lower tf
    exec_ok = False
    for e in result['execution']:
        if direction == 'long' and (e.get('bullish_engulfing') or e.get('hammer')):
            exec_ok = True; notes.append(f"Execution candle on {e['timeframe']}")
            break
        if direction == 'short':
            # Placeholder: we didn't implement bearish patterns; skip
            pass

    score = (1 if direction != 'none' else 0) + (1 if bias_ok else 0) + (1 if exec_ok else 0)
    result['confluence'] = {
        'direction': direction,
        'bias_ok': bias_ok,
        'execution_ok': exec_ok,
        'score': score,
        'notes': notes + (["Errors:"] + errors if errors else []) + (["⚠️ Using mock data - API unavailable"] if use_mock_data else []),
    }
    return result

################################################################################
# Portfolio/Orders
################################################################################

class Broker:
    def __init__(self, exchange, paper: bool):
        self.exchange = exchange
        self.paper = paper
        self.paper_positions = {}  # symbol -> {"qty", "entry", "tp", "sl", "time"}

    def fetch_balance_quote(self, quote):
        bal = self.exchange.fetch_balance()
        total = bal["total"].get(quote, 0)
        free = bal["free"].get(quote, 0)
        return total, free

    def market_buy(self, symbol, amount_quote, markets):
        # Determine qty by price and lot step
        ticker = self.exchange.fetch_ticker(symbol)
        price = ticker["last"]
        market = markets[symbol]
        lot_step = market["limits"]["amount"]["step"]
        qty = floor_to_step(amount_quote / price, lot_step)

        if qty <= 0:
            raise ValueError(f"Calculated qty {qty} too small for {symbol} at price {price}")

        if self.paper:
            return {"id": f"paper-{symbol}-{int(time.time())}", "symbol": symbol, "price": price, "amount": qty}

        order = self.exchange.create_order(symbol, "market", "buy", qty)
        return order

    def place_tp_sl(self, symbol, entry_price, qty, tp_pct, sl_pct, markets):
        market = markets[symbol]
        tick = market["limits"]["price"]["min"] or market.get("precision", {}).get("price")

        tp_price = round_price(entry_price * (1 + tp_pct), tick or 1e-6)
        sl_price = round_price(entry_price * (1 - sl_pct), tick or 1e-6)

        if self.paper:
            # store in paper positions
            self.paper_positions[symbol] = {
                "qty": qty, "entry": entry_price, "tp": tp_price, "sl": sl_price, "time": now_utc().isoformat()
            }
            return {"tp": tp_price, "sl": sl_price}

        # Place a limit sell (TP)
        self.exchange.create_order(symbol, "limit", "sell", qty, tp_price)
        # Place a stop-market (SL)
        params = {"stopPrice": sl_price, "type": "STOP_MARKET"}
        self.exchange.create_order(symbol, "market", "sell", qty, params=params)
        return {"tp": tp_price, "sl": sl_price}

################################################################################
# Strategy
################################################################################

def generate_signal(df: pd.DataFrame, min_rsi=55, max_rsi=80, lookback=50):
    if len(df) < max(lookback, 50):
        return False

    df = df.copy()
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    last = df.iloc[-1]
    recent_high_val = recent_high(df["high"], lookback)

    conditions = [
        last["close"] > recent_high_val * 1.001,  # tiny breakout buffer
        last["ema20"] > last["ema50"],
        min_rsi <= last["rsi14"] <= max_rsi,
    ]
    return all(conditions)

################################################################################
# Logging
################################################################################

def ensure_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    trade_log = os.path.join(LOG_DIR, "trades.csv")
    if not os.path.exists(trade_log):
        with open(trade_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time","symbol","side","qty","entry","tp","sl","note"])

def ensure_signal_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    sig_log = os.path.join(LOG_DIR, "signals.csv")
    if not os.path.exists(sig_log):
        with open(sig_log, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "time","symbol","close","ema20","ema50","rsi14","recent_high","breakout","ema_trend","rsi_ok","signal"
            ])

def log_trade(symbol, side, qty, entry, tp, sl, note=""):
    ensure_logs()
    with open(os.path.join(LOG_DIR, "trades.csv"), "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([datetime.utcnow().isoformat(), symbol, side, qty, entry, tp, sl, note])

################################################################################
# Main loop (single pass)
################################################################################

def run_scan(cfg_overrides: Optional[Dict[str, Any]] = None, execute_trades: bool = True) -> Dict[str, Any]:
    """Run a single scan with optional config overrides.
    execute_trades=False will not place orders (dry-run), but will compute would-trade opportunities.
    """
    cfg = load_config()
    if cfg_overrides:
        # Merge overrides (basic types expected)
        for k, v in cfg_overrides.items():
            if k in cfg:
                cfg[k] = v

    ensure_logs()
    ensure_signal_logs()

    ex = make_exchange(cfg["TESTNET"])
    messages: List[str] = []
    try:
        markets = ex.load_markets()
        tickers = ex.fetch_tickers()
    except ccxt.NetworkError as e:
        if cfg["TESTNET"]:
            msg = "Testnet unreachable (" + str(e) + "). Falling back to live endpoints for market data. PAPER_TRADING remains enabled."
            print(msg)
            messages.append(msg)
            ex = make_exchange(False)
            try:
                markets = ex.load_markets()
                tickers = ex.fetch_tickers()
            except ccxt.NetworkError as e_live:
                alt_hosts = ["api1.binance.com","api2.binance.com","api3.binance.com","api-gcp.binance.com"]
                last_err = e_live
                for host in alt_hosts:
                    print(f"Trying alternate Binance host: {host}")
                    messages.append(f"Trying alternate host: {host}")
                    set_binance_host(ex, host)
                    try:
                        markets = ex.load_markets()
                        tickers = ex.fetch_tickers()
                        print(f"Connected via {host}")
                        messages.append(f"Connected via {host}")
                        break
                    except ccxt.NetworkError as e_alt:
                        last_err = e_alt
                        continue
                else:
                    msg = "Network error connecting to Binance. Consider HTTP_PROXY/HTTPS_PROXY or VPN."
                    print(msg)
                    messages.append(msg)
                    raise last_err
        else:
            alt_hosts = ["api1.binance.com","api2.binance.com","api3.binance.com","api-gcp.binance.com"]
            last_err = e
            for host in alt_hosts:
                print(f"Trying alternate Binance host: {host}")
                messages.append(f"Trying alternate host: {host}")
                set_binance_host(ex, host)
                try:
                    markets = ex.load_markets()
                    tickers = ex.fetch_tickers()
                    print(f"Connected via {host}")
                    messages.append(f"Connected via {host}")
                    break
                except ccxt.NetworkError as e_alt:
                    last_err = e_alt
                    continue
            else:
                msg = "Network error connecting to Binance. Consider HTTP_PROXY/HTTPS_PROXY or VPN."
                print(msg)
                messages.append(msg)
                raise last_err

    universe = filter_symbols(tickers, base_quote=cfg["BASE_QUOTE"], min_quote_vol=cfg["MIN_24H_QUOTE_VOLUME"])
    universe = universe[: cfg["UNIVERSE_SIZE"]]
    print(f"Universe ({len(universe)}): {universe}")

    total_quote, _ = Broker(ex, cfg["PAPER_TRADING"]).fetch_balance_quote(cfg["BASE_QUOTE"])
    equity = total_quote if total_quote > 0 else 100.0
    risk_amount = equity * cfg["RISK_PER_TRADE"]

    open_positions = 0
    cooldown_map: Dict[str, datetime] = {}
    broker = Broker(ex, cfg["PAPER_TRADING"]) if execute_trades else None

    signals_true = 0
    signals_out: List[Dict[str, Any]] = []
    trades_out: List[Dict[str, Any]] = []

    for symbol in universe:
        if open_positions >= cfg["MAX_OPEN_POSITIONS"]:
            break

        last_time = cooldown_map.get(symbol)
        if last_time and now_utc() - last_time < timedelta(minutes=cfg["COOLDOWN_MINUTES"]):
            continue

        try:
            df = fetch_ohlcv_df(ex, symbol, cfg["TIMEFRAME"], limit=max(cfg["LOOKBACK_BARS"]+60, 150))
        except Exception as e:
            print(f"Failed to fetch OHLCV for {symbol}: {e}")
            messages.append(f"OHLCV error {symbol}: {e}")
            continue

        signal = generate_signal(df, cfg["MIN_RSI"], cfg["MAX_RSI"], cfg["LOOKBACK_BARS"])

        # Compose diagnostics row
        df_tmp = df.copy()
        df_tmp["ema20"] = ema(df_tmp["close"], 20)
        df_tmp["ema50"] = ema(df_tmp["close"], 50)
        df_tmp["rsi14"] = rsi(df_tmp["close"], 14)
        last = df_tmp.iloc[-1]
        rh = recent_high(df_tmp["high"], cfg["LOOKBACK_BARS"])
        breakout = bool(last["close"] > rh * 1.001)
        ema_trend = bool(last["ema20"] > last["ema50"])
        rsi_ok = bool(cfg["MIN_RSI"] <= last["rsi14"] <= cfg["MAX_RSI"])
        sig_row = {
            "symbol": symbol,
            "close": float(last["close"]),
            "ema20": float(last["ema20"]),
            "ema50": float(last["ema50"]),
            "rsi14": float(last["rsi14"]),
            "recent_high": float(rh),
            "breakout": breakout,
            "ema_trend": ema_trend,
            "rsi_ok": rsi_ok,
            "signal": bool(signal),
        }
        signals_out.append(sig_row)

        # Append to CSV signals log
        try:
            with open(os.path.join(LOG_DIR, "signals.csv"), "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.utcnow().isoformat(), symbol,
                    sig_row["close"], sig_row["ema20"], sig_row["ema50"], sig_row["rsi14"],
                    sig_row["recent_high"], sig_row["breakout"], sig_row["ema_trend"], sig_row["rsi_ok"], sig_row["signal"]
                ])
        except Exception as e:
            messages.append(f"Signal log write failed {symbol}: {e}")

        if not signal:
            continue
        signals_true += 1

        # Position sizing
        last_close = float(df["close"].iloc[-1])
        stop_distance = last_close * cfg["STOP_LOSS_PCT"]
        if stop_distance <= 0:
            continue
        qty = floor_to_step(risk_amount / stop_distance, markets[symbol]["limits"]["amount"]["step"])
        if qty <= 0:
            continue

        if not execute_trades:
            trades_out.append({
                "symbol": symbol,
                "qty": qty,
                "entry": last_close,
                "tp": round_price(last_close * (1 + cfg["TAKE_PROFIT_PCT"]), markets[symbol]["limits"]["price"]["min"] or 1e-6),
                "sl": round_price(last_close * (1 - cfg["STOP_LOSS_PCT"]), markets[symbol]["limits"]["price"]["min"] or 1e-6),
                "note": "breakout",
                "executed": False,
            })
            continue

        # Execute order flow
        try:
            order = broker.market_buy(symbol, amount_quote=qty*last_close, markets=markets)
            entry_price = order.get("price") or last_close
            levels = broker.place_tp_sl(symbol, entry_price, qty, cfg["TAKE_PROFIT_PCT"], cfg["STOP_LOSS_PCT"], markets)
            log_trade(symbol, "buy", qty, entry_price, levels["tp"], levels["sl"], note="breakout")
            open_positions += 1
            cooldown_map[symbol] = now_utc()
            print(f"TRADE {symbol}: qty={qty} entry={entry_price} tp={levels['tp']} sl={levels['sl']}")
            trades_out.append({
                "symbol": symbol, "qty": qty, "entry": entry_price, "tp": levels["tp"], "sl": levels["sl"], "note": "breakout", "executed": True
            })
        except Exception as e:
            print(f"Order failed for {symbol}: {e}")
            messages.append(f"Order failed {symbol}: {e}")
            continue

    summary = {
        "universe": universe,
        "signals_true": signals_true,
        "universe_size": len(universe),
        "trades_taken": open_positions,
        "messages": messages,
        "paper_trading": cfg["PAPER_TRADING"],
        "testnet": cfg["TESTNET"],
    }
    return {"summary": summary, "signals": signals_out, "trades": trades_out, "config": cfg}


def main():
    result = run_scan()
    print(f"Scan complete. Signals true: {result['summary']['signals_true']}/{result['summary']['universe_size']}. Trades taken: {result['summary']['trades_taken']}. See logs/signals.csv and logs/trades.csv.")

if __name__ == "__main__":
    main()
