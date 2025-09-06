# Binance Spot Momentum Bot (CCXT)

**DISCLAIMER:** This is an educational scaffold. Crypto trading is high-risk. No guarantees of profit.

## Features
- Spot testnet or live
- Paper trading switch (no live orders)
- Universe selection by 24h quote volume (USDT pairs)
- 5m momentum breakout with EMA20>EMA50 and RSI filter
- Risk per trade sizing with fixed % stop and take-profit
- Simple CSV logging

## Quickstart
1. **Download** this folder or the ZIP.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set keys:
   - Create Binance **spot testnet** keys (recommended) and set `TESTNET=true`.
   - For dry run without orders, leave `PAPER_TRADING=true`.
4. Fund your testnet wallet with test USDT if needed.
5. Run the bot (single scan pass):
   ```bash
   python bot.py
   ```

### Connectivity Troubleshooting (Windows)
- If you see DNS errors resolving Binance hosts, your network may block them.
- Options:
   - Use alternate hosts: the bot automatically tries api1/api2/api3/api-gcp.
   - Configure a proxy in `.env`:
      ```powershell
      # PowerShell example
      $env:HTTP_PROXY = "http://127.0.0.1:8888"
      $env:HTTPS_PROXY = "http://127.0.0.1:8888"
      ```
      Or set HTTP_PROXY/HTTPS_PROXY lines in `.env` and restart the shell.
   - Use a VPN that allows Binance connectivity.

## Config Notes
- `RISK_PER_TRADE=0.01` means risking 1% of your equity per trade.
- `STOP_LOSS_PCT` and `TAKE_PROFIT_PCT` are symmetric by default (e.g., 3% SL, 6% TP).
- `UNIVERSE_SIZE` chooses the top-N liquid USDT pairs by 24h quote volume.
- Increase `MIN_24H_QUOTE_VOLUME` to avoid illiquid assets.

## Production Tips
- Run in a loop with sleep (e.g., every 2–5 minutes).
- Add persistence for open orders and position reconciliation.
- Expand to include trailing stops, multi-timeframe confirmation, and blacklist of leveraged tokens.

