from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, Optional
import glob
import os
import csv

from bot import load_config, run_scan, ensure_logs, ensure_signal_logs, LOG_DIR, trinity_analyze, make_exchange, set_binance_host
import traceback
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

app = FastAPI(title="Binance Spot Bot API")

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    overrides: Optional[Dict[str, Any]] = None
    dry_run: bool = False


class TrinityRequest(BaseModel):
    symbol: str
    macro_tfs: Optional[list[str]] = None
    bias_tfs: Optional[list[str]] = None
    exec_tfs: Optional[list[str]] = None


@app.get("/config")
def get_config():
    return load_config()


@app.post("/scan")
def post_scan(req: ScanRequest):
    try:
        res = run_scan(req.overrides or {}, execute_trades=not req.dry_run)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trinity/analyze")
def trinity_api(req: TrinityRequest):
    try:
        cfg = load_config()
        symbol = (req.symbol or '').strip().upper()
        if not symbol or '/' not in symbol:
            raise ValueError("Symbol must be like BTC/USDT")

        ex = make_exchange(cfg.get('TESTNET', True))
        try:
            ex.load_markets()
        except Exception:
            # Fallback to live
            ex = make_exchange(False)
            try:
                ex.load_markets()
            except Exception as e_live:
                # Try alternate hosts
                last_err = e_live
                for host in ["api1.binance.com","api2.binance.com","api3.binance.com","api-gcp.binance.com"]:
                    try:
                        set_binance_host(ex, host)
                        ex.load_markets()
                        break
                    except Exception as e_alt:
                        last_err = e_alt
                        continue
                else:
                    raise last_err

        res = trinity_analyze(
            symbol=symbol,
            exchange=ex,
            macro_tfs=req.macro_tfs or ['1w','1d'],
            bias_tfs=req.bias_tfs or ['4h','1h'],
            exec_tfs=req.exec_tfs or ['15m','5m']
        )
        return JSONResponse(content=res, headers={"Cache-Control": "no-store"})
    except Exception as e:
        logging.error("Trinity analyze failed: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/signals")
def get_signals_log(limit: int = 200):
    path = os.path.join(LOG_DIR, "signals.csv")
    if not os.path.exists(path):
        headers = [
            "time","symbol","close","ema20","ema50","rsi14","recent_high","breakout","ema_trend","rsi_ok","signal"
        ]
        resp = {"headers": headers, "rows": []}
        return JSONResponse(content=resp, headers={"Cache-Control": "no-store"})
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        for r in reader:
            rows.append(r)
    resp = {"headers": headers, "rows": rows[-limit:]}
    return JSONResponse(content=resp, headers={"Cache-Control": "no-store"})


@app.get("/logs/trades")
def get_trades_log(limit: int = 200):
    path = os.path.join(LOG_DIR, "trades.csv")
    if not os.path.exists(path):
        return JSONResponse(content={"headers": ["time","symbol","side","qty","entry","tp","sl","note"], "rows": []}, headers={"Cache-Control": "no-store"})
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        for r in reader:
            rows.append(r)
    resp = {"headers": headers, "rows": rows[-limit:]}
    return JSONResponse(content=resp, headers={"Cache-Control": "no-store"})


@app.post("/logs/clear")
def clear_logs():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        def reset_csv(path: str, headers: list[str]):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(headers)

        # Clear known logs by overwriting headers (works even if file is locked for deletion on Windows)
        reset_csv(os.path.join(LOG_DIR, "signals.csv"), [
            "time","symbol","close","ema20","ema50","rsi14","recent_high","breakout","ema_trend","rsi_ok","signal"
        ])
        reset_csv(os.path.join(LOG_DIR, "trades.csv"), [
            "time","symbol","side","qty","entry","tp","sl","note"
        ])

        # Optionally remove any other stray CSVs in logs directory
        for other in glob.glob(os.path.join(LOG_DIR, "*.csv")):
            if os.path.basename(other) not in ("signals.csv", "trades.csv"):
                try:
                    os.remove(other)
                except Exception:
                    # Ignore failures for unknown files
                    pass

        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logs/signals/clear")
def clear_signals_log():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, "signals.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time","symbol","close","ema20","ema50","rsi14","recent_high","breakout","ema_trend","rsi_ok","signal"])
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trinity", include_in_schema=False)
@app.get("/trinity/", include_in_schema=False)
def trinity_page():
    path = os.path.join(WEB_DIR, "trinity.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Trinity page not found")
    return FileResponse(path)

# Serve minimal UI from ./web (mount LAST to avoid swallowing API routes)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
