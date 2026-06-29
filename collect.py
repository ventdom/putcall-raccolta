#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  collect.py  -  raccolta giornaliera per GitHub Actions (gira nel cloud)
#  Scatta il put/call dei 10 ticker (yfinance), aggancia gli indici di
#  volatilita'/mercato (^VIX,^VXN,^OVX,^GSPC,^IXIC) e ACCODA a pc_archive.csv
#  nella cartella del repository. Idempotente. Didattico, NON consulenza.
# ============================================================================
import os, time, datetime as dt
import numpy as np, pandas as pd
import yfinance as yf

ARCHIVE      = "pc_archive.csv"   # nel repo (il workflow fa il commit)
MAX_DTE      = 60
MAX_EXPIRIES = 8

TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM","WMT"]
BASE_COLS = ["date","ticker","spot","call_vol","put_vol","call_oi","put_oi",
             "call_prem","put_prem","pc_vol","pc_oi","n_expiries"]
IDX = {"vix":"^VIX","vxn":"^VXN","ovx":"^OVX","sp500":"^GSPC","nasdaq":"^IXIC"}
IDX_COLS = list(IDX.keys())


def load_archive(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=BASE_COLS)
    d = pd.read_csv(path); d["date"] = pd.to_datetime(d["date"], errors="coerce"); return d

def save_archive(df, path):
    out = df.copy(); out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    cols = [c for c in BASE_COLS if c in out.columns] + [c for c in out.columns if c not in BASE_COLS]
    out[cols].sort_values(["date","ticker"]).to_csv(path, index=False)

def append_rows(path, rows):
    old = load_archive(path); new = pd.DataFrame(rows); new["date"] = pd.to_datetime(new["date"])
    if old.empty:
        return new
    keys = set(zip(new["date"], new["ticker"]))
    old = old[[(d,t) not in keys for d,t in zip(old["date"], old["ticker"])]]
    return new if old.empty else pd.concat([old, new], ignore_index=True)


def fetch_snapshot(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d", auto_adjust=True)
    spot = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
    today = dt.date.today(); keep = []
    for e in (tk.options or []):
        try: d = dt.date.fromisoformat(e)
        except ValueError: continue
        if 0 <= (d - today).days <= MAX_DTE: keep.append(e)
    keep = keep[:MAX_EXPIRIES]
    cv=pv=coi=poi=cprem=pprem=0.0
    for e in keep:
        try: ch = tk.option_chain(e)
        except Exception: continue
        for side, d in (("c", ch.calls), ("p", ch.puts)):
            if d is None or d.empty: continue
            vol = pd.to_numeric(d.get("volume"), errors="coerce").fillna(0)
            oi  = pd.to_numeric(d.get("openInterest"), errors="coerce").fillna(0)
            last= pd.to_numeric(d.get("lastPrice"), errors="coerce").fillna(0)
            prem = (vol*last*100).sum()
            if side=="c": cv+=vol.sum(); coi+=oi.sum(); cprem+=prem
            else:         pv+=vol.sum(); poi+=oi.sum(); pprem+=prem
        time.sleep(0.05)
    return {"date":today.isoformat(),"ticker":ticker,"spot":round(spot,4),
            "call_vol":int(cv),"put_vol":int(pv),"call_oi":int(coi),"put_oi":int(poi),
            "call_prem":round(cprem,2),"put_prem":round(pprem,2),
            "pc_vol":round(pv/cv,4) if cv>0 else np.nan,
            "pc_oi":round(poi/coi,4) if coi>0 else np.nan,"n_expiries":len(keep)}


def fetch_idx_close(symbol, start, end):
    h = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True)
    if h.empty: raise ValueError("vuoto")
    s = h["Close"].copy(); s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()

def enrich(arch):
    if arch.empty: return arch
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(arch["date"]).unique()))
    start = (dates.min()-pd.Timedelta(days=10)).date().isoformat()
    end   = (dates.max()+pd.Timedelta(days=2)).date().isoformat()
    tab = pd.DataFrame(index=dates)
    for name, sym in IDX.items():
        try:
            tab[name] = fetch_idx_close(sym, start, end).reindex(dates, method="ffill")
            print(f"  [idx] {name:<7}{sym:<7} ok")
        except Exception as e:
            print(f"  [idx] {name:<7}{sym:<7} ERRORE: {e}")
    arch = arch.drop(columns=[c for c in IDX_COLS if c in arch.columns], errors="ignore")
    return arch.merge(tab, left_on="date", right_index=True, how="left")


if __name__ == "__main__":
    print("Raccolta giornaliera (GitHub Actions)...")
    rows = []
    for t in TICKERS:
        try:
            r = fetch_snapshot(t); rows.append(r)
            print(f"  {t:<6} spot={r['spot']:<9} pc_vol={r['pc_vol']}")
        except Exception as e:
            print(f"  {t:<6} ERRORE: {e}")
    if rows:
        arch = append_rows(ARCHIVE, rows)
        arch = enrich(arch)
        save_archive(arch, ARCHIVE)
        print(f"[ok] archivio: {len(arch)} righe, {arch['date'].nunique()} giorni -> {ARCHIVE}")
    else:
        print("[!] nessuno scatto riuscito.")
