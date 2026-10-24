#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  collect.py v4 - raccolta giornaliera per GitHub Actions
#
#  NOVITA' v4 (correzioni importanti):
#   1) PREZZO ROBUSTO: lo prende dalla catena opzioni ("underlying"), che su
#      GitHub funziona anche quando la richiesta dei prezzi storici fallisce.
#      Ripiego: storico prezzi. Con 3 tentativi per ogni richiesta.
#   2) SEDUTA ESPLICITA: salva 'session' (il giorno di borsa a cui si
#      riferiscono i dati) e 'run_utc' (l'ora esatta dell'esecuzione).
#      Cosi' la data della riga non puo' piu' essere ambigua.
#   3) PROTEZIONE: se parte mentre la borsa USA e' aperta (dati parziali),
#      non salva niente ed esce.
#   4) Meno richieste a Yahoo: gli indici (VIX, S&P...) non vengono piu'
#      scaricati qui, li scarica il modello in Colab.
#  Le righe vecchie dell'archivio restano intatte.
#  Idempotente. Didattico, NON consulenza finanziaria.
# ============================================================================
import os, sys, time, warnings, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

ARCHIVE       = "pc_archive.csv"
MAX_DTE       = 60          # NON cambiare: deve restare uguale ai dati passati
MAX_EXPIRIES  = 8           # NON cambiare: idem
IV_TARGET_DTE = 30
NY = ZoneInfo("America/New_York")

TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","XOM","WMT"]
BASE_COLS = ["date","session","run_utc","ticker","spot","open","spot_src","market_state",
             "call_vol","put_vol","call_oi","put_oi","call_prem","put_prem",
             "pc_vol","pc_oi","n_expiries","atm_iv","iv_skew","exp_move"]


# ----------------------------- utilita' -------------------------------------
def retry(fn, tries=3, wait=4):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(wait * (i + 1))
    raise last


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _to_ny_date(t):
    """regularMarketTime puo' arrivare come epoch (int) o come datetime."""
    if t is None:
        return None
    if isinstance(t, (int, float, np.integer, np.floating)):
        return dt.datetime.fromtimestamp(int(t), tz=NY).date()
    if hasattr(t, "astimezone"):
        try:
            return t.astimezone(NY).date()
        except Exception:
            return None
    try:
        return pd.Timestamp(t).tz_convert(NY).date()
    except Exception:
        return None


# ----------------------------- archivio -------------------------------------
def load_archive(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=BASE_COLS)
    return pd.read_csv(path, dtype={"date": str, "session": str, "run_utc": str})


def save_archive(df, path):
    cols = [c for c in BASE_COLS if c in df.columns] + [c for c in df.columns if c not in BASE_COLS]
    df[cols].sort_values(["date", "ticker"]).to_csv(path, index=False)


def append_rows(path, rows):
    """Aggiunge le righe nuove. Sostituisce le vecchie con stessa (data, ticker)
    oppure stessa (seduta, ticker): niente doppioni anche se rilanci."""
    old = load_archive(path)
    new = pd.DataFrame(rows)
    if old.empty:
        return new
    k_date = set(zip(new["date"], new["ticker"]))
    k_sess = set((s, t) for s, t in zip(new["session"], new["ticker"]) if isinstance(s, str))
    keep = []
    for _, r in old.iterrows():
        same_day = (r["date"], r["ticker"]) in k_date
        s = r.get("session") if "session" in old.columns else None
        same_sess = isinstance(s, str) and (s, r["ticker"]) in k_sess
        keep.append(not (same_day or same_sess))
    old = old[keep]
    return new if old.empty else pd.concat([old, new], ignore_index=True)


# ----------------------------- IV ------------------------------------------
def _iv_at_strike(df, target):
    if df is None or df.empty or "impliedVolatility" not in df.columns:
        return np.nan
    d = pd.DataFrame({"strike": pd.to_numeric(df["strike"], errors="coerce"),
                      "iv": pd.to_numeric(df["impliedVolatility"], errors="coerce")}).dropna()
    d = d[(d["iv"] > 0.01) & (d["iv"] < 5.0)]
    if d.empty:
        return np.nan
    return float(d.iloc[(d["strike"] - target).abs().argmin()]["iv"])


# ----------------------------- scatto --------------------------------------
def fetch_snapshot(ticker, run_utc):
    tk = yf.Ticker(ticker)
    today = run_utc.date()                      # stesso riferimento delle versioni precedenti
    expiries = retry(lambda: list(tk.options or []))
    keep = []
    for e in expiries:
        try:
            d = dt.date.fromisoformat(e)
        except ValueError:
            continue
        if 0 <= (d - today).days <= MAX_DTE:
            keep.append((e, (d - today).days))
    keep = keep[:MAX_EXPIRIES]

    cv = pv = coi = poi = cprem = pprem = 0.0
    chains, underlying = {}, None
    for e, dte in keep:
        try:
            ch = retry(lambda: tk.option_chain(e))
        except Exception:
            continue
        chains[e] = ch
        if underlying is None:
            u = getattr(ch, "underlying", None)
            if isinstance(u, dict) and u:
                underlying = u
        for side, d in (("c", ch.calls), ("p", ch.puts)):
            if d is None or d.empty:
                continue
            vol = pd.to_numeric(d.get("volume"), errors="coerce").fillna(0)
            oi = pd.to_numeric(d.get("openInterest"), errors="coerce").fillna(0)
            last = pd.to_numeric(d.get("lastPrice"), errors="coerce").fillna(0)
            prem = (vol * last * 100).sum()
            if side == "c":
                cv += vol.sum(); coi += oi.sum(); cprem += prem
            else:
                pv += vol.sum(); poi += oi.sum(); pprem += prem
        time.sleep(0.3)

    # --- prezzo e seduta: 1) dalla catena opzioni  2) dallo storico ---
    spot = open_px = np.nan; src = "nessuno"; session = None; state = None
    if underlying:
        state = underlying.get("marketState")
        p = _num(underlying.get("regularMarketPrice"))
        s = _to_ny_date(underlying.get("regularMarketTime"))
        if np.isfinite(p) and s is not None:
            spot, session, src = p, s, "opzioni"
            open_px = _num(underlying.get("regularMarketOpen"))
    if not np.isfinite(spot):
        try:
            h = retry(lambda: tk.history(period="10d", auto_adjust=False))
            if h is not None and not h.empty:
                spot = float(h["Close"].iloc[-1])
                open_px = float(h["Open"].iloc[-1])
                ix = pd.Timestamp(h.index[-1])
                session = (ix.tz_convert(NY) if ix.tzinfo else ix).date()
                src = "storico"
        except Exception:
            pass

    # --- IV sulla scadenza piu' vicina a 30 giorni ---
    atm_iv = iv_skew = exp_move = np.nan
    if keep and np.isfinite(spot):
        ref_e, ref_dte = min(keep, key=lambda x: abs(x[1] - IV_TARGET_DTE))
        ch = chains.get(ref_e)
        if ch is not None:
            vals = [v for v in (_iv_at_strike(ch.calls, spot), _iv_at_strike(ch.puts, spot))
                    if np.isfinite(v)]
            if vals:
                atm_iv = float(np.mean(vals))
                exp_move = float(atm_iv * np.sqrt(max(ref_dte, 1) / 365.0))
            ip, ic = _iv_at_strike(ch.puts, spot * 0.95), _iv_at_strike(ch.calls, spot * 1.05)
            if np.isfinite(ip) and np.isfinite(ic):
                iv_skew = float(ip - ic)

    r4 = lambda x: round(x, 4) if np.isfinite(x) else np.nan
    return {"date": today.isoformat(),
            "session": session.isoformat() if session else np.nan,
            "run_utc": run_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ticker": ticker, "spot": r4(spot), "open": r4(open_px),
            "spot_src": src, "market_state": state,
            "call_vol": int(cv), "put_vol": int(pv), "call_oi": int(coi), "put_oi": int(poi),
            "call_prem": round(cprem, 2), "put_prem": round(pprem, 2),
            "pc_vol": round(pv / cv, 4) if cv > 0 else np.nan,
            "pc_oi": round(poi / coi, 4) if coi > 0 else np.nan,
            "n_expiries": len(keep),
            "atm_iv": r4(atm_iv), "iv_skew": r4(iv_skew), "exp_move": r4(exp_move)}


def borsa_aperta(now_utc):
    ny = now_utc.astimezone(NY)
    return ny.weekday() < 5 and dt.time(9, 25) <= ny.time() < dt.time(16, 15)


def main():
    now = dt.datetime.now(dt.timezone.utc)
    print(f"Raccolta v4 - esecuzione {now:%Y-%m-%d %H:%M} UTC")
    if borsa_aperta(now):
        print("[!] Borsa USA aperta: i dati sarebbero parziali. Non salvo nulla.")
        print("    Nessun problema: la prossima esecuzione automatica raccogliera' i dati.")
        return
    rows = []
    for t in TICKERS:
        try:
            r = fetch_snapshot(t, now)
            if r["market_state"] == "REGULAR":
                print(f"  {t:<6} saltato: mercato in corso secondo Yahoo"); continue
            rows.append(r)
            print(f"  {t:<6} seduta={r['session']} spot={r['spot']} ({r['spot_src']}) "
                  f"pc_vol={r['pc_vol']} iv={r['atm_iv']}")
        except Exception as e:
            print(f"  {t:<6} ERRORE: {e}")
        time.sleep(1.0)
    if not rows:
        print("[!] Nessuno scatto riuscito. Riprovera' la prossima esecuzione.")
        return
    arch = append_rows(ARCHIVE, rows)
    save_archive(arch, ARCHIVE)
    n_spot = sum(1 for r in rows if np.isfinite(r["spot"]))
    print(f"[ok] {len(rows)} ticker salvati ({n_spot} con prezzo) | archivio: {len(arch)} righe")


if __name__ == "__main__":
    main()
