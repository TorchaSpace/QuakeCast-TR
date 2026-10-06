"""ISC Bülteni (kapsamlı) — Türkiye ve çevresi, tüm kurumların tüm büyüklükleriyle, aylık parçalar.

Kaynak: International Seismological Centre (2026), On-line Bulletin, https://doi.org/10.31905/D808B830
Sorgu: web-db-run, out_format=CATCSV, request=COMPREHENSIVE, 34–45K / 24–47D, min_mag=2.2 (herhangi bir tür),
include_magnitudes=on. Çıktı: data/raw/isc/isc_YYYYMM.csv (ham CATCSV gövdesi). Kaldığı yerden devam eder.
Kullanım: BUDGET=150 python3 src/data/fetch_isc.py <baslangic_yil> <bitis_yil>
"""
import os, sys, time, urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/isc"; OUT.mkdir(parents=True, exist_ok=True)
BASE = ("http://www.isc.ac.uk/cgi-bin/web-db-run?out_format=CATCSV&request=COMPREHENSIVE&searchshape=RECT"
        "&bot_lat=34&top_lat=45&left_lon=24&right_lon=47&min_mag=2.2&req_mag_agcy=Any&req_mag_type=Any&include_magnitudes=on")


def query(b, e):
    u = BASE + (f"&start_year={b.year}&start_month={b.month}&start_day=1&start_time=00:00:00"
                f"&end_year={e.year}&end_month={e.month}&end_day=1&end_time=00:00:00")
    for k in range(4):
        try:
            with urllib.request.urlopen(u, timeout=600) as r:
                txt = r.read().decode("utf-8", errors="replace")
            if "STOP" in txt or "No events were found" in txt:
                return txt
            raise RuntimeError("yanıt tamamlanmamış")
        except Exception as ex:
            print(f"  hata {b}: {ex} (deneme {k+1})", flush=True); time.sleep(20 * (k + 1))
    raise RuntimeError(f"{b} indirilemedi")


def body(txt):
    s = txt.find("EVENTID,TYPE"); e = txt.find("STOP", s)
    return "" if s < 0 else txt[s:e]


if __name__ == "__main__":
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    t0 = time.time(); budget = float(os.environ.get("BUDGET", "1e9")); today = date.today()
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            b = date(y, m, 1); e = date(y + (m == 12), m % 12 + 1, 1)
            if b > today:
                break
            p = OUT / f"isc_{y}{m:02d}.csv"
            if p.exists() and (e < date(today.year, today.month, 1) or time.time() - p.stat().st_mtime < 86400):
                continue
            if time.time() - t0 > budget:
                print("SURE DOLDU", flush=True); sys.exit(0)
            txt = query(b, e); bd = body(txt)
            tmp = p.with_suffix(".tmp"); tmp.write_text(bd, encoding="utf-8"); tmp.rename(p)
            print(f"{y}-{m:02d}: {max(bd.count(chr(10)) - 1, 0)} olay ({time.time()-t0:.0f} s)", flush=True)
    print("ISC BITTI", flush=True)
