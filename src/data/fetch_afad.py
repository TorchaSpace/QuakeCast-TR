"""AFAD Event Web Service'ten deprem kataloğunu aylık parçalar halinde indirir (1990 -> bugün).

Çıktı: data/raw/afad/afad_YYYYMM.txt, sekme ayrımlı.
"""
import json, os, sys, time, urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "afad"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://servisnet.afad.gov.tr/apigateway/deprem/apiv2/event/filter"
COLS = ["eventID", "date", "latitude", "longitude", "depth", "type", "magnitude", "rms",
        "location", "country", "province", "district", "neighborhood", "isEventUpdate", "lastUpdateDate"]


def query(b: date, e: date):
    u = f"{URL}?start={b:%Y-%m-%d}T00:00:00&end={e:%Y-%m-%d}T00:00:00&format=json&orderby=time"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(u, timeout=300) as r:
                return json.load(r)
        except Exception as ex:
            print(f"  hata {b}: {ex} (deneme {attempt+1})", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{b} indirilemedi")


def clean(v):
    return "" if v is None else str(v).replace("\t", " ").replace("\n", " ")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1990
    today = date.today()
    y, m = start, 1
    t0 = time.time(); budget = float(os.environ.get("BUDGET", "1e9"))
    while date(y, m, 1) <= today:
        if time.time() - t0 > budget:
            print("SURE DOLDU", flush=True); sys.exit(0)
        b = date(y, m, 1)
        y, m = (y + (m == 12), m % 12 + 1)
        e = date(y, m, 1)
        p = OUT / f"afad_{b:%Y%m}.txt"
        if p.exists() and e <= today:
            continue
        rows = query(b, e)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\t".join(COLS) + "\n")
            for r in rows:
                f.write("\t".join(clean(r.get(c)) for c in COLS) + "\n")
        tmp.rename(p)
        print(f"{b:%Y-%m}: {len(rows)}", flush=True)
        time.sleep(0.5)
    print("AFAD BITTI", flush=True)
