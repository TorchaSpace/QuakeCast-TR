"""Kandilli (KOERI) deprem kataloğunu aylık parçalar halinde indirir.

Çıktı: data/raw/kandilli/kandilli_YYYYMM.txt (+ 1900-1989 tek parça), sekme ayrımlı.
Kullanım: python3 src/data/fetch_kandilli.py [baslangic_yil]
"""
import os, re, sys, time, urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "kandilli"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://www.koeri.boun.edu.tr/sismo/zeqdb/submitRecSearchT.asp?"
COLS = ["no", "event_code", "date", "time", "lat", "lon", "depth", "xM", "MD", "ML", "Mw", "Ms", "Mb", "type", "location"]
START = re.compile(r"(?=\d{6}\t\d{14}\t)")


def parse(html: str) -> list:
    i = html.find("Bulunan")
    j = html.find("clerWaitMessage", i)
    block = html[i:j if j > 0 else None]
    rows = []
    for rec in START.split(block)[1:]:
        f = [x.strip() for x in re.sub(r"<[^>]*>", " ", rec).split("\t")]
        if len(f) < 15:
            continue
        rows.append(tuple(f[:14] + [" ".join(f[14:]).strip()]))
    return rows


def query(b: date, e: date) -> list:
    q = (f"bYear={b.year}&bMont={b.month:02d}&bDay={b.day:02d}&eYear={e.year}&eMont={e.month:02d}&eDay={e.day:02d}"
         f"&EnMin=34&EnMax=45&BoyMin=24&BoyMax=47&MAGMin=0&MAGMax=9.9&DerMin=0&DerMax=999&Tip=Hepsi&ofName=qc{b:%Y%m%d}{e:%Y%m%d}.txt")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(URL + q, timeout=300) as r:
                html = r.read().decode("windows-1254", errors="replace")
            m = re.search(r"Bulunan:\s*(\d+)", html)
            rows = parse(html)
            if not m:
                raise RuntimeError("yanıtta 'Bulunan' yok (boş/bozuk yanıt)")
            n = int(m.group(1))
            if n != len(rows):
                print(f"  UYARI {b}..{e}: Bulunan={n} ayrıştırılan={len(rows)}", flush=True)
            return rows
        except Exception as ex:  # ağ hatası -> tekrar dene
            print(f"  hata {b}..{e}: {ex} (deneme {attempt+1})", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{b}..{e} indirilemedi")


def save(path: Path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    tmp.rename(path)


def fetch_year(y: int):
    """2016 öncesi yıllık, sonrası çeyrek yıllık parça (sunucu büyük sorgularda kopuyor)."""
    from datetime import timedelta
    today = date.today()
    if y < 2016:
        parts = [(f"{y}", date(y, 1, 1), date(y, 12, 31))]
    else:
        parts = []
        for q in range(4):
            b = date(y, 3 * q + 1, 1)
            e = (date(y + 1, 1, 1) if q == 3 else date(y, 3 * q + 4, 1)) - timedelta(days=1)
            if b <= today:
                parts.append((f"{y}Q{q+1}", b, min(e, today)))
    msgs = []
    for tag, b, e in parts:
        p = OUT / f"kandilli_{tag}.txt"
        if p.exists() and e < today - timedelta(days=1):
            msgs.append(f"{tag}: zaten var"); continue
        rows = query(b, e)
        save(p, rows)
        msgs.append(f"{tag}: {len(rows)}")
        print(msgs[-1], flush=True)
    return " | ".join(msgs)


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    old = OUT / "kandilli_1900_1989.txt"
    if not old.exists():
        save(old, query(date(1900, 1, 1), date(1989, 12, 31)))
        print("1900-1989 tamam", flush=True)
    years = [int(a) for a in sys.argv[1:]] or list(range(1990, date.today().year + 1))
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "1"))) as ex:
        for msg in ex.map(fetch_year, years):
            print(msg, flush=True)
    print("KANDILLI BITTI", flush=True)
