"""AFAD web kataloğu (deprem.afad.gov.tr/event-catalog) servisinden tam kayıt indirir.

Public API'ye (fetch_afad.py) göre farkı: eski kayıtlarda hassas koordinat, olay türü
(Deprem / Patlama / Bilinmeyen ...), konum hataları (erh, erz, gap) ve kabuk modeli.
Servis sayfa başına en fazla 20 kayıt veriyor; her ay için indirilen benzersiz kayıt sayısı
sitenin bildirdiği toplamla (totalCount) karşılaştırılır, tutmazsa ay gün gün yeniden çekilir.
Çıktı: data/raw/afad_web/afad_web_YYYYMM.txt (sekme ayrımlı)
"""
import json, math, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "afad_web"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://deprem.afad.gov.tr/EventData/GetEventsByFilter"
PAGE = 20
COLS = ["refId", "id", "eventDate", "latitude", "longitude", "depth", "magnitude", "magnitudeType",
        "eventType", "eventTypeId", "rms", "erh", "erz", "gap", "crustModelId", "location"]


def post(t0: datetime, t1: datetime, skip: int, take: int, etype=None):
    fl = [{"FilterType": 8, "Value": t0.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
          {"FilterType": 9, "Value": t1.strftime("%Y-%m-%dT%H:%M:%S.000Z")}]
    if etype is not None:
        fl.append({"FilterType": 13, "Value": str(etype)})
    body = {"EventSearchFilterList": fl,
            "Skip": skip, "Take": take, "SortDescriptor": {"field": "eventDate", "dir": "asc"}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except Exception as ex:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"istek başarısız {t0} {skip}")


def fetch_window(t0, t1, pool, etype=None):
    n = post(t0, t1, 0, 1, etype)["totalCount"]
    pages = math.ceil(n / PAGE)
    res = list(pool.map(lambda k: post(t0, t1, k * PAGE, PAGE, etype), range(pages)))
    ev = {}
    for j in res:
        for e in j.get("eventList") or []:
            ev[e["id"]] = e
    return n, ev


def fetch_month(b: date, pool):
    e = date(b.year + (b.month == 12), b.month % 12 + 1, 1)
    t0, t1 = datetime(b.year, b.month, 1), datetime(e.year, e.month, 1)
    n, ev = fetch_window(t0, t1, pool)
    if len(ev) != n:  # sıralama kararsızlığı -> gün gün çek
        ev = {}; tot = 0; d = t0
        while d < t1:
            dn, dev = fetch_window(d, d + timedelta(days=1), pool)
            if len(dev) != dn:
                raise RuntimeError(f"{d:%Y-%m-%d}: {len(dev)} != {dn}")
            ev.update(dev); tot += dn; d += timedelta(days=1)
        n = tot
    return n, ev


def clean(v):
    return "" if v is None else str(v).replace("\t", " ").replace("\n", " ")


def write(p, evs):
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        for e in sorted(evs, key=lambda e: e["eventDate"]):
            f.write("\t".join(clean(e.get(c)) for c in COLS) + "\n")
    tmp.rename(p)


def mode_types(years, pool):
    """Deprem dışı olaylar (1 Bilinmeyen, 3 Patlama, 4 Göçük, 5 Volkanik), aylık dosyalar."""
    budget = float(os.environ.get("BUDGET", "1e9")); t_start = time.time()
    for yr in years:
        if (OUT / f"afad_web_digertur_{yr}.txt").exists():
            continue
        for mo in range(1, 13):
            t0 = datetime(yr, mo, 1); t1 = datetime(yr + (mo == 12), mo % 12 + 1, 1)
            if t0.date() > date.today():
                break
            p = OUT / f"afad_web_digertur_{yr}{mo:02d}.txt"
            if p.exists() and t1.date() <= date.today():
                continue
            if time.time() - t_start > budget:
                print("SURE DOLDU", flush=True); return
            allev = {}
            for et in (1, 3, 4, 5):
                n, ev = fetch_window(t0, t1, pool, et)
                if len(ev) != n:
                    raise RuntimeError(f"{yr}-{mo} tür {et}: {len(ev)} != {n}")
                allev.update(ev)
            write(p, allev.values())
            print(f"{yr}-{mo:02d} deprem-dışı: {len(allev)}", flush=True)


def mode_counts(pool):
    """Her ay için sitenin bildirdiği toplam olay sayısı."""
    p = OUT / "afad_web_aylik_toplamlar.txt"
    months = []; d = date(1990, 1, 1)
    while d <= date.today():
        months.append(d); d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    def cnt(b):
        e = date(b.year + (b.month == 12), b.month % 12 + 1, 1)
        return b, post(datetime(b.year, b.month, 1), datetime(e.year, e.month, 1), 0, 1)["totalCount"]
    res = list(pool.map(cnt, months))
    p.write_text("ay\tsite_toplam\n" + "".join(f"{b:%Y-%m}\t{n}\n" for b, n in res), encoding="utf-8")
    print("aylik toplamlar yazildi", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("types", "counts"):
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        if sys.argv[1] == "counts":
            mode_counts(pool)
        else:
            mode_types([int(a) for a in sys.argv[2:]], pool)
    sys.exit(0)

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1990
    budget = float(os.environ.get("BUDGET", "1e9")); t_start = time.time()
    today = date.today(); y, m = start, 1
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        end_year = int(os.environ.get("END_YEAR", "9999"))
        while date(y, m, 1) <= today and y <= end_year:
            b = date(y, m, 1); y, m = (y + (m == 12), m % 12 + 1)
            p = OUT / f"afad_web_{b:%Y%m}.txt"
            if p.exists() and date(y, m, 1) <= today:
                continue
            if time.time() - t_start > budget:
                print("SURE DOLDU", flush=True); sys.exit(0)
            n, ev = fetch_month(b, pool)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\t".join(COLS) + "\n")
                for e in sorted(ev.values(), key=lambda e: e["eventDate"]):
                    f.write("\t".join(clean(e.get(c)) for c in COLS) + "\n")
            tmp.rename(p)
            print(f"{b:%Y-%m}: site={n} indirilen={len(ev)}", flush=True)
    print("AFAD_WEB BITTI", flush=True)
