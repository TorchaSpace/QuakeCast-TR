"""Kandilli indirmesinin bağımsız doğrulaması.

Her test siteye farklı parametrelerle yeni sorgu atar ve sonucu indirilen dosyalarla karşılaştırır.
Sonuçlar: data/interim/kandilli_kontrol/<test>.txt (her test bir kez çalışır, kaldığı yerden devam eder)
  parca_*  : aynı dönem farklı parçalarla (aylık / yarı dönem) -> olay kümeleri aynı mı
  kutu_*   : daha geniş kutu (30-50K, 20-50D) -> indirme kutusunun dışında kalan olaylar
  tur_*    : Tip=Deprem ve Tip=Patlatma toplamı = Tip=Hepsi mi; Sm = Patlatma mı
  derinlik_*: negatif derinlik / 999 km üstü var mı
"""
import glob, json, re, sys, time, urllib.request
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import fetch_kandilli as fk
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "interim" / "kandilli_kontrol"
OUT.mkdir(parents=True, exist_ok=True)


def q(b, e, tip="Hepsi", box=(34, 45, 24, 47), dep=(0, 999), mag=(0, 9.9)):
    ck = OUT / "cache" / f"{b}_{e}_{tip}_{'_'.join(map(str, box))}_{dep[0]}_{dep[1]}.json"
    ck.parent.mkdir(exist_ok=True)
    if ck.exists():
        n, rows = json.loads(ck.read_text()); return n, [tuple(r) for r in rows]
    n, rows = _q(b, e, tip, box, dep, mag)
    ck.write_text(json.dumps([n, rows]))
    return n, rows


def _q(b, e, tip, box, dep, mag):
    s = (f"bYear={b.year}&bMont={b.month:02d}&bDay={b.day:02d}&eYear={e.year}&eMont={e.month:02d}&eDay={e.day:02d}"
         f"&EnMin={box[0]}&EnMax={box[1]}&BoyMin={box[2]}&BoyMax={box[3]}&MAGMin={mag[0]}&MAGMax={mag[1]}"
         f"&DerMin={dep[0]}&DerMax={dep[1]}&Tip={tip}&ofName=k{b:%Y%m%d}{e:%Y%m%d}{tip[:1]}{int(time.time())%100000}.txt")
    for attempt in range(6):
        try:
            with urllib.request.urlopen(fk.URL + s, timeout=150) as r:
                html = r.read().decode("windows-1254", errors="replace")
            m = re.search(r"Bulunan:\s*(\d+)", html)
            if not m:
                if "Liste sonu" in html or "bulunamad" in html.lower():
                    return 0, []
                raise RuntimeError("Bulunan yok")
            rows = fk.parse(html)
            if int(m.group(1)) != len(rows):
                raise RuntimeError(f"Bulunan {m.group(1)} != {len(rows)}")
            return int(m.group(1)), rows
        except Exception as ex:
            print(f"   tekrar ({ex})", flush=True); time.sleep(8 * (attempt + 1))
    raise RuntimeError("sorgu başarısız")


def local(b, e, tip=None):
    fs = glob.glob(str(ROOT / "data" / "raw" / "kandilli" / "kandilli_*.txt"))
    d = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in fs])
    dd = pd.to_datetime(d["date"].str.replace(".", "-", regex=False))
    d = d[(dd >= pd.Timestamp(b)) & (dd <= pd.Timestamp(e))]
    if tip:
        d = d[d["type"] == tip]
    return set(d.event_code + "|" + d.lat + "|" + d.lon)


def key(rows):
    return set(r[1] + "|" + r[4] + "|" + r[5] for r in rows)


TESTS = {
    # en yoğun günler: büyük parçalarla (yıllık/çeyreklik) indirilen veri, gün gün yeniden sorgulanır
    "parca_2023_Subat06-07": (date(2023, 2, 6), date(2023, 2, 7), "D"),
    "parca_1999_Kasim12-13": (date(1999, 11, 12), date(1999, 11, 13), "D"),
    "parca_2011_Ekim23-24": (date(2011, 10, 23), date(2011, 10, 24), "D"),
    "parca_2025_Agustos10-11": (date(2025, 8, 10), date(2025, 8, 11), "D"),
    "parca_2020_Ocak24-25": (date(2020, 1, 24), date(2020, 1, 25), "D"),
    "parca_1977_Ocak-Haziran": (date(1977, 1, 1), date(1977, 6, 30), "Y"),
    "parca_2005_Ocak10-16": (date(2005, 1, 10), date(2005, 1, 16), "Y"),
}


def split(b, e, mode):
    parts = []; d = b
    while d <= e:
        if mode == "D":
            n = d + timedelta(days=1)
        elif mode == "M":
            n = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        elif mode == "Y":
            n = date(d.year + 1, 1, 1)
        else:  # H: yarım ay
            n = d + timedelta(days=15) if d.day == 1 else date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        parts.append((d, min(n - timedelta(days=1), e))); d = n
    return parts


def done(name):
    return (OUT / f"{name}.txt").exists()


def save(name, lines):
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def run(name):
    if done(name):
        return
    if name.startswith("parca_"):
        b, e, mode = TESTS[name]
        got = set(); tot = 0
        for pb, pe in split(b, e, mode):
            n, rows = q(pb, pe); tot += n; got |= key(rows)
        loc = local(b, e)
        save(name, [f"{name}: dönem {b}..{e}, {mode}-parçalı yeni sorgu toplamı={tot}, benzersiz={len(got)}, indirilen dosyalarda={len(loc)}",
                    f"  yeni sorguda olup dosyada olmayan: {len(got - loc)}   dosyada olup yeni sorguda olmayan: {len(loc - got)}",
                    "  SONUÇ: " + ("TUTARLI" if got == loc else "FARK VAR")] + [f"  eksik: {x}" for x in sorted(got - loc)[:20]] + [f"  fazla: {x}" for x in sorted(loc - got)[:20]])
    elif name.startswith("kutu_"):
        y = int(name.split("_")[1]); b, e = date(y, 3, 1), date(y, 3, 7)
        n, rows = q(b, e, box=(30, 50, 20, 50))
        loc = local(b, e); got = key(rows)
        out = [r for r in rows if not (34 <= float(r[4]) <= 45 and 24 <= float(r[5]) <= 47)]
        inside = got - set(r[1] + "|" + r[4] + "|" + r[5] for r in out)
        save(name, [f"{name}: geniş kutu (30-50K, 20-50D) {b}..{e}: {n} olay; indirme kutusu dışında: {len(out)}; kutu içi = dosya: {inside == loc} ({len(inside)} / {len(loc)})",
                    "  kutu dışı örnekler: " + "; ".join(f"{r[2]} {r[4]},{r[5]} M{r[7]} {r[14]}" for r in out[:8])])
    elif name.startswith("tur_"):
        # Not: sitenin Tip=Patlatma sorgusu sunucuda takılıyor (1 günlük sorgu bile >90 s); Deprem filtresiyle test edilir
        y = int(name.split("_")[1]); b, e = date(y, 3, 1), date(y, 3, 7)
        nd, rd = q(b, e, tip="Deprem")
        loc = local(b, e); lke = local(b, e, "Ke"); lsm = local(b, e, "Sm")
        save(name, [f"{name}: {b}..{e}  Tip=Deprem sorgusu {nd} olay; dosyada Hepsi {len(loc)} = Ke {len(lke)} + Sm {len(lsm)}",
                    f"  Tip=Deprem kümesi = dosyadaki Ke kümesi: {key(rd) == lke};  Deprem sorgusunda Sm kodlu kayıt: {sum(r[13] == 'Sm' for r in rd)}",
                    "  -> Sm kodlu kayıtlar sitenin 'Deprem' saymadığı (patlatma) kayıtlardır." if key(rd) == lke else "  -> TUTARSIZ"])
    elif name.startswith("derinlik_"):
        y = int(name.split("_")[1]); b, e = date(y, 1, 24), date(y, 1, 26)
        n1, r1 = q(b, e, dep=(-50, 0)); n2, r2 = q(b, e, dep=(999, 9999))
        neg = [r for r in r1 if float(r[6]) < 0]
        save(name, [f"{name}: {b}..{e} derinlik -50..0 sorgusu {n1} olay (bunlardan negatif derinlikli: {len(neg)}), 999..9999 sorgusu {n2} olay"])


if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS) + ["kutu_2010", "kutu_2024", "tur_2015", "tur_2024", "derinlik_2020"]
    t0 = time.time()
    for n in names:
        if time.time() - t0 > 60:
            print("SURE DOLDU"); break
        run(n)
