"""Prospektif (canlı) katalog: dondurulmuş geçmiş + son dönemin AYNI kurallarla artımlı yeniden kurulumu.

- T_CUT öncesi: v6 dondurulduğunda kullanılan ETAS girdi kataloğu (data/processed/prospektif/model_v6/etas_girdi_dondurulmus.csv).
- T_CUT sonrası: AFAD + Kandilli ham verisi yeniden indirilir; eşleştirme dondurulmuş kalibrasyonla (match_catalogs.run_pair,
  par = eslestirme_kalibrasyonu_v6.json), birleştirme dondurulmuş GOR dönüşümleriyle (build_catalog.build_events,
  conv = gor_donusumleri_v6.json) yapılır; filtre: olay_turu = deprem, Mw >= 2.5 (araştırma kataloğuyla aynı).
Çıktı: data/processed/prospektif/etas_girdi_canli.csv
Kullanım: python3 src/prospective/live_catalog.py [--indirme-yok]
"""
import hashlib, json, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
import match_catalogs as mc  # noqa: E402
import build_catalog as BC  # noqa: E402

PK = ROOT / "data/processed/prospektif"
T_CUT = pd.Timestamp("2026-08-01")  # Kandilli veritabanı 31 Temmuz 2026'da bitiyordu (~2 ay gecikme)
OUT = PK / "etas_girdi_canli.csv"


def fetch_recent():
    import fetch_afad as FA, fetch_kandilli as FK
    today = date.today()
    y, m = T_CUT.year, T_CUT.month
    while date(y, m, 1) <= today:
        b = date(y, m, 1); y2, m2 = (y + (m == 12), m % 12 + 1); e = date(y2, m2, 1)
        rows = FA.query(b, e)
        p = FA.OUT / f"afad_{b:%Y%m}.txt"; tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\t".join(FA.COLS) + "\n")
            for r in rows:
                f.write("\t".join(FA.clean(r.get(c)) for c in FA.COLS) + "\n")
        tmp.rename(p); print(f"AFAD {b:%Y-%m}: {len(rows)}", flush=True)
        y, m = y2, m2
    # AFAD web kataloğundan deprem-dışı olay türleri (patlatma vb.) — aynı aylar yeniden
    import fetch_afad_web as FW
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    with ThreadPoolExecutor(4) as pool:
        y, m = T_CUT.year, T_CUT.month
        while date(y, m, 1) <= today:
            t0 = datetime(y, m, 1); y2, m2 = (y + (m == 12), m % 12 + 1); t1 = datetime(y2, m2, 1)
            allev = {}
            try:
                for et in (1, 3, 4, 5):
                    n, ev = FW.fetch_window(t0, t1, pool, et)
                    allev.update(ev)
                FW.write(FW.OUT / f"afad_web_digertur_{y}{m:02d}.txt", allev.values())
                print(f"AFAD web deprem-dışı {y}-{m:02d}: {len(allev)}", flush=True)
            except Exception as ex:
                print(f"UYARI AFAD web {y}-{m:02d}: {ex} (mevcut dosya korunuyor)", flush=True)
            y, m = y2, m2
    # Kandilli ~2 ay gecikmeli yayınlıyor: T_CUT'tan bugüne kadar her çeyrek baştan indirilir (yeni yayınlananlar girsin)
    from datetime import timedelta
    for yy in range(T_CUT.year, today.year + 1):
        for q in range(4):
            b = date(yy, 3 * q + 1, 1)
            e = (date(yy + 1, 1, 1) if q == 3 else date(yy, 3 * q + 4, 1)) - timedelta(days=1)
            if e < T_CUT.date() or b > today:
                continue
            rows = kandilli_query(FK, b, min(e, today))
            if rows is None:   # henüz yayınlanmamış / sunucu yanıt vermedi -> mevcut dosya korunur
                print(f"Kandilli {yy}Q{q+1}: veri yok ya da yanıt yok (mevcut dosya korunuyor)", flush=True); continue
            FK.save(FK.OUT / f"kandilli_{yy}Q{q+1}.txt", rows)
            print(f"Kandilli {yy}Q{q+1}: {len(rows)}", flush=True)


def kandilli_query(FK, b, e, attempts=2):
    """FK.query'nin kısa sürümü: 'Bulunan' yoksa (yayınlanmamış dönem) None döner, süreci durdurmaz."""
    import re, time, urllib.request
    q = (f"bYear={b.year}&bMont={b.month:02d}&bDay={b.day:02d}&eYear={e.year}&eMont={e.month:02d}&eDay={e.day:02d}"
         f"&EnMin=34&EnMax=45&BoyMin=24&BoyMax=47&MAGMin=0&MAGMax=9.9&DerMin=0&DerMax=999&Tip=Hepsi&ofName=qc{b:%Y%m%d}{e:%Y%m%d}.txt")
    for k in range(attempts):
        try:
            with urllib.request.urlopen(FK.URL + q, timeout=300) as r:
                html = r.read().decode("windows-1254", errors="replace")
            if re.search(r"Bulunan:\s*(\d+)", html):
                return FK.parse(html)
        except Exception as ex:
            print(f"  Kandilli hata {b}..{e}: {ex}", flush=True)
        time.sleep(5)
    return None


def frozen_par():
    J = json.load(open(PK / "eslestirme_kalibrasyonu_v6.json"))
    return {int(k): v for k, v in J.items()}


def build():
    log = []
    A = mc.drop_exact_duplicates(mc.load_afad().dropna(subset=["time", "lat", "lon"]), "AFAD", log)
    B = mc.drop_exact_duplicates(mc.load_kandilli().dropna(subset=["time", "lat", "lon"]), "KANDILLI", log)
    buf = T_CUT - pd.Timedelta(days=2)
    As = A[A.time >= buf].reset_index(drop=True); Bs = B[B.time >= buf].reset_index(drop=True)
    if len(Bs) and len(As):
        m = mc.run_pair(As, Bs, "AFAD", "KANDILLI", par=frozen_par())
        pairs = pd.DataFrame({"AFAD_id": As.id.values[m["ia"]], "KANDILLI_id": Bs.id.values[m["ib"]]})
    else:
        pairs = pd.DataFrame({"AFAD_id": pd.Series([], dtype=str), "KANDILLI_id": pd.Series([], dtype=str)})
    conv = json.load(open(PK / "gor_donusumleri_v6.json"))
    out, _ = BC.build_events(As, Bs, pairs, conv=conv)
    out = out[(pd.to_datetime(out.time) >= T_CUT) & (out.olay_turu == "deprem") & (out.Mw >= 2.5)]
    new = pd.DataFrame({"time": out.time, "latitude": out.lat, "longitude": out.lon, "depth": out.depth,
                        "magnitude": out.Mw, "Mw_kaynak": out.Mw_kaynak, "kaynaklar": out.kaynaklar})
    hist = pd.read_csv(PK / "model_v6/etas_girdi_dondurulmus.csv")
    hist = hist[pd.to_datetime(hist.time) < T_CUT]
    live = pd.concat([hist, new], ignore_index=True)
    live = live.iloc[np.argsort(pd.to_datetime(live.time).values, kind="stable")]
    live.to_csv(OUT, index=False)
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    kt = B.time.max()
    info = dict(son_olay=str(live.time.iloc[-1]), n=len(live), n_yeni=len(new), sha256=h, kandilli_son=str(kt),
                kaynaklar_yeni=new.kaynaklar.value_counts().to_dict(),
                n_M35_son30g=int(((pd.to_datetime(live.time) > pd.to_datetime(live.time.iloc[-1]) - pd.Timedelta(days=30)) & (live.magnitude >= 3.45)).sum()))
    json.dump(info, open(PK / "etas_girdi_canli.json", "w"), indent=1)
    print(info, flush=True)
    return info


if __name__ == "__main__":
    if "--indirme-yok" not in sys.argv:
        fetch_recent()
    build()
