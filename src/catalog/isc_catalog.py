"""ISC Bülteni'nden TURHEC benzeri homojen Mw* kataloğu (Tan 2021, NHESS yöntemine benzer).

1) Ham CATCSV satırları ayrıştırılır: ISC asal (prime) hiposantr + tüm kurumların (kurum, tür, değer) büyüklükleri.
2) Olay türü: 'ke' (bilinen deprem) ve 'se' (şüpheli deprem); patlatma/maden (x, m, h…) dışarıda.
3) Doğrudan Mw referansı: moment tensörü kaynaklı Mw'lerin medyanı (GCMT MW, NEIC Mww/Mwc/Mwb/Mwr, GFZ Mw,
   MED_RCMT Mw, AFAD MW, ATH MW/Mw, THE Mw, ISK MW/Mw, IPGP Mw, ROM Mw).
4) Diğer türler için Mw = a + b·M genel ortogonal regresyon (η=1), kurum × tür × dönem (≤2006, 2007–2013, ≥2014)
   bazında, en az 100 çift; çift bulunamazsa dönemsiz.
5) Olay Mw*: doğrudan Mw varsa o; yoksa dönüştürülmüş değerlerin ters-varyans ağırlıklı ortalaması
   (en düşük σ'lı en fazla 3 tür). Kaynak ve σ saklanır.
Çıktılar: data/processed/isc/isc_olaylar.parquet-benzeri csv, isc_donusumler.json, isc_katalog_Mw.csv
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/isc"; OUT = ROOT / "data/processed/isc"; OUT.mkdir(parents=True, exist_ok=True)
DIRECT = {("GCMT", "MW"), ("NEIC", "MWW"), ("NEIC", "MWC"), ("NEIC", "MWB"), ("NEIC", "MWR"), ("NEIC", "MW"),
          ("GFZ", "MW"), ("MED_RCMT", "MW"), ("AFAD", "MW"), ("ATH", "MW"), ("THE", "MW"), ("ISK", "MW"),
          ("IPGP", "MW"), ("ROM", "MW"), ("USGS", "MWW")}
ERAS = [(1900, 2006), (2007, 2013), (2014, 2100)]


def parse():
    rows, mags = [], []
    for f in sorted(glob.glob(str(RAW / "isc_*.csv"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip() or line.startswith("EVENTID"):
                continue
            p = [x.strip() for x in line.rstrip("\n").split(",")]
            if len(p) < 9 or not p[0].isdigit():
                continue
            eid = int(p[0])
            rows.append((eid, p[1], p[2], p[3] + "T" + p[4], p[5], p[6], p[7], p[8]))
            for i in range(9, len(p) - 2, 3):
                if p[i] and p[i + 1] and p[i + 2]:
                    try:
                        mags.append((eid, p[i], p[i + 1].upper(), float(p[i + 2])))
                    except ValueError:
                        pass
    ev = pd.DataFrame(rows, columns=["eid", "tur", "yazar", "zaman", "enlem", "boylam", "derinlik", "derinlik_sabit"]).drop_duplicates("eid")
    ev["zaman"] = pd.to_datetime(ev.zaman, errors="coerce", format="mixed")
    for c in ["enlem", "boylam", "derinlik"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    mg = pd.DataFrame(mags, columns=["eid", "kurum", "tip", "M"]).drop_duplicates()
    mg["tip"] = mg.tip.replace({"MD": "MD", "ML": "ML", "MS": "MS", "MB": "MB"})
    return ev, mg


def gor(x, y):
    mx, my = x.mean(), y.mean()
    sxx, syy, sxy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean(), ((x - mx) * (y - my)).mean()
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy); a = my - b * mx
    s = float(np.std(y - (a + b * x)))
    return dict(a=float(a), b=float(b), n=int(len(x)), s=s, lo=float(np.percentile(x, 1)), hi=float(np.percentile(x, 99)))


def era(y):
    return np.select([y <= 2006, y <= 2013], [0, 1], 2)


def main():
    ev, mg = parse()
    ev = ev[ev.tur.isin(["ke", "se"]) & ev.zaman.notna()].reset_index(drop=True)
    mg = mg[mg.eid.isin(ev.eid)]
    yr = ev.set_index("eid").zaman.dt.year
    mg["donem"] = era(yr.reindex(mg.eid).values)
    key = list(zip(mg.kurum, mg.tip))
    mg["dogrudan"] = [k in DIRECT for k in key]
    # --- TURHEC yöntemi (Tan 2021, NHESS): ölçek türü bazında kurum ortalaması, birleşik GOR denklemleri ---
    ref = mg[mg.dogrudan].groupby("eid").M.median()                       # gözlenen Mw (moment tensörü)
    def scale(t):
        t = t.upper()
        if t in ("ML", "MLH", "MLV", "ML1", "MLR"): return "ML"
        if t in ("MD", "MD1", "MDL"): return "MD"
        if t in ("MB", "MBTMP", "MB1", "MB1MX", "MB_BB", "MBLG"): return "MB" if t in ("MB", "MBTMP") else None
        if t in ("MS", "MS_20", "MS1", "MS7", "MS1MX"): return "MS" if t in ("MS", "MS_20") else None
        if t in ("M",): return "M"
        return None
    o = mg[~mg.dogrudan].copy(); o["olcek"] = o.tip.map(scale); o = o.dropna(subset=["olcek"])
    avg = o.groupby(["eid", "olcek"]).M.mean().unstack()                 # olay × ölçek: kurumlar ortalaması
    MC_TYPE = {"ML": 2.8, "MD": 2.8, "MB": 4.0, "MS": 4.0, "M": 2.8}
    CUT = {"MD": 0.85, "ML": 0.67, "MB": 0.57, "MS": 1.1, "M": 0.9}
    conv = {}
    for sc in avg.columns:
        x = avg[sc]; r = ref.reindex(avg.index)
        ok = x.notna() & r.notna() & (x >= MC_TYPE[sc]) & ((x - r).abs() <= CUT[sc])
        if ok.sum() < 100:
            continue
        xx, yy = x[ok].values.astype(float), r[ok].values.astype(float)
        c = gor(xx, yy)
        res = yy - (c["a"] + c["b"] * xx); keep = np.abs(res) <= 2 * c["s"]     # ±2σ aykırı ayıklama
        conv[sc] = gor(xx[keep], yy[keep])
    ev = ev.set_index("eid")
    mw = ref.reindex(ev.index); kay = pd.Series(np.where(mw.notna(), "Mw", ""), index=ev.index); sig = pd.Series(np.where(mw.notna(), 0.1, np.nan), index=ev.index)
    for sc in ["MS", "MB", "ML", "MD", "M"]:                              # doyma sırası önceliği
        if sc not in conv or sc not in avg.columns:
            continue
        c = conv[sc]; x = avg[sc].reindex(ev.index)
        # Ms ve mb yalnız kendi tamlık eşiğinin üstünde kullanılır (küçük olaylarda seçilim yanlılığı)
        use = mw.isna() & x.notna() & (x >= (MC_TYPE[sc] if sc in ("MS", "MB") else -9))
        mw[use] = c["a"] + c["b"] * x[use]; kay[use] = sc + "->Mw"; sig[use] = c["s"]
    ev["Mw"] = mw.round(2); ev["Mw_kaynak"] = kay; ev["Mw_sigma"] = sig.round(3)
    ev = ev.reset_index()
    ev = ev.dropna(subset=["Mw"]).sort_values("zaman")
    ev.to_csv(OUT / "isc_katalog_Mw.csv", index=False)
    json.dump(conv, open(OUT / "isc_donusumler.json", "w"), indent=1)
    rep = [f"ISC Mw* kataloğu: {len(ev):,} deprem (ke/se), {ev.zaman.min()} → {ev.zaman.max()}",
           "  Mw kaynağı: " + ", ".join(f"{k}={v:,}" for k, v in ev.Mw_kaynak.value_counts().items()),
           f"  TURHEC yöntemi: ölçek başına kurum ortalaması, birleşik GOR (Mc ML/md 2.8, mb/Ms 4.0; ±2σ ayıklama); öncelik Mw>Ms>mb>ML>md>M"]
    for k, c in sorted(conv.items(), key=lambda kv: -kv[1]["n"])[:40]:
        rep.append(f"    {k:4s} Mw = {c['a']:+.3f} + {c['b']:.3f}·M  n={c['n']:6d} σ={c['s']:.2f} aralık {c['lo']:.1f}-{c['hi']:.1f}")
    (OUT / "ISC_KATALOG_RAPORU.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
