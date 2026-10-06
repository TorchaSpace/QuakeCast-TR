"""Birleşik katalog v2 (araştırma; dondurulmuş v1/prospektif dosyalarına dokunmaz).

v1'den farkı (ISC karşılaştırmasıyla bulunan hata): AFAD web kataloğu 2011–2012'de çok sayıda gerçek depremi
"Bilinmeyen" türüyle işaretlemiş; v1 kuralı Kandilli 'Deprem' dese bile bunları 'bilinmeyen' yapıp ETAS girdisinden
düşürüyordu (M>=3.5'te 2011: 858, 2012: 497 olay; Van 2011 artçılarının çoğu). v2 kuralı:
  patlatma: AFAD 'Patlama' ya da Kandilli 'Patlatma(Sm)' (v1 ile aynı)
  bilinmeyen: AFAD 'Bilinmeyen' VE Kandilli 'Deprem' değil VE ISC'de deprem (ke/se) olarak yok
  diğerleri: deprem
İsteğe bağlı (--isc-ekle): ISC'de olup bizde olmayan depremler (ISC Mw*, yıl bazında bizim ölçeğe kaydırılarak).
Çıktılar: data/processed/v2/katalog_birlesik_v2.txt, etas_girdi_v2_2000_M25.csv, etas_girdi_v2_2000_M25_mcvar.csv
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
import match_catalogs as mc  # noqa: E402
import build_catalog as BC  # noqa: E402
from mc_field import McField  # noqa: E402

OUT = ROOT / "data/processed/v2"; OUT.mkdir(parents=True, exist_ok=True)


def isc_match(t, lat, lon, I):
    """Her olay için en yakın ISC olayının indeksi (|Δt|<=5 s, <=50 km), yoksa -1."""
    ti = I.zaman.values.astype("datetime64[ms]").astype(np.int64) / 1e3
    tq = pd.to_datetime(t).values.astype("datetime64[ms]").astype(np.int64) / 1e3
    j = np.searchsorted(ti, tq); best = np.full(len(tq), -1); bd = np.full(len(tq), 1e9)
    for off in [-2, -1, 0, 1]:
        k = np.clip(j + off, 0, len(I) - 1); dt = np.abs(ti[k] - tq)
        d = np.hypot((I.enlem.values[k] - lat) * 111.2, (I.boylam.values[k] - lon) * 111.2 * np.cos(np.radians(lat)))
        ok = (dt <= 5) & (d <= 50) & (dt + d / 10 < bd); best[ok] = k[ok]; bd[ok] = (dt + d / 10)[ok]
    return best


def main(add_isc=False):
    log = []
    A = mc.drop_exact_duplicates(mc.load_afad().dropna(subset=["time", "lat", "lon"]), "AFAD", log)
    B = mc.drop_exact_duplicates(mc.load_kandilli().dropna(subset=["time", "lat", "lon"]), "KANDILLI", log)
    E = BC.E
    pairs = pd.concat([pd.read_csv(E / f, sep="\t", dtype={"AFAD_id": str, "KANDILLI_id": str}, usecols=["AFAD_id", "KANDILLI_id", "guven"])
                       for f in ["eslesen_afad_kandilli.txt", "zayif_veya_belirsiz_afad_kandilli.txt"]]).drop_duplicates("AFAD_id").drop_duplicates("KANDILLI_id")
    conv = json.load(open(ROOT / "data/processed/prospektif/gor_donusumleri_v6.json"))
    out, _ = BC.build_events(A, B, pairs, conv=conv)
    out = out.reset_index(drop=True)
    I = pd.read_csv(ROOT / "data/processed/isc/isc_katalog_Mw.csv", parse_dates=["zaman"]).sort_values("zaman").reset_index(drop=True)
    kt = out.kandilli_id.map(B.set_index("id").olay_turu)
    unk = out.olay_turu == "bilinmeyen"
    k_dep = kt == "Deprem"
    im = isc_match(out.time, out.lat.values, out.lon.values, I)
    isc_dep = (im >= 0) & np.isin(I.tur.values[np.maximum(im, 0)], ["ke", "se"])
    fix = unk & (k_dep | isc_dep)
    out.loc[fix, "olay_turu"] = "deprem"
    rep = [f"v2 düzeltmesi: 'bilinmeyen' → 'deprem' {fix.sum():,} olay (Kandilli 'Deprem': {(unk & k_dep).sum():,}, ISC ke/se: {(unk & ~k_dep & isc_dep).sum():,}); "
           f"kalan bilinmeyen {(out.olay_turu == 'bilinmeyen').sum():,}"]
    t = pd.to_datetime(out.time)
    big = out.Mw >= 3.45
    rep.append("  M>=3.5 düzeltilen olay, yıllara göre: " + ", ".join(f"{y}:{n}" for y, n in out[fix & big].groupby(t[fix & big].dt.year).size().items()))
    if add_isc:
        used = set(im[im >= 0])
        Io = I[~I.index.isin(used) & I.tur.isin(["ke", "se"])].copy()
        # bizim ölçeğe kaydırma: eşleşen olaylarda (ISC Mw*>=3) yıl bazında medyan (biz − ISC)
        mm = pd.DataFrame({"y": t.dt.year[im >= 0].values, "biz": out.Mw[im >= 0].values, "isc": I.Mw.values[im[im >= 0]]})
        off = mm[mm.isc >= 3].groupby("y").apply(lambda s: (s.biz - s.isc).median())
        Io["Mw"] = (Io.Mw + Io.zaman.dt.year.map(off).fillna(0)).round(2)
        add = pd.DataFrame({"time": Io.zaman.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-4], "lat": Io.enlem, "lon": Io.boylam,
                            "depth": Io.derinlik, "Mw": Io.Mw, "Mw_kaynak": "ISC:" + Io.Mw_kaynak, "olay_turu": "deprem",
                            "kaynaklar": "ISC"})
        out = pd.concat([out, add], ignore_index=True)
        rep.append(f"  ISC-yalnız eklenen deprem: {len(add):,} (M>=3.5: {(add.Mw >= 3.45).sum():,})")
    out = out.iloc[np.argsort(pd.to_datetime(out.time).values, kind="stable")].reset_index(drop=True)
    tag = "v2b" if add_isc else "v2"
    out.to_csv(OUT / f"katalog_birlesik_{tag}.txt", sep="\t", index=False)
    g = out[(pd.to_datetime(out.time) >= "2000-01-01") & (out.olay_turu == "deprem") & (out.Mw >= 2.5)]
    g = pd.DataFrame({"time": g.time, "latitude": g.lat, "longitude": g.lon, "depth": g.depth, "magnitude": g.Mw,
                      "Mw_kaynak": g.Mw_kaynak, "kaynaklar": g.kaynaklar})
    g.to_csv(OUT / f"etas_girdi_{tag}_2000_M25.csv", index=False)
    tn = (pd.to_datetime(g.time) - pd.Timestamp("2010-01-01")).dt.total_seconds().values / 86400
    F = McField(pd.DataFrame({"time": tn, "latitude": g.latitude.values, "longitude": g.longitude.values, "magnitude": g.magnitude.values}))
    g["mc_current"] = np.ceil(F(tn, g.latitude.values, g.longitude.values) * 10 - 1e-6) / 10
    g.to_csv(OUT / f"etas_girdi_{tag}_2000_M25_mcvar.csv", index=False)
    rep.append(f"  ETAS girdisi ({tag}, 2000+, Mw>=2.5, deprem): {len(g):,} olay")
    (OUT / f"RAPOR_{tag}.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main("--isc-ekle" in sys.argv)
