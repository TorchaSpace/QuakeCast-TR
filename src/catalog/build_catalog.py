"""Birleşik QuakeCast-TR kataloğu: AFAD + Kandilli (+ ileride TURHEC), tek olay / tek büyüklük ölçeği (Mw).

Adımlar
1) Kaynaklar yüklenir, birebir tekrarlar ayıklanır (match_catalogs ile aynı kurallar).
2) Eşleştirme sonuçlarından (eslesen + zayif_veya_belirsiz) aynı deprem olan kayıtlar tek olaya indirgenir.
   Zayıf/belirsiz çiftler de birleştirilir: aynı depremi iki kez saymak ETAS'ta sahte kümelenme yaratır.
3) Büyüklük dönüşümleri: her (kurum, tür) -> Mw için genel ortogonal regresyon (GOR, hata oranı 1),
   referans Mw = olayın kurumlardaki Mw değerlerinin ortalaması. Eğitim aralığı dışına çıkan değerler işaretlenir.
4) Olay başına Mw: doğrudan Mw (kurumların ortalaması) > dönüştürülmüş ML > dönüştürülmüş MD > diğer.
   Birden fazla kurum varsa dönüştürülmüş değerlerin ortalaması alınır.
5) Konum/zaman: 2007 ve sonrası AFAD, öncesi Kandilli (AFAD'ın eski koordinatları kaba). Diğerinin değerleri saklanır.
Çıktılar: data/processed/katalog_birlesik.txt, data/processed/donusum_raporu.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
import match_catalogs as mc  # noqa: E402

E = ROOT / "data" / "interim" / "eslestirme"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
REP = []


def gor(x, y):
    """Genel ortogonal regresyon (eta=1): y = a + b x. (a, b, n, sigma_dik, x_min, x_max)"""
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    mx, my = x.mean(), y.mean()
    sxx, syy, sxy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean(), ((x - mx) * (y - my)).mean()
    b = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    a = my - b * mx
    res = (y - (a + b * x)) / np.sqrt(1 + b ** 2)
    return dict(a=a, b=b, n=int(ok.sum()), s=float(np.std(res) * np.sqrt(1 + b ** 2)),
                lo=float(np.percentile(x, 1)), hi=float(np.percentile(x, 99)))


def build_events(A, B, pairs, conv=None):
    """Kaynaklar (id indeksli olmayan) + eşleşme çiftleri -> birleşik olay tablosu. conv verilirse (dondurulmuş GOR
    katsayıları) yeniden uydurulmaz; prospektif artımlı güncelleme bunu kullanır."""
    A = A.set_index("id"); B = B.set_index("id")
    ev = pd.DataFrame({"afad_id": pairs.AFAD_id.values, "kandilli_id": pairs.KANDILLI_id.values})
    ev = pd.concat([ev,
                    pd.DataFrame({"afad_id": A.index.difference(pairs.AFAD_id)}),
                    pd.DataFrame({"kandilli_id": B.index.difference(pairs.KANDILLI_id)})], ignore_index=True)

    def col(src, ids, c):
        s = pd.Series(np.nan, index=ev.index, dtype=object if c in ("olay_turu", "place", "magtype") else float)
        m = ids.notna()
        s[m] = src.loc[ids[m], c].values
        return s

    def tcol(src, ids):
        s = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns]")
        m = ids.notna(); s[m] = src.loc[ids[m], "time"].values
        return s

    f = {}
    for tag, src, ids in [("A", A, ev.afad_id), ("K", B, ev.kandilli_id)]:
        f[tag + "_time"] = tcol(src, ids)
        for c in ["lat", "lon", "depth", "mag", "ML", "MD", "MW", "olay_turu", "place", "magtype"]:
            f[f"{tag}_{c}"] = col(src, ids, c)
    ev = pd.concat([ev, pd.DataFrame(f)], axis=1)

    # --- büyüklük dönüşümleri ---
    mw_ref = ev[["A_MW", "K_MW"]].mean(axis=1)
    if conv is None:
        conv = {}
        for name, x in [("AFAD_ML", ev.A_ML), ("AFAD_MD", ev.A_MD), ("KANDILLI_ML", ev.K_ML), ("KANDILLI_MD", ev.K_MD)]:
            sel = x.notna() & mw_ref.notna() & (mw_ref >= 2.5)
            if sel.sum() >= 200:
                conv[name] = gor(x[sel].values.astype(float), mw_ref[sel].values.astype(float))
        # MD için doğrudan Mw çifti azsa: MD -> (aynı olayın) ML -> Mw zinciri
        for ag, mlc, mdc in [("AFAD", "A_ML", "A_MD"), ("KANDILLI", "K_ML", "K_MD")]:
            if f"{ag}_MD" not in conv:
                other_ml = ev.K_ML if ag == "AFAD" else ev.A_ML
                sel = ev[mdc].notna() & other_ml.notna() & (other_ml >= 2.5)
                oth = "KANDILLI_ML" if ag == "AFAD" else "AFAD_ML"
                if sel.sum() >= 200 and oth in conv:
                    r1 = gor(ev[mdc][sel].values.astype(float), other_ml[sel].values.astype(float)); r2 = conv[oth]
                    conv[f"{ag}_MD"] = dict(a=r2["a"] + r2["b"] * r1["a"], b=r2["b"] * r1["b"], n=r1["n"],
                                            s=float(np.hypot(r1["s"] * r2["b"], r2["s"])), lo=r1["lo"], hi=r1["hi"], zincir=oth)
    REP.append("\nBüyüklük dönüşümleri (GOR, Mw = a + b·M; referans = kurumların Mw ortalaması, Mw>=2.5):")
    for k, r in conv.items():
        REP.append(f"  {k:12s} -> Mw: a={r['a']:+.3f} b={r['b']:.3f}  n={r['n']:,}  σ={r['s']:.2f}  geçerli aralık ≈ {r['lo']:.1f}-{r['hi']:.1f}" + (f"  (zincir: {r['zincir']})" if "zincir" in r else ""))

    def apply(name, x):
        r = conv.get(name)
        return (r["a"] + r["b"] * x) if r else pd.Series(np.nan, index=x.index)

    est = pd.DataFrame({
        "MW": mw_ref,
        "ML": pd.concat([apply("AFAD_ML", ev.A_ML), apply("KANDILLI_ML", ev.K_ML)], axis=1).mean(axis=1),
        "MD": pd.concat([apply("AFAD_MD", ev.A_MD), apply("KANDILLI_MD", ev.K_MD)], axis=1).mean(axis=1),
        "diger": ev[["A_mag", "K_mag"]].mean(axis=1),
    })
    ev["Mw"] = est.MW.fillna(est.ML).fillna(est.MD).fillna(est.diger)
    ev["Mw_kaynak"] = np.select([est.MW.notna(), est.ML.notna(), est.MD.notna(), est.diger.notna()],
                                ["Mw", "ML->Mw", "MD->Mw", "donusumsuz"], "yok")
    sig = {k: v["s"] for k, v in conv.items()}
    ev["Mw_sigma"] = np.select([ev.Mw_kaynak == "Mw", ev.Mw_kaynak == "ML->Mw", ev.Mw_kaynak == "MD->Mw"],
                               [0.1, np.nanmean([sig.get("AFAD_ML", np.nan), sig.get("KANDILLI_ML", np.nan)]),
                                np.nanmean([sig.get("AFAD_MD", np.nan), sig.get("KANDILLI_MD", np.nan)])], 0.3)
    lo = min(r["lo"] for r in conv.values())
    ev["Mw_aralik_disi"] = (ev.Mw_kaynak.isin(["ML->Mw", "MD->Mw"])) & (est.ML.fillna(est.MD) < lo)

    # --- konum / zaman ---
    use_a = ev.A_time.notna() & ((ev.A_time.dt.year >= 2007) | ev.K_time.isna())
    for c in ["lat", "lon", "depth"]:
        ev[c] = np.where(use_a, ev[f"A_{c}"], ev[f"K_{c}"]).astype(float)
    ev["time"] = ev.A_time.where(use_a, ev.K_time)
    ev["konum_kaynagi"] = np.where(use_a, "AFAD", "KANDILLI")
    ev["kaynaklar"] = np.select([ev.afad_id.notna() & ev.kandilli_id.notna(), ev.afad_id.notna()], ["AFAD+KANDILLI", "AFAD"], "KANDILLI")
    blast = (ev.A_olay_turu == "Patlama") | (ev.K_olay_turu == "Patlatma(Sm)")
    unk = (ev.A_olay_turu == "Bilinmeyen") & ~blast
    ev["olay_turu"] = np.select([blast, unk], ["patlatma", "bilinmeyen"], "deprem")
    ev["yer"] = ev.K_place.fillna(ev.A_place)
    out = ev.sort_values("time")[["time", "lat", "lon", "depth", "Mw", "Mw_kaynak", "Mw_sigma", "Mw_aralik_disi",
                                  "olay_turu", "kaynaklar", "konum_kaynagi", "afad_id", "kandilli_id",
                                  "A_mag", "A_magtype", "K_mag", "K_ML", "K_MD", "K_MW", "yer"]]
    out["time"] = out.time.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-4]
    out.Mw = out.Mw.round(2)
    return out, conv


def main():
    A = mc.load_afad(); B = mc.load_kandilli()
    log = []
    A = mc.drop_exact_duplicates(A.dropna(subset=["time", "lat", "lon"]), "AFAD", log)
    B = mc.drop_exact_duplicates(B.dropna(subset=["time", "lat", "lon"]), "KANDILLI", log)
    REP.extend(log)
    pairs = pd.concat([pd.read_csv(E / f, sep="\t", dtype={"AFAD_id": str, "KANDILLI_id": str},
                                   usecols=["AFAD_id", "KANDILLI_id", "guven"])
                       for f in ["eslesen_afad_kandilli.txt", "zayif_veya_belirsiz_afad_kandilli.txt"]])
    pairs = pairs.drop_duplicates("AFAD_id").drop_duplicates("KANDILLI_id")
    REP.append(f"Birleştirilen çift: {len(pairs):,} (" + ", ".join(f"{k}={v:,}" for k, v in pairs.guven.value_counts().items()) + ")")

    out, conv = build_events(A, B, pairs)
    out.to_csv(OUT / "katalog_birlesik.txt", sep="\t", index=False)
    REP.append(f"\nBirleşik katalog: {len(out):,} olay  -> data/processed/katalog_birlesik.txt")
    REP.append("  kaynak: " + ", ".join(f"{k}={v:,}" for k, v in out.kaynaklar.value_counts().items()))
    REP.append("  olay türü: " + ", ".join(f"{k}={v:,}" for k, v in out.olay_turu.value_counts().items()))
    REP.append("  Mw kaynağı: " + ", ".join(f"{k}={v:,}" for k, v in out.Mw_kaynak.value_counts().items()))
    REP.append(f"  dönüşüm aralığı dışında kalan (küçük) olay: {out.Mw_aralik_disi.sum():,}")
    (OUT / "donusum_raporu.txt").write_text("\n".join(REP) + "\n", encoding="utf-8")
    print("\n".join(REP))


if __name__ == "__main__":
    main()
