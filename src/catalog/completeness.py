"""Tamlık büyüklüğü (Mc) ve b-değeri analizi — birleşik katalog, sadece depremler, çalışma alanı 35-44K / 25-46D.

Yöntemler
- MAXC + 0.2 düzeltmesi (Wiemer & Wyss 2000; Woessner & Wiemer 2005) ve
  b-değeri kararlılığı (MBS; Cao & Gao 2002 / Woessner & Wiemer 2005) — ikisinin büyüğü raporlanır.
- b: Aki-Utsu (ΔM=0.1 düzeltmeli), belirsizlik Shi & Bolt (1982).
Çıktılar: data/processed/mc/ (mc_yillik.txt, mc_harita_*.txt, rapor, grafikler)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "mc"
OUT.mkdir(parents=True, exist_ok=True)
DM = 0.1
BOX = (35, 44, 25, 46)


def bval(m, mc):
    x = m[m >= mc - DM / 2]
    n = len(x)
    if n < 50:
        return np.nan, np.nan, n
    b = np.log10(np.e) / (x.mean() - (mc - DM / 2))
    sb = 2.3 * b ** 2 * np.sqrt(((x - x.mean()) ** 2).sum() / (n * (n - 1)))
    return b, sb, n


def maxc(m):
    r = np.round(m / DM) * DM
    v, c = np.unique(np.round(r, 1), return_counts=True)
    return v[np.argmax(c)] + 0.2


def mbs(m, start):
    """b-değeri kararlılığı: Mc adayından itibaren b, sonraki 0.5 birimlik ortalama b'den ±δb içinde olmalı."""
    for mc in np.arange(start - 0.5, start + 1.5, DM):
        b, sb, n = bval(m, mc)
        if n < 100 or np.isnan(b):
            return np.nan
        bs = [bval(m, mc + k * DM)[0] for k in range(1, 6)]
        if np.isnan(bs).any():
            return np.nan
        if abs(np.mean(bs) - b) <= sb:
            return round(mc, 1)
    return np.nan


def mc_of(m):
    if len(m) < 100:
        return np.nan, np.nan, np.nan
    a = maxc(m); b = mbs(m, a)
    return a, b, np.nanmax([a, b])


def main():
    d = pd.read_csv(ROOT / "data" / "processed" / "katalog_birlesik.txt", sep="\t", low_memory=False)
    d["time"] = pd.to_datetime(d.time, format="ISO8601")
    d = d[(d.olay_turu == "deprem") & d.lat.between(BOX[0], BOX[1]) & d.lon.between(BOX[2], BOX[3]) & d.Mw.notna()]
    rep = [f"Mc analizi: {len(d):,} deprem (35-44K, 25-46D), büyüklük = birleşik Mw", ""]
    rows = []
    for y, g in d.groupby(d.time.dt.year):
        if y < 1960:
            continue
        a, s, mcv = mc_of(g.Mw.values)
        b, sb, n = bval(g.Mw.values, mcv) if not np.isnan(mcv) else (np.nan, np.nan, 0)
        rows.append(dict(yil=y, olay=len(g), Mc_MAXC=a, Mc_MBS=s, Mc=mcv, b=b, b_hata=sb, n_ustu=n))
    yr = pd.DataFrame(rows)
    yr.round(3).to_csv(OUT / "mc_yillik.txt", sep="\t", index=False)
    rep.append("Yıllık Mc (MAXC+0.2 ve MBS'nin büyüğü) ve b-değeri:")
    rep += ["  " + s for s in yr.round(2).to_string(index=False).splitlines()]

    # mekânsal Mc: dönemler x 1° hücre
    for t0, t1 in [(2000, 2006), (2007, 2014), (2015, 2026)]:
        g = d[(d.time.dt.year >= t0) & (d.time.dt.year <= t1)]
        gx = (np.floor(g.lat)).astype(int); gy = (np.floor(g.lon)).astype(int)
        cells = []
        for (la, lo), h in g.groupby([gx, gy]):
            a, s, mcv = mc_of(h.Mw.values)
            cells.append(dict(enlem=la, boylam=lo, olay=len(h), Mc=mcv))
        cm = pd.DataFrame(cells)
        cm.to_csv(OUT / f"mc_harita_{t0}_{t1}.txt", sep="\t", index=False)
        ok = cm.Mc.dropna()
        rep.append(f"\nMekânsal Mc {t0}-{t1} (1°x1° hücre, >=100 olay): {len(ok)} hücre; medyan {ok.median():.1f}, %90 {ok.quantile(.9):.1f}, en yüksek {ok.max():.1f}")
        top = cm.dropna().sort_values("Mc", ascending=False).head(5)
        rep.append("  en yüksek Mc hücreleri: " + "; ".join(f"{r.enlem}K {r.boylam}D Mc={r.Mc:.1f} (n={r.olay})" for r in top.itertuples()))
    # grafik
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax[0].plot(yr.yil, yr.Mc, "o-", label="Mc"); ax[0].set_ylabel("Mc (Mw)"); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].errorbar(yr.yil, yr.b, yr.b_hata, fmt="o-", label="b"); ax[1].set_ylabel("b"); ax[1].grid(alpha=.3)
        ax[1].set_xlabel("Yıl"); fig.suptitle("Birleşik katalog: yıllık Mc ve b-değeri (35-44K, 25-46D)")
        fig.tight_layout(); fig.savefig(OUT / "mc_b_yillik.png", dpi=120)
        rep.append("\nGrafik: data/processed/mc/mc_b_yillik.png")
    except Exception as ex:
        rep.append(f"(grafik çizilemedi: {ex})")
    (OUT / "mc_raporu.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
