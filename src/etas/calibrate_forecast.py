"""Simülasyon tahminlerinin kalibrasyonu (CSEP N-testi tarzı) — ardışık 30 günlük pencereler.

Her pencere için model (o tarihte geçerli olan: temel ya da ardışık güncellenmiş) N_SIM katalog üretir;
gözlenen olay sayısının simülasyon dağılımındaki yeri (PIT, rastgeleleştirilmiş) ve %5/%95 dışında kalma oranı.
İyi kalibre bir modelde PIT ~ Uniform(0,1), aralık dışı oran ~%10.
Kullanım: python3 src/etas/calibrate_forecast.py <temel_config> [ardisik]   (ardisik: güncellemeli modelleri kullan)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import simulate_forecast as SF  # noqa
import evaluate_etas as E  # noqa
from evaluate_sequential import DATES  # noqa

H = 30; N_SIM = 400; THR = [3.5, 4.0, 5.0]


def main(base, mode="sabit"):
    cat = pd.read_csv(ROOT / "data/processed/etas_girdi_2010_M25.csv", parse_dates=["time"])
    cat = cat[E.in_poly(cat.latitude.values, cat.longitude.values, [[35, 25], [44, 25], [44, 46], [35, 46], [35, 25]])]
    starts = pd.date_range("2022-01-01", "2026-06-30", freq=f"{H}D")
    out = ROOT / f"data/processed/etas/kalibrasyon_{mode}.txt"
    rows = pd.read_csv(out, sep="\t").to_dict("records") if out.exists() else []
    done = {r["baslangic"] for r in rows}
    sims = {}
    rng = np.random.default_rng(7)
    for T0 in starts:
        if str(T0.date()) in done:
            continue
        cfgp = base
        if mode == "ardisik":
            past = [d for d in DATES if pd.Timestamp(d) <= T0]
            if past:
                cfgp = f"configs/ardisik_{past[-1]}.json"
        if cfgp not in sims:
            sims[cfgp] = SF.Simulator(cfgp, seed=11)
        ev, _ = sims[cfgp].run(str(T0.date()), H, N_SIM)
        obsw = cat[(cat.time >= T0) & (cat.time < T0 + pd.Timedelta(days=H))]
        r = dict(baslangic=str(T0.date()), model=cfgp)
        for thr in THR:
            cnt = ev[ev.m >= thr - 0.05].groupby("sim").size().reindex(range(N_SIM), fill_value=0).values
            o = int((obsw.magnitude >= thr - 0.05).sum())
            pit = (cnt < o).mean() + rng.uniform() * (cnt == o).mean()
            r.update({f"gozlenen_M{thr}": o, f"medyan_M{thr}": float(np.median(cnt)), f"q05_M{thr}": float(np.percentile(cnt, 5)),
                      f"q95_M{thr}": float(np.percentile(cnt, 95)), f"PIT_M{thr}": round(pit, 3)})
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(r["baslangic"], cfgp, {k: v for k, v in r.items() if k.startswith(("gozlenen", "medyan"))}, flush=True)
    df = pd.DataFrame(rows)
    for thr in THR:
        p = df[f"PIT_M{thr}"]
        out_rate = ((df[f"gozlenen_M{thr}"] < df[f"q05_M{thr}"]) | (df[f"gozlenen_M{thr}"] > df[f"q95_M{thr}"])).mean()
        from scipy import stats
        print(f"M>={thr}: pencere {len(df)}, PIT ortalama {p.mean():.2f} (0.5 ideal), KS p={stats.kstest(p, 'uniform').pvalue:.2g}, "
              f"%5-%95 dışında %{100*out_rate:.0f} (~%10 ideal)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sabit")
