"""Çok katlı (rolling-origin) doğrulama sürücüsü — dondurulmuş modülleri değiştirmeden (pencereler çalışma anında verilir).

Katlar: A eğitim 2014–2018 → doğrulama 2018–2020; B 2014–2020 → 2020–2022; C 2014–2022 → 2022–2024.
Her (kat, varyant) için: karışım arka plan ağırlığı (eğitim hedeflerinde LOO), maskeli taban LL, dizi-özgü verimlilik (ν ızgarası),
dizi-özgü Omori p (σ ızgarası). Sonuçlar: data/processed/etas/cv/sonuclar.csv (kaldığı yerden devam eder).
Kullanım: python3 src/etas/cv_degerlendir.py <kat> <varyant> [adim]
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
FOLDS = {"A": (("2018-01-01", "2020-01-01"), ("2020-01-01", "2022-01-01")),
         "B": (("2020-01-01", "2022-01-01"), ("2022-01-01", "2024-01-01")),
         "C": (("2022-01-01", "2024-01-01"), ("2024-01-01", "2026-08-01"))}
OUT = ROOT / "data/processed/etas/cv/sonuclar.csv"


def mix_cfg(f, v):
    base = ROOT / f"configs/cv/{f}_{v}.json"
    c = json.load(open(base))
    if v == "sabit":
        return str(base.relative_to(ROOT))
    md = ROOT / c["out_dir"]
    if not (md / "arka_plan_karisim.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src/etas/select_background.py"), str(base), "karisim"], check=True,
                       stdout=subprocess.DEVNULL)
    c["bg_mix"] = True; c["name"] += " + karışım"
    p = ROOT / f"configs/cv/{f}_{v}_mix.json"; json.dump(c, open(p, "w"), indent=1)
    return str(p.relative_to(ROOT))


def run(f, v):
    import seq_update as SU, seq_omori as SO
    val, nxt = FOLDS[f]
    SU.VAL, SU.TEST = val, nxt
    cfg = mix_cfg(f, v)
    P = SU.prepare_sparse(cfg)
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    key = f"{val[0]}..{val[1]}"; b = base[key]
    rows = [dict(kat=f, varyant=v, guncelleme="yok", nu=np.nan, m_p=np.nan, sig=np.nan, n=b["n_gozlenebilir"],
                 LL=b["LL"], LL_olay=b["LL_olay_basi"])]
    for nu in [0.2, 0.3, 0.5, 1.0]:
        r = SU.score_sparse(P, nu, 3.5, windows=(val,))[0]
        rows.append(dict(kat=f, varyant=v, guncelleme="verimlilik", nu=nu, n=b["n_gozlenebilir"], LL=b["LL"] + r["dLL"],
                         LL_olay=(b["LL"] + r["dLL"]) / b["n_gozlenebilir"]))
    Q = SO.prepare_omori(cfg)
    for nu in [0.3]:
        for m_p in [3.5, 4.5]:
            for sig in [0.1, 0.2, 0.3]:
                r = SO.score_omori(P, Q, nu, 0.0, sig, m_p, windows=(val,))[0]
                rows.append(dict(kat=f, varyant=v, guncelleme="verimlilik+omori", nu=nu, m_p=m_p, sig=sig, n=b["n_gozlenebilir"],
                                 LL=b["LL"] + r["dLL"], LL_olay=(b["LL"] + r["dLL"]) / b["n_gozlenebilir"]))
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(old):
        old = old[~((old.kat == f) & (old.varyant == v))]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(OUT, index=False)
    best = max(rows, key=lambda r: r["LL_olay"])
    print(f"{f} {v}: taban {rows[0]['LL_olay']:.4f}  en iyi {best['guncelleme']} ν={best['nu']} m_p={best.get('m_p')} σ={best.get('sig')} {best['LL_olay']:.4f}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
