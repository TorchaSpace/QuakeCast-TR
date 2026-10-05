"""Ardışık güncellemeli eğitim dışı test: her dönem, o döneme kadar uydurulmuş modelle puanlanır.

Dönemler: [2022-01-01, d1) temel model; [d_k, d_{k+1}) 'configs/ardisik_<d_k>.json' modeli.
Puanlama evaluate_masked ile aynıdır (gözlenebilir olaylar, maske düzeltmeli kompanzatör).
Kullanım: python3 src/etas/evaluate_sequential.py configs/etas_mcvar_bw15_sonlu_kirik.json
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_masked as EM  # noqa
import evaluate_etas as E  # noqa

DATES = ["2023-02-07", "2023-02-13", "2023-03-06", "2023-06-01", "2024-01-01", "2025-01-01", "2026-01-01"]
START, END = "2022-01-01", "2026-08-01"


def main(base_cfg):
    segs = [(START, DATES[0], base_cfg)] + [(d, (DATES + [END])[i + 1], f"configs/ardisik_{d}.json") for i, d in enumerate(DATES)]
    out = ROOT / "data/processed/etas/test_ardisik.txt"
    done = pd.read_csv(out, sep="\t") if out.exists() else pd.DataFrame()
    rows = [] if done.empty else done.to_dict("records")
    for a, b, cfg in segs:
        key = f"{a}..{b}"
        if any(r["pencere"] == key and r["config"] == cfg for r in rows):
            continue
        res = EM.evaluate(cfg, windows=[(a, b)])
        r = res[0]; r["config"] = cfg
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
        print(r, flush=True)
    df = pd.DataFrame(rows)
    df = df[df.config.isin([s[2] for s in segs])]
    print("\nArdışık güncellemeli toplam: n =", df.n_gozlenebilir.sum(), " LL =", round(df.LL.sum(), 1),
          " olay başına =", round(df.LL.sum() / df.n_gozlenebilir.sum(), 4), " beklenen =", round(df.beklenen.sum(), 1))


if __name__ == "__main__":
    main(sys.argv[1])
