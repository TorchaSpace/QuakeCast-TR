"""Klasik ETAS uyumu (Mizrahi vd. `etas` paketi, EM ile) — kaldığı yerden devam edebilen sürücü.

Paket: https://github.com/lmizrahi/etas (ETAS_LIB ortam değişkeni ile yolu verilir).
Her çağrıda BUDGET saniye kadar EM iterasyonu yapılır, parametreler durum dosyasına yazılır;
sonraki çağrı oradan devam eder (yakınsama ölçütü paketle aynı: fark < 0.001).

Kullanım: BUDGET=140 python3 src/etas/fit_etas.py configs/etas_temel.json
"""
import json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
from etas.inversion import (ETASParameterCalculation, branching_ratio,  # noqa: E402
                            calc_diff_to_before, parameter_array2dict, parameter_dict2array)

ROOT = Path(__file__).resolve().parents[2]


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    out = ROOT / cfg["out_dir"]; out.mkdir(parents=True, exist_ok=True)
    state_f = out / "durum.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"iter": 0, "theta": cfg["theta_0"], "gecmis": []}
    if state.get("yakinsadi"):
        print("zaten yakınsadı:", json.dumps(state["theta"], indent=1)); return
    budget = float(os.environ.get("BUDGET", "140")); t0 = time.time()

    cat = pd.read_csv(ROOT / cfg["catalog"], parse_dates=["time"])
    meta = dict(
        catalog=cat[["time", "latitude", "longitude", "magnitude"]].copy(),
        auxiliary_start=cfg["auxiliary_start"], timewindow_start=cfg["timewindow_start"],
        timewindow_end=cfg["timewindow_end"], mc=cfg["mc"], delta_m=cfg["delta_m"],
        coppersmith_multiplier=cfg["coppersmith_multiplier"], shape_coords=cfg["shape_coords"],
        theta_0=state["theta"], name=cfg.get("name", "QuakeCast-TR ETAS"))
    logging.basicConfig(level=logging.WARNING)
    calc = ETASParameterCalculation(meta)
    calc.prepare()
    t_prep = time.time() - t0
    theta_old = parameter_dict2array(state["theta"])
    print(f"hazırlık {t_prep:.0f} s; hedef olay {len(calc.target_events)}, kaynak olay {len(calc.source_events)}", flush=True)
    while time.time() - t0 < budget:
        ti = time.time()
        calc.pij, calc.target_events, calc.source_events, calc.n_hat, calc.i_hat = \
            calc.expectation_step(theta_old, calc.m_ref - calc.delta_m / 2)
        theta_new = calc.optimize_parameters(theta_old)
        diff = float(calc_diff_to_before(theta_old, theta_new))
        state["iter"] += 1
        state["theta"] = {k: (None if v is None else float(v)) for k, v in parameter_array2dict(theta_new).items()}
        try:
            br = float(branching_ratio(theta_new, calc.beta))
        except Exception:
            br = None
        state["gecmis"].append({"iter": state["iter"], "fark": diff, "n_hat": float(calc.n_hat), "dallanma_orani": br})
        state["beta"] = float(calc.beta); state["b"] = float(calc.beta / np.log(10))
        state_f.write_text(json.dumps(state, indent=1))
        print(f"iter {state['iter']}: fark={diff:.4f}  n_hat={calc.n_hat:.1f}  dallanma={br}  ({time.time()-ti:.0f} s)", flush=True)
        theta_old = theta_new
        if diff < 0.001:
            state["yakinsadi"] = True; state_f.write_text(json.dumps(state, indent=1))
            print("YAKINSADI", flush=True)
            break
    print(json.dumps(state["theta"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
