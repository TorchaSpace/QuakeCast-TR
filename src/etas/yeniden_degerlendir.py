"""Doğrulama (2022–23) / test (2024–26.07) ayrımıyla maskeli puanlama — düzeltilmiş (kaynak-merkezli) kompanzatörle.
Sonuçlar: data/processed/etas/dogrulama_test_v2.txt (kaldığı yerden devam eder; BUDGET saniye)."""
import os, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_masked as EM  # noqa: E402

OUT = ROOT / "data/processed/etas/dogrulama_test_v2.txt"
W = [("2022-01-01", "2024-01-01"), ("2024-01-01", "2026-08-01")]

if __name__ == "__main__":
    t0 = time.time(); budget = float(os.environ.get("BUDGET", 140))
    done = pd.read_csv(OUT, sep="\t") if OUT.exists() else pd.DataFrame(columns=["config"])
    rows = done.to_dict("records")
    for c in sys.argv[1:]:
        if c in set(done.config):
            continue
        if time.time() - t0 > budget - 30:
            print("SURE DOLDU"); break
        r = EM.evaluate(c, windows=W)
        rows.append(dict(config=c, model=r[0]["model"], dogrulama=r[0]["LL_olay_basi"], test=r[1]["LL_olay_basi"],
                         LL_dog=r[0]["LL"], LL_test=r[1]["LL"], bek_dog=r[0]["beklenen"], n_dog=r[0]["n_gozlenebilir"],
                         bek_test=r[1]["beklenen"], n_test=r[1]["n_gozlenebilir"]))
        pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
        print(f"{c}: doğ {r[0]['LL_olay_basi']}  test {r[1]['LL_olay_basi']}", flush=True)
    else:
        print("TUMU BITTI")
