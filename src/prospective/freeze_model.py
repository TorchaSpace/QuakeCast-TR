"""v6 topluluk modelini dondurur: model ve kod dosyalarının SHA256 listesi -> prospektif/MODEL_V6_MANIFEST.json.
Yayın betiği her çalıştığında bu listeyi doğrular; tek bir bayt değişse bile yayın durur."""
import hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "configs/v6_topluluk.json", "configs/v6_dizi_omori.json", "configs/v6_sabitMc_dizi_omori.json",
    "data/processed/etas/v3_pbig13/durum.json", "data/processed/etas/v3_pbig13/P_background.csv",
    "data/processed/etas/v3_pbig13/arka_plan_karisim.json",
    "data/processed/etas/duy_mc35_serbest_bw15/durum.json", "data/processed/etas/duy_mc35_serbest_bw15/P_background.csv",
    "data/processed/etas/rupturler.csv", "data/processed/arka_plan/uzun_donem.npz",
    "data/processed/etas_girdi_2010_M25_mcvar.csv", "data/processed/etas_girdi_2010_M25.csv",
    "data/processed/prospektif/model_v6/etas_girdi_dondurulmus.csv",
    "data/processed/prospektif/gor_donusumleri_v6.json", "data/processed/prospektif/eslestirme_kalibrasyonu_v6.json",
] + [f"src/etas/{m}.py" for m in ["simulate_forecast", "seq_omori", "seq_update", "obs_comp", "evaluate_masked",
                                   "evaluate_etas", "finite_source", "residuals", "select_background", "longterm_background"]] \
  + ["src/catalog/mc_field.py", "src/catalog/build_catalog.py", "src/data/match_catalogs.py", "src/data/fetch_afad.py",
     "src/data/fetch_kandilli.py", "src/data/fetch_afad_web.py", "src/prospective/live_catalog.py",
     "src/prospective/issue_forecast.py"]

if __name__ == "__main__":
    import numpy, scipy
    etas_commit = subprocess.run(["git", "-C", str(Path.home() / "etasrepo"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    M = dict(model="QuakeCast-TR v6 topluluk (0.8 v6_dizi_omori + 0.2 v6_sabitMc_dizi_omori)",
             dondurma_zamani_utc=str(pd.Timestamp.utcnow().tz_localize(None)),
             etas_kutuphanesi="https://github.com/lmizrahi/etas@" + etas_commit,
             python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__, pandas=pd.__version__,
             dosyalar={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES})
    out = ROOT / "prospektif/MODEL_V6_MANIFEST.json"
    if out.exists() and "--yeniden" not in sys.argv:
        raise SystemExit("Manifest zaten var (model dondurulmuş). Bilinçli yeniden dondurma için --yeniden.")
    json.dump(M, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in M.items() if k != "dosyalar"}, ensure_ascii=False, indent=1), len(FILES), "dosya")
