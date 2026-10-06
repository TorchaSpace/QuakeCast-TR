"""N-testi kalibrasyon adaylarının maskeli LL özeti (doğrulama 2022–24, test 2024–26.07).
Her yapılandırmanın kendi seq_nu / seq_omori ayarlarıyla dizi güncellemeli LL'yi hesaplar (önbellekli hazırlıkları kullanır).
Kullanım: python3 src/etas/aday_puanla.py configs/a_dizi_omori.json ...  → data/processed/etas/ntest_adaylari_LL.csv"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import seq_update as SU, seq_omori as SO  # noqa: E402
VAL, TEST = ("2022-01-01", "2024-01-01"), ("2024-01-01", "2026-08-01")
OUT = ROOT / "data/processed/etas/ntest_adaylari_LL.csv"

if __name__ == "__main__":
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["config"])
    rows = old.to_dict("records")
    for cfg in sys.argv[1:]:
        if cfg in set(old.config):
            continue
        c = json.load(open(ROOT / cfg)); so = c.get("seq_omori", dict(mu=0.0, sig=0.2, m_p=4.5)); nu = c.get("seq_nu", 0.3)
        SU.VAL, SU.TEST = VAL, TEST
        P = SU.prepare_sparse(cfg); Q = SO.prepare_omori(cfg)
        base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
        out = SO.score_omori(P, Q, nu, so["mu"], so["sig"], so["m_p"], windows=(VAL, TEST))
        r = {"config": cfg, "a": json.load(open(ROOT / c["out_dir"] / "durum.json"))["theta"]["a"]}
        for x, nm in zip(out, ["dogrulama", "test"]):
            b = base[x["pencere"]]
            r[f"{nm}_taban"] = round(b["LL_olay_basi"], 4)
            r[f"{nm}_guncel"] = round((b["LL"] + x["dLL"]) / b["n_gozlenebilir"], 4)
        rows.append(r); pd.DataFrame(rows).to_csv(OUT, index=False); print(r, flush=True)
