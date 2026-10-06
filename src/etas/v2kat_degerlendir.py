"""Katalog v2 deneylerinin değerlendirmesi (QC_KATALOG = v2 ETAS girdisi). Pencereler: doğrulama 2022–24, test 2024–26.07.
Adımlar: karışım arka plan (değişken-Mc üye), maskeli taban LL, dizi güncellemesi (v6 ayarları + küçük ızgara),
iki üyeli topluluk (w=0.8). Kullanım: QC_KATALOG=data/processed/v2/etas_girdi_v2_2000_M25.csv python3 src/etas/v2kat_degerlendir.py E1"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
assert os.environ.get("QC_KATALOG"), "QC_KATALOG ayarlanmalı"
import seq_update as SU, seq_omori as SO, evaluate_masked as EM  # noqa
VAL, TEST = ("2022-01-01", "2024-01-01"), ("2024-01-01", "2026-08-01")


def mixcfg(tag):
    base = ROOT / f"configs/v2kat_{tag}.json"; c = json.load(open(base)); md = ROOT / c["out_dir"]
    if not (md / "arka_plan_karisim.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src/etas/select_background.py"), str(base), "karisim"], check=True, stdout=subprocess.DEVNULL)
    c["bg_mix"] = True; p = ROOT / f"configs/v2kat_{tag}_mix.json"; json.dump(c, open(p, "w"), indent=1)
    return str(p.relative_to(ROOT))


def member(cfg, nu=0.3, mu=0.0, sig=0.2, m_p=4.5):
    SU.VAL, SU.TEST = VAL, TEST
    P = SU.prepare_sparse(cfg); Q = SO.prepare_omori(cfg)
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    out, pr = SO.score_omori(P, Q, nu, mu, sig, m_p, windows=(VAL, TEST), return_parts=True)
    X = EM.evaluate(cfg, expose=True); ev = X["ev"]
    q = np.searchsorted(ev.t.values, P["tO"]); lobs = -X["beta"] * (np.maximum(ev.mcf.values[q], 3.5) - 3.5)
    res = {}
    for r in out:
        b = base[r["pencere"]]
        res[r["pencere"]] = dict(taban=b["LL_olay_basi"], guncel=(b["LL"] + r["dLL"]) / b["n_gozlenebilir"], n=b["n_gozlenebilir"],
                                 comp=b["beklenen"] + r["dcomp"])
    return P["tO"], np.log(P["lam_tot"]) + pr["dl"] + lobs, res


if __name__ == "__main__":
    tag = sys.argv[1]
    c1 = mixcfg(tag); c2 = f"configs/v2kat_{tag}_sabit.json"
    t1, l1, r1 = member(c1); t2, l2, r2 = member(c2)
    rows = []
    for w0, w1 in (VAL, TEST):
        key = f"{w0}..{w1}"; T0, T1 = SU._days(w0), SU._days(w1)
        i1 = np.nonzero((t1 >= T0) & (t1 < T1))[0]; i2 = np.searchsorted(t2, t1[i1]); assert np.allclose(t2[i2], t1[i1])
        ens = logsumexp(np.vstack([np.log(0.8) + l1[i1], np.log(0.2) + l2[i2]]), axis=0).sum() - (0.8 * r1[key]["comp"] + 0.2 * r2[key]["comp"])
        rows.append(dict(deney=tag, pencere=key, n=r1[key]["n"], taban=r1[key]["taban"], guncel=round(r1[key]["guncel"], 4),
                         sabit_guncel=round(r2[key]["guncel"], 4), topluluk=round(ens / len(i1), 4)))
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    out = ROOT / "data/processed/etas/v2kat_sonuclar.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    pd.concat([old[old.deney != tag] if len(old) else old, df]).to_csv(out, index=False)
