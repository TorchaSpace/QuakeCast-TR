"""Dizi-özgü verimlilik güncellemesi (Bayesçi Gamma–Poisson), maskeli değerlendirme üzerine.

Fikir (Page vd. 2016 BSSA; USGS OAF; Mizrahi vd. 2023 flETAS'ın büzülmeli hali):
genel (generic) ETAS her M>=m_seq kaynağa aynı verimlilik yasasını uygular; gerçek diziler bundan
log10'da ~0.5 sapar (diziler-arası değişkenlik). Tahmin anına kadar gözlenen artçılar bu sapmayı ölçer.

Kaynak j için çarpan g_j(t) ~ Gamma(ν, ν) ön-bilgisi (ortalama 1; ν≈1.2 ↔ σ_log10≈0.49, Page 2016 ANSR):
    g_j(t) = (ν + n_j(t⁻)) / (ν + E_j(t))
  n_j(t⁻) = Σ_{i gözlendi, t_j<t_i<t} P(i←j)          (taban modelin sorumluluk olasılıkları)
  E_j(t)  = taban modelde j'nin (t_j, t] içinde gözlenebilir beklenen doğrudan artçısı
            (obs_comp.ObsComp: kendi mekânsal çekirdeğinden örneklenen noktalarda Mc(t,x) ve bölge sınırı; zaman analitik)
Güncellenmiş yoğunluk: λ'(t,x) = λ(t,x) + Σ_j (g_j(t) − 1) λ_j(t,x)
Log-olabilirlik farkı: Σ_i log(1 + Σ_j (g_j(t_i)−1) P_ij)  −  Σ_j ∫ (g_j − 1) dE_j
  olaylar arası n sabitken ∫ = (ν+N) ln((ν+E_b)/(ν+E_a)) − (E_b − E_a)  (kapalı biçim).
Hiçbir ek veri gerekmez; yalnızca tahmin anından önceki katalog kullanılır (ileriye bakış yok).

Kullanım:
  python3 src/etas/seq_update.py configs/v3_referans.json seyrek     # tüm kaynaklar (önerilen), ν/m_seq taraması
  python3 src/etas/seq_update.py configs/v3_referans.json            # yoğun sürüm (M>=M_CAND kaynaklar)
  QC_T_END=2026-10-06 ...                                            # operasyonel tahmin için hazırlık bitişi
Seçim (doğrulama 2022–23): tüm kaynaklar, ν = 0.3; öz-tutarlı sorumluluk (ITER>0) ve arka-plan-olasılığına bağlı
ön-bilgi ortalaması doğrulamada daha kötü → kullanılmıyor.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
import evaluate_masked as EM  # noqa: E402
import evaluate_etas as E  # noqa: E402
import finite_source as fs  # noqa: E402
from etas.inversion import expected_aftershocks, upper_gamma_ext  # noqa: E402
from mc_field import hav  # noqa: E402

T_END = os.environ.get("QC_T_END", "2026-08-01")
M_CAND = 4.5
VAL = ("2022-01-01", "2024-01-01"); TEST = ("2024-01-01", "2026-08-01")


def _days(s):
    return (pd.Timestamp(s) - EM.T_ORIGIN).days


def inside_fraction(lat, lon, D, rho, poly, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    r = np.sqrt(D * (rng.random(n) ** (-1 / rho) - 1)); a = rng.random(n) * 2 * np.pi
    la = lat + r * np.cos(a) / 111.2; lo = lon + r * np.sin(a) / (111.2 * np.cos(np.radians(lat)))
    return E.in_poly(la, lo, poly).mean()


def prepare(cfg_path, m_cand=M_CAND, force=False):
    cfg0 = json.load(open(cfg_path)) if str(cfg_path).endswith(".json") else None
    out_dir = ROOT / (cfg0["out_dir"] if cfg0 else cfg_path)
    tag = "_bgmix" if (cfg0 or {}).get("bg_mix") else ""
    fn = out_dir / f"dizi_hazirlik2{tag}_m{m_cand:.1f}.npz"
    if fn.exists() and not force:
        return dict(np.load(fn, allow_pickle=True))
    t0 = time.time()
    X = EM.evaluate(cfg_path, expose=True)
    ev, S, obs, lam35, F = X["ev"], X["S"], X["observed"], X["lam35"], X["F"]
    th, beta = X["th"], X["beta"]
    ts = X["ts"]; T_end = _days(T_END)
    seq = np.nonzero(S.magnitude.values >= m_cand - 0.05)[0]
    O = np.nonzero(obs & (ev.t.values > ts[seq].min()) & (ev.t.values < T_end))[0]
    tO = ev.t.values[O]
    lam_tot, lam_seq = lam35(tO, ev.latitude.values[O], ev.longitude.values[O], cols=seq)
    print(f"  λ: {len(O)} olay × {len(seq)} kaynak ({time.time()-t0:.0f} s)", flush=True)
    # gözlenebilir beklenen doğrudan artçı E_j(t): kaynak-merkezli kompanzatör (obs_comp)
    import obs_comp as OC
    oc = OC.ObsComp(X, T_end)
    BT = np.array([_days(VAL[0]), _days(VAL[1]), _days(TEST[1])], float)
    EO = np.zeros((len(O), len(seq)), np.float32); EB = np.zeros((len(BT), len(seq)))
    for q, j in enumerate(seq):
        EO[:, q] = oc.E_source(j, tO); EB[:, q] = oc.E_source(j, BT)
    print(f"  E: hazır ({time.time()-t0:.0f} s)", flush=True)
    # taban maskeli sonuçlar (karşılaştırma için)
    base = EM.evaluate(cfg_path, windows=[VAL, TEST])
    th8 = [th["log10_k0"], th["a"], th["log10_c"], th["omega"], th["log10_tau"], th["log10_d"], th["gamma"], th["rho"]]
    tb = np.zeros(len(seq), bool)
    if X["omega_big"] is not None:
        tbs = set(X["tb_idx"]); tb = np.array([j in tbs for j in seq])
    d = dict(seq=seq, O=O, tO=tO, ev_t=ev.t.values, observed=obs, lam_tot=lam_tot, lam_seq=lam_seq.astype(np.float32),
             EO=EO, EB=EB, BT=BT, ts_seq=ts[seq], m_seq=S.magnitude.values[seq], xi_seq=X["xi"][seq], tb=tb,
             th8=np.array(th8), mref=X["mref"], scale35=X["scale35"],
             omega_big=np.nan if X["omega_big"] is None else X["omega_big"],
             base=np.array(json.dumps([{k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()} for r in base])),
             lat_seq=S.latitude.values[seq], lon_seq=S.longitude.values[seq], time_seq=S.time.astype(str).values[seq])
    np.savez_compressed(fn, **d)
    print(f"  hazırlık kaydedildi: {fn.relative_to(ROOT)} ({time.time()-t0:.0f} s)", flush=True)
    return dict(np.load(fn, allow_pickle=True))


def expected_obs(P, cols, times):
    """Hazırlıkta saklanan E matrislerinden: times ya O zamanları (tO) ya da sınır zamanları (BT) olmalı."""
    tO, BT = P["tO"], P["BT"]
    out = np.zeros((len(times), len(cols)))
    for r, t in enumerate(times):
        hit = np.nonzero(BT == t)[0]
        if len(hit):
            out[r] = P["EB"][hit[0], cols]
        else:
            q = np.searchsorted(tO, t)
            assert q < len(tO) and tO[q] == t, "E yalnızca hazırlanan zamanlarda"
            out[r] = P["EO"][q, cols]
    return out


def score(P, nu, m_seq=5.0, windows=(VAL, TEST), self_consistent=False, return_g=False):
    cols = np.nonzero(P["m_seq"] >= m_seq - 0.05)[0]
    lt = P["lam_tot"]; L = P["lam_seq"][:, cols].astype(float)
    tO = P["tO"]
    Pij = L / lt[:, None]
    EO = P["EO"][:, cols].astype(float)
    for _ in range(3 if self_consistent else 1):
        ncum = np.cumsum(Pij, axis=0)
        nbefore = np.vstack([np.zeros(len(cols)), ncum[:-1]])
        g = (nu + nbefore) / (nu + EO)
        if self_consistent:
            lt_new = lt + ((g - 1) * L).sum(1)
            Pij = g * L / lt_new[:, None]
    ncum = np.cumsum(Pij, axis=0)
    nbefore = np.vstack([np.zeros(len(cols)), ncum[:-1]])
    g = (nu + nbefore) / (nu + EO)
    dlog_all = np.log1p(((g - 1) * L).sum(1) / lt)
    out = []
    for w0, w1 in windows:
        T0, T1 = _days(w0), _days(w1)
        iw = np.nonzero((tO >= T0) & (tO < T1))[0]
        dlog = dlog_all[iw].sum()
        # kompanzatör farkı: sınırlar T0, olay zamanları, T1
        Eb = np.vstack([expected_obs(P, cols, [float(T0)]), EO[iw], expected_obs(P, cols, [float(T1)])])
        # her aralığın başındaki n (aralık başı olay dahil)
        n0 = ncum[iw[0] - 1] if iw[0] > 0 else np.zeros(len(cols))
        Nq = np.vstack([n0, ncum[iw]])  # len(iw)+1 aralık
        Ea, Eb2 = Eb[:-1], Eb[1:]
        dcomp = ((nu + Nq) * np.log((nu + Eb2) / (nu + Ea)) - (Eb2 - Ea)).sum()
        out.append(dict(pencere=f"{w0}..{w1}", n=len(iw), dlog=dlog, dcomp=dcomp, dLL=dlog - dcomp))
    if return_g:
        return out, cols, g
    return out


def main(cfg_path):
    P = prepare(cfg_path, m_cand=float(os.environ.get("M_CAND", M_CAND)))
    base = json.loads(str(P["base"]))
    bmap = {r["pencere"]: r for r in base}
    rows = []
    grid_m = [float(x) for x in os.environ.get("M_SEQ", "4.0,4.5,5.0,6.0").split(",")]
    grid_nu = [float(x) for x in os.environ.get("NU", "0.1,0.2,0.3,0.5,1.0,1.2,2.0").split(",")]
    sc = os.environ.get("TUTARLI", "0") == "1"
    for m_seq in grid_m:
        for nu in grid_nu:
            for r in score(P, nu, m_seq, self_consistent=sc):
                b = bmap[r["pencere"]]
                rows.append(dict(m_seq=m_seq, nu=nu, pencere=r["pencere"], n=r["n"], dLL=round(r["dLL"], 1),
                                 LL_olay_basi=round((b["LL"] + r["dLL"]) / b["n_gozlenebilir"], 4),
                                 taban=b["LL_olay_basi"]))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["m_seq", "nu"], columns="pencere", values="LL_olay_basi")
    print(piv.to_string())
    return df




# ---------------------------------------------------------------------------
# Seyrek sürüm: TÜM kaynaklar (M>=3.5) için; P_ij > eşik çiftleri saklanır.
# Kompanzatör, kaynak başına yalnızca n'nin değiştiği olaylarda bölünür (aralarda kapalı biçim).
# ---------------------------------------------------------------------------
def prepare_sparse(cfg_path, thr=1e-4, force=False):
    cfg0 = json.load(open(cfg_path))
    out_dir = ROOT / cfg0["out_dir"]; tag = "_bgmix" if cfg0.get("bg_mix") else ""
    fn = out_dir / (f"dizi_seyrek{tag}.npz" if T_END == "2026-08-01" else f"dizi_seyrek{tag}_{T_END}.npz")
    if fn.exists() and not force:
        return dict(np.load(fn, allow_pickle=True))
    t0 = time.time()
    X = EM.evaluate(cfg_path, expose=True)
    ev, S, obs, lam35 = X["ev"], X["S"], X["observed"], X["lam35"]
    ts = X["ts"]; T_end = _days(T_END); nS = len(S)
    O = np.nonzero(obs & (ev.t.values > ts.min()) & (ev.t.values < T_end))[0]
    tO = ev.t.values[O]; laO = ev.latitude.values[O]; loO = ev.longitude.values[O]
    lam_tot = np.empty(len(O)); SRC = []; EVT = []; LIJ = []
    allc = np.arange(nS)
    for s in range(0, len(O), 400):
        sl = slice(s, s + 400)
        tot, M = lam35(tO[sl], laO[sl], loO[sl], cols=allc)
        lam_tot[sl] = tot
        r, c = np.nonzero(M > thr * tot[:, None])
        SRC.append(c.astype(np.int32)); EVT.append((r + s).astype(np.int32)); LIJ.append(M[r, c].astype(np.float32))
    SRC = np.concatenate(SRC); EVT = np.concatenate(EVT); LIJ = np.concatenate(LIJ)
    print(f"  λ: {len(O)} olay, {len(SRC)} çift ({time.time()-t0:.0f} s)", flush=True)
    import obs_comp as OC
    oc = OC.ObsComp(X, T_end)
    BT = np.array([_days(VAL[0]), _days(VAL[1]), _days(TEST[1])], float)
    order = np.lexsort((EVT, SRC)); SRC, EVT, LIJ = SRC[order], EVT[order], LIJ[order]
    EIJ = np.zeros(len(SRC), np.float32); EB = np.zeros((len(BT), nS))
    bounds = np.r_[0, np.cumsum(np.bincount(SRC, minlength=nS))]
    for j in range(nS):
        EB[:, j] = oc.E_source(j, BT)
        a, b = bounds[j], bounds[j + 1]
        if b > a:
            EIJ[a:b] = oc.E_source(j, tO[EVT[a:b]])
    print(f"  E: hazır ({time.time()-t0:.0f} s)", flush=True)
    base = EM.evaluate(cfg_path, windows=[VAL, TEST])
    d = dict(tO=tO, lam_tot=lam_tot, SRC=SRC, EVT=EVT, LIJ=LIJ, EIJ=EIJ, EB=EB, BT=BT, ts=ts, m=S.magnitude.values,
             base=np.array(json.dumps([{k: (float(v) if not isinstance(v, str) else v) for k, v in r.items()} for r in base])))
    np.savez_compressed(fn, **d)
    print(f"  kaydedildi: {fn.relative_to(ROOT)} ({time.time()-t0:.0f} s)", flush=True)
    return dict(np.load(fn, allow_pickle=True))


def score_sparse(P, nu, m_seq=3.5, windows=(VAL, TEST), return_parts=False, n_iter=0, prior_mean=None):
    """prior_mean: kaynak başına ön-bilgi ortalaması m_j (None → 1). Gamma(ν, ν/m_j)."""
    m = P["m"]; keep = m[P["SRC"]] >= m_seq - 0.05
    SRC, EVT = P["SRC"][keep], P["EVT"][keep]
    L = P["LIJ"][keep].astype(float); Ei = P["EIJ"][keep].astype(float)
    lt = P["lam_tot"]; tO = P["tO"]; ts = P["ts"]; BT = P["BT"]
    Pij = L / lt[EVT]
    pm = np.ones(len(m)) if prior_mean is None else np.asarray(prior_mean, float)
    first = np.r_[True, SRC[1:] != SRC[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(SRC)), 0))
    for it in range(1 + int(n_iter)):
        # kaynak içi (zaman sıralı) dışlayıcı kümülatif
        cs = np.cumsum(Pij)
        cs_before_grp = np.where(grp_start > 0, cs[grp_start - 1], 0.0)
        ncum = cs - cs_before_grp               # dahil
        nbef = ncum - Pij                       # hariç
        g = (nu + nbef) / (nu / pm[SRC] + Ei)
        if it < n_iter:  # öz-tutarlı sorumluluklar: P_ij = g_j λ_ij / λ'_i
            lt_new = lt + np.bincount(EVT, weights=(g - 1) * L, minlength=len(lt))
            Pij = g * L / lt_new[EVT]
    dl = np.log1p(np.bincount(EVT, weights=(g - 1) * L, minlength=len(tO)) / lt)
    srcs = np.nonzero(m >= m_seq - 0.05)[0]
    out = []
    for w0, w1 in windows:
        T0, T1 = float(_days(w0)), float(_days(w1))
        b0 = np.nonzero(BT == T0)[0][0]; b1 = np.nonzero(BT == T1)[0][0]
        iw = (tO >= T0) & (tO < T1)
        dlog = dl[iw].sum()
        # kompanzatör: kaynak başına satırlar [T0 başlangıç] + [penceredeki çiftler] + [T1 bitiş]
        sj = srcs[ts[srcs] < T1]
        inwin = iw[EVT]
        N0 = np.bincount(SRC[(tO[EVT] < T0)], weights=Pij[(tO[EVT] < T0)], minlength=len(m))
        rs = np.r_[sj, SRC[inwin], sj]
        rk = np.r_[np.zeros(len(sj)), 1 + np.arange(inwin.sum()), np.full(len(sj), 1e12)]
        rE = np.r_[P["EB"][b0, sj], Ei[inwin], P["EB"][b1, sj]]
        rdN = np.r_[N0[sj], Pij[inwin], np.zeros(len(sj))]
        o = np.lexsort((rk, rs)); rs, rE, rdN = rs[o], rE[o], rdN[o]
        cN = np.cumsum(rdN); fst = np.r_[True, rs[1:] != rs[:-1]]
        gs = np.maximum.accumulate(np.where(fst, np.arange(len(rs)), 0))
        Nr = cN - np.where(gs > 0, cN[gs - 1], 0.0)
        same = rs[1:] == rs[:-1]
        Ea, Eb_ = rE[:-1][same], rE[1:][same]; Eb_ = np.maximum(Eb_, Ea)
        rpm = np.r_[pm[sj], pm[SRC[inwin]], pm[sj]][o][:-1][same]
        nb_ = nu / rpm
        dcomp = ((nu + Nr[:-1][same]) * np.log((nb_ + Eb_) / (nb_ + Ea)) - (Eb_ - Ea)).sum()
        out.append(dict(pencere=f"{w0}..{w1}", n=int(iw.sum()), dlog=dlog, dcomp=dcomp, dLL=dlog - dcomp))
    if return_parts:
        return out, dict(g=g, SRC=SRC, EVT=EVT, dl=dl)
    return out


def main_sparse(cfg_path):
    P = prepare_sparse(cfg_path)
    base = json.loads(str(P["base"])); bmap = {r["pencere"]: r for r in base}
    rows = []
    for m_seq in [float(x) for x in os.environ.get("M_SEQ", "3.5,4.0,4.5").split(",")]:
        for nu in [float(x) for x in os.environ.get("NU", "0.1,0.2,0.3,0.5,1.0").split(",")]:
            for r in score_sparse(P, nu, m_seq, n_iter=int(os.environ.get("ITER", 0))):
                b = bmap[r["pencere"]]
                rows.append(dict(m_seq=m_seq, nu=nu, pencere=r["pencere"], dLL=round(r["dLL"], 1),
                                 LL_olay_basi=round((b["LL"] + r["dLL"]) / b["n_gozlenebilir"], 4)))
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["m_seq", "nu"], columns="pencere", values="LL_olay_basi").to_string())
    return df


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "seyrek":
        main_sparse(sys.argv[1])
    else:
        main(sys.argv[1])


class Posterior:
    """Tahmin anı T0'da her geçmiş kaynak için verimlilik çarpanının sonsalı: Gamma(ν + n_j(T0), ν + E_j(T0)).
    Simülasyonda (simulate_forecast) geçmiş kaynakların doğrudan artçı beklentisi bu sonsaldan çekilen çarpanla,
    yeni (simüle) olaylarınki Gamma(ν, ν) ön-bilgisinden çekilen çarpanla ölçeklenir (tutarlı üretici model)."""
    def __init__(self, cfg_path, nu):
        import obs_comp as OC
        self.P = prepare_sparse(cfg_path)
        X = EM.evaluate(cfg_path, expose=True)
        self.oc = OC.ObsComp(X, _days(T_END))
        self.nu = float(nu); P = self.P
        self.Pij = P["LIJ"].astype(float) / P["lam_tot"][P["EVT"]]
        self.tE = P["tO"][P["EVT"]]; self.SRC = P["SRC"]; self.nS = len(P["m"]); self.t_last = float(P["tO"].max())

    def at(self, T0, jj):
        sel = self.tE < T0
        n = np.bincount(self.SRC[sel], weights=self.Pij[sel], minlength=self.nS)[jj]
        E = self.oc.E(np.asarray(jj), np.full(len(jj), float(T0)))
        return self.nu + n, self.nu + E
