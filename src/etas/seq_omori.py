"""Dizi-özgü Omori üssü + verimlilik: Bayesçi (ω-ızgarası × Gamma) karışım sonsalı.

seq_update.py'deki Gamma–Poisson verimlilik güncellemesinin genellemesi. Her kaynak j için:
  ω_j = ω_cur,j + δ,  δ ∈ ızgara, önsel π(δ) ∝ N(μ_δ, σ_δ²)   (Omi vd. 2016: p ~ N(1.05, 0.13))
  g_j ~ Gamma(ν, ν) (verimlilik çarpanı, ortalama 1);  ω değişince toplam verimlilik korunur, yalnız zaman dağılımı değişir.
Sonsal ağırlık (G üzerinden integral alınmış), t anına kadar:
  log w_k(t) = log π_k + Σ_{i<t} P_ij log r_k(t_i − t_j) − (ν + N_j) log(ν + E_j^(k)(t))
  r_k = pdf_k / pdf_cur  (normalize Omori-τ zaman yoğunluklarının oranı)
Öngörü yoğunluğu:  λ'_j(t) = λ_j(t) · Σ_k w_k (ν+N)/(ν+E^(k)) r_k(t − t_j)
Kompanzatör (n sabit aralık a→b):  −log Σ_k w_k(a) ((ν+E_k(a))/(ν+E_k(b)))^{ν+N}   (kapalı biçim, sağkalım olasılığı)
δ = 0 tek noktalı ızgara seq_update.score_sparse ile aynı sonucu verir (test edildi).
Kullanım: python3 src/etas/seq_omori.py configs/v3_referans.json   (hazırlık + σ/μ taraması)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import seq_update as SU  # noqa: E402
import evaluate_masked as EM  # noqa: E402
import finite_source as fs  # noqa: E402
from etas.inversion import upper_gamma_ext  # noqa: E402

DELTAS = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def _logr(dt, om_cur, om_new, c, tau):
    return np.log(fs.big_time_ratio(dt, om_cur, om_new, c, tau, upper_gamma_ext))


def prepare_omori(cfg_path, deltas=DELTAS, force=False):
    cfg0 = json.load(open(cfg_path)); out_dir = ROOT / cfg0["out_dir"]
    tag = "_bgmix" if cfg0.get("bg_mix") else ""
    suf = "" if SU.T_END == "2026-08-01" else f"_{SU.T_END}"
    fn = out_dir / f"dizi_omori{tag}{suf}.npz"
    if fn.exists() and not force:
        return dict(np.load(fn, allow_pickle=True))
    t0 = time.time()
    P = SU.prepare_sparse(cfg_path)
    X = EM.evaluate(cfg_path, expose=True)
    import obs_comp as OC
    oc = OC.ObsComp(X, SU._days(SU.T_END))
    th = X["th"]; om0 = th["omega"]; ob = X["omega_big"]
    c, tau = 10 ** th["log10_c"], 10 ** th["log10_tau"]
    om_cur = np.where(oc.tb, ob if ob is not None else om0, om0)
    SRC, EVT, tO, ts, BT = P["SRC"], P["EVT"], P["tO"], P["ts"], P["BT"]
    nS = len(P["m"]); K = len(deltas)
    bounds = np.r_[0, np.cumsum(np.bincount(SRC, minlength=nS))]
    EIJ = np.zeros((len(SRC), K), np.float32); EB = np.zeros((len(BT), nS, K)); LR = np.zeros((len(SRC), K), np.float32)
    dt_e = tO[EVT] - ts[SRC]
    for k, dl in enumerate(deltas):
        for grp in [oc.tb[SRC], ~oc.tb[SRC]]:
            if grp.any():
                oc_ = float(om_cur[SRC[grp]][0])
                LR[grp, k] = _logr(dt_e[grp], oc_, oc_ + dl, c, tau)
        for j in range(nS):
            EB[:, j, k] = oc.E_source(j, BT, omega=om_cur[j] + dl)
            a, b = bounds[j], bounds[j + 1]
            if b > a:
                EIJ[a:b, k] = oc.E_source(j, tO[EVT[a:b]], omega=om_cur[j] + dl)
        print(f"  δ={dl:+.1f} hazır ({time.time()-t0:.0f} s)", flush=True)
    np.savez_compressed(fn, deltas=deltas, EIJ=EIJ, EB=EB, LR=LR, om_cur=om_cur)
    print(f"  kaydedildi: {fn.relative_to(ROOT)} ({time.time()-t0:.0f} s)", flush=True)
    return dict(np.load(fn, allow_pickle=True))


def _grp_excl_cumsum(x, first):
    """Grup içi dışlayıcı kümülatif toplam (satırlar grup → zaman sıralı); x (n,) ya da (n,K)."""
    cs = np.cumsum(x, axis=0)
    gs = np.maximum.accumulate(np.where(first, np.arange(len(first)), 0))
    before = np.where((gs > 0)[:, None] if x.ndim == 2 else gs > 0, cs[gs - 1], 0.0)
    return cs - before - x


def log_prior(deltas, mu, sig, m, m_p):
    K = len(deltas)
    lp = -0.5 * ((deltas - mu) / sig) ** 2; lp -= logsumexp(lp)
    lp0 = np.full(K, -np.inf); lp0[np.argmin(np.abs(deltas))] = 0.0
    return np.where((m >= m_p - 0.05)[:, None], lp[None, :], lp0[None, :])  # nS × K


def score_omori(P, Q, nu, mu=0.0, sig=0.15, m_p=3.5, windows=(SU.VAL, SU.TEST), return_parts=False):
    SRC, EVT = P["SRC"], P["EVT"]; L = P["LIJ"].astype(float); lt = P["lam_tot"]; tO = P["tO"]; ts = P["ts"]; BT = P["BT"]
    m = P["m"]; deltas = Q["deltas"]; K = len(deltas); k0 = int(np.argmin(np.abs(deltas)))
    EIJ = Q["EIJ"].astype(float); LR = Q["LR"].astype(float); EB = Q["EB"]
    LP = log_prior(deltas, mu, sig, m, m_p)
    Pij = L / lt[EVT]
    first = np.r_[True, SRC[1:] != SRC[:-1]]
    Nb = _grp_excl_cumsum(Pij, first)
    Sb = _grp_excl_cumsum(Pij[:, None] * LR, first)
    lw = LP[SRC] + Sb - (nu + Nb)[:, None] * np.log(nu + EIJ)
    lw -= logsumexp(lw, axis=1, keepdims=True)
    gpred = (np.exp(lw) * (nu + Nb)[:, None] / (nu + EIJ) * np.exp(LR)).sum(1)
    dl = np.log1p(np.bincount(EVT, weights=(gpred - 1) * L, minlength=len(tO)) / lt)
    srcs = np.arange(len(m)); out = []
    tE = tO[EVT]
    for w0, w1 in windows:
        T0, T1 = float(SU._days(w0)), float(SU._days(w1))
        b0 = np.nonzero(BT == T0)[0][0]; b1 = np.nonzero(BT == T1)[0][0]
        iw = (tO >= T0) & (tO < T1)
        sj = srcs[ts < T1]
        pre = tE < T0; inw = (tE >= T0) & (tE < T1)
        N0 = np.bincount(SRC[pre], weights=Pij[pre], minlength=len(m))
        S0 = np.stack([np.bincount(SRC[pre], weights=(Pij * LR[:, k])[pre], minlength=len(m)) for k in range(K)], 1)
        rs = np.r_[sj, SRC[inw], sj]
        rk = np.r_[np.zeros(len(sj)), 1 + np.arange(inw.sum()), np.full(len(sj), 1e12)]
        rN = np.r_[N0[sj], (Nb + Pij)[inw], np.zeros(len(sj))]
        rS = np.vstack([S0[sj], (Sb + Pij[:, None] * LR)[inw], np.zeros((len(sj), K))])
        rE = np.vstack([EB[b0, sj, :], EIJ[inw], EB[b1, sj, :]])
        o = np.lexsort((rk, rs)); rs, rN, rS, rE = rs[o], rN[o], rS[o], rE[o]
        same = rs[1:] == rs[:-1]
        Na, Sa, Ea, Eb_ = rN[:-1][same], rS[:-1][same], rE[:-1][same], np.maximum(rE[1:][same], rE[:-1][same])
        lwa = LP[rs[:-1][same]] + Sa - (nu + Na)[:, None] * np.log(nu + Ea)
        lwa -= logsumexp(lwa, axis=1, keepdims=True)
        term = -logsumexp(lwa + (nu + Na)[:, None] * (np.log(nu + Ea) - np.log(nu + Eb_)), axis=1)
        dcomp = (term - (Eb_[:, k0] - Ea[:, k0])).sum()
        out.append(dict(pencere=f"{w0}..{w1}", n=int(iw.sum()), dlog=dl[iw].sum(), dcomp=dcomp, dLL=dl[iw].sum() - dcomp))
    if return_parts:
        return out, dict(dl=dl, lw=lw)
    return out


def main(cfg_path):
    P = SU.prepare_sparse(cfg_path); Q = prepare_omori(cfg_path)
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    def fmt(res):
        return [round((base[r["pencere"]]["LL"] + r["dLL"]) / base[r["pencere"]]["n_gozlenebilir"], 4) for r in res]
    # kontrol: δ=0 tek nokta ≡ seq_update
    print("kontrol (yalnız verimlilik, ν=0.3):", fmt(score_omori(P, Q, 0.3, sig=1e-6)), "  seq_update:", fmt(SU.score_sparse(P, 0.3, 3.5)))
    rows = []
    for nu in [float(x) for x in os.environ.get("NU", "0.3").split(",")]:
        for m_p in [float(x) for x in os.environ.get("M_P", "3.5,4.5").split(",")]:
            for mu in [float(x) for x in os.environ.get("MU", "0.0,0.1,0.2").split(",")]:
                for sig in [float(x) for x in os.environ.get("SIG", "0.1,0.2,0.3").split(",")]:
                    v, t = fmt(score_omori(P, Q, nu, mu, sig, m_p))
                    rows.append(dict(nu=nu, m_p=m_p, mu=mu, sig=sig, dogrulama=v, test=t)); print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = main(sys.argv[1])
    print(df.sort_values("dogrulama", ascending=False).head(10).to_string(index=False))


class OmoriPosterior:
    """T0 anında geçmiş kaynaklar için ortak sonsal: w_k (δ ağırlıkları), Gamma(ν+N, ν+E_k(T0))."""
    def __init__(self, cfg_path, nu, mu=0.0, sig=0.2, m_p=4.5):
        import obs_comp as OC
        self.P = SU.prepare_sparse(cfg_path); self.Q = prepare_omori(cfg_path)
        X = EM.evaluate(cfg_path, expose=True)
        self.oc = OC.ObsComp(X, SU._days(SU.T_END))
        self.nu = float(nu); self.deltas = self.Q["deltas"]; self.om_cur = self.Q["om_cur"]
        P = self.P; self.m = P["m"]; self.m_p = m_p
        self.LP = log_prior(self.deltas, mu, sig, self.m, m_p)
        self.lp_new = log_prior(self.deltas, mu, sig, np.array([9.0]), m_p)[0]  # yeni olaylar (M>=m_p) için önsel
        self.Pij = P["LIJ"].astype(float) / P["lam_tot"][P["EVT"]]
        self.tE = P["tO"][P["EVT"]]; self.SRC = P["SRC"]; self.nS = len(self.m); self.t_last = float(P["tO"].max())
        self.k0 = int(np.argmin(np.abs(self.deltas)))

    def at(self, T0, jj):
        jj = np.asarray(jj); K = len(self.deltas)
        sel = self.tE < T0
        N = np.bincount(self.SRC[sel], weights=self.Pij[sel], minlength=self.nS)[jj]
        S = np.stack([np.bincount(self.SRC[sel], weights=(self.Pij * self.Q["LR"][:, k])[sel], minlength=self.nS)[jj]
                      for k in range(K)], 1)
        E = np.repeat(self.oc.E(jj, np.full(len(jj), float(T0)))[:, None], K, 1)
        full = self.m[jj] >= self.m_p - 0.05
        for q in np.nonzero(full)[0]:
            j = jj[q]
            E[q] = [self.oc.E_source(j, [float(T0)], omega=self.om_cur[j] + d)[0] for d in self.deltas]
        lw = self.LP[jj] + S - (self.nu + N)[:, None] * np.log(self.nu + E)
        lw -= logsumexp(lw, axis=1, keepdims=True)
        return self.nu + N, self.nu + E, np.exp(lw)


# ---------------------------------------------------------------------------
# GÜNLÜK güncelleme kipi (literatürdeki 1 günlük tahminlerle karşılaştırılabilir):
#  - gün d'deki hedefler için yalnızca gün başından (UTC 00:00) önceki kaynaklar kullanılır (aynı gün tetiklemesi yok);
#  - dizi sonsalındaki sayımlar (N, S) gün başında dondurulur; E_k(t) deterministik olduğundan sürekli kalır
#    (ileriye bakış yok); kompanzatör aynı kapalı biçimle, sayım değişimleri ertesi gece yarısında.
#  ν → ∞ ve tek δ = 0 ile güncellemesiz ETAS'ın günlük sürümü elde edilir.
# ---------------------------------------------------------------------------
def prepare_daily(cfg_path, deltas=DELTAS, force=False):
    cfg0 = json.load(open(cfg_path)); out_dir = ROOT / cfg0["out_dir"]
    tag = "_bgmix" if cfg0.get("bg_mix") else ""
    fn = out_dir / f"dizi_gunluk{tag}_K{len(deltas)}.npz"
    if fn.exists() and not force:
        return dict(np.load(fn, allow_pickle=True))
    t0 = time.time()
    P = SU.prepare_sparse(cfg_path)
    X = EM.evaluate(cfg_path, expose=True)
    import obs_comp as OC
    oc = OC.ObsComp(X, SU._days(SU.T_END))
    th = X["th"]; om0 = th["omega"]; ob = X["omega_big"]
    om_cur = np.where(oc.tb, ob if ob is not None else om0, om0)
    SRC, EVT, tO, ts = P["SRC"], P["EVT"], P["tO"], P["ts"]
    nS = len(P["m"]); K = len(deltas)
    bounds = np.r_[0, np.cumsum(np.bincount(SRC, minlength=nS))]
    EC = np.zeros((len(SRC), K), np.float32); ES = np.zeros((nS, K))
    cE = np.ceil(tO[EVT]); cS = np.ceil(ts)
    for k, dl in enumerate(deltas):
        for j in range(nS):
            ES[j, k] = oc.E_source(j, [cS[j]], omega=om_cur[j] + dl)[0]
            a, b = bounds[j], bounds[j + 1]
            if b > a:
                EC[a:b, k] = oc.E_source(j, cE[a:b], omega=om_cur[j] + dl)
        print(f"  günlük δ={dl:+.1f} ({time.time()-t0:.0f} s)", flush=True)
    np.savez_compressed(fn, deltas=deltas, EC=EC, ES=ES)
    return dict(np.load(fn, allow_pickle=True))


def score_daily(P, Q, D, nu, mu=0.0, sig=0.2, m_p=4.5, windows=(SU.VAL, SU.TEST), base_only=False):
    """Günlük güncellemeli öngörü: olay başına Δlog (sürekli taban λ'ya göre) ve pencere kompanzatör farkı."""
    SRC, EVT = P["SRC"], P["EVT"]; L = P["LIJ"].astype(float); lt = P["lam_tot"]; tO = P["tO"]; ts = P["ts"]; BT = P["BT"]
    m = P["m"]; deltas = Q["deltas"]
    if base_only:  # güncellemesiz ETAS: tek δ=0, ν → ∞
        kk = [int(np.argmin(np.abs(deltas)))]; nu = 1e9; sig = 1e-6; m_p = 99
    else:
        kk = list(range(len(deltas)))
    dsel = np.asarray(deltas)[kk]; K = len(kk); k0 = int(np.argmin(np.abs(dsel)))
    EIJ = Q["EIJ"][:, kk].astype(float); LR = Q["LR"][:, kk].astype(float); EB = Q["EB"][:, :, kk]
    dk = [int(np.argmin(np.abs(D["deltas"] - d))) for d in dsel]
    EC = D["EC"][:, dk].astype(float); ES = D["ES"][:, dk]
    LP = log_prior(dsel, mu, sig, m, m_p)
    Pij = L / lt[EVT]; tE = tO[EVT]; ds = np.floor(tE)
    same = ts[SRC] >= ds                       # kaynak hedefle aynı gün → günlük tahminde yok
    # gün başına kadar (tE < ds_e) aynı kaynağın çocuk sayımları
    key = SRC * 1e5 + tE; csP = np.r_[0.0, np.cumsum(Pij)]; csS = np.vstack([np.zeros(K), np.cumsum(Pij[:, None] * LR, 0)])
    first = np.r_[True, SRC[1:] != SRC[:-1]]; gstart = np.maximum.accumulate(np.where(first, np.arange(len(SRC)), 0))
    pos = np.searchsorted(key, SRC * 1e5 + ds, side="left")
    Nds = csP[pos] - csP[gstart]; Sds = csS[pos] - csS[gstart]
    lw = LP[SRC] + Sds - (nu + Nds)[:, None] * np.log(nu + EIJ); lw -= logsumexp(lw, axis=1, keepdims=True)
    gpred = (np.exp(lw) * (nu + Nds)[:, None] / (nu + EIJ) * np.exp(LR)).sum(1)
    contrib = np.where(same, -L, (gpred - 1) * L)
    lam_d = lt + np.bincount(EVT, weights=contrib, minlength=len(tO))
    dl = np.log(np.maximum(lam_d, 1e-300) / lt)
    out = []
    posE = np.searchsorted(key, SRC * 1e5 + np.ceil(tE), side="left")   # gece yarısı c_e'den önceki tüm çocuklar
    Nc = csP[posE] - csP[gstart]; Sc = csS[posE] - csS[gstart]
    for w0, w1 in windows:
        T0, T1 = float(SU._days(w0)), float(SU._days(w1))
        b0 = np.nonzero(BT == T0)[0][0]; b1 = np.nonzero(BT == T1)[0][0]
        iw = (tO >= T0) & (tO < T1)
        sj = np.nonzero(ts < T1)[0]
        st = np.where(ts[sj] < T0, T0, np.ceil(ts[sj])); okj = st < T1; sj, st = sj[okj], st[okj]
        Est = np.where((ts[sj] < T0)[:, None], EB[b0, sj, :], ES[sj, :])
        # başlangıçtaki sayımlar: st'den önceki çocuklar
        kst = np.searchsorted(key, sj * 1e5 + st, side="left")
        gst = np.searchsorted(key, sj * 1e5 - 0.5, side="left")
        Nst = csP[kst] - csP[gst]; Sst = csS[kst] - csS[gst]
        ce = np.ceil(tE); chg = (ce > np.maximum(T0, np.ceil(ts[SRC]))) & (ce < T1)
        rs = np.r_[sj, SRC[chg], sj]
        rt = np.r_[st, ce[chg], np.full(len(sj), T1)]
        rk = np.r_[np.zeros(len(sj)), 1 + np.arange(chg.sum()), np.full(len(sj), 1e12)]
        rN = np.r_[Nst, Nc[chg], np.zeros(len(sj))]
        rS = np.vstack([Sst, Sc[chg], np.zeros((len(sj), K))])
        rE = np.vstack([Est, EC[chg], EB[b1, sj, :]])
        o = np.lexsort((rk, rt, rs)); rs, rN, rS, rE = rs[o], rN[o], rS[o], rE[o]
        sm = rs[1:] == rs[:-1]
        Na, Sa, Ea = rN[:-1][sm], rS[:-1][sm], rE[:-1][sm]; Eb_ = np.maximum(rE[1:][sm], Ea)
        lwa = LP[rs[:-1][sm]] + Sa - (nu + Na)[:, None] * np.log(nu + Ea); lwa -= logsumexp(lwa, axis=1, keepdims=True)
        if base_only:
            term = (Eb_ - Ea)[:, 0]
        else:
            term = -logsumexp(lwa + (nu + Na)[:, None] * (np.log(nu + Ea) - np.log(nu + Eb_)), axis=1)
        base_trig = (EB[b1, ts < T1, k0] - EB[b0, ts < T1, k0]).sum()
        dcomp = term.sum() - base_trig
        out.append(dict(pencere=f"{w0}..{w1}", n=int(iw.sum()), dlog=dl[iw].sum(), dcomp=dcomp, dLL=dl[iw].sum() - dcomp))
    return out, dl
