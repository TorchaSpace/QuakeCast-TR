"""Sonlu kaynaklı (çizgisel) ETAS çekirdeği — büyük depremler için.

Neden: dairesel (nokta kaynak) çekirdek, yüzlerce km'lik yırtılmalarda artçıları merkez üssünde yoğunlaştırır;
bu hem mekânsal tahmini bozar hem de verimlilik üssünü (α) aşağı çeker (Hainzl vd. 2008; Guo vd. 2019 GRL).

Geometri: M >= M_FS her ana şok için ilk 24 saatteki artçıların (M>=2.5) asal ekseni (PCA), uzunluk = projeksiyonların
%2.5-%97.5 aralığı. Eş zamanlı daha büyük bir ana şokun hattına 15 km'den yakın artçılar dışarıda bırakılır
(ör. 6 Şubat 2023: önce M7.8 Doğu Anadolu Fayı, sonra M7.6 Çardak). En az 20 artçı yoksa nokta kaynak kalır.
Not: geometri ilk 24 saatin verisiyle belirlenir; gerçek zamanda sonlu fay modelleri saatler içinde yayınlanır.

Çekirdek: uzaklık r = hedefin yırtılma doğru parçasına uzaklığı.
  f(r) = (r^2 + D)^-(1+ρ) / Z,  Z = π D^-ρ / ρ + L √π Γ(ρ+½)/Γ(ρ+1) D^-(ρ+½)
Nokta kaynakla aynı toplam beklenen artçı sayısını korumak için tetiklenme yoğunluğu κ = Z0/Z ile çarpılır
(Z0 = π D^-ρ / ρ); M-adımı olabilirliğine de -log(Z/Z0) terimi eklenir.
"""
import numpy as np
import pandas as pd
from scipy.special import gammaln

M_FS = 6.0
R_EARTH = 6378.137


def wc_length(m):
    return 10 ** (-2.44 + 0.59 * np.asarray(m))


def _xy(lat, lon, lat0, lon0):
    return (np.radians(np.asarray(lon) - lon0) * R_EARTH * np.cos(np.radians(lat0)),
            np.radians(np.asarray(lat) - lat0) * R_EARTH)


def build_ruptures(cat, m_min=M_FS, hours=24, out_csv=None):
    """cat: time (datetime), latitude, longitude, magnitude."""
    cat = cat.sort_values("time").reset_index(drop=True)
    mains = cat[cat.magnitude >= m_min].sort_values("magnitude", ascending=False)
    rups = []
    for _, ms in mains.iterrows():
        L_wc = wc_length(ms.magnitude)
        w = cat[(cat.time > ms.time) & (cat.time <= ms.time + pd.Timedelta(hours=hours))]
        x, y = _xy(w.latitude.values, w.longitude.values, ms.latitude, ms.longitude)
        sel = np.hypot(x, y) <= max(1.5 * L_wc, 20)
        # eş zamanlı daha büyük ana şokların hatlarına yakın olanları çıkar
        for r in rups:
            if abs((r["time"] - ms.time).total_seconds()) < 3 * 86400 and r["magnitude"] >= ms.magnitude and r["finite"]:
                xa, ya = _xy([r["lat1"], r["lat2"]], [r["lon1"], r["lon2"]], ms.latitude, ms.longitude)
                d = _seg_dist(x, y, xa[0], ya[0], xa[1], ya[1])
                sel &= d > 15
        x, y = x[sel], y[sel]
        row = dict(time=ms.time, latitude=ms.latitude, longitude=ms.longitude, magnitude=ms.magnitude,
                   n_artci=int(sel.sum()), L_wc=L_wc, finite=False, lat1=ms.latitude, lon1=ms.longitude,
                   lat2=ms.latitude, lon2=ms.longitude, L=0.0, kirik_cizgi="")
        if len(x) >= 20:
            P = np.c_[x, y]; cen = P.mean(0)
            u = np.linalg.svd(P - cen, full_matrices=False)[2][0]
            proj = (P - cen) @ u
            lo, hi = np.percentile(proj, [2.5, 97.5])
            L = hi - lo
            if L >= 0.3 * L_wc:
                p1 = cen + lo * u; p2 = cen + hi * u
                lat_of = lambda q: ms.latitude + np.degrees(q[1] / R_EARTH)
                lon_of = lambda q: ms.longitude + np.degrees(q[0] / (R_EARTH * np.cos(np.radians(ms.latitude))))
                # kırık çizgi: eksen boyunca bölmelerde dik sapmanın medyanı (eğrisel yırtılmalar için)
                v = np.array([-u[1], u[0]]); perp = (P - cen) @ v
                nb = int(np.clip(L / 25, 1, 12)); edges = np.linspace(lo, hi, nb + 1)
                verts = []
                for b in range(nb):
                    mb = (proj >= edges[b]) & (proj <= edges[b + 1])
                    off = np.median(perp[mb]) if mb.sum() >= 5 else 0.0
                    verts.append((0.5 * (edges[b] + edges[b + 1]), off))
                pts = [(lo, verts[0][1])] + verts + [(hi, verts[-1][1])]
                xy = [cen + a_ * u + o_ * v for a_, o_ in pts]
                poly = ";".join(f"{lat_of(q):.5f},{lon_of(q):.5f}" for q in xy)
                Lp = float(sum(np.hypot(*(np.array(xy[i + 1]) - np.array(xy[i]))) for i in range(len(xy) - 1)))
                row.update(finite=True, lat1=lat_of(p1), lon1=lon_of(p1), lat2=lat_of(p2), lon2=lon_of(p2), L=Lp, kirik_cizgi=poly)
        rups.append(row)
    R = pd.DataFrame(rups).sort_values("time").reset_index(drop=True)
    if out_csv is not None:
        R.to_csv(out_csv, index=False)
    return R


def _seg_dist(x, y, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / np.where(L2 > 0, L2, 1), 0, 1) if L2 > 0 else 0
    return np.hypot(x - (x1 + t * dx), y - (y1 + t * dy))


def dist2_to_rupture(lat, lon, r):
    """r: satır (kirik_cizgi 'lat,lon;lat,lon;...' ya da lat1..lon2). Dönüş km^2 (en yakın segmente)."""
    pl = r.get("kirik_cizgi", "") if hasattr(r, "get") else ""
    if isinstance(pl, str) and pl:
        V = np.array([[float(a) for a in q.split(",")] for q in pl.split(";")])
    else:
        V = np.array([[r["lat1"], r["lon1"]], [r["lat2"], r["lon2"]]])
    lat0, lon0 = V[:, 0].mean(), V[:, 1].mean()
    x, y = _xy(lat, lon, lat0, lon0)
    vx, vy = _xy(V[:, 0], V[:, 1], lat0, lon0)
    d = np.full(np.shape(x), np.inf)
    for i in range(len(V) - 1):
        d = np.minimum(d, _seg_dist(x, y, vx[i], vy[i], vx[i + 1], vy[i + 1]))
    return d ** 2


def log_Z_ratio(L, D, rho):
    """log(Z/Z0) = log(1 + L ρ Γ(ρ+½)/(√π Γ(ρ+1)) D^-½)"""
    L = np.asarray(L, float)
    coef = np.exp(np.log(rho) + gammaln(rho + 0.5) - gammaln(rho + 1) - 0.5 * np.log(np.pi))
    return np.log1p(L * coef / np.sqrt(D))


def match_sources(times, mags, R, tol_s=2.0):
    """Kaynak olayları (zaman, büyüklük) yırtılma tablosuyla eşleştir; eşleşmeyen -> -1."""
    idx = np.full(len(times), -1)
    rt = pd.to_datetime(R.time).values.astype("datetime64[ms]").astype(np.int64) / 1000
    tt = pd.to_datetime(times).values.astype("datetime64[ms]").astype(np.int64) / 1000
    for k in np.nonzero(R.finite.values)[0]:
        j = np.nonzero((np.abs(tt - rt[k]) <= tol_s) & (np.abs(np.asarray(mags) - R.magnitude.values[k]) <= 0.15))[0]
        if len(j):
            idx[j[np.argmin(np.abs(tt[j] - rt[k]))]] = k
    return idx


def patch_calc(calc, R, inversion_module):
    """Paket nesnesine sonlu kaynak uygular: uzaklıkları değiştirir, çekirdek ve olabilirliği sarmalar."""
    cat = calc.catalog
    src_ids = calc.source_events.index.values
    k_of = match_sources(cat.loc[src_ids, "time"].values, cat.loc[src_ids, "magnitude"].values, R)
    L_by_src = pd.Series(np.where(k_of >= 0, R.L.values[np.maximum(k_of, 0)], 0.0), index=src_ids)
    dist = calc.distances
    te = calc.target_events
    for s, k in zip(src_ids, k_of):
        if k < 0:
            continue
        try:
            sub = dist.xs(s, level="source_id", drop_level=False)
        except KeyError:
            continue
        tids = sub.index.get_level_values("target_id")
        d2 = dist2_to_rupture(te.loc[tids, "latitude"].values, te.loc[tids, "longitude"].values, R.iloc[k])
        dist.loc[sub.index, "spatial_distance_squared"] = d2
    calc.distances = dist
    orig_kernel = inversion_module.triggering_kernel
    orig_nll = inversion_module.neg_log_likelihood

    def kernel(metrics, params):
        res = orig_kernel(metrics, params)
        m = metrics[2]
        if isinstance(m, pd.Series) and isinstance(m.index, pd.MultiIndex):
            theta, mc = params
            d = 10 ** theta[7]; g = theta[8]; rho = theta[9]
            Ls = L_by_src.reindex(m.index.get_level_values("source_id")).fillna(0).values
            D = d * np.exp(g * (m.values - mc))
            res = res * np.exp(-log_Z_ratio(Ls, D, rho))
        return res

    def nll(theta, Pij, source_events, mc_min):
        base = orig_nll(theta, Pij, source_events, mc_min)
        log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho = theta
        Ls = L_by_src.reindex(Pij.index.get_level_values("source_id")).fillna(0).values
        if not (Ls > 0).any():
            return base
        D = 10 ** log10_d * np.exp(gamma * (Pij["source_magnitude"].values - mc_min))
        corr = (Pij["Pij"].values * Pij["zeta_plus_1"].values * log_Z_ratio(Ls, D, rho))[Ls > 0].sum()
        return base + corr

    inversion_module.triggering_kernel = kernel
    inversion_module.neg_log_likelihood = nll
    return int((k_of >= 0).sum())
