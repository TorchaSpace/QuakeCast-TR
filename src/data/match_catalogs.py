"""Kataloglar arası olay eşleştirme (v2) ve çakışan (aynı deprem, farklı değer) kayıtların işaretlenmesi.

Kaynaklar: AFAD (public API + web kataloğundan olay türü), Kandilli (KOERI), varsa TURHEC.
Çıktılar: data/interim/eslestirme/*.txt (sekme ayrımlı) ve OZET.txt

Yöntem
1) Aday çiftler: |Δt| <= 60 s ve mesafe <= 150 km.
2) Dönem bazlı (1990-99, 2000-06, 2007-14, 2015+) sistematik zaman kayması ve saçılımlar
   (σt, σd, σM) güvenilir ön eşleşmelerden (tek adaylı, M>=3) robust olarak ölçülür.
3) Maliyet z = ((Δt-kayma)/σt)^2 + (mesafe/σd)^2 + ((ΔM-bias)/σM)^2  (ΔM yoksa terim 1 kabul edilir)
4) Aday grafiğinin her bağlı bileşeninde global en iyi bire-bir atama (Macar algoritması);
   z > Z_MAX ise eşleşme yapılmaz (eşleşmemek eşleşmekten "ucuz" olur).
5) Güven: en iyi alternatifin maliyet farkı (marj) < 4 ise BELIRSIZ, z > 9 ise ORTA, diğerleri YUKSEK.
6) Sahte eşleşme testi: Kandilli zamanları ±3600 s kaydırılıp aynı prosedür çalıştırılır;
   bulunan eşleşmeler rastlantısaldır -> gerçek koşudaki yanlış eşleşme sayısının tahmini.
Fark işaretleri (eşleşen çiftlerde): ZAMAN >2 s, KONUM >10 km, DERINLIK >10 km, BUYUKLUK >0.2 (aynı tür).
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "eslestirme"
OUT.mkdir(parents=True, exist_ok=True)

T_CAND, D_CAND = 60.0, 150.0
Z_MAX = 80.0
ZBINS = np.array([0, 2, 4, 6, 9, 12, 16, 20, 25, 30, 40, 50, 60, 80.0])
SHIFTS = (3600.0, -3600.0, 7200.0, -7200.0, 10800.0, -10800.0, 14400.0, -14400.0)
MBINS = np.array([-1, 2.0, 3.0, 4.0, 10.0])  # yanlışlık olasılığı büyüklük sınıfına göre ayrı ölçülür
P_KESIN, P_MUHTEMEL = 0.005, 0.05
THR = dict(dt=2.0, dist=10.0, ddep=10.0, dmag=0.2)
ERAS = [(1900, 1999), (2000, 2006), (2007, 2014), (2015, 2100)]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def ndec(s):
    return s.str.split(".").str[1].fillna("").str.rstrip("0").str.len()


def load_afad():
    fs = sorted(glob.glob(str(RAW / "afad" / "afad_*.txt")))
    d = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    d = d.drop_duplicates("eventID")
    mt = d.type.str.upper().replace({"MWP": "MW", "MS(BB)": "MS"})
    out = pd.DataFrame({
        "id": d.eventID, "time": pd.to_datetime(d.date, format="ISO8601"),
        "lat": num(d.latitude), "lon": num(d.longitude), "depth": num(d.depth),
        "mag": num(d.magnitude), "magtype": mt, "place": d.location,
        "koord_ondalik": np.minimum(ndec(d.latitude), ndec(d.longitude)),
    })
    out["kaba_zaman"] = (out.time.dt.second == 0) & (out.time.dt.microsecond == 0)
    for t in ["ML", "MD", "MW"]:
        out[t] = np.where(mt == t, out.mag, np.nan)
    # olay türü: web kataloğundaki deprem-dışı kayıtlar
    ws = glob.glob(str(RAW / "afad_web" / "afad_web_digertur_*.txt"))
    typ = {}
    if ws:
        w = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in ws])
        typ = dict(zip(w.refId, w.eventType))
    out["olay_turu"] = out.id.map(typ).fillna("Deprem")
    return out


def load_kandilli():
    fs = sorted(glob.glob(str(RAW / "kandilli" / "kandilli_*.txt")))
    d = pd.concat([pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    d = d.drop_duplicates(["event_code", "time", "lat", "lon"])
    t = pd.to_datetime(d["date"].str.replace(".", "-", regex=False) + " " + d["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    z = lambda c: num(d[c]).where(num(d[c]) > 0)
    out = pd.DataFrame({
        "id": d.event_code + "_" + d["no"], "time": t,
        "lat": num(d.lat), "lon": num(d.lon), "depth": num(d.depth),
        "mag": z("xM"), "magtype": "xM", "place": d.location,
        "ML": z("ML"), "MD": z("MD"), "MW": z("Mw"),
        "koord_ondalik": np.minimum(ndec(d.lat), ndec(d.lon)),
        "olay_turu": d["type"].map({"Ke": "Deprem", "Sm": "Patlatma(Sm)"}).fillna(d["type"]),
    })
    out["kaba_zaman"] = (out.time.dt.second == 0) & (out.time.dt.microsecond == 0)
    return out


def drop_exact_duplicates(C, name, log):
    """Aynı katalogda birebir tekrar (Δt<0.05 s, <0.5 km, ΔM<0.05) kayıtlar: ilki tutulur."""
    C = C.sort_values(["time", "id"]).reset_index(drop=True)
    t = C.time.values.astype("datetime64[ms]").astype(np.int64) / 1000.0
    i = np.nonzero(np.diff(t) < 0.05)[0]
    dist = haversine(C.lat.values[i], C.lon.values[i], C.lat.values[i + 1], C.lon.values[i + 1])
    dup = i[(dist < 0.5) & (np.abs(np.nan_to_num(C.mag.values[i] - C.mag.values[i + 1], nan=9)) < 0.05)] + 1
    if len(dup):
        C.iloc[dup].to_csv(OUT / f"tekrar_kayit_{name.lower()}.txt", sep="\t", index=False)
    log.append(f"{name}: birebir tekrar kayıt {len(dup)} adet ayıklandı (tekrar_kayit_{name.lower()}.txt)")
    return C.drop(index=dup).reset_index(drop=True)


def load_turhec():
    fs = sorted(glob.glob(str(RAW / "turhec" / "*.txt")))
    if not fs:
        return None
    raise NotImplementedError("TURHEC sütun eşlemesi dosya geldikten sonra eklenecek")


def haversine(lat1, lon1, lat2, lon2):
    r = np.pi / 180
    a = np.sin((lat2 - lat1) * r / 2) ** 2 + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin((lon2 - lon1) * r / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def same_type_mag(A, B, ia, ib):
    """Aynı türde büyüklük (MW>ML>MD), yoksa ana büyüklük. (ma, mb, tür) döner."""
    ma = A.mag.values[ia].astype(float); mb = B.mag.values[ib].astype(float)
    mt = np.full(len(ia), "ana", dtype=object)
    for t in ["MW", "ML", "MD"]:
        x = A[t].values[ia].astype(float); y = B[t].values[ib].astype(float)
        ok = ~np.isnan(x) & ~np.isnan(y) & (mt == "ana")
        ma = np.where(ok, x, ma); mb = np.where(ok, y, mb); mt = np.where(ok, t, mt)
    return ma, mb, mt


def era_of(years):
    e = np.zeros(len(years), int)
    for k, (a, b) in enumerate(ERAS):
        e[(years >= a) & (years <= b)] = k
    return e


def candidates(A, B, shift=0.0):
    ta = A.time.values.astype("datetime64[ms]").astype(np.int64) / 1000.0
    tb = B.time.values.astype("datetime64[ms]").astype(np.int64) / 1000.0 + shift
    order = np.argsort(tb, kind="stable"); tbs = tb[order]
    lo = np.searchsorted(tbs, ta - T_CAND, "left"); hi = np.searchsorted(tbs, ta + T_CAND, "right")
    cnt = hi - lo
    ia = np.repeat(np.arange(len(A)), cnt)
    ib = order[np.concatenate([np.arange(l, h) for l, h in zip(lo, hi)])] if cnt.sum() else np.array([], int)
    dist = haversine(A.lat.values[ia], A.lon.values[ia], B.lat.values[ib], B.lon.values[ib])
    k = dist <= D_CAND
    ia, ib, dist = ia[k], ib[k], dist[k]
    dt = tb[ib] - ta[ia]
    ma, mb, mt = same_type_mag(A, B, ia, ib)
    return ia, ib, dt, dist, mb - ma, mt


def calibrate(A, B, ia, ib, dt, dist, dm, mt):
    """Güvenilir ön eşleşmelerden dönem bazlı kayma ve saçılımlar."""
    ca = np.bincount(ia, minlength=len(A)); cb = np.bincount(ib, minlength=len(B))
    uniq = (ca[ia] == 1) & (cb[ib] == 1) & (np.fmax(A.mag.values[ia], B.mag.values[ib]) >= 3.0)
    era = era_of(A.time.dt.year.values[ia])
    par = {}
    for e in range(len(ERAS)):
        s = uniq & (era == e) & (np.abs(dt) <= 30) & (dist <= 100)
        if s.sum() < 30:
            s = uniq & (np.abs(dt) <= 30) & (dist <= 100)
        x = dt[s]; off = np.median(x)
        st = max(1.4826 * np.median(np.abs(x - off)), 0.3)
        sd = max(np.median(dist[s]) / 1.1774, 2.0)
        def tmean(v):
            v = v[np.abs(v - np.nanmedian(v)) <= 0.5]
            return float(np.nanmean(v)) if len(v) else 0.0
        bias = {t: tmean(dm[s & (mt == t)]) if (s & (mt == t)).sum() > 20 else 0.0 for t in ["MW", "ML", "MD", "ana"]}
        y = dm[s] - np.array([bias[t] for t in mt[s]])
        sm = max(1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y))), 0.1)
        par[e] = dict(off=off, st=st, sd=sd, sm=sm, bias=bias, n=int(s.sum()))
    return par


def cost(A, ia, dt, dist, dm, mt, par, extra=None):
    era = era_of(A.time.dt.year.values[ia])
    off = np.array([par[e]["off"] for e in era]); st = np.array([par[e]["st"] for e in era])
    sd = np.array([par[e]["sd"] for e in era]); sm = np.array([par[e]["sm"] for e in era])
    bias = np.array([par[e]["bias"][t] for e, t in zip(era, mt)])
    zm = ((dm - bias) / sm) ** 2
    zm = np.where(np.isnan(zm), 1.0, np.minimum(zm, 4.0))  # büyüklük farkı eşleşmeyi tek başına reddetmesin
    if extra is not None:  # yuvarlanmış zaman/koordinat -> belirsizlik payı (karesel toplam)
        st = np.sqrt(st ** 2 + np.where(extra["kt"], 30.0 ** 2 / 3, 0.0))
        sd = np.sqrt(sd ** 2 + np.where(extra["kd0"], 35.0 ** 2, np.where(extra["kd1"], 6.0 ** 2, 0.0)))
    return ((dt - off) / st) ** 2 + (dist / sd) ** 2 + zm


def assign(nA, nB, ia, ib, z):
    """Bileşen bazlı global en iyi bire-bir atama. Seçilen aday indekslerini ve marjları döner."""
    keep = z <= Z_MAX
    idx = np.nonzero(keep)[0]
    ia_k, ib_k = ia[idx], ib[idx]
    g = coo_matrix((np.ones(len(idx)), (ia_k, nA + ib_k)), shape=(nA + nB, nA + nB))
    ncomp, lab = connected_components(g, directed=False)
    comp = lab[ia_k]
    order = np.argsort(comp, kind="stable")
    sel = []
    bounds = np.flatnonzero(np.diff(comp[order])) + 1
    for grp in np.split(order, bounds):
        if len(grp) == 0:
            continue
        ua, ra = np.unique(ia_k[grp], return_inverse=True)
        ub, rb = np.unique(ib_k[grp], return_inverse=True)
        if len(grp) == 1:
            sel.append(idx[grp[0]]); continue
        na, nb = len(ua), len(ub); n = na + nb
        BIG = 1e6
        C = np.full((n, n), BIG)
        C[ra, rb] = z[idx[grp]]
        C[:na, nb:] = np.where(np.eye(na) > 0, Z_MAX / 2, BIG)
        C[na:, :nb] = np.where(np.eye(nb) > 0, Z_MAX / 2, BIG)
        C[na:, nb:] = 0.0
        r, c = linear_sum_assignment(C)
        lookup = {(a, b): k for k, a, b in zip(idx[grp], ra, rb)}
        for i, j in zip(r, c):
            if i < na and j < nb and C[i, j] < BIG:
                sel.append(lookup[(i, j)])
    sel = np.array(sorted(sel), int)
    # marj: seçilen çiftin A veya B olayı için en iyi alternatif adayla maliyet farkı
    def two_best(key):
        o = np.lexsort((z, key)); k = key[o]; zz = z[o]
        first = np.r_[True, k[1:] != k[:-1]]
        n = int(key.max()) + 1 if len(key) else 0
        b1 = np.full(n, np.inf); b2 = np.full(n, np.inf)
        b1[k[first]] = zz[first]
        sec = np.r_[False, first[:-1]] & ~first
        b2[k[sec]] = zz[sec]
        return b1, b2
    best_other = np.full(len(sel), np.inf)
    for key in (ia, ib):
        b1, b2 = two_best(key)
        kk = key[sel]
        alt = np.where(np.isclose(z[sel], b1[kk]), b2[kk], b1[kk])
        best_other = np.minimum(best_other, alt)
    margin = best_other - z[sel]
    return sel, margin


def run_pair(A, B, na, nb, shift=0.0, par=None):
    ia, ib, dt, dist, dm, mt = candidates(A, B, shift)
    if par is None:
        par = calibrate(A, B, ia, ib, dt, dist, dm, mt)
    kd = np.minimum(A.koord_ondalik.values[ia], B.koord_ondalik.values[ib])
    extra = dict(kt=A.kaba_zaman.values[ia] | B.kaba_zaman.values[ib], kd0=kd <= 0, kd1=kd == 1)
    z = cost(A, ia, dt, dist, dm, mt, par, extra)
    sel, margin = assign(len(A), len(B), ia, ib, z)
    mmax = np.nanmean(np.vstack([A.mag.values[ia[sel]], B.mag.values[ib[sel]]]).astype(float), axis=0)  # çiftin ortalama büyüklüğü
    return dict(ia=ia[sel], ib=ib[sel], dt=dt[sel], dist=dist[sel], dm=dm[sel], mt=mt[sel], z=z[sel], margin=margin, par=par, mmax=mmax)


def false_prob(A, m, fakes):
    """Dönem x büyüklük sınıfı x skor aralığı bazında sahte/gerçek oranı -> her çift için yanlışlık olasılığı."""
    def strata(x):
        e = era_of(A.time.dt.year.values[x["ia"]])
        mc_ = np.clip(np.digitize(np.nan_to_num(x["mmax"], nan=0.0), MBINS) - 1, 0, len(MBINS) - 2)
        return e, mc_
    er, mr = strata(m)
    fs = [strata(f) for f in fakes]
    p = np.zeros(len(m["z"])); table = {}
    for e in range(len(ERAS)):
        for k in range(len(MBINS) - 1):
            sr = (er == e) & (mr == k)
            real = np.histogram(m["z"][sr], ZBINS)[0].astype(float)
            fake = np.zeros(len(ZBINS) - 1)
            for f, (fe, fm) in zip(fakes, fs):
                fake += np.histogram(f["z"][(fe == e) & (fm == k)], ZBINS)[0]
            fake /= len(fakes)
            ratio = np.minimum((fake + 0.5 / len(fakes)) / np.maximum(real, 1), 1.0)
            ratio = np.maximum.accumulate(ratio)
            table[(e, k)] = (real, fake, ratio)
            b_ = np.clip(np.digitize(m["z"][sr], ZBINS) - 1, 0, len(ZBINS) - 2)
            p[sr] = ratio[b_]
    return p, table


def build_table(A, B, na, nb, m):
    a = A.iloc[m["ia"]].reset_index(drop=True); b = B.iloc[m["ib"]].reset_index(drop=True)
    ma, mb, mt = same_type_mag(A, B, m["ia"], m["ib"])
    kaba = np.minimum(a.koord_ondalik.values, b.koord_ondalik.values) <= 1
    r = pd.DataFrame({
        f"{na}_id": a.id, f"{nb}_id": b.id,
        f"{na}_zaman": a.time.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4],
        f"{nb}_zaman": b.time.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4],
        "dt_s": np.round(m["dt"], 2),
        f"{na}_enlem": a.lat, f"{na}_boylam": a.lon, f"{nb}_enlem": b.lat, f"{nb}_boylam": b.lon,
        "mesafe_km": np.round(m["dist"], 2),
        f"{na}_derinlik": a.depth, f"{nb}_derinlik": b.depth, "d_derinlik_km": np.round(b.depth - a.depth, 1),
        f"{na}_M": a.mag, f"{na}_Mtur": a.magtype, f"{nb}_M": b.mag, f"{nb}_Mtur": b.magtype,
        "karsilastirilan_M_turu": mt, f"{na}_Mk": ma, f"{nb}_Mk": mb, "dM": np.round(mb - ma, 2),
        f"{na}_olay_turu": a.olay_turu, f"{nb}_olay_turu": b.olay_turu,
        f"{na}_yer": a.place, f"{nb}_yer": b.place,
        "z_skor": np.round(m["z"], 2), "marj": np.round(np.minimum(m["margin"], 999), 2),
        "yanlis_eslesme_olasiligi": np.round(m["pfalse"], 4),
        "kaba_koordinat": kaba,
    })
    r["guven"] = np.where(r.marj < 4, "BELIRSIZ", np.where(r.yanlis_eslesme_olasiligi < P_KESIN, "KESIN",
                          np.where(r.yanlis_eslesme_olasiligi < P_MUHTEMEL, "MUHTEMEL", "ZAYIF")))
    f_t = r.dt_s.abs() > THR["dt"]; f_d = r.mesafe_km > THR["dist"]
    f_z = r.d_derinlik_km.abs() > THR["ddep"]; f_m = r.dM.abs() > THR["dmag"]
    f_y = (r[f"{na}_olay_turu"] != r[f"{nb}_olay_turu"]) & ~((r[f"{na}_olay_turu"] == "Bilinmeyen"))
    r["fark_ZAMAN"], r["fark_KONUM"], r["fark_DERINLIK"], r["fark_BUYUKLUK"], r["fark_TUR"] = f_t, f_d, f_z, f_m, f_y
    r["isaret"] = np.where(f_t | f_d | f_z | f_m | f_y, "FARKLI", "AYNI")
    r["farklar"] = (np.where(f_t, "ZAMAN;", "") + np.where(f_d & ~kaba, "KONUM;", "") + np.where(f_d & kaba, "KONUM(kaba_koord);", "")
                    + np.where(f_z, "DERINLIK;", "") + np.where(f_m, "BUYUKLUK;", "") + np.where(f_y, "OLAY_TURU;", ""))
    r["farklar"] = r.farklar.str.rstrip(";")
    return r.sort_values(f"{na}_zaman").reset_index(drop=True)


def overlap(A, B):
    t0 = max(A.time.min(), B.time.min()); t1 = min(A.time.max(), B.time.max())
    return (A[(A.time >= t0) & (A.time <= t1)].reset_index(drop=True),
            B[(B.time >= t0) & (B.time <= t1)].reset_index(drop=True), t0, t1)


def write(df, name):
    df.to_csv(OUT / name, sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    cats = {"AFAD": load_afad(), "KANDILLI": load_kandilli()}
    t = load_turhec()
    if t is not None:
        cats["TURHEC"] = t
    duplog = []
    for k_ in list(cats):
        cats[k_] = drop_exact_duplicates(cats[k_].dropna(subset=["time", "lat", "lon"]), k_, duplog)
    names = list(cats)
    L0 = duplog
    L = ["QuakeCast-TR — Katalog eşleştirme özeti (v2)", f"Oluşturulma: {pd.Timestamp.now():%Y-%m-%d %H:%M}", "",
         "Yöntem: aday |Δt|<=60 s & <=150 km; dönem bazlı kalibre edilmiş maliyet z; bileşen bazlı global en iyi atama; z<=%g." % Z_MAX,
         "Güven: her çiftin yanlış eşleşme olasılığı, saat kaydırma testinden skor aralığı ve dönem bazında ölçülür.",
         "  KESIN <%0.5, MUHTEMEL <%5, ZAYIF >=%5, BELIRSIZ = başka bir adayla maliyet farkı <4. Ana sonuçlar = KESIN + MUHTEMEL.",
         f"FARKLI eşikleri: zaman >{THR['dt']} s, konum >{THR['dist']} km, derinlik >{THR['ddep']} km, büyüklük >{THR['dmag']} (aynı tür), olay türü farklı.",
         "KONUM(kaba_koord): kaynaklardan biri koordinatı <=1 ondalıkla veriyor (AFAD'ın 2000 öncesi kayıtları kaynağında böyle).",
         "Dosyalar: eslesen_*.txt (KESIN+MUHTEMEL), cakisan_farkli_*.txt (bunların FARKLI olanları), zayif_veya_belirsiz_*.txt (elle bakılmalı), eslesmeyen_*.txt", ""]
    for n in names:
        c = cats[n]
        L.append(f"{n}: {len(c):,} olay, {c.time.min()} → {c.time.max()}; olay türleri: " + ", ".join(f"{k}={v:,}" for k, v in c.olay_turu.value_counts().items()))
    L += L0 + [""]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            A, B, t0, t1 = overlap(cats[na], cats[nb])
            m = run_pair(A, B, na, nb)
            fakes = [run_pair(A, B, na, nb, shift=sh, par=m["par"]) for sh in SHIFTS]
            m["pfalse"], ptab = false_prob(A, m, fakes)
            rall = build_table(A, B, na, nb, m)
            ok = rall.guven.isin(["KESIN", "MUHTEMEL"])
            r = rall[ok].reset_index(drop=True)
            tag = f"{na.lower()}_{nb.lower()}"
            idmapA = set(r[f"{na}_id"]); idmapB = set(r[f"{nb}_id"])
            usedA = A.id.isin(idmapA).values; usedB = B.id.isin(idmapB).values
            write(r, f"eslesen_{tag}.txt")
            write(r[r.isaret == "FARKLI"], f"cakisan_farkli_{tag}.txt")
            write(rall[~ok], f"zayif_veya_belirsiz_{tag}.txt")
            write(A[~usedA], f"eslesmeyen_{na.lower()}_vs_{nb.lower()}.txt")
            write(B[~usedB], f"eslesmeyen_{nb.lower()}_vs_{na.lower()}.txt")
            exp_false = rall.yanlis_eslesme_olasiligi[ok].sum()
            # elle kontrol: eşleşmemiş M>=3 olay için tek, eşleşmemiş, benzer büyüklükte aday (<=60 s, <=50 km)
            ia_c, ib_c, dt_c, dist_c, dm_c, mt_c = candidates(A, B)
            fa = ~usedA[ia_c] & ~usedB[ib_c] & (dist_c <= 50) & (np.abs(A.mag.values[ia_c] - B.mag.values[ib_c]) < 0.5) & (np.fmax(A.mag.values[ia_c], B.mag.values[ib_c]) >= 3)
            ia_c, ib_c, dt_c, dist_c = ia_c[fa], ib_c[fa], dt_c[fa], dist_c[fa]
            u1 = np.bincount(ia_c, minlength=len(A))[ia_c] == 1; u2 = np.bincount(ib_c, minlength=len(B))[ib_c] == 1
            kk = u1 & u2
            ek = pd.DataFrame({f"{na}_id": A.id.values[ia_c[kk]], f"{nb}_id": B.id.values[ib_c[kk]],
                               f"{na}_zaman": A.time.values[ia_c[kk]], f"{nb}_zaman": B.time.values[ib_c[kk]],
                               "dt_s": np.round(dt_c[kk], 2), "mesafe_km": np.round(dist_c[kk], 1),
                               f"{na}_M": A.mag.values[ia_c[kk]], f"{nb}_M": B.mag.values[ib_c[kk]],
                               f"{na}_yer": A.place.values[ia_c[kk]], f"{nb}_yer": B.place.values[ib_c[kk]]})
            write(ek, f"elle_kontrol_{tag}.txt")
            fr = r[r.isaret == "FARKLI"]
            L += [f"=== {na} ↔ {nb}  (ortak dönem {t0:%Y-%m-%d} → {t1:%Y-%m-%d}) ===",
                  f"  {na} olay: {len(A):,}   {nb} olay: {len(B):,}",
                  "  Kalibrasyon (dönem: zaman kayması, σt, σmesafe, σM, ön eşleşme sayısı):"]
            for e, (a_, b_) in enumerate(ERAS):
                p = m["par"][e]
                L.append(f"    {a_}-{min(b_, 2026)}: kayma {p['off']:+.2f} s, σt {p['st']:.2f} s, σd {p['sd']:.1f} km, σM {p['sm']:.2f}, n={p['n']:,}; ΔM bias " +
                         ", ".join(f"{k}={v:+.2f}" for k, v in p["bias"].items()))
            L += [f"  Eşleşen (aynı deprem): {len(r):,}",
                  f"    güven: " + ", ".join(f"{k}={v:,}" for k, v in r.guven.value_counts().items()) +
                  f"   (ayrıca ZAYIF/BELIRSIZ, ana sonuçlara katılmadı: " + ", ".join(f"{k}={v:,}" for k, v in rall[~ok].guven.value_counts().items()) + ")",
                  f"    AYNI değerler: {(r.isaret=='AYNI').sum():,}   FARKLI değerler: {len(fr):,}",
                  f"      zaman: {r.fark_ZAMAN.sum():,}  konum: {r.fark_KONUM.sum():,} (kaba koordinat kaynaklı: {(r.fark_KONUM & r.kaba_koordinat).sum():,})  derinlik: {r.fark_DERINLIK.sum():,}  büyüklük: {r.fark_BUYUKLUK.sum():,}  olay türü: {r.fark_TUR.sum():,}",
                  f"  Sadece {na}'da: {(~usedA).sum():,}   Sadece {nb}'da: {(~usedB).sum():,}",
                  f"  Elle kontrol önerilen (eşleşmemiş M>=3, tek benzer aday, büyük zaman/konum farkı): {len(ek):,} (elle_kontrol_{tag}.txt)",
                  f"  SAHTE EŞLEŞME TESTİ ({nb} zamanları ±1, ±2, ±3, ±4 saat kaydırılarak): kaydırılmış verilerde ortalama {np.mean([len(f['ia']) for f in fakes]):,.0f} rastlantısal eşleşme",
                  f"    -> KESIN+MUHTEMEL içindeki tahmini yanlış eşleşme: ~{exp_false:.0f} / {len(r):,} (%{100*exp_false/max(len(r),1):.3f});  sadece KESIN: ~{rall.yanlis_eslesme_olasiligi[rall.guven=='KESIN'].sum():.0f} / {(rall.guven=='KESIN').sum():,}",
                  "    Skor aralığına göre gerçek / sahte(ort.) / yanlışlık oranı, dönem ve büyüklük sınıfı bazında:"]
            mlab = ["M<2", "2<=M<3", "3<=M<4", "M>=4"]
            for (e, k_), (real, fake, ratio) in ptab.items():
                a_, b_ = ERAS[e]
                L.append(f"    {a_}-{min(b_, 2026)} {mlab[k_]}: " + "  ".join(f"[{ZBINS[k]:g}-{ZBINS[k+1]:g}] {int(real[k])}/{fake[k]:.1f}/{100*ratio[k]:.2f}%" for k in range(len(real)) if real[k] > 0))
            L += [
                  f"  Medyan |Δt|: {r.dt_s.abs().median():.2f} s   medyan mesafe: {r.mesafe_km.median():.1f} km",
                  "  Büyüklük türüne göre ortalama ΔM (n):"]
            for mt_, g in r.groupby("karsilastirilan_M_turu"):
                L.append(f"    {mt_}: {g.dM.mean():+.3f}  (n={len(g):,})")
            yy = r.assign(yil=r[f"{na}_zaman"].str[:4]).groupby("yil").agg(
                eslesen=("isaret", "size"), farkli=("isaret", lambda s: (s == "FARKLI").sum()), belirsiz=("guven", lambda s: (s == "MUHTEMEL").sum()))
            L.append("  Yıllara göre eşleşen / farklı / (KESIN+MUHTEMEL içinde) :")
            L += [f"    {y}: {row.eslesen:,} / {row.farkli:,} / {row.belirsiz:,}" for y, row in yy.iterrows()]
            big = fr[(fr[f"{na}_M"] >= 5) | (fr[f"{nb}_M"] >= 5)]
            L.append(f"  M>=5 olup farklı değerli eşleşmeler: {len(big)} (ilk 25):")
            cols = [f"{na}_zaman", "dt_s", "mesafe_km", "d_derinlik_km", f"{na}_M", f"{na}_Mtur", f"{nb}_M", "guven", "farklar", f"{nb}_yer"]
            L += ["    " + s for s in big[cols].head(25).to_string(index=False).splitlines()]
            L.append("")
    (OUT / "OZET.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:45]))
