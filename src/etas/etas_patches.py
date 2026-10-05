"""etas paketine (Mizrahi vd., MIT) çalışma anında uygulanan küçük yamalar.

1) Arka plan şekli (ETASbg, Han vd. 2025): E-adımında sabit μ yerine μ·A·f(x_j) kullanılır; f bölge üzerinde
   integrali 1 olan sabit yoğunluk (longterm_background.py). μ̂ = n̂/(A·T) kapalı formu tutarlı kalır.
2) τ sabitleme sınırlarla: paketin fixed_parameters + alpha kısıtı birlikte kullanıldığında indeks kayması olduğundan
   (fixed_parameters 'alpha' varken starting_index=3), τ'yu optimize sınırlarıyla sabitliyoruz.
"""
import inspect
import textwrap
import numpy as np


def patch_bg_shape(calc, inv_module, f, area):
    te = calc.target_events
    calc.target_events["bg_shape"] = area * f(te["latitude"].values, te["longitude"].values)
    cls = type(calc)
    if getattr(cls, "_bg_shape_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(cls.expectation_step))
    new = src.replace('target_events_0["mu"] = mu\n',
                      'target_events_0["mu"] = mu * target_events_0["bg_shape"] if "bg_shape" in target_events_0.columns else mu\n')
    assert new != src, "yama noktası bulunamadı"
    ns = dict(inv_module.__dict__)
    exec(compile(new, "<etas_patch_bg_shape>", "exec"), ns)
    cls.expectation_step = ns["expectation_step"]
    cls._bg_shape_patched = True


def fix_tau_by_bounds(inv_module, log10_tau):
    cls = inv_module.ETASParameterCalculation
    rng = list(inv_module.RANGES)
    rng[6] = (log10_tau, log10_tau)
    cls.optimize_parameters.__defaults__ = (tuple(rng),)
