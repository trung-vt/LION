"""Bounded log-coordinate ternary search for costly measurement-only risk.

Ternary shrink is valid for unimodal *one-dimensional sections*.
The overall nonconvex 2D/3D PnP risk is not guaranteed to be unimodal.
Cache expensive evaluations. Always choose best evaluated finite candidate,
not an untested extrapolated midpoint or a PSNR-selected configuration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Probe:
    params: dict[str, float]
    loss: float
    status: str
    coordinate: str
    cycle: int
    step: int


def stable_params(params):
    assert all(float(v) > 0 and math.isfinite(float(v)) for v in params.values())
    return {k: float(f"{float(v):.12g}") for k, v in params.items()}


def search(
    objective: Callable[[dict[str, float]], float],
    ranges: dict[str, list[float]],
    orders: list[list[str]],
    contractions: int = 3,
    absolute_tie_tolerance: float = 1e-10,
):
    """Evaluate candidates through objective and return a complete trace.

    objective returns +inf for numerically invalid candidates. The caller must
    persist a separate error record and never use GT or test-case metrics.
    """
    assert contractions >= 1
    assert len(orders) == 2
    assert set(orders[0]) == set(orders[1]) == set(ranges)
    assert len(orders[0]) == len(orders[1]) == len(ranges)
    for name, (low, high) in ranges.items():
        assert 0 < low < high and math.isfinite(low + high)
    cache: dict[tuple, float] = {}
    trace: list[Probe] = []
    warnings = []

    def evaluate(p, coordinate, cycle, step):
        stable = stable_params(p)
        key = tuple(sorted(stable.items()))
        if key not in cache:
            result = float(objective(stable))
            if math.isnan(result) or result < 0:
                raise ValueError("Invalid SSDU risk")
            cache[key] = result
            trace.append(
                Probe(
                    stable,
                    result,
                    "valid" if math.isfinite(result) else "invalid",
                    coordinate,
                    cycle,
                    step,
                )
            )
        return cache[key]

    current = stable_params(
        {name: math.sqrt(lo * hi) for name, (lo, hi) in ranges.items()}
    )
    evaluate(current, "initial", -1, -1)
    for cycle, coordinates in enumerate(orders):
        for name in coordinates:
            lo, hi = (math.log10(float(v)) for v in ranges[name])
            candidates = [(evaluate(current, name, cycle, -1), dict(current))]
            for step in range(contractions):
                m1 = lo + (hi - lo) / 3.0
                m2 = hi - (hi - lo) / 3.0
                v1, v2 = 10.0**m1, 10.0**m2
                a = dict(current)
                a[name] = v1
                b = dict(current)
                b[name] = v2
                fa = evaluate(a, name, cycle, step)
                fb = evaluate(b, name, cycle, step)
                candidates.extend([(fa, stable_params(a)), (fb, stable_params(b))])
                if not math.isfinite(fa) and not math.isfinite(fb):
                    warnings.append(
                        {
                            "cycle": cycle,
                            "parameter": name,
                            "step": step,
                            "issue": "both_ternary_probes_invalid",
                        }
                    )
                    break
                if fa <= fb:
                    hi = m2
                else:
                    lo = m1
            # Set current using only evaluations on this fixed-coordinate slice.
            valid = [(loss, x) for loss, x in candidates if math.isfinite(loss)]
            if valid:
                valid.sort(
                    key=lambda q: (
                        round(q[0] / absolute_tie_tolerance),
                        q[0],
                        tuple(sorted(q[1].items())),
                    )
                )
                best_loss, best_params = valid[0]
                incumbent = evaluate(current, name, cycle, -1)
                if best_loss < incumbent - absolute_tie_tolerance:
                    current = best_params
            else:
                warnings.append(
                    {
                        "cycle": cycle,
                        "parameter": name,
                        "issue": "no_valid_probe_on_coordinate",
                    }
                )
    valid_trace = [(i, p) for i, p in enumerate(trace) if math.isfinite(p.loss)]
    if not valid_trace:
        raise RuntimeError("All PnP ternary candidates numerically invalid")
    # Explicit stable tie break: first evaluated candidate within tolerance.
    minimum = min(p.loss for _, p in valid_trace)
    tied = [
        (i, p) for i, p in valid_trace if p.loss <= minimum + absolute_tie_tolerance
    ]
    tied.sort(key=lambda x: x[0])
    winner = tied[0][1]
    assert all(ranges[k][0] <= winner.params[k] <= ranges[k][1] for k in ranges)
    max_evals = 1 + 2 * contractions * sum(len(x) for x in orders)
    assert len(trace) <= max_evals
    return {
        "winner": winner.params,
        "winner_measurement_mse": winner.loss,
        "evaluations": len(trace),
        "valid_evaluations": len(valid_trace),
        "invalid_evaluations": len(trace) - len(valid_trace),
        "maximum_evaluations": max_evals,
        "coordinate_cycles": len(orders),
        "contractions_per_coordinate": contractions,
        "ranges": ranges,
        "warnings": warnings,
        "trace": [
            {
                "params": x.params,
                "loss": x.loss if math.isfinite(x.loss) else None,
                "valid": math.isfinite(x.loss),
                "coordinate": x.coordinate,
                "cycle": x.cycle,
                "step": x.step,
            }
            for x in trace
        ],
        "unimodality_assumed_not_proven": True,
        "no_image_ground_truth_used": True,
        "no_test_metric_used": True,
    }
