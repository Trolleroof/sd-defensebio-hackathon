"""Small executable baseline programs for the recovery forecast interface."""


PERSISTENCE_CODE = '''\
def fit(support):
    """Persistence needs no fitted parameters."""
    return None

def predict(history, forecast_times_s):
    if not history:
        raise ValueError("history must contain at least one observed heart rate")
    last = history[-1]
    value = float(last["hr_bpm"])
    return [{"time_s": float(t), "mean": value, "lower": value - 5.0, "upper": value + 5.0}
            for t in forecast_times_s]
'''


LINEAR_RECOVERY_CODE = '''\
def fit(support):
    """Fit a common within-session HR slope from permitted support recordings."""
    global fitted_slope
    slopes = []
    for session in support:
        rows = session["rows"]
        if len(rows) < 2:
            continue
        t0 = float(rows[0]["time_s"])
        xs = [float(row["time_s"]) - t0 for row in rows]
        ys = [float(row["hr_bpm"]) for row in rows]
        xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        if denom:
            slopes.append(sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom)
    fitted_slope = sum(slopes) / len(slopes) if slopes else 0.0
    return fitted_slope

def predict(history, forecast_times_s):
    if not history:
        raise ValueError("history must contain at least one observed heart rate")
    last = history[-1]
    t0, y0 = float(last["time_s"]), float(last["hr_bpm"])
    slope = float(globals().get("fitted_slope", 0.0))
    return [{"time_s": float(t), "mean": y0 + slope * (float(t) - t0),
             "lower": y0 + slope * (float(t) - t0) - 5.0,
             "upper": y0 + slope * (float(t) - t0) + 5.0}
            for t in forecast_times_s]
'''


def persistence_code():
    """Return an executable no-parameter fit/predict implementation."""
    return PERSISTENCE_CODE


def linear_recovery_code():
    """Return executable source; caller binds ``fitted_slope = fit(support)``."""
    return LINEAR_RECOVERY_CODE


def baseline_codes():
    """Return baseline names mapped to standalone, executable code strings."""
    return {"persistence": persistence_code(), "linear_recovery": linear_recovery_code()}
