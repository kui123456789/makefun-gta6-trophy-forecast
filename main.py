"""Forecast GTA VI trophy earn rates on Exophase.

Reads every <id>.in case in the input directory (MAKEFUN_INPUT_DIR, else
/input) and writes one <id>.out forecast per case to the output directory
(MAKEFUN_OUTPUT_DIR, else /output): five quantiles (5/25/50/75/95%) of each
trophy's earn rate at the case's target, as JSON
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Method, per trophy:
  1. Work in logit space, logit(p) = ln(p / (100 - p)), on the trophy's
     rate series over the case history, against x = ln(1 + hours since the
     release day began, 2026-11-19T00:00:00Z). Early launch growth is fast
     and then bends over, which log-time captures; a straight line in time
     would run away over a multi-day horizon.
  2. Fit that line by weighted least squares, weights decaying with a
     12-hour half-life so the recent reads (the current regime) dominate.
     The fitted slope is damped by one half and the extrapolated move from
     the last observed rate is capped at 2 logit units: a launch-week
     curve saturates, and an undamped slope would happily forecast a 1%
     trophy at 80% a week out.
  3. A frozen page gets a refresh prior. When the whole history is one
     unchanged page (no trophy moved at all), the fitted slope is zero
     only because Exophase has not refreshed, not because the true rates
     are still: players have been joining and playing the whole time, and
     the next refresh reveals that as a dilution (new players hold far
     fewer trophies). The median is then shifted down 1.0 logit unit
     (about a 2.7x cut in rate) for every trophy below 100%, except the
     trophy already at 100%, which a refresh leaves at 100%.
  4. Spread grows with the horizon: sigma = sqrt(sigma_jump^2 +
     (0.35 * sqrt(days))^2). sigma_jump is the per-refresh logit surprise
     measured from the case's own history (the RMS size of the pooled
     per-trophy logit jumps at refreshes, weighted toward the recent
     ones: launch night's first refresh, when the tracked-player base
     grew six-fold at once, must not set the spread for a week later),
     floored at 0.20 and capped at 0.8; with no refresh in the history
     it falls back to 0.55, the scale of the one refresh the practice
     cases straddle (launch night's page, flat for hours, then a refresh
     that cut every sub-100% rate several-fold). The 0.35-per-root-day
     term is below the persistence baseline's 0.5 because the trend
     already carries the drift, so only the residual error grows with
     the horizon.

Only the standard library. No network, no clock, no randomness: two runs
on the same case write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back
# first so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release day began 2026-11-19T00:00:00Z; the log-time axis starts there.
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Recency half-life (hours) of the trend fit's weights.
HALF_LIFE_HOURS = 12.0
# The fitted log-time slope is halved before extrapolation.
SLOPE_DAMP = 0.5
# No trophy's median may move more than this many logit units from its
# last observed rate.
MAX_LOGIT_MOVE = 2.0
# Frozen-page refresh prior: logit shift applied to the median when the
# whole history is one unchanged page.
FROZEN_SHIFT = -1.0
# Per-refresh surprise when the history shows no refresh at all.
JUMP_PRIOR = 0.55
JUMP_MIN, JUMP_MAX = 0.20, 0.8
# Recency half-life (hours) for weighting refresh events.
JUMP_HALF_LIFE_HOURS = 24.0
# Random-walk spread, logit units per square-root day.
SIGMA_RW = 0.35


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l: float) -> float:
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, halves up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment: datetime) -> float:
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def refresh_events(case: dict):
    """Reads where Exophase refreshed the page.

    A refresh shows as a rise in the tracked-player count, or as any
    trophy's rate moving after the page sat unchanged. Returns the
    indices into history of the reads just after each refresh.
    """
    events = []
    history = case["history"]
    for k in range(1, len(history)):
        before = history[k - 1].get("players_tracked")
        after = history[k].get("players_tracked")
        grew = (before is not None and after is not None
                and after > before)
        moved = history[k]["rates"] != history[k - 1]["rates"]
        if grew or moved:
            events.append(k)
    return events


def jump_sigma(case: dict, n_trophies: int) -> float:
    """Typical per-refresh logit surprise, from the history's own refreshes.

    For each refresh event, the mean squared per-trophy logit change;
    the events are combined with weights decaying with a 24-hour
    half-life, so launch night's first refresh (a six-fold jump in the
    tracked-player base) does not set the spread a week later. Falls
    back to JUMP_PRIOR when the history holds no refresh.
    """
    events = refresh_events(case)
    if not events:
        return JUMP_PRIOR
    history = case["history"]
    last_at = parse_time(history[-1]["at"])
    weighted_sum = 0.0
    weight_total = 0.0
    for k in events:
        prev = history[k - 1]["rates"]
        cur = history[k]["rates"]
        squares = []
        for i in range(n_trophies):
            a, b = prev[i], cur[i]
            if a is None or b is None:
                continue
            if a >= 99.99 and b >= 99.99:
                continue
            d = logit(b) - logit(a)
            squares.append(d * d)
        if not squares:
            continue
        age = (last_at - parse_time(history[k]["at"])).total_seconds() / 3600.0
        weight = 0.5 ** (max(0.0, age) / JUMP_HALF_LIFE_HOURS)
        weighted_sum += weight * (sum(squares) / len(squares))
        weight_total += weight
    if weight_total <= 0:
        return JUMP_PRIOR
    value = math.sqrt(weighted_sum / weight_total)
    return min(JUMP_MAX, max(JUMP_MIN, value))


def page_frozen(case: dict) -> bool:
    """True when no trophy's rate moved at all across the whole history."""
    history = case["history"]
    if len(history) < 2:
        return False
    first = history[0]["rates"]
    for read in history[1:]:
        if read["rates"] != first:
            return False
    return True


def series_for(case: dict, index: int):
    """(x, y, weight) points for one trophy: log-time, logit rate, recency."""
    history = case["history"]
    last_at = parse_time(history[-1]["at"])
    points = []
    for read in history:
        rate = read["rates"][index]
        if rate is None:
            continue
        moment = parse_time(read["at"])
        age = (last_at - moment).total_seconds() / 3600.0
        weight = 0.5 ** (max(0.0, age) / HALF_LIFE_HOURS)
        x = math.log1p(hours_since_release(moment))
        points.append((x, logit(rate), weight))
    return points


def trend_move(points, x_target: float) -> float:
    """Damped extrapolated logit move from the last observed rate.

    Weighted least squares of logit rate on ln(1 + hours since release).
    Returns 0 when the series carries no usable slope (too few distinct
    values or reads, or no spread in x).
    """
    distinct = {round(y, 9) for _, y, _ in points}
    if len(points) < 3 or len(distinct) < 3:
        return 0.0
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return 0.0
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    x_last = points[-1][0]
    move = SLOPE_DAMP * slope * (x_target - x_last)
    return min(MAX_LOGIT_MOVE, max(-MAX_LOGIT_MOVE, move))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)
    last = case["history"][-1]
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target))
    days = case["horizon_minutes"] / 1440.0
    sigma_jump = jump_sigma(case, n)
    sigma = math.sqrt(sigma_jump ** 2 + (SIGMA_RW * math.sqrt(days)) ** 2)
    frozen = page_frozen(case)
    frozen_span = 0.0
    if frozen:
        first_at = parse_time(case["history"][0]["at"])
        last_at = parse_time(last["at"])
        frozen_span = (last_at - first_at).total_seconds() / 60.0

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last["rates"][i]
        if rate is None:
            # Fall back to the most recent read that had this trophy.
            for read in reversed(case["history"]):
                if read["rates"][i] is not None:
                    rate = read["rates"][i]
                    break
        if rate is None:
            rate = 1.0
        median_logit = logit(rate) + trend_move(series_for(case, i), x_target)
        if frozen and frozen_span >= 60.0 and rate < 99.99:
            # The page has not refreshed in over an hour and the history
            # shows no movement at all: the next refresh will dilute.
            median_logit += FROZEN_SHIFT
        quantiles = []
        for z in Z:
            q = inv_logit(median_logit + z * sigma)
            quantiles.append(round4(min(100.0, max(0.0, q))))
        # Rounding is monotone, but enforce non-decreasing explicitly.
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def persistence_fallback(case: dict) -> dict:
    """Last-rate forecast with the house baseline's spread. Used only if
    the trend model fails on a case, so the output stays valid."""
    last = case["history"][-1] if case["history"] else {"rates": []}
    spread = SIGMA_RW * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        if rate is None:
            rate = 1.0
        lg = logit(rate)
        quantiles = [round4(min(100.0, max(0.0, inv_logit(lg + z * spread)))) for z in Z]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # never leave a case without a valid file
            print(f"warning: {path.name}: {exc}; using persistence fallback",
                  file=sys.stderr)
            forecast = persistence_fallback(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                            encoding="utf-8")


if __name__ == "__main__":
    main()
