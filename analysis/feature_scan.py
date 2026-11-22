"""
Which team metrics predict future performance BEYOND what EPA already says? And does adding the
ones that hold up improve a game-margin model or beat the spread?

Anti-overfitting protocol (decided before looking at results):
  1. Discovery   2019–2023 (FTN charting features: 2022–2023)  -> Benjamini–Hochberg FDR 10%
  2. Validation  2024–2025: a discovered feature must keep its sign with p < .05
  3. Holdout     2026: reported only, never used to choose anything
  * Partial correlations control for the team's own pre-game EPA/play and the opponent's
    pre-game EPA on the other side of the ball, plus season fixed effects.
  * CIs from a cluster bootstrap by team-season (games of one team aren't independent).
  * PFF / Sumer / paid-FTN exist only for 2026 -> scanned separately as EXPLORATORY.
    They can't be validated or used in a model until another season exists.
  * The game model uses only replicated features, ridge-regularized, trained walk-forward
    (each season predicted from earlier seasons only).

Outputs: data/private/derived/ (git-ignored)
Usage:  .venv/bin/python analysis/feature_scan.py
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import features as F  # noqa: E402

OUT = F.PRIVATE / "derived"
SEASONS = list(range(2019, 2027))
DISC, VALID, HOLD = [2019, 2020, 2021, 2022, 2023], [2024, 2025], [2026]
MIN_WEEK, MIN_PRIOR = 5, 3
BOOT = 400
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ build team-game table
def build() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    pre_all, priv = [], None
    for s in SEASONS:
        p = F.load_plays(s)
        tg = F.team_games(p)
        pre = F.pregame(tg)
        pre_all.append(pre)
        if s == 2026:
            ptg = F.private_team_games(p)
            if ptg is not None:
                priv = F.pregame(ptg)
    pre = pd.concat(pre_all, ignore_index=True)
    # opponent's pre-game numbers on the same row
    opp = pre[["game_id", "team"] + [c for c in pre.columns if c.startswith("pre_") and not c.endswith("_n")]]
    pairs = pre[["game_id", "team", "home_team", "away_team"]].copy()
    pairs["opp"] = np.where(pairs.team == pairs.home_team, pairs.away_team, pairs.home_team)
    opp = opp.rename(columns={"team": "opp"}).rename(columns=lambda c: "opp_" + c if c.startswith("pre_") else c)
    pre = pre.merge(pairs[["game_id", "team", "opp"]], on=["game_id", "team"]).merge(opp, on=["game_id", "opp"], how="left")
    return pre, priv


# ------------------------------------------------------------------ partial correlation with cluster bootstrap
def resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def partial(df: pd.DataFrame, f: str, y: str, controls: list[str]) -> dict | None:
    d = df[[f, y, *controls, "season", "team"]].dropna()
    if len(d) < 80 or d[f].std() == 0:
        return None
    X = np.column_stack([pd.get_dummies(d.season, dtype=float).to_numpy(), d[controls].to_numpy()])
    rf, ry = resid(d[f].to_numpy(), X), resid(d[y].to_numpy(), X)
    r = np.corrcoef(rf, ry)[0, 1]
    cl = (d.season.astype(str) + d.team).to_numpy()
    ids = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in ids}
    bs = []
    for _ in range(BOOT):
        take = np.concatenate([idx[c] for c in RNG.choice(ids, len(ids))])
        bs.append(np.corrcoef(rf[take], ry[take])[0, 1])
    se = np.nanstd(bs)
    z = r / se if se > 0 else 0
    return {"r": r, "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5), "p": math.erfc(abs(z) / math.sqrt(2)), "n": len(d)}


def bh(p: pd.Series, q: float = 0.10) -> pd.Series:
    p = p.dropna()
    order = p.sort_values()
    m = len(order)
    thresh = order.values <= q * np.arange(1, m + 1) / m
    k = np.max(np.where(thresh)[0]) + 1 if thresh.any() else 0
    passed = pd.Series(False, index=p.index)
    passed[order.index[:k]] = True
    return passed


def stickiness(df: pd.DataFrame, f: str) -> float:
    g = "game_" + f[4:]
    if g not in df:
        return np.nan
    d = df[[f, g]].dropna()
    return d[f].corr(d[g]) if len(d) > 50 else np.nan


def scan(pre: pd.DataFrame, feats: list[str], seasons: list[int]) -> pd.DataFrame:
    d = pre[pre.season.isin(seasons) & (pre.week >= MIN_WEEK) & (pre.prior_games >= MIN_PRIOR)]
    out = []
    for f in feats:
        unit = f.split("_")[1]           # pre_off_x / pre_def_x
        y = f"game_{unit}_epa"
        ctrl = [f"pre_{unit}_epa", f"opp_pre_{'def' if unit == 'off' else 'off'}_epa"]
        if f in ctrl:
            continue
        res = partial(d, f, y, ctrl)
        if res:
            out.append({"feature": f[4:], **res, "sticky": stickiness(d, f)})
    return pd.DataFrame(out).set_index("feature") if out else pd.DataFrame()


# ------------------------------------------------------------------ game-level walk-forward model
def game_table(pre: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    g = F.games_frame()
    cols = ["game_id", "team", "prior_games", *[f"pre_{x}" for x in feats]]
    h = pre[cols].rename(columns=lambda c: "h_" + c if c not in ("game_id",) else c)
    a = pre[cols].rename(columns=lambda c: "a_" + c if c not in ("game_id",) else c)
    m = g.merge(h, left_on=["game_id", "home_team"], right_on=["game_id", "h_team"]) \
         .merge(a, left_on=["game_id", "away_team"], right_on=["game_id", "a_team"])
    m = m[(m.week >= MIN_WEEK) & (m.h_prior_games >= MIN_PRIOR) & (m.a_prior_games >= MIN_PRIOR)]
    for x in feats:   # matchup differential, signed so + favors home
        sign = 1 if x.startswith("off_") else -1
        m[f"d_{x}"] = sign * (m[f"h_pre_{x}"] - m[f"a_pre_{x}"])
    m["hfa"] = (m.location != "Neutral").astype(float)
    return m


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    b = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return lambda Xn: y.mean() + ((Xn - mu) / sd) @ b, b / sd


def walk_forward(m: pd.DataFrame, cols: list[str], alphas=(1, 10, 100, 1000)) -> pd.DataFrame:
    preds = []
    for s in sorted(m.season.unique()):
        tr, te = m[(m.season < s) & m.result.notna()].dropna(subset=cols), m[m.season == s].dropna(subset=cols)
        if tr.season.nunique() < 2 or te.empty:
            continue
        # choose alpha by leave-one-season-out inside the training seasons only
        best = min(alphas, key=lambda a: np.mean([
            np.mean(np.abs(ridge_fit(tr[tr.season != v][cols + ["hfa"]].to_numpy(), tr[tr.season != v].result.to_numpy(), a)[0](
                tr[tr.season == v][cols + ["hfa"]].to_numpy()) - tr[tr.season == v].result.to_numpy()))
            for v in tr.season.unique()]))
        f, _ = ridge_fit(tr[cols + ["hfa"]].to_numpy(), tr.result.to_numpy(), best)
        preds.append(te.assign(pred=f(te[cols + ["hfa"]].to_numpy()), alpha=best))
    return pd.concat(preds)


def evaluate(pr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, d in list(pr.groupby("season")) + [("all", pr)]:
        d = d[d.result.notna()]
        if d.empty:
            continue
        ats = d[(d.result - d.spread_line) != 0]
        win = (np.sign(ats.pred - ats.spread_line) == np.sign(ats.result - ats.spread_line)).mean()
        rows.append({"season": s, "games": len(d), "r_margin": d.pred.corr(d.result),
                     "mae": (d.pred - d.result).abs().mean(), "spread_mae": (d.spread_line - d.result).abs().mean(),
                     "ats": win, "ats_n": len(ats)})
    return pd.DataFrame(rows).set_index("season")


# ------------------------------------------------------------------ main
def fmt_scan(t: pd.DataFrame, extra: list[str] = ()) -> str:
    cols = ["r", "lo", "hi", "p", "n", "sticky", *extra]
    return t[[c for c in cols if c in t]].round(3).to_string()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pre, priv = build()
    pre.to_parquet(OUT / "team_game_pregame_2019_2026.parquet")
    feats = sorted(c for c in pre.columns if c.startswith("pre_") and not c.endswith("_n") and not c.startswith("pre_opp"))
    FTN_KEYS = {"motion", "pa", "pa_epa", "screen", "blitz", "box", "rushers", "int_worthy", "oop", "drop",
                "catchable", "nohuddle", "rpo"}
    ftn_feats = [f for f in feats if f.split("_", 2)[2] in FTN_KEYS]
    base_feats = [f for f in feats if f not in ftn_feats]
    lines = ["# Feature scan: what predicts future EPA beyond EPA itself?\n",
             f"Team-games from week {MIN_WEEK}+, teams with {MIN_PRIOR}+ prior games. Partial r = correlation between a "
             "team's pre-game metric and its EPA/play in the next game, after removing its own pre-game EPA/play, the "
             "opponent's pre-game EPA on the other side and season effects. |r| of 0.05 is small; 0.10 is meaningful.\n"]

    # 1-2. discovery + validation for nflverse and free-FTN features
    results = {}
    for label, fl, disc in [("nflverse", base_feats, DISC), ("FTN free charting", ftn_feats, [2022, 2023])]:
        d = scan(pre, fl, disc)
        d["fdr_pass"] = bh(d.p)
        v = scan(pre, fl, VALID)
        h = scan(pre, fl, HOLD)
        d = d.join(v[["r", "p"]].rename(columns={"r": "r_valid", "p": "p_valid"})).join(h[["r", "p"]].rename(columns={"r": "r_2026", "p": "p_2026"}))
        d["replicated"] = d.fdr_pass & (np.sign(d.r) == np.sign(d.r_valid)) & (d.p_valid < 0.05)
        d = d.sort_values("p")
        results[label] = d
        lines += [f"\n## {label}: discovery {disc[0]}–{disc[-1]}, validation 2024–25, holdout 2026\n", "```",
                  d[["r", "lo", "hi", "p", "fdr_pass", "r_valid", "p_valid", "replicated", "r_2026", "sticky"]].round(3).to_string(), "```"]
    rep = pd.concat(results.values())
    replicated = list(rep.index[rep.replicated])

    # 3. 2026-only exploratory: PFF team grades, Sumer tags, paid FTN
    expl_rows = []
    if priv is not None:
        sum_feats = [c for c in priv.columns if c.startswith("pre_") and not c.endswith("_n")]
        pv = pre[pre.season == 2026].merge(priv[["game_id", "team", *sum_feats]], on=["game_id", "team"], how="left")
        expl_rows.append(scan(pv, sum_feats, HOLD).assign(source="Sumer"))
    pff = F.pff_team_week(2026)
    if pff is not None:
        pff = pff.sort_values(["team", "week"])
        gcols = [c for c in pff.columns if c.startswith("pff_")]
        cum = pff.groupby("team")[gcols].transform(lambda s: s.shift(1).expanding().mean())
        pff_pre = pd.concat([pff[["team", "week"]], cum.add_prefix("pre_")], axis=1)
        # name PFF features by the unit they describe so the scan picks the right target
        unit = {"pass_block": "off", "run_block": "off", "pressures_allowed_rate": "off", "qb_pass": "off",
                "qb_twp_rate": "off", "route": "off", "rush": "off", "pass_rush": "def", "run_def": "def",
                "tackle": "def", "def": "def", "pressure_rate": "def"}
        ren = {f"pre_pff_{k}": f"pre_{u}_pff_{k}" for k, u in unit.items()}
        pff_pre = pff_pre.rename(columns=ren)
        pv = pre[pre.season == 2026].merge(pff_pre, on=["team", "week"], how="left")
        expl_rows.append(scan(pv, list(ren.values()), HOLD).assign(source="PFF"))
        pff.to_csv(OUT / "pff_team_week_2026.csv", index=False)
    if expl_rows:
        ex = pd.concat(expl_rows)
        ex["fdr_pass"] = bh(ex.p)
        ex = ex.sort_values("p")
        lines += ["\n## 2026 only, EXPLORATORY (PFF grades, Sumer tags): cannot be validated yet\n", "```",
                  ex[["source", "r", "lo", "hi", "p", "fdr_pass", "n", "sticky"]].round(3).to_string(), "```"]

    # 4. market test: does any matchup differential predict the cover margin against the closing spread?
    all_x = [f[4:] for f in feats]
    gm = game_table(pre, all_x)
    gm["ats_margin"] = gm.result - gm.spread_line
    def ats_scan(seasons):
        d = gm[gm.season.isin(seasons) & gm.ats_margin.notna()]
        out = {}
        for x in all_x:
            c = d[[f"d_{x}", "ats_margin", "season", "week"]].dropna()
            if len(c) < 80:
                continue
            r = c[f"d_{x}"].corr(c.ats_margin)
            cl = (c.season.astype(str) + "_" + c.week.astype(str)).to_numpy()
            ids = np.unique(cl); idx = {k: np.where(cl == k)[0] for k in ids}
            xv, yv = c[f"d_{x}"].to_numpy(), c.ats_margin.to_numpy()
            bs = [np.corrcoef(xv[t], yv[t])[0, 1] for t in (np.concatenate([idx[k] for k in RNG.choice(ids, len(ids))]) for _ in range(BOOT))]
            se = np.nanstd(bs)
            out[x] = {"r": r, "p": math.erfc(abs(r / se) / math.sqrt(2)) if se else 1, "n": len(c)}
        return pd.DataFrame(out).T
    a_d = pd.concat([ats_scan(DISC).loc[[x for x in all_x if f"pre_{x}" in base_feats]],
                     ats_scan([2022, 2023]).loc[[x for x in all_x if f"pre_{x}" in ftn_feats]]])
    a_d["fdr_pass"] = bh(a_d.p)
    a_v, a_h = ats_scan(VALID), ats_scan(HOLD)
    a_d = a_d.join(a_v[["r", "p"]].rename(columns={"r": "r_valid", "p": "p_valid"})).join(a_h[["r"]].rename(columns={"r": "r_2026"}))
    a_d["replicated"] = a_d.fdr_pass & (np.sign(a_d.r) == np.sign(a_d.r_valid)) & (a_d.p_valid < 0.05)
    a_d = a_d.sort_values("p")
    lines += ["\n## Against the spread: home-minus-away pre-game differential vs cover margin (result − closing spread)\n",
              "Positive r = the side that's better on this metric covers more than the line expects.\n", "```",
              a_d[["r", "p", "n", "fdr_pass", "r_valid", "p_valid", "replicated", "r_2026"]].astype(float, errors="ignore").round(3).to_string(), "```"]

    # 5. game model: EPA-only baseline vs baseline + replicated features
    base = ["off_epa", "def_epa"]
    extra = [f for f in replicated if f not in base]
    m = game_table(pre, sorted(set(base + extra + ["off_sr", "def_sr"])))
    cols0 = [f"d_{x}" for x in base]
    ev0 = evaluate(walk_forward(m, cols0))
    lines += ["\n## Game model, walk-forward (each season predicted from earlier seasons only)\n",
              "Baseline: home-field + pre-game offensive and defensive EPA/play differentials.\n", "```", ev0.round(3).to_string(), "```"]
    if extra:
        cols1 = cols0 + [f"d_{x}" for x in extra]
        pr1 = walk_forward(m, cols1)
        ev1 = evaluate(pr1)
        lines += [f"\nBaseline + replicated features ({', '.join(extra)}):\n", "```", ev1.round(3).to_string(), "```"]
        pr1.to_csv(OUT / "game_model_predictions.csv", index=False)
    else:
        lines += ["\nNo feature replicated, so there is nothing to add to the model."]
    lines += ["\nBreakeven against the spread at -110 is 52.4%. Spread MAE is the closing line's own error, for scale."]

    # 6. one-row-per-team 2026 reference table (season to date): nflverse + free FTN + PFF + Sumer
    last = pre[pre.season == 2026].sort_values("week").groupby("team").tail(1)
    keep = ["off_epa", "def_epa", "off_sr", "def_sr", "off_pass_epa", "off_run_epa", "def_pass_epa", "def_run_epa",
            "off_proe", "off_expl", "def_expl", "off_motion", "off_pa", "off_screen", "off_nohuddle", "def_blitz", "def_box",
            "def_rushers", "off_int_worthy", "off_sack_rate", "def_sack_rate", "off_to_rate", "def_to_rate"]
    # season-to-date through the latest game = pre-game value of the next game isn't available, so recompute from sums
    tgs = F.team_games(F.load_plays(2026))
    sums = tgs.groupby("team")[[c for c in tgs.columns if c.endswith("__s") or c.endswith("__n")]].sum()
    ref = pd.DataFrame({k: sums[f"{k}__s"] / sums[f"{k}__n"] for k in keep if f"{k}__s" in sums})
    ref["games"] = tgs.groupby("team").game_id.nunique()
    if pff is not None:
        ref = ref.join(pff.groupby("team")[[c for c in pff.columns if c.startswith("pff_")]].mean())
    if priv is not None:
        ps = F.private_team_games(F.load_plays(2026))
        psum = ps.groupby("team")[[c for c in ps.columns if c.endswith("__s") or c.endswith("__n")]].sum()
        for k in sorted({c[:-3] for c in psum.columns}):
            ref[f"sumer_{k}"] = psum[f"{k}__s"] / psum[f"{k}__n"]
    ref.round(4).to_csv(OUT / "league_reference_2026.csv")

    report = "\n".join(lines)
    (OUT / "feature_scan_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
