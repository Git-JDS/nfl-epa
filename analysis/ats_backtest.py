"""
Does EPA beat the closing spread?  Walk-forward backtest, 2020–present.

For every regular-season game:
  1. Rate each team using ONLY plays from earlier weeks that season, blended with a
     regressed version of last season's rating (prior fades as games accumulate).
  2. Convert the rating gap to a predicted margin with a regression fit ONLY on
     earlier seasons.
  3. Edge = predicted home margin - closing spread. Bet the side the edge points to.
Breakeven at standard -110 juice is 52.38%.

Also tests: ATS streaks (does covering predict covering?) and "point-differential luck"
(teams whose scores lag their EPA).

Usage:  .venv/bin/python analysis/ats_backtest.py
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEASONS = list(range(2019, 2027))
PRIOR_PLAYS = 250      # last season's rating counts like this many plays (~4 games)
PRIOR_REGRESS = 0.6    # ...after shrinking it 40% toward average
BREAKEVEN = 110 / 210

# ------------------------------------------------------------------ data
games = pd.read_csv(DATA / "games.csv")
games = games[(games.season.isin(SEASONS)) & (games.game_type == "REG")].copy()
games["neutral"] = (games.location == "Neutral").astype(int)
# nflverse uses LA / LAC / LV etc. consistently across pbp and games.

frames = []
for s in SEASONS:
    p = pd.read_parquet(DATA / f"pbp_{s}.parquet",
                        columns=["game_id", "season", "week", "season_type", "posteam", "defteam", "pass", "rush", "epa"])
    p = p[(p.season_type == "REG") & ((p["pass"] == 1) | (p["rush"] == 1))].dropna(subset=["epa", "posteam"])
    frames.append(p)
pbp = pd.concat(frames)

off = pbp.groupby(["season", "week", "posteam"]).epa.agg(["sum", "count"]).rename_axis(["season", "week", "team"])
dfn = pbp.groupby(["season", "week", "defteam"]).epa.agg(["sum", "count"]).rename_axis(["season", "week", "team"])
tw = off.add_suffix("_off").join(dfn.add_suffix("_def"), how="outer").fillna(0).reset_index()

# final-season ratings, for next year's prior
fin = tw.groupby(["season", "team"])[["sum_off", "count_off", "sum_def", "count_def"]].sum()
fin["net"] = fin.sum_off / fin.count_off - fin.sum_def / fin.count_def


def rating(season: int, week: int, team: str) -> float:
    """Net EPA/play (offense minus defense allowed) entering `week`, no lookahead."""
    cur = tw[(tw.season == season) & (tw.week < week) & (tw.team == team)]
    so, no_, sd, nd = cur.sum_off.sum(), cur.count_off.sum(), cur.sum_def.sum(), cur.count_def.sum()
    prior = fin.net.get((season - 1, team), 0.0) * PRIOR_REGRESS if season - 1 in SEASONS else 0.0
    # blend offense and defense separately toward prior/2 each
    o = (so + prior / 2 * PRIOR_PLAYS) / (no_ + PRIOR_PLAYS)
    d = (sd - prior / 2 * PRIOR_PLAYS) / (nd + PRIOR_PLAYS)
    return o - d


def avg_margin(season: int, week: int, team: str) -> float:
    g = games[(games.season == season) & (games.week < week) & games.result.notna()
              & ((games.home_team == team) | (games.away_team == team))]
    if g.empty:
        return np.nan
    m = np.where(g.home_team == team, g.result, -g.result)
    return float(m.mean())


rows = []
for g in games.itertuples():
    rh, ra = rating(g.season, g.week, g.home_team), rating(g.season, g.week, g.away_team)
    rows.append({
        "season": g.season, "week": g.week, "home": g.home_team, "away": g.away_team,
        "spread": g.spread_line, "result": g.result, "neutral": g.neutral,
        "diff": rh - ra,
        "mdiff": avg_margin(g.season, g.week, g.home_team) - avg_margin(g.season, g.week, g.away_team),
    })
df = pd.DataFrame(rows).dropna(subset=["spread"])

# ------------------------------------------------------------------ walk-forward model
def fit(train):
    X = np.column_stack([np.ones(len(train)), 1 - train.neutral, train["diff"]])
    return np.linalg.lstsq(X, train.result, rcond=None)[0]

df["pred"] = np.nan
for s in SEASONS[1:]:
    coef = fit(df[(df.season < s) & df.result.notna()])
    m = df.season == s
    df.loc[m, "pred"] = coef[0] * 0 + coef[1] * (1 - df.loc[m, "neutral"]) + coef[2] * df.loc[m, "diff"]
    # intercept dropped: home-field already in coef[1]; league-wide margin intercept ~0

df["edge"] = df.pred - df.spread
played = df[df.result.notna() & df.pred.notna()].copy()
played["ats"] = played.result - played.spread                  # >0 home covered
played = played[played.ats != 0]                                # drop pushes
played["model_win"] = np.sign(played.edge) == np.sign(played.ats)


def rate(sub):
    n = len(sub)
    w = sub.model_win.sum()
    p = w / n if n else np.nan
    se = np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n) if n else np.nan
    return f"{w:>4}-{n - w:<4} {p:6.1%}  (n={n:4}, ±{1.96 * se:4.1%} 95% noise band)"


print("=" * 78)
print("EPA MODEL vs CLOSING SPREAD: walk-forward, regular season 2020–2026")
print("=" * 78)
print(f"Breakeven at -110: {BREAKEVEN:.2%}\n")
print("How well does each predict the final margin? (correlation with result)")
print(f"  closing spread : r = {np.corrcoef(played.spread, played.result)[0,1]:.3f}")
print(f"  EPA model      : r = {np.corrcoef(played.pred, played.result)[0,1]:.3f}")
print(f"  model vs spread: r = {np.corrcoef(played.pred, played.spread)[0,1]:.3f}  (how much the market already knows)\n")

X = np.column_stack([np.ones(len(played)), played.spread, played.pred])
b = np.linalg.lstsq(X, played.result, rcond=None)[0]
resid = played.result - X @ b
cov = np.linalg.inv(X.T @ X) * resid.var(ddof=3)
print("Result ~ spread + EPA model (does EPA add anything beyond the line?)")
print(f"  spread weight {b[1]:.2f} (se {np.sqrt(cov[1,1]):.2f}),  EPA-model weight {b[2]:.2f} (se {np.sqrt(cov[2,2]):.2f})\n")

print("Betting the side the EPA edge points to:")
print(f"  all games          {rate(played)}")
for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4.5), (4.5, 99)]:
    sub = played[(played.edge.abs() >= lo) & (played.edge.abs() < hi)]
    print(f"  edge {lo:>3}–{hi:<4} pts   {rate(sub)}")
print()
print("By season (all edges):")
for s, sub in played.groupby("season"):
    print(f"  {s}  {rate(sub)}")
print()
print("By part of season:")
for name, sub in [("wks 1–4", played[played.week <= 4]), ("wks 5–9", played[played.week.between(5, 9)]),
                  ("wks 10–18", played[played.week >= 10])]:
    print(f"  {name:<9} {rate(sub)}")
big = played[played.edge.abs() >= 3]
print(f"\nBig edges (≥3 pts), betting dogs vs favorites:")
dog = big[np.sign(big.edge) != np.sign(big.spread)]
fav = big[np.sign(big.edge) == np.sign(big.spread)]
print(f"  model says dog     {rate(dog)}")
print(f"  model says fav     {rate(fav)}")

# ------------------------------------------------------------------ luck signal
played["luck"] = played.mdiff - played.pred      # scoring margin ran ahead of EPA => regress
l = played.dropna(subset=["luck"])
l = l[l.week >= 4]
print("\n" + "=" * 78)
print("POINT-DIFFERENTIAL 'LUCK' (weeks 4+): teams whose scoring margin outran their EPA")
print("=" * 78)
print(f"  corr(luck gap, ATS margin) r = {np.corrcoef(l.luck, l.result - l.spread)[0,1]:+.3f}  (negative = lucky teams fail to cover)")
for q in [5, 8]:
    sub = l[l.luck.abs() >= q].copy()
    sub["model_win"] = np.sign(-sub.luck) == np.sign(sub.ats)
    print(f"  fade the 'lucky' side when gap ≥ {q} pts: {rate(sub)}")

# ------------------------------------------------------------------ ATS streaks
long = pd.concat([
    pd.DataFrame({"season": played.season, "week": played.week, "team": played.home, "cover": played.ats > 0}),
    pd.DataFrame({"season": played.season, "week": played.week, "team": played.away, "cover": played.ats < 0}),
]).sort_values(["team", "season", "week"])
long["prev"] = long.groupby(["team", "season"]).cover.shift(1)
long["prev2"] = long.groupby(["team", "season"]).cover.shift(2)
long["ytd"] = long.groupby(["team", "season"]).cover.transform(lambda s: s.shift(1).expanding().mean())
x = long.dropna(subset=["prev"])
print("\n" + "=" * 78)
print("ATS MOMENTUM: does covering last week predict covering this week?")
print("=" * 78)
print(f"  covered last game  → cover rate next {x[x.prev == True].cover.mean():.1%}  (n={int((x.prev == True).sum())})")
print(f"  failed last game   → cover rate next {x[x.prev == False].cover.mean():.1%}  (n={int((x.prev == False).sum())})")
x2 = long.dropna(subset=["prev", "prev2"])
for v, name in [(True, "covered last 2"), (False, "failed last 2")]:
    s = x2[(x2.prev == v) & (x2.prev2 == v)]
    print(f"  {name:<18} → cover rate next {s.cover.mean():.1%}  (n={len(s)})")
y = long[(long.week >= 6)].dropna(subset=["ytd"])
print(f"  corr(season-to-date ATS %, next cover), wk 6+: r = {np.corrcoef(y.ytd, y.cover)[0,1]:+.3f}")

# ------------------------------------------------------------------ this week
up = df[df.result.isna() & df.pred.notna()]
if not up.empty:
    wk = up.week.min()
    cur = up[up.week == wk].copy()
    cur["pick"] = np.where(cur.edge > 0, cur.home, cur.away)
    cur = cur.reindex(cur.edge.abs().sort_values(ascending=False).index)
    print("\n" + "=" * 78)
    print(f"{int(cur.season.iloc[0])} WEEK {wk}: EPA model vs current line (home perspective, + = home favored)")
    print("=" * 78)
    for r in cur.itertuples():
        print(f"  {r.away:>3} @ {r.home:<3}  line {r.spread:+5.1f}  model {r.pred:+5.1f}  edge {r.edge:+5.1f}  → {r.pick}")

played.to_csv(DATA / "ats_backtest_games.csv", index=False)
