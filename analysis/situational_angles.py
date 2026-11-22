"""
Test well-known "sharp" situational ATS angles on every regular-season game since 1999.

An angle is only believable if it:
  * beats 50% in BOTH eras (1999–2012 and 2013–present), since markets learn and angles die,
  * has a large sample, and
  * survives the fact that we're testing ~25 angles (expect ~1 false "p<0.05" by chance).

Uses closing spreads from nflverse games.csv. Pushes excluded.
Usage:  .venv/bin/python analysis/situational_angles.py
"""
import math
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
g = pd.read_csv(ROOT / "data" / "games.csv")
g = g[(g.game_type == "REG") & g.result.notna() & g.spread_line.notna()].copy()
g = g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

# nflverse: spread_line > 0 means HOME favored by that many; result = home - away
g["ats"] = g.result - g.spread_line                  # > 0 home covered
g = g[g.ats != 0]
g["home_cov"] = g.ats > 0
g["hour"] = pd.to_numeric(g.gametime.str[:2], errors="coerce")
g["prime"] = (g.hour >= 19) | g.weekday.isin(["Monday", "Thursday"])
g["neutral"] = g.location == "Neutral"

WEST = {"SEA", "SF", "LA", "LAC", "SD", "OAK", "LV", "STL_never"}
PT_ERA = lambda team, season: team in WEST and not (team == "LA" and season < 2016)
DOME = {"ATL", "DET", "IND", "MIN", "NO", "HOU", "ARI", "DAL", "LV", "LA", "STL"}  # roughly; outdoor history ignored

# ---- per-team game log for "last game" angles
rows = []
for x in g.itertuples():
    for side, team, opp, margin, cov, qb in [
        ("home", x.home_team, x.away_team, x.result, x.home_cov, x.home_qb_id),
        ("away", x.away_team, x.home_team, -x.result, not x.home_cov, x.away_qb_id),
    ]:
        rows.append({"idx": x.Index, "season": x.season, "week": x.week, "team": team, "side": side,
                     "margin": margin, "cover": cov, "ats_m": x.ats if side == "home" else -x.ats,
                     "qb": qb, "ot": x.overtime == 1})
log = pd.DataFrame(rows).sort_values(["team", "season", "week"])
grp = log.groupby(["team", "season"])
log["prev_margin"] = grp.margin.shift(1)
log["prev_ats_m"] = grp.ats_m.shift(1)
log["prev_qb"] = grp.qb.shift(1)
log["prev_ot"] = grp.ot.shift(1)
log["ytd_cov"] = grp.cover.transform(lambda s: s.shift(1).expanding().mean())
log["ytd_n"] = grp.cover.cumcount()
for side in ["home", "away"]:
    sub = log[log.side == side].set_index("idx")
    for c in ["prev_margin", "prev_ats_m", "prev_qb", "prev_ot", "ytd_cov", "ytd_n"]:
        g[f"{side}_{c}"] = sub[c]

# ---- helpers: an angle = boolean mask + which side it picks (True = home)
angles = []
def angle(name, mask, pick_home, why=""):
    mask = mask.fillna(False).astype(bool)
    pick_home = pd.Series(pick_home, index=g.index) if not isinstance(pick_home, pd.Series) else pick_home
    win = np.where(pick_home[mask], g.home_cov[mask], ~g.home_cov[mask])
    angles.append((name, mask, pick_home, why))

hfav = g.spread_line > 0
dog_home = g.spread_line < 0          # home team is the underdog
sp = g.spread_line.abs()

# Classic favorites / dogs
angle("All home teams", ~g.neutral, True)
angle("All underdogs", sp > 0, ~hfav)
angle("Home underdogs", dog_home & ~g.neutral, True)
angle("Road favorites (fade them)", hfav.eq(False) & ~g.neutral & (sp > 0), True, "same as home dogs")
angle("Big dogs +7 or more", sp >= 7, ~hfav)
angle("Big dogs +10 or more", sp >= 10, ~hfav)
angle("Small dogs +1 to +3", sp.between(1, 3), ~hfav)
angle("Dogs getting exactly +3", sp == 3, ~hfav, "key number")
angle("Dogs getting exactly +7", sp == 7, ~hfav, "key number")
angle("Dogs getting +3.5 (hook past 3)", sp == 3.5, ~hfav, "key number")

# Division / familiarity
angle("Divisional dogs", (g.div_game == 1) & (sp > 0), ~hfav)
angle("Divisional home dogs", (g.div_game == 1) & dog_home, True)

# Rest / schedule
rest_adv = g.home_rest - g.away_rest
angle("Team off bye vs. team not off bye", (g.home_rest >= 13) ^ (g.away_rest >= 13), g.home_rest >= 13)
angle("Rest edge ≥ 3 days (pick rested team)", rest_adv.abs() >= 3, rest_adv > 0)
angle("Thursday road dogs", (g.weekday == "Thursday") & hfav, False)
angle("Thursday home teams", g.weekday == "Thursday", True)

# Travel / body clock
west_early_east = g.away_team.apply(lambda t: t in WEST) & (g.hour == 13) & ~g.home_team.isin(list(WEST)) & ~g.neutral
angle("West-coast team at 1pm ET game (fade)", west_early_east, True)
angle("Primetime dogs", g.prime & (sp > 0), ~hfav)
angle("Primetime home teams", g.prime & ~g.neutral, True)

# Weather
cold = (g.temp <= 32) & (g.roof.isin(["outdoors", "open"]))
angle("Freezing game: pick home team", cold, True)
angle("Freezing game: fade dome team on road", cold & g.away_team.isin(list(DOME)), True)
angle("Wind ≥ 15 mph: pick the dog", (g.wind >= 15) & (sp > 0), ~hfav, "low-scoring = fewer pts to cover")

# Overreaction to last week
for t, n in [(20, "20+")]:
    a = g.away_prev_margin <= -t; h = g.home_prev_margin <= -t
    angle(f"Team lost by {n} last week (back them)", a ^ h, h)
    a = g.away_prev_margin >= t; h = g.home_prev_margin >= t
    angle(f"Team won by {n} last week (fade them)", a ^ h, a)
a = g.away_prev_ats_m >= 14; h = g.home_prev_ats_m >= 14
angle("Team beat spread by 14+ last week (fade)", a ^ h, a)
a = g.away_ytd_cov.ge(0.7) & g.away_ytd_n.ge(6); h = g.home_ytd_cov.ge(0.7) & g.home_ytd_n.ge(6)
angle("Hot ATS team (≥70% covers, 6+ games) – fade", a ^ h, a, "public loves hot teams")
a = g.away_prev_ot.eq(True); h = g.home_prev_ot.eq(True)
angle("Team coming off OT game (fade)", a ^ h, a)

# QB change
hq = g.home_prev_qb.notna() & (g.home_qb_id != g.home_prev_qb)
aq = g.away_prev_qb.notna() & (g.away_qb_id != g.away_prev_qb)
angle("Team starting a different QB than last week (back them)", hq ^ aq, hq, "market may overreact to backups")

# Late season
angle("Week 17/18 dogs", (g.week >= 17) & (sp > 0) & ~((g.season < 2021) & (g.week == 18)), ~hfav)
angle("December+ home dogs", (g.week >= 13) & dog_home, True)

# ---- evaluate
def stats(mask, pick_home, sel):
    m = mask & sel
    if m.sum() == 0:
        return 0, 0, np.nan
    win = np.where(pick_home[m], g.home_cov[m], ~g.home_cov[m])
    return int(win.sum()), int(m.sum()), win.mean()

early, late = g.season <= 2012, g.season >= 2013
recent = g.season >= 2019
out = []
for name, mask, pick, why in angles:
    w, n, p = stats(mask, pick, slice(None) if False else pd.Series(True, index=g.index))
    _, n1, p1 = stats(mask, pick, early)
    _, n2, p2 = stats(mask, pick, late)
    _, n3, p3 = stats(mask, pick, recent)
    z = (p - 0.5) / math.sqrt(0.25 / n) if n else 0
    pval = math.erfc(abs(z) / math.sqrt(2))
    out.append((name, w, n, p, p1, n1, p2, n2, p3, n3, pval, why))

out.sort(key=lambda r: -abs(r[3] - 0.5) * math.sqrt(r[2]))
print(f"{len(g):,} regular-season games {g.season.min()}–{g.season.max()}, closing lines, pushes excluded")
print(f"Testing {len(out)} angles → expect ~{len(out)*0.05:.1f} to look 'significant' (p<.05) by pure luck.\n")
print(f"{'Angle':<52}{'All':>13}{'1999–2012':>12}{'2013–now':>11}{'2019–now':>11}{'p':>7}  verdict")
for name, w, n, p, p1, n1, p2, n2, p3, n3, pval, why in out:
    both = (p1 > 0.5) == (p2 > 0.5) and min(abs(p1 - .5), abs(p2 - .5)) > 0.01
    verdict = ("consistent" if both and pval < 0.05 else "weak/consistent" if both else "inconsistent") if n >= 200 else "small n"
    print(f"{name:<52}{p:6.1%} {n:>5} {p1:8.1%} {p2:10.1%} {p3:10.1%}  {pval:6.3f}  {verdict}")
