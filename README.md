# NFL EPA Tiers

My take on [rbsdm.com](https://rbsdm.com): an NFL efficiency site built on
[nflverse](https://github.com/nflverse) play-by-play and FTN charting data, plus a few views
I wanted that rbsdm doesn't have. It's plain HTML/JS on top of a Python data build, updates
itself twice a week, and works offline.

## Running it

```bash
./update.sh --open
```

Needs Python 3.10+. The first run sets up a virtualenv in `.venv/`, downloads the current
season and opens `site/index.html`. No server needed.

## Tabs

| Tab | |
|---|---|
| Teams | Offense vs defense EPA/play tiers, net EPA, success rate, pass/run splits |
| Quarterbacks | EPA per dropback vs CPOE, plus a composite (0.2×EPA + 0.01×CPOE) |
| Team trend | One team's offense and defense EPA by week with a 3-game rolling average |
| Tendencies | Pass rate over expected, shotgun, no-huddle, motion, play-action, RPO, screens, blitz rate, with EPA splits |
| 4th downs | Go-for-it attempts over expected by team, conversion rate, gutsiest and most conservative calls |
| Matchup | How an opponent played a team vs how that opponent plays everyone else, both sides of the ball, by half |
| Pick'em | This week's games with a lean for each, for ATS or straight-up pick'em leagues |

Filters at the top cover week range, play type, down, win probability (to drop garbage time)
and playoffs.

## Notes on method

- Only pass and run plays count, so no-play penalties, kneels and spikes are out. EPA and CPOE
  are nflfastR's.
- PROE is actual pass rate minus nflverse's `xpass`.
- 4th downs: `build_data.py` fits a logistic regression on the previous 5 seasons (distance,
  field position, FG range, score, clock, win prob) and sums `went for it - P(typical team goes)`
  per team. That measures aggressiveness relative to league habit, not whether the call was
  right. nfl4th is the tool for that. The fit is cached in `data/fourth_model.json`
  (`--retrain` to refit).
- Matchup: "normal" is the opponent's other games, by default only the ones before the matchup.
  Neutral early downs = 1st/2nd down, win prob 20-80%, more than 2:00 left in the half. The
  signal column flags a change as clear at roughly 2 standard errors and marks anything under
  20 plays as a small sample. One game is a small sample.
- If you have a local file of charted coverage/personnel tags matched to nflverse play ids,
  drop it in `data/private/` (ignored by git) and the Matchup tab picks up man/zone, coverage
  shell, nickel/dime and personnel rows.
- Pick'em leans use the only 4 of 30 situational angles that held up in both 1999-2012 and
  2013-now (bounce-back after a 20+ point loss, division dog, dog of 7+, road team). Since they
  overlap, the page looks up how each exact combination has covered and shrinks small samples
  toward 50%. Even the best combos are tiebreakers, not locks.

## Backtests

| Script | |
|---|---|
| `analysis/ats_backtest.py` | Does a weekly EPA rating beat the closing spread? (walk-forward, 2020-now) |
| `analysis/situational_angles.py` | 30 common betting angles on every game since 1999, split by era |
| `analysis/feature_scan.py` | ~45 team metrics tested for predicting next-game EPA beyond EPA itself, and for predicting cover margin |

The feature scan finds candidates in 2019-2023, checks them on 2024-2025 and only reports 2026,
with a false-discovery correction and team-clustered bootstrap CIs. Short version of what came
out:

- Nothing replicated. Once you know a team's EPA/play, success rate, explosives, PROE, motion,
  play-action, blitz rate, box counts and turnover-worthy throws don't add anything for the
  next game, and none of them predict covering the spread.
- The EPA model alone went 48% ATS from 2021-2026. The closing line is better.
- Tendencies (motion, no-huddle, PROE, blitz rate) are stable week to week. Turnovers and drops
  are mostly noise.
- The "home underdog" angle covered exactly 50.0% since 1999.

## Updating

nflverse posts new play-by-play overnight after games.

- By hand: `./update.sh`
- On a Mac, automatically: `schedule/install_mac.sh` installs a launchd job for Tuesday and
  Friday at 7am (pass an hour to change it; `schedule/uninstall_mac.sh` removes it). Keep the
  folder out of iCloud/Desktop/Documents, since macOS won't let background jobs read those.
- GitHub Pages: push the repo, set Settings > Pages > Source to GitHub Actions, and
  `.github/workflows/update.yml` rebuilds and redeploys twice a week during the season.

Other seasons: `./update.sh --season 2024 --open`. FTN charting starts in 2022.

Every update also writes `site/NFL-EPA.html`, the whole site in one file with the data and logos
inlined, so it can be sent to someone and opened with a double-click.

## Layout

```
build_data.py         download + aggregate into site/data.js
make_standalone.py    bundle the site into one html file
update.sh             set up venv if needed, build, optionally open
site/                 index.html (the app), data.js (generated), logos/
analysis/             backtests
data/                 cached downloads, 4th-down model
schedule/             macOS launchd install/uninstall
.github/workflows/    scheduled rebuild + Pages deploy
```
