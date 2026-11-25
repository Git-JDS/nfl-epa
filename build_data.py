"""
Pull nflverse play-by-play (+ FTN charting) for a season and write a compact,
filterable dataset to site/data.js so site/index.html works offline (file://).

Usage:
    python build_data.py                  # current season, re-downloads current-season data
    python build_data.py --season 2025
    python build_data.py --no-download    # reuse cached files in data/
    python build_data.py --retrain        # refit the 4th-down model
"""
import argparse
import datetime as dt
import json
import math
import pathlib
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
SITE = ROOT / "site"
LOGOS = SITE / "logos"

REL = "https://github.com/nflverse/nflverse-data/releases/download"
PBP_URL = REL + "/pbp/play_by_play_{season}.parquet"
FTN_URL = REL + "/ftn_charting/ftn_charting_{season}.parquet"
TEAMS_URL = REL + "/teams/teams_colors_logos.csv"
LOGO_URL = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{abbr}.png&w=96&h=96"

CURRENT = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB",
    "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

# Play flags packed into one int per play (decoded in index.html).
F_SHOTGUN, F_NOHUDDLE, F_PA, F_MOTION, F_RPO, F_SCREEN, F_BLITZ, F_CHARTED = (1 << i for i in range(8))

FOURTH_TRAIN_YEARS = 5  # seasons before the current one used to fit the 4th-down model


def default_season() -> int:
    today = dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def download(url: str, dest: pathlib.Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "nfl-epa-local/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as e:
        print(f"  download failed ({url.rsplit('/', 1)[-1]}): {e}")
        return False


def fetch(url: str, dest: pathlib.Path, refresh: bool) -> bool:
    if dest.exists() and not refresh:
        return True
    return download(url, dest) or dest.exists()


def r(x, nd):
    return None if pd.isna(x) else round(float(x), nd)


# ---------------------------------------------------------------- coverage tags (optional)
# If a local file of charted coverage/personnel tags matched to nflverse play ids exists under
# data/private/ (git-ignored), join it. Otherwise these fields are -1 and the site hides them.
TAG_COV = ["COVER 0", "COVER 1", "COVER 2", "COVER 3", "COVER 4", "COVER 6", "COVER-RZ"]
TAG_MZ = ["ZONE", "MAN", "OTHER"]
TAG_DP = ["BASE", "NICKEL", "DIME"]
TAG_RUN = ["OUTSIDE ZONE", "INSIDE ZONE", "POWER", "PULL LEAD", "COUNTER", "TRAP", "OPTION", "TRICK RUN"]


def coverage_tags(season: int) -> pd.DataFrame | None:
    files = sorted((DATA / "private").glob(f"**/coverage_tags_{season}.json"))
    if not files:
        return None
    s = pd.DataFrame(json.loads(files[-1].read_text()))
    s["play_id"] = pd.to_numeric(s.matched_play_id, errors="coerce")
    s = s.dropna(subset=["play_id"]).sort_values("match_score").drop_duplicates(["game_id", "play_id"], keep="last")
    idx = lambda vals, v: vals.index(v) if v in vals else -1
    pers = lambda v: int(v.rstrip("*")) if v.rstrip("*").isdigit() else -1
    def pt(v: str) -> int:
        return (1 if "BLITZ" in v else 0) | (2 if "STUNT" in v else 0) | (4 if "PRESSURE" in v else 0)
    return pd.DataFrame({
        "game_id": s.game_id, "play_id": s.play_id.astype(int),
        "s_cov": [idx(TAG_COV, v) for v in s.coverage], "s_mz": [idx(TAG_MZ, v) for v in s.man_zone],
        "s_dp": [idx(TAG_DP, v) for v in s.defensive_personnel], "s_op": [pers(v) for v in s.personnel],
        "s_run": [idx(TAG_RUN, v) for v in s.run_concept],
        "s_pt": [pt(v) if c else -1 for v, c in zip(s.pressure_tag, s.coverage)],
    })


# ---------------------------------------------------------------- 4th downs
FOURTH_COLS = [
    "game_id", "season_type", "week", "posteam", "defteam", "down", "play_type", "ydstogo",
    "yardline_100", "wp", "score_differential", "game_seconds_remaining", "half_seconds_remaining", "qtr", "epa",
    "fourth_down_converted", "qb_kneel", "desc",
]


def fourth_downs(pbp: pd.DataFrame) -> pd.DataFrame:
    d = pbp[(pbp.down == 4) & pbp.play_type.isin(["pass", "run", "punt", "field_goal"])]
    d = d[(d.qb_kneel != 1)].dropna(subset=["ydstogo", "yardline_100", "wp", "score_differential", "game_seconds_remaining"])
    d = d.copy()
    d["go"] = d.play_type.isin(["pass", "run"]).astype(int)
    return d


def features(d: pd.DataFrame) -> np.ndarray:
    yl = d.yardline_100.to_numpy(float) / 100
    ytg = np.log1p(d.ydstogo.to_numpy(float))
    sd = np.clip(d.score_differential.to_numpy(float), -24, 24) / 10
    t = d.game_seconds_remaining.to_numpy(float) / 3600
    wp = d.wp.to_numpy(float)
    trailing_late = (sd < 0) * (1 - t) ** 2 * -sd
    fg_range = (d.yardline_100.to_numpy(float) <= 37).astype(float)
    # End of first half: teams kick rather than go when the clock is nearly out.
    hs = d.half_seconds_remaining.fillna(900).to_numpy(float)
    eoh = ((d.qtr.to_numpy() == 2) & (hs <= 60)).astype(float)
    eoh_final = ((d.qtr.to_numpy() == 2) & (hs <= 15)).astype(float)
    return np.column_stack([
        np.ones(len(d)), ytg, ytg ** 2, yl, yl ** 2, yl * ytg, fg_range, fg_range * ytg,
        sd, sd * (1 - t), trailing_late, wp, t, eoh, eoh * fg_range, eoh_final,
    ])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Plain IRLS / Newton logistic regression, so we don't need scikit-learn."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def fourth_model(season: int, refresh: bool, retrain: bool) -> np.ndarray:
    path = DATA / "fourth_model.json"
    if path.exists() and not retrain:
        m = json.loads(path.read_text())
        if m.get("season") == season:
            return np.array(m["coef"])
    years = list(range(season - FOURTH_TRAIN_YEARS, season))
    print(f"Fitting 4th-down go model on {years[0]}–{years[-1]}…")
    frames = []
    for y in years:
        p = DATA / f"pbp_{y}.parquet"
        if fetch(PBP_URL.format(season=y), p, refresh=False):
            frames.append(fourth_downs(pd.read_parquet(p, columns=FOURTH_COLS)))
    train = pd.concat(frames)
    train = train[train.season_type == "REG"]
    X, y = features(train), train.go.to_numpy(float)
    coef = fit_logistic(X, y)
    p = 1 / (1 + np.exp(-X @ coef))
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"  {len(train):,} 4th downs, go rate {y.mean():.1%}, log loss {ll:.3f}")
    path.write_text(json.dumps({"season": season, "years": years, "n": len(train), "coef": coef.tolist()}))
    return coef


# ---------------------------------------------------------------- pick'em
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
# Net-EPA to margin conversion, fit walk-forward in analysis/ats_backtest.py on 2019–2025.
HFA_PTS, PTS_PER_EPA = 1.72, 48.66
PRIOR_PLAYS, PRIOR_REGRESS = 250, 0.6


def side_keys(spread, div, neutral, last_margin, home: bool) -> list:
    """Which angles apply to one side of a game. spread: + = home favored."""
    k = []
    dog = spread < 0 if home else spread > 0
    if last_margin is not None and not pd.isna(last_margin) and last_margin <= -20: k.append("bounce")
    if div and dog: k.append("divdog")
    if abs(spread) >= 7 and dog: k.append("bigdog")
    if not home and not neutral: k.append("road")
    return k


def combo_history(h: pd.DataFrame) -> dict:
    """Home cover rate for each exact combination of angles (home keys | away keys).
    Angles overlap and don't add up, so the page looks up the real combined record."""
    out = {}
    for x in h.itertuples():
        neutral = x.location == "Neutral"
        sig = ",".join(side_keys(x.spread_line, x.div_game == 1, neutral, x.hp, True)) + "|" + \
              ",".join(side_keys(x.spread_line, x.div_game == 1, neutral, x.ap, False))
        c = out.setdefault(sig, [0, 0, 0, 0])   # home covers, n, home covers since 2013, n since 2013
        c[0] += int(x.hc); c[1] += 1
        if x.season >= 2013:
            c[2] += int(x.hc); c[3] += 1
    return out


def angle_history(g: pd.DataFrame, season: int) -> dict:
    """ATS hit rate of each angle across all seasons before `season` (see analysis/situational_angles.py)."""
    h = g[(g.game_type == "REG") & g.result.notna() & g.spread_line.notna() & (g.season < season)].copy()
    h = h.sort_values(["season", "week", "gameday"])
    h["ats"] = h.result - h.spread_line
    h = h[h.ats != 0]
    h["hc"] = h.ats > 0
    sp = h.spread_line
    prev = {}
    rows = []
    for x in h.sort_values(["season", "week"]).itertuples():
        rows.append((prev.get((x.season, x.home_team)), prev.get((x.season, x.away_team))))
        prev[(x.season, x.home_team)] = x.result
        prev[(x.season, x.away_team)] = -x.result
    h["hp"], h["ap"] = zip(*rows)
    h["hp"], h["ap"] = h.hp.astype(float), h.ap.astype(float)
    early = h.season <= 2012

    def rate(mask, pick_home):
        m = mask.fillna(False)
        win = np.where(pick_home[m], h.hc[m], ~h.hc[m])
        e = np.where(pick_home[m & early], h.hc[m & early], ~h.hc[m & early])
        l = np.where(pick_home[m & ~early], h.hc[m & ~early], ~h.hc[m & ~early])
        return {"rate": round(float(win.mean()), 4), "n": int(m.sum()),
                "early": round(float(e.mean()), 4), "late": round(float(l.mean()), 4)}

    true = pd.Series(True, index=h.index)
    hb, ab = h.hp <= -20, h.ap <= -20
    angles = [
        {"key": "bounce", "label": "Bounce-back", "desc": "Lost by 20+ last game", **rate(hb ^ ab, hb)},
        {"key": "divdog", "label": "Division dog", "desc": "Underdog in a division game", **rate((h.div_game == 1) & (sp != 0), sp < 0)},
        {"key": "bigdog", "label": "Big dog", "desc": "Getting 7+ points", **rate(sp.abs() >= 7, sp < 0)},
        {"key": "road", "label": "Road team", "desc": "Visiting team (lines slightly overrate home field)",
         **rate(h.location != "Neutral", ~true)},
    ]
    return {"angles": angles, "combos": combo_history(h)}


def net_ratings(pbp_cur: pd.DataFrame, prior_path: pathlib.Path, before_week: int) -> dict:
    """Net EPA/play (offense − defense) per team entering `before_week`, blended with last season."""
    cur = pbp_cur[pbp_cur.week < before_week]
    prior = {}
    if prior_path.exists():
        p = pd.read_parquet(prior_path, columns=["season_type", "posteam", "defteam", "pass", "rush", "epa"])
        p = p[(p.season_type == "REG") & ((p["pass"] == 1) | (p["rush"] == 1))].dropna(subset=["epa"])
        o, d = p.groupby("posteam").epa.mean(), p.groupby("defteam").epa.mean()
        prior = ((o - d) * PRIOR_REGRESS).to_dict()
    so, no_ = cur.groupby("posteam").epa.sum(), cur.groupby("posteam").epa.count()
    sd, nd = cur.groupby("defteam").epa.sum(), cur.groupby("defteam").epa.count()
    out = {}
    for t in CURRENT:
        pr = prior.get(t, 0.0)
        o = (so.get(t, 0) + pr / 2 * PRIOR_PLAYS) / (no_.get(t, 0) + PRIOR_PLAYS)
        d = (sd.get(t, 0) - pr / 2 * PRIOR_PLAYS) / (nd.get(t, 0) + PRIOR_PLAYS)
        out[t] = o - d
    return out


def ml_prob(home_ml, away_ml):
    """No-vig home win probability from American moneylines."""
    def imp(ml):
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None
    h, a = imp(home_ml), imp(away_ml)
    return h / (h + a)


def pickem(season: int, pbp_cur: pd.DataFrame, refresh: bool) -> dict | None:
    gpath = DATA / "games.csv"
    if not fetch(GAMES_URL, gpath, refresh):
        return None
    g = pd.read_csv(gpath)
    sg = g[(g.season == season) & g.game_type.isin(["REG", "WC", "DIV", "CON", "SB"])]
    open_ = sg[sg.result.isna()]
    if open_.empty:
        return None
    week = int(open_.week.min())
    wk = sg[sg.week == week].sort_values(["gameday", "gametime"])
    prior_path = DATA / f"pbp_{season - 1}.parquet"
    fetch(PBP_URL.format(season=season - 1), prior_path, refresh=False)
    ratings = net_ratings(pbp_cur, prior_path, week)

    done = sg[sg.result.notna() & (sg.week < week)].sort_values(["week"])
    last = {}
    for x in done.itertuples():
        last[x.home_team] = (int(x.week), x.away_team, int(x.result))
        last[x.away_team] = (int(x.week), x.home_team, -int(x.result))

    games = []
    for x in wk.itertuples():
        neutral = x.location == "Neutral"
        model = (0 if neutral else HFA_PTS) + PTS_PER_EPA * (ratings[x.home_team] - ratings[x.away_team])
        p_home = ml_prob(x.home_moneyline, x.away_moneyline)
        if p_home is None and not pd.isna(x.spread_line):  # fall back to spread, normal approx, sd 13.5
            p_home = 0.5 * (1 + math.erf(x.spread_line / (13.5 * math.sqrt(2))))
        games.append({
            "id": x.game_id, "day": x.weekday, "date": x.gameday, "time": x.gametime,
            "home": x.home_team, "away": x.away_team, "neutral": neutral, "div": int(x.div_game) == 1,
            "spread": None if pd.isna(x.spread_line) else float(x.spread_line),
            "total": None if pd.isna(x.total_line) else float(x.total_line),
            "pHome": None if p_home is None else round(p_home, 3),
            "model": round(model, 1),
            "homeLast": last.get(x.home_team), "awayLast": last.get(x.away_team),
            "homeScore": None if pd.isna(x.home_score) else int(x.home_score),
            "awayScore": None if pd.isna(x.away_score) else int(x.away_score),
        })
    # every finished game this season, for grading saved picks: [id, week, home, away, home pts, away pts, closing spread]
    results = [[x.game_id, int(x.week), x.home_team, x.away_team, int(x.home_score), int(x.away_score),
                None if pd.isna(x.spread_line) else float(x.spread_line)]
               for x in sg[sg.result.notna()].itertuples()]
    return {"season": season, "week": week, "games": games, "results": results, **angle_history(g, season)}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=default_season())
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    refresh = not args.no_download

    DATA.mkdir(exist_ok=True)
    LOGOS.mkdir(parents=True, exist_ok=True)

    pbp_path = DATA / f"pbp_{args.season}.parquet"
    ftn_path = DATA / f"ftn_{args.season}.parquet"
    teams_path = DATA / "teams.csv"
    print(f"Season {args.season}: {'downloading' if refresh else 'using cached'} play-by-play…")
    if not fetch(PBP_URL.format(season=args.season), pbp_path, refresh):
        raise SystemExit("No play-by-play available for that season.")
    has_ftn = fetch(FTN_URL.format(season=args.season), ftn_path, refresh)
    fetch(TEAMS_URL, teams_path, refresh)

    teams = pd.read_csv(teams_path)
    teams = teams[teams.team_abbr.isin(CURRENT)].sort_values("team_abbr")
    for abbr in teams.team_abbr:
        p = LOGOS / f"{abbr}.png"
        if not p.exists():
            download(LOGO_URL.format(abbr=abbr.lower()), p)
    team_idx = {a: i for i, a in enumerate(teams.team_abbr)}

    cols = sorted(set(FOURTH_COLS) | {
        "play_id", "pass", "rush", "cpoe", "qb_dropback", "id", "name",
        "xpass", "shotgun", "no_huddle", "yards_gained",
    })
    raw = pd.read_parquet(pbp_path, columns=cols)
    raw = raw[raw.posteam.isin(CURRENT) & raw.defteam.isin(CURRENT)]

    # ---- pass/run plays, with FTN charting flags joined on
    pbp = raw[((raw["pass"] == 1) | (raw["rush"] == 1))].dropna(subset=["epa"]).copy()
    for c in ["is_play_action", "is_motion", "is_rpo", "is_screen_pass", "n_blitzers", "is_no_huddle",
              "n_defense_box", "n_pass_rushers"]:
        pbp[c] = np.nan
    if has_ftn:
        ftn = pd.read_parquet(ftn_path).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn = ftn.drop_duplicates(["game_id", "play_id"])
        ftn_cols = ["is_play_action", "is_motion", "is_rpo", "is_screen_pass", "n_blitzers", "is_no_huddle",
                    "n_defense_box", "n_pass_rushers"]
        pbp = pbp.drop(columns=ftn_cols)
        pbp = pbp.merge(ftn[["game_id", "play_id", *ftn_cols]], on=["game_id", "play_id"], how="left")

    S_COLS = ["s_cov", "s_mz", "s_dp", "s_op", "s_run", "s_pt"]
    tags = coverage_tags(args.season)
    tags_week = None
    if tags is not None:
        pbp = pbp.merge(tags, on=["game_id", "play_id"], how="left")
        tagged = pbp.s_cov.notna()
        tags_week = int(pbp.loc[tagged, "week"].max()) if tagged.any() else None
    for c in S_COLS:
        pbp[c] = pbp[c].fillna(-1).astype(int) if c in pbp else -1

    db = pbp[(pbp.qb_dropback == 1) & pbp["id"].notna()]
    qb_team = db.groupby("id").posteam.agg(lambda s: s.value_counts().index[0])
    qb_name = db.groupby("id")["name"].agg(lambda s: s.value_counts().index[0])
    qb_ids = list(qb_name.index)
    qb_idx = {q: i for i, q in enumerate(qb_ids)}

    def flags(row) -> int:
        f = 0
        if row["shotgun"] == 1: f |= F_SHOTGUN
        if row["no_huddle"] == 1: f |= F_NOHUDDLE
        if not pd.isna(row["is_motion"]):
            f |= F_CHARTED
            if row["is_play_action"]: f |= F_PA
            if row["is_motion"]: f |= F_MOTION
            if row["is_rpo"]: f |= F_RPO
            if row["is_screen_pass"]: f |= F_SCREEN
            if (row["n_blitzers"] or 0) > 0: f |= F_BLITZ
        return f

    plays = []
    for row in pbp.to_dict("records"):
        is_db = row["qb_dropback"] == 1 and isinstance(row["id"], str)
        plays.append([
            int(row["week"]), team_idx[row["posteam"]], team_idx[row["defteam"]],
            0 if pd.isna(row["down"]) else int(row["down"]),
            r(row["wp"], 3), 1 if row["pass"] == 1 else 0, r(row["epa"], 3), r(row["cpoe"], 1),
            qb_idx[row["id"]] if is_db else -1, 1 if row["season_type"] == "POST" else 0,
            r(row["xpass"], 3), flags(row),
            # matchup-tab extras: half (3 = OT), seconds left in half, yards, box count, pass rushers (-1 = not charted)
            min(3, (int(row["qtr"]) + 1) // 2), 0 if pd.isna(row["half_seconds_remaining"]) else int(row["half_seconds_remaining"]),
            0 if pd.isna(row["yards_gained"]) else int(row["yards_gained"]),
            -1 if pd.isna(row["n_defense_box"]) else int(row["n_defense_box"]),
            -1 if pd.isna(row["n_pass_rushers"]) else int(row["n_pass_rushers"]),
            *(row[c] for c in S_COLS),
        ])

    # ---- 4th downs vs. model
    coef = fourth_model(args.season, refresh, args.retrain)
    fd = fourth_downs(raw)
    fd["xgo"] = 1 / (1 + np.exp(-features(fd) @ coef))
    ptype = {"pass": 0, "run": 1, "punt": 2, "field_goal": 3}
    fourth = [[
        int(x.week), team_idx[x.posteam], team_idx[x.defteam], int(x.ydstogo), int(x.yardline_100),
        r(x.wp, 3), int(x.go), r(x.xgo, 3),
        1 if x.fourth_down_converted == 1 else 0, r(x.epa, 3), 1 if x.season_type == "POST" else 0,
        int(x.qtr), int(x.score_differential), ptype[x.play_type],
        int(x.game_seconds_remaining),
    ] for x in fd.itertuples()]

    out = {
        "meta": {
            "season": args.season,
            "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "maxWeek": int(pbp.week.max()),
            "nPlays": len(plays),
            "charting": bool(has_ftn),
            "fields": ["week", "off", "def", "down", "wp", "pass", "epa", "cpoe", "qb", "post", "xpass", "flags",
                       "half", "halfSecs", "yards", "box", "rushers",
                       "sCov", "sManZone", "sDefPers", "sOffPers", "sRunConcept", "sPressure"],
            "tags": None if tags_week is None else {
                "throughWeek": tags_week, "coverages": TAG_COV, "manZone": TAG_MZ,
                "defPersonnel": TAG_DP, "runConcepts": TAG_RUN},
            "fourthFields": ["week", "team", "opp", "ydstogo", "yardline", "wp", "go", "xgo", "conv", "epa",
                             "post", "qtr", "scoreDiff", "playType", "secsLeft"],
            "fourthModel": json.loads((DATA / "fourth_model.json").read_text())["years"],
        },
        "teams": [{
            "abbr": t.team_abbr, "name": t.team_name, "conf": t.team_conf, "div": t.team_division,
            "color": t.team_color, "color2": t.team_color2,
            "logo": f"logos/{t.team_abbr}.png" if (LOGOS / f"{t.team_abbr}.png").exists() else None,
        } for t in teams.itertuples()],
        "qbs": [{"id": q, "name": qb_name[q], "team": qb_team[q]} for q in qb_ids],
        "plays": plays,
        "fourth": fourth,
        "pickem": pickem(args.season, pbp, refresh),
    }

    js = "window.EPA_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n"
    (SITE / "data.js").write_text(js)
    print(f"Wrote site/data.js: {len(plays):,} plays, {len(fourth):,} 4th downs, weeks 1–{out['meta']['maxWeek']}, "
          f"{len(qb_ids)} QBs, charting={'yes' if has_ftn else 'no'}, {len(js)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
