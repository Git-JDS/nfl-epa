"""
Team-game feature tables for backtesting. Every "pre" feature uses only a team's earlier games
that season (no lookahead), shrunk toward the league average so early-season samples don't swing.

League-wide sources:
  * nflverse play-by-play (2019+)            -> EPA, success, pass/run, PROE, explosives, turnovers...
  * nflverse FTN charting (free, 2022+)       -> motion, play-action, screens, blitz, box, rushers,
                                                 turnover-worthy throws, drops, out-of-pocket...
  * Optional local-only extras in data/private/ (not in the repo):
      PFF weekly player grades -> team-week unit grades (snap-weighted)
      Sumer play tags          -> man / two-high / nickel / stunts / personnel
      FTN paid CSV             -> defensive backs on field, stunts
"""
from __future__ import annotations

import json
import pathlib
import sqlite3

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRIVATE = DATA / "private"
SHRINK = 120          # plays (or dropbacks/runs) of league-average data blended into each pre-game rate

PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
            "pass", "rush", "qb_dropback", "epa", "success", "down", "xpass", "yards_gained", "sack",
            "interception", "fumble_lost", "third_down_converted", "third_down_failed", "cpoe", "wp",
            "complete_pass", "pass_attempt"]
FTN_COLS = ["is_motion", "is_play_action", "is_screen_pass", "is_rpo", "is_no_huddle", "is_qb_out_of_pocket",
            "is_interception_worthy", "is_throw_away", "is_catchable_ball", "is_drop", "n_blitzers",
            "n_pass_rushers", "n_defense_box", "is_created_reception"]


def load_plays(season: int) -> pd.DataFrame:
    import build_data as B  # download any missing nflverse files (first run on a new machine)
    B.fetch(B.PBP_URL.format(season=season), DATA / f"pbp_{season}.parquet", False)
    if season >= 2022:
        B.fetch(B.FTN_URL.format(season=season), DATA / f"ftn_{season}.parquet", False)
    p = pd.read_parquet(DATA / f"pbp_{season}.parquet", columns=PBP_COLS)
    p = p[(p.season_type == "REG") & ((p["pass"] == 1) | (p["rush"] == 1))].dropna(subset=["epa", "posteam"]).copy()
    f = DATA / f"ftn_{season}.parquet"
    if f.exists():
        ftn = pd.read_parquet(f).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        ftn = ftn.drop_duplicates(["game_id", "play_id"])[["game_id", "play_id", *FTN_COLS]]
        p = p.merge(ftn, on=["game_id", "play_id"], how="left")
    else:
        for c in FTN_COLS:
            p[c] = np.nan
    p["charted"] = p.is_motion.notna()
    return p


# ------------------------------------------------------------------ per-play indicator columns
# name -> (unit, denominator mask, value). Unit "off" aggregates by posteam, "def" by defteam.
def _defs(p: pd.DataFrame) -> dict:
    ps, ru, db, ch = p["pass"] == 1, p["rush"] == 1, p.qb_dropback == 1, p.charted
    att = p.pass_attempt == 1
    expl = np.where(ps, p.yards_gained >= 20, p.yards_gained >= 10).astype(float)
    early = p.down.isin([1, 2])
    third = p.third_down_converted.eq(1) | p.third_down_failed.eq(1)
    to = (p.interception.fillna(0) + p.fumble_lost.fillna(0)).clip(upper=1)
    d = {}
    for u in ("off", "def"):
        d.update({
            f"{u}_epa": (u, True, p.epa), f"{u}_sr": (u, True, p.success),
            f"{u}_pass_epa": (u, ps, p.epa), f"{u}_run_epa": (u, ru, p.epa),
            f"{u}_early_pass_epa": (u, ps & early, p.epa),
            f"{u}_expl": (u, True, expl), f"{u}_sack_rate": (u, db, p.sack),
            f"{u}_to_rate": (u, True, to), f"{u}_third_conv": (u, third, p.third_down_converted),
            # FTN (2022+)
            f"{u}_motion": (u, ch, p.is_motion), f"{u}_pa": (u, ch & db, p.is_play_action),
            f"{u}_screen": (u, ch & db, p.is_screen_pass), f"{u}_blitz": (u, ch & db, (p.n_blitzers > 0)),
            f"{u}_box": (u, ch & p.n_defense_box.notna(), p.n_defense_box),
            f"{u}_rushers": (u, ch & db & p.n_pass_rushers.notna(), p.n_pass_rushers),
            f"{u}_int_worthy": (u, ch & att, p.is_interception_worthy),
            f"{u}_oop": (u, ch & db, p.is_qb_out_of_pocket),
            f"{u}_drop": (u, ch & att, p.is_drop), f"{u}_catchable": (u, ch & att, p.is_catchable_ball),
            f"{u}_pa_epa": (u, ch & db & p.is_play_action.eq(True), p.epa),
        })
    d["off_proe"] = ("off", p.xpass.notna(), p["pass"] - p.xpass)
    d["off_cpoe"] = ("off", p.cpoe.notna(), p.cpoe / 100)
    d["off_nohuddle"] = ("off", ch, p.is_no_huddle)
    d["off_rpo"] = ("off", ch, p.is_rpo)
    return d


def team_games(p: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: sum and count for every indicator, for that game."""
    rows = []
    for name, (unit, mask, val) in _defs(p).items():
        team = p.posteam if unit == "off" else p.defteam
        m = pd.Series(mask, index=p.index) if not isinstance(mask, pd.Series) else mask
        v = pd.Series(val, index=p.index).astype(float)
        keep = m.fillna(False).astype(bool) & v.notna()
        g = pd.DataFrame({"game_id": p.game_id[keep], "team": team[keep], "v": v[keep]}).groupby(["game_id", "team"]).v
        rows.append(pd.DataFrame({f"{name}__s": g.sum(), f"{name}__n": g.count()}))
    tg = pd.concat(rows, axis=1).reset_index()
    meta = p.groupby("game_id")[["season", "week", "home_team", "away_team"]].first()
    return tg.join(meta, on="game_id")


def pregame(tg: pd.DataFrame, k: int = SHRINK) -> pd.DataFrame:
    """Season-to-date rate entering each game, shrunk toward the league mean of games already played."""
    tg = tg.sort_values(["season", "team", "week"]).copy()
    names = sorted({c[:-3] for c in tg.columns if c.endswith("__s")})
    base = tg[["game_id", "season", "week", "team", "home_team", "away_team"]].copy()
    out = {}
    for n in names:
        s, c = tg[f"{n}__s"].fillna(0), tg[f"{n}__n"].fillna(0)
        cs = s.groupby([tg.season, tg.team]).cumsum() - s
        cn = c.groupby([tg.season, tg.team]).cumsum() - c
        # league mean using only weeks before this one
        wk = tg.groupby(["season", "week"]).agg(S=(f"{n}__s", "sum"), N=(f"{n}__n", "sum")).groupby(level=0).cumsum()
        wk = wk.groupby(level=0).shift(1)
        lg = (wk.S / wk.N).reindex(pd.MultiIndex.from_arrays([tg.season, tg.week])).to_numpy()
        pre = np.where(cn > 0, (cs + np.nan_to_num(lg) * k) / (cn + k), np.nan)
        pre[np.isnan(lg) | (cn.to_numpy() == 0)] = np.nan
        out[f"pre_{n}"] = pre
        out[f"game_{n}"] = (s / c.replace(0, np.nan)).to_numpy()
        out[f"pre_{n}_n"] = cn.to_numpy()
    out = pd.concat([base.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    out["prior_games"] = out.groupby(["season", "team"]).cumcount()
    return out


def games_frame() -> pd.DataFrame:
    g = pd.read_csv(DATA / "games.csv")
    return g[g.game_type == "REG"][["game_id", "season", "week", "home_team", "away_team", "result", "spread_line",
                                    "location", "div_game"]]


# ------------------------------------------------------------------ optional local extras (2026)
PFF_TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def pff_team_week(season: int = 2026) -> pd.DataFrame | None:
    """Snap-weighted team unit grades per week from PFF weekly player reports."""
    f = next(PRIVATE.glob(f"**/pff_{season}.sqlite"), None)
    if f is None:
        return None
    c = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    recs = []
    for scope, rep, payload in c.execute("select scope, report, payload from reports where scope like 'week_%'"):
        d = json.loads(payload)
        recs.append({**d, "week": int(scope[5:]), "report": rep, "team": PFF_TEAM.get(d.get("team"), d.get("team"))})
    df = pd.DataFrame(recs)

    def wavg(sub, grade, w):
        sub = sub.dropna(subset=[grade, w])
        sub = sub[sub[w] > 0]
        return (sub[grade] * sub[w]).sum() / sub[w].sum() if len(sub) else np.nan

    out = []
    for (wk, tm), g in df.groupby(["week", "team"]):
        de, bl, pa, re, ru = (g[g.report == r] for r in ["defense", "blocking", "passing", "receiving", "rushing"])
        qb = pa.sort_values("dropbacks", ascending=False).head(1) if len(pa) else pa
        out.append({
            "week": wk, "team": tm,
            "pff_pass_block": wavg(bl, "gradesPassBlock", "snapCountsPassBlock"),
            "pff_run_block": wavg(bl, "gradesRunBlock", "snapCountsRunBlock"),
            "pff_pressures_allowed_rate": bl.pressuresAllowed.sum() / max(1, bl.snapCountsPassBlock.sum()) * 5,
            "pff_qb_pass": qb.gradesPass.iloc[0] if len(qb) else np.nan,
            "pff_qb_twp_rate": qb.twpRate.iloc[0] if len(qb) and "twpRate" in qb else np.nan,
            "pff_route": wavg(re, "gradesPassRoute", "routes"),
            "pff_rush": wavg(ru, "gradesRun", "attempts"),
            "pff_pass_rush": wavg(de, "gradesPassRushDefense", "snapCountsPassRush"),
            "pff_run_def": wavg(de, "gradesRunDefense", "snapCountsRunDefense"),
            "pff_tackle": wavg(de, "gradesTackle", "snapCountsDefense"),
            "pff_def": wavg(de, "gradesDefense", "snapCountsDefense"),
            "pff_pressure_rate": de.totalPressures.sum() / max(1, de.snapCountsPassRush.sum()) * 4,
        })
    return pd.DataFrame(out)


def sumer_plays(season: int = 2026) -> pd.DataFrame | None:
    f = next(PRIVATE.glob(f"**/coverage_tags_{season}.json"), None)
    if f is None:
        return None
    s = pd.DataFrame(json.loads(f.read_text()))
    s["play_id"] = pd.to_numeric(s.matched_play_id, errors="coerce")
    s = s.dropna(subset=["play_id"]).drop_duplicates(["game_id", "play_id"])
    return s.assign(play_id=s.play_id.astype(int))[["game_id", "play_id", "coverage", "man_zone",
                                                     "defensive_personnel", "personnel", "pressure_tag", "run_concept"]]


def ftn_paid_plays(season: int = 2026) -> pd.DataFrame | None:
    f = next(PRIVATE.glob(f"**/historical/{season}/plays.csv"), None)
    if f is None:
        return None
    return pd.read_csv(f, usecols=["gid", "pid", "off", "def", "type", "db_count", "stunt"], low_memory=False)


def private_team_games(p: pd.DataFrame) -> pd.DataFrame | None:
    """2026-only Sumer rates per team-game (offense and defense side), joined on nflverse play ids."""
    s = sumer_plays(int(p.season.iloc[0]))
    if s is None:
        return None
    q = p.merge(s, on=["game_id", "play_id"], how="left")
    db = (q.qb_dropback == 1) & q.coverage.fillna("").ne("")
    feats = {
        "def_man": ("def", (q.qb_dropback == 1) & q.man_zone.isin(["MAN", "ZONE"]), q.man_zone.eq("MAN")),
        "def_two_high": ("def", db, q.coverage.isin(["COVER 2", "COVER 4", "COVER 6"])),
        "def_cover1": ("def", db, q.coverage.eq("COVER 1")),
        "def_nickel_plus": ("def", q.defensive_personnel.fillna("").ne(""), q.defensive_personnel.isin(["NICKEL", "DIME"])),
        "def_stunt": ("def", db, q.pressure_tag.fillna("").str.contains("STUNT")),
        "def_sumer_blitz": ("def", db, q.pressure_tag.fillna("").str.contains("BLITZ")),
        "off_11": ("off", q.personnel.fillna("").ne(""), q.personnel.str.rstrip("*").eq("11")),
        "off_epa_vs_man": ("off", (q.qb_dropback == 1) & q.man_zone.eq("MAN"), q.epa),
        "off_epa_vs_zone": ("off", (q.qb_dropback == 1) & q.man_zone.eq("ZONE"), q.epa),
        "off_outside_zone": ("off", (q.rush == 1) & q.run_concept.fillna("").ne(""), q.run_concept.eq("OUTSIDE ZONE")),
    }
    rows = []
    for name, (unit, mask, val) in feats.items():
        team = q.posteam if unit == "off" else q.defteam
        keep = mask.fillna(False)
        g = pd.DataFrame({"game_id": q.game_id[keep], "team": team[keep], "v": val[keep].astype(float)}).groupby(["game_id", "team"]).v
        rows.append(pd.DataFrame({f"{name}__s": g.sum(), f"{name}__n": g.count()}))
    tg = pd.concat(rows, axis=1).reset_index()
    return tg.join(p.groupby("game_id")[["season", "week", "home_team", "away_team"]].first(), on="game_id")
