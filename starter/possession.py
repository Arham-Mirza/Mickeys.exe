"""Generate possession labels from tracking and Smart Tagging."""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

HOME, AWAY = 0, 1
LABELS = np.array(["H", "A"])
ATTACKING_PHASES = {
    "BALL IN FINAL THIRD", "BALL IN THE BOX", "BUILD UP", "CREATING CHANCES",
    "LONG BALL", "PLAYERS IN THE BOX", "PROGRESSION", "ATTACKING TRANSITION",
}
DEFENDING_PHASES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD", "DEFENSIVE TRANSITION",
}
STRONG_PHASES = {"BUILD UP", "PROGRESSION", "DEFENDING IN MIDDLE THIRD"}
MIN_TAG_EVIDENCE_FRAMES = 100
TAG_CONFIDENCE_THRESHOLD = 0.20
TAG_CONTRADICTION_THRESHOLD = 0.25
RESTART_TYPES = {
    "CORNER KICK", "FREE KICK", "GOAL KICK", "THROW-IN",
    "KICK OFF (START)", "KICK OFF (AFTER GOAL)",
}
SPARSE_TRACKING_RESET_FRAMES = 50


def track_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    players = list(root.iter("Player"))
    if len(teams) != 2 or not players:
        raise ValueError("{}: expected two teams and player slots".format(meta_path))
    return np.array([HOME if p.get("teamId") == teams[0] else AWAY for p in players],
                    dtype=np.int8)


def track_team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    names = [team.findtext("Name") for team in root.iter("Team")]
    if len(names) != 2 or any(not name for name in names):
        raise ValueError("{}: expected names for two teams".format(meta_path))
    return tuple(names)


def read_tracking(raw_path, team_by_slot, n_frames, pitch):
    length, width = pitch
    distances = np.full((n_frames, 2), np.inf, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    player_count = np.zeros(n_frames, dtype=np.uint8)

    with open(raw_path) as source:
        for expected_frame, line in enumerate(source):
            fields = line.rstrip().split(":")
            if len(fields) != 3:
                raise ValueError("Malformed ATD record at line {}".format(expected_frame + 1))
            frame = int(fields[0]) - 1
            if frame != expected_frame or frame >= n_frames:
                raise ValueError("ATD frame sequence is not 1..{}".format(n_frames))

            nearest_squared = [float("inf"), float("inf")]
            values = fields[2].split(",", 2)
            has_ball = values[0] != "NaN"
            if has_ball:
                ball[frame] = (float(values[0]), float(values[1]))
                bx, by = ball[frame, 0] * length, ball[frame, 1] * width

            for slot, item in enumerate(fields[1].split(";")):
                if item[0] == "N":
                    continue
                x, y = item.split(",", 2)[:2]
                player_count[frame] += 1
                if has_ball:
                    px, py = float(x) * length, float(y) * width
                    dx, dy = px - bx, py - by
                    team = team_by_slot[slot]
                    nearest_squared[team] = min(
                        nearest_squared[team], dx * dx + dy * dy)

            if has_ball:
                distances[frame] = np.sqrt(nearest_squared)
    if expected_frame + 1 != n_frames:
        raise ValueError("ATD has {} records; expected {}".format(expected_frame + 1, n_frames))
    return distances, ball, player_count


def tag_evidence(events, n_frames, team_names):
    tagged = np.zeros(n_frames, dtype=np.float32)
    home_name, away_name = team_names
    for event in events:
        code = (event["code"] or "").strip().upper()
        team = (event["tags"].get("Team") or "").strip()
        if team not in (home_name, away_name):
            continue
        side = HOME if team == home_name else AWAY
        if code in ATTACKING_PHASES:
            owner = side
        elif code in DEFENDING_PHASES:
            owner = 1 - side
        else:
            continue
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if end > start:
            weight = 0.25 if code in STRONG_PHASES else 0.10
            tagged[start:end] += weight if owner == HOME else -weight
    np.clip(tagged, -0.5, 0.5, out=tagged)
    return tagged


def restart_evidence(dead, hint, ball, events, n_frames, pitch, fps):
    length, width = pitch
    for event in events:
        if (event["code"] or "").strip().upper() != "SET PIECES":
            continue
        kind = (event["tags"].get("Type") or "").strip().upper()
        if kind not in RESTART_TYPES:
            continue
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if end <= start:
            continue

        first = max(start + 7, 7)
        found = None
        for frame in range(first, end):
            previous = frame - 7
            if not np.isfinite(ball[frame]).all() or not np.isfinite(ball[previous]).all():
                continue
            dx = (ball[frame, 0] - ball[previous, 0]) * length
            dy = (ball[frame, 1] - ball[previous, 1]) * width
            if np.hypot(dx, dy) > 1.5:
                found = frame
                break
        if found is not None:
            dead[start:found] = True
            team = (event["tags"].get("Team") or "").strip()
            if team in ("Barcelona", "Atletic Club"):
                owner = HOME if team == "Barcelona" else AWAY
                hint[found:min(end, found + 3 * fps)] += \
                    0.2 if owner == HOME else -0.2


def centered_mean(values, radius):
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    low = np.maximum(0, index - radius)
    high = np.minimum(len(values), index + radius + 1)
    return (cumulative[high] - cumulative[low]) / (high - low)


def proximity_score(distances):
    home = np.exp(-distances[:, HOME] / 2.5)
    away = np.exp(-distances[:, AWAY] / 2.5)
    score = home - away
    score[~np.isfinite(score)] = 0.0
    return score


def possession_observations(distances, max_control_distance=3.5, min_distance_margin=0.5):
    nearest = np.argmin(distances, axis=1).astype(np.int8)
    rows = np.arange(len(distances))
    own = distances[rows, nearest]
    other = distances[rows, 1 - nearest]
    finite = np.isfinite(own) & np.isfinite(other)
    margin = np.zeros_like(own)
    np.subtract(other, own, out=margin, where=finite)
    confident = finite & (own <= max_control_distance) & (margin >= min_distance_margin)
    observations = np.full(len(distances), -1, dtype=np.int8)
    observations[confident] = nearest[confident]
    return observations


def track_possession(observations, phase_hint, dead, player_count,
                     direct_confirm=15, phase_confirm=50):
    labels = np.full(len(observations), "X", dtype="<U1")
    owner = -1
    pending = -1
    pending_count = 0
    pending_gap = 0
    sparse_player_frames = 0

    for frame, observation in enumerate(observations):
        if player_count[frame] < 8:
            sparse_player_frames += 1
        else:
            sparse_player_frames = 0
        if sparse_player_frames >= SPARSE_TRACKING_RESET_FRAMES:
            owner = -1
            pending = -1
            pending_count = 0
            pending_gap = 0

        if dead[frame]:
            labels[frame] = "D"
            pending = -1
            pending_count = 0
            pending_gap = 0
            continue
        if player_count[frame] < 8:
            pending_gap += 1
            if pending_gap > 8:
                pending = -1
                pending_count = 0
            continue

        candidate = int(observation)
        direct = candidate >= 0
        required = direct_confirm
        if candidate < 0 and abs(phase_hint[frame]) >= 0.16:
            candidate = HOME if phase_hint[frame] > 0 else AWAY
            required = phase_confirm

        if owner < 0 and direct:
            owner = candidate
            pending = -1
            pending_count = 0
            pending_gap = 0
        elif candidate < 0:
            pending_gap += 1
            if pending_gap > 8:
                pending = -1
                pending_count = 0
        elif candidate == owner:
            pending = -1
            pending_count = 0
            pending_gap = 0
        elif candidate == pending:
            pending_count += 1
            pending_gap = 0
            if pending_count >= required:
                owner = candidate
                pending = -1
                pending_count = 0
                pending_gap = 0
        else:
            pending = candidate
            pending_count = 1
            pending_gap = 0

        if owner == HOME:
            labels[frame] = "H"
        elif owner == AWAY:
            labels[frame] = "A"
    return labels


def predict(distances, ball, player_count, events, n_frames, fps, team_names):
    tracking_score = proximity_score(distances)
    phases = centered_mean(tag_evidence(events, n_frames, team_names), 25)
    restart_hint = np.zeros(n_frames, dtype=np.float32)
    dead = np.zeros(n_frames, dtype=bool)
    restart_evidence(
        dead, restart_hint, ball, events, n_frames, GAMES[GAME]["pitch"], fps)
    restart_hint = centered_mean(restart_hint, 5)
    phases += restart_hint
    phases = np.clip(phases, -0.5, 0.5)
    observations = possession_observations(distances)

    outside = (ball[:, 0] < -0.01) | (ball[:, 0] > 1.01) | \
              (ball[:, 1] < -0.01) | (ball[:, 1] > 1.01)
    dead[outside] = True
    labels = track_possession(observations, phases, dead, player_count)
    return labels, tracking_score


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="") as output:
        output.write("frame,label\n")
        output.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))


def denoise_tagging(source_path, output_path, tracking_score, team_names, n_frames, fps):
    tree = ET.parse(source_path)
    tracking = centered_mean(tracking_score, 25)
    team_to_side = {team_names[0]: HOME, team_names[1]: AWAY}
    removed = 0
    evaluated = 0

    for event in tree.getroot().iter("instance"):
        code = (event.findtext("code") or "").strip().upper()
        if code in ATTACKING_PHASES:
            defending = False
        elif code in DEFENDING_PHASES:
            defending = True
        else:
            continue

        team_label = next((label for label in event.findall("label")
                           if label.findtext("group") == "Team"), None)
        if team_label is None:
            continue
        team = (team_label.findtext("text") or "").strip()
        if team not in team_to_side:
            continue

        start = max(0, int(round(float(event.findtext("start")) * fps)))
        end = min(n_frames, int(round(float(event.findtext("end")) * fps)))
        if end <= start:
            continue
        expected = team_to_side[team]
        if defending:
            expected = 1 - expected

        evidence = tracking[start:end]
        evidence = evidence[np.abs(evidence) >= TAG_CONFIDENCE_THRESHOLD]
        if len(evidence) < MIN_TAG_EVIDENCE_FRAMES:
            continue
        evaluated += 1
        agreement = np.mean(evidence * (1 if expected == HOME else -1) > 0)
        if agreement < TAG_CONTRADICTION_THRESHOLD:
            event.remove(team_label)
            removed += 1

    folder = os.path.dirname(output_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return removed, evaluated


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(
        ROOT, "evaluation", "results", "Mickeys.exe.csv"))
    parser.add_argument("--tagging-out", default=os.path.join(
        ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging_denoised.xml"))
    args = parser.parse_args()

    meta_path, raw_path, tagging_path = require(["atd_meta", "atd_raw", "smart_tagging"])
    n_frames = source_frames("tactical", GAME)
    print("Reading tracking...")
    distances, ball, players = read_tracking(
        raw_path, track_teams(meta_path), n_frames, GAMES[GAME]["pitch"])
    events = load_tags()
    team_names = track_team_names(meta_path)
    labels, tracking_score = predict(
        distances, ball, players, events, n_frames, GAMES[GAME]["fps"], team_names)
    write_submission(args.out, labels)
    removed, evaluated = denoise_tagging(
        tagging_path, args.tagging_out, tracking_score, team_names,
        n_frames, GAMES[GAME]["fps"])
    print("Wrote {} labels to {}".format(len(labels), args.out))
    print("Wrote denoised tagging to {}".format(args.tagging_out))
    print("Removed {} strongly contradicted Team tags ({} events had enough tracking evidence)."
          .format(removed, evaluated))
    print("H={} A={} D={} X={}".format(
        *(int(np.sum(labels == label)) for label in ("H", "A", "D", "X"))))


if __name__ == "__main__":
    main()
