# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""One-off selection of 3DCoMPaT++ shapes that pass Prelim A, then freeze.

Reads the ranked candidates written by ``make_recipes.py --candidates`` and, per
class, builds and walks candidates in order until enough pass. The output is the
frozen recipe (the dataset definition every later experiment uses) plus a
selection log listing every candidate tried and why it was rejected. Selection is
not re-run for experiments.

Each walk is a real 200-step surface-agent episode (``prelim_a_walk.py``) under
ambient light, from two start rotations ([0, 0, 0] and [90, 0, 0]).

3DC-Texture candidate (base composition + 4 slot variants) passes when:
    * on-object: mean fraction of steps with the patch centre on the object
      > ``--min-on-object`` (0.9);
    * slot visits: fraction of steps where the base and the most different
      variant's patches differ (mean |diff| > 4 of 255) >= ``--min-slot-visit``
      (0.4). The surface policy uses geometry only, so variants of one shape
      walk identical trajectories and differing patches mark steps on the
      varied slot (checked: identical on-object sequences are required);
    * brightness: for every variant, the median HSV value of its patch on
      slot steps >= ``--min-brightness`` (0.15).
3DC-Mixed candidate (one shape, composition 0) passes on on-object alone.

Usage::

    python scripts/compat3d_test/select_dataset_shapes.py \
        --prep ~/3dcompat-habitat-prep --work /tmp/select --parallel 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).parents[2]
WALK_SCRIPT = Path(__file__).with_name("prelim_a_walk.py")
BASE_EXPERIMENT = "journal/phase2_pre_training_texture_p2_012_amb"
ROTATIONS = ([0, 0, 0], [90, 0, 0])
DIFF_THRESHOLD = 4.0
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def build(prep: Path, entries: list[dict], outdir: Path, target_size: float) -> list[str]:
    """Build ``entries`` with compat3d_helper; return the built object names."""
    outdir.mkdir(parents=True, exist_ok=True)
    recipe = outdir / "recipe_in.yaml"
    recipe_text = yaml.safe_dump({
        "split": "train", "level": "fine", "target_size": target_size,
        "outdir": str(outdir), "entries": entries,
    }, sort_keys=False)
    names_path = outdir / "object_names.yaml"
    if names_path.exists() and recipe.exists() and recipe.read_text() == recipe_text:
        return yaml.safe_load(names_path.read_text())["object_names"]  # built before
    recipe.write_text(recipe_text)
    with open(outdir / "build.log", "w") as f:
        subprocess.run(
            [str(prep / ".venv/bin/python"), "compat3d_helper.py",
             "--zip", "3DCoMPaT_3D/3DCoMPaT_ZIP.zip", "--recipe", str(recipe), "--build"],
            cwd=prep, stdout=f, stderr=subprocess.STDOUT, check=True,
        )
    return yaml.safe_load((outdir / "object_names.yaml").read_text())["object_names"]


def walk(data_path: Path, name: str, rotation: list[int], out: Path) -> dict:
    """One 200-step ambient walk; returns per-step stats and patches."""
    if not Path(f"{out}.json").exists():
        cmd = [
            sys.executable, str(WALK_SCRIPT),
            f"experiment={BASE_EXPERIMENT}",
            f"experiment.config.environment.env_init_args.data_path={data_path}",
            f"experiment.config.train_env_interface_args.object_names=[{name}]",
            "experiment.config.train_env_interface_args.object_init_sampler.rotations="
            f"[[{rotation[0]},{rotation[1]},{rotation[2]}]]",
            "experiment.config.n_train_epochs=1",
            "experiment.config.monty_config.monty_args.num_exploratory_steps=200",
            "experiment.config.max_total_steps=200",
            "experiment.config.max_train_steps=200",
            f"experiment.config.logging.output_dir={out.parent / 'runs'}",
            f"experiment.config.logging.run_name={out.name}",
        ]
        env = {**os.environ, "PRELIM_A_OUT": str(out), "PRELIM_A_SAVE_PATCHES": "1",
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2"}
        with open(f"{out}.log", "w") as f:
            subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                           check=False)
    steps = json.loads(Path(f"{out}.json").read_text())
    patches = np.load(f"{out}_patches.npz")["patches"]
    return {"on": np.array([s["center_on_object"] for s in steps]), "patches": patches}


def patch_v(patches: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hsv

    return np.array([rgb2hsv(p)[..., 2].mean() for p in patches])


def on_object_fraction(walks: list[dict]) -> float:
    return float(np.mean(np.concatenate([w["on"] for w in walks])))


def evaluate_texture(cls: str, candidates: list[dict], args, rows: list) -> list | None:
    for i, cand in enumerate(candidates):
        tag = f"{cls}_{i:02d}_{cand['shape_id']}_c{cand['comp']}"
        row = {"dataset": "texture", "class": cls, "rank": i,
               "shape_id": cand["shape_id"], "comp": cand["comp"],
               "slot": "+".join(cand["slot"]),
               "slot_visible_fraction": cand["slot_visible_fraction"]}
        work = args.work / "texture" / tag
        try:
            # Stage 1: base vs the most different (last) variant.
            names = build(args.prep, [cand["entries"][0], cand["entries"][-1]],
                          work / "stage1", args.target_size)
            base = [walk(work / "stage1", names[0], r, work / f"base_r{j}")
                    for j, r in enumerate(ROTATIONS)]
            other = [walk(work / "stage1", names[1], r, work / f"last_r{j}")
                     for j, r in enumerate(ROTATIONS)]
            same_path = all(np.array_equal(b["on"], o["on"]) for b, o in zip(base, other))
            slot_steps = [
                (np.abs(b["patches"].astype(float) - o["patches"].astype(float))
                 .mean(axis=(1, 2, 3)) > DIFF_THRESHOLD) & b["on"]
                for b, o in zip(base, other)
            ]
            row["on_object"] = round(on_object_fraction(base), 3)
            row["slot_visit"] = round(float(np.mean(np.concatenate(slot_steps))), 3)
            row["same_path"] = same_path
            reasons = []
            if not same_path:
                reasons.append("variant walks diverged")
            if row["on_object"] <= args.min_on_object:
                reasons.append(f"on-object {row['on_object']}")
            if row["slot_visit"] < args.min_slot_visit:
                reasons.append(f"slot visit {row['slot_visit']}")
            if not reasons:
                # Stage 2: brightness of every variant on slot steps.
                names = build(args.prep, cand["entries"], work / "stage2", args.target_size)
                v_min = 1.0
                for name in names:
                    vs = []
                    for j, r in enumerate(ROTATIONS):
                        w = walk(work / "stage2", name, r, work / f"{name}_r{j}")
                        if slot_steps[j].any():
                            vs.append(patch_v(w["patches"][slot_steps[j]]))
                    v_min = min(v_min, float(np.median(np.concatenate(vs))))
                row["min_variant_slot_v"] = round(v_min, 3)
                if v_min < args.min_brightness:
                    reasons.append(f"darkest variant V {row['min_variant_slot_v']}")
        except subprocess.CalledProcessError as e:
            reasons = [f"build failed ({e})"]
        row["accepted"] = not reasons
        row["reject_reason"] = "; ".join(reasons)
        rows.append(row)
        log(f"[texture] {tag}: {'ACCEPT' if not reasons else 'reject: ' + row['reject_reason']}")
        if not reasons:
            return cand["entries"]
    log(f"[texture] {cls}: no candidate passed")
    return None


def evaluate_mixed(cls: str, candidates: list[dict], args, rows: list, need: int) -> list:
    accepted = []
    for i, entry in enumerate(candidates):
        tag = f"{cls}_{i:02d}_{entry['shape_id']}"
        row = {"dataset": "mixed", "class": cls, "rank": i, "shape_id": entry["shape_id"],
               "comp": 0}
        try:
            names = build(args.prep, [entry], args.work / "mixed" / tag, args.target_size)
            walks = [walk(args.work / "mixed" / tag, names[0], r,
                          args.work / "mixed" / tag / f"r{j}")
                     for j, r in enumerate(ROTATIONS)]
            row["on_object"] = round(on_object_fraction(walks), 3)
            reason = "" if row["on_object"] > args.min_on_object else f"on-object {row['on_object']}"
        except subprocess.CalledProcessError as e:
            reason = f"build failed ({e})"
        row["accepted"] = not reason
        row["reject_reason"] = reason
        rows.append(row)
        log(f"[mixed] {tag}: {'ACCEPT' if not reason else 'reject: ' + reason}")
        if not reason:
            accepted.append(entry)
            if len(accepted) >= need:
                break
    return accepted


def write_recipe(path: Path, outdir: str, target_size: float, entries: list, header: list):
    body = yaml.safe_dump({"split": "train", "level": "fine", "target_size": target_size,
                           "outdir": outdir, "entries": entries}, sort_keys=False)
    path.write_text("".join(f"# {h}\n" for h in header) + body)
    log(f"wrote {path} ({len(entries)} objects)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", type=Path, default=Path("~/3dcompat-habitat-prep"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=12)
    parser.add_argument("--target-size", type=float, default=0.12)
    parser.add_argument("--min-on-object", type=float, default=0.9)
    parser.add_argument("--min-slot-visit", type=float, default=0.4)
    parser.add_argument("--min-brightness", type=float, default=0.15)
    parser.add_argument("--mixed-per-class", type=int, default=2)
    parser.add_argument("--only", choices=["texture", "mixed"], default=None)
    parser.add_argument("--classes", nargs="+", default=None,
                        help="3DC-Texture only: re-select just these classes and merge "
                             "them into the existing frozen recipe (other classes kept); "
                             "their rows are appended to the selection log.")
    parser.add_argument("--max-new", type=int, default=None,
                        help="With --classes: merge at most this many passing classes, "
                             "taken in --classes order (priority order).")
    args = parser.parse_args()
    args.prep = args.prep.expanduser()
    args.work = args.work.expanduser().resolve()
    recipes = args.prep / "recipes"
    thresholds = (f"on-object > {args.min_on_object}, slot visit >= {args.min_slot_visit}, "
                  f"variant slot brightness >= {args.min_brightness}; 200-step ambient "
                  f"walks from rotations {list(ROTATIONS)}")
    header = ["FROZEN dataset definition, written once by tbp.monty "
              "scripts/compat3d_test/select_dataset_shapes.py; do not regenerate.",
              f"Selection rule: {thresholds}. Log: selection_log.csv"]
    rows: list[dict] = []

    if args.classes:
        args.only = "texture"
    if args.only in (None, "texture"):
        cands = yaml.safe_load((recipes / "candidates_texture.yaml").read_text())
        run = {c: v for c, v in cands.items() if not args.classes or c in args.classes}
        with ThreadPoolExecutor(args.parallel) as pool:
            chosen = dict(zip(run, pool.map(
                lambda c: evaluate_texture(c, run[c], args, rows), run)))
        if args.classes and args.max_new is not None:
            passing = [c for c in args.classes if chosen.get(c)]
            for c in passing[args.max_new:]:
                chosen[c] = None
                for r in rows:
                    if r["class"] == c and r["accepted"]:
                        r["reject_reason"] = (f"passed, but not needed: first "
                                              f"{args.max_new} passing classes were "
                                              f"{passing[:args.max_new]}")
                        r["accepted"] = False
        if args.classes:
            # Keep the already-frozen classes; the class of an entry is its shape's.
            from_shape = {cand["shape_id"]: c for c, v in cands.items() for cand in v}
            kept = yaml.safe_load((recipes / "compat3d_texture.yaml").read_text())["entries"]
            for e in kept:
                chosen.setdefault(from_shape[e["shape_id"]], []).append(e)
        missing = [c for c in cands if not chosen.get(c)
                   and not (args.classes and c in args.classes)]
        entries = [e for c in cands if chosen.get(c) for e in chosen[c]]
        write_recipe(recipes / "compat3d_texture.yaml", "~/tbp/data/compat3d_texture",
                     args.target_size, entries,
                     header + ["3DC-Texture: dataset composition + 4 same-family swaps "
                               "of the most visible slot"]
                     + ([f"Classes with no passing candidate: {missing}"] if missing else []))

    if args.only in (None, "mixed"):
        cands = yaml.safe_load((recipes / "candidates_mixed.yaml").read_text())
        with ThreadPoolExecutor(args.parallel) as pool:
            chosen = dict(zip(cands, pool.map(
                lambda c: evaluate_mixed(c, cands[c], args, rows, args.mixed_per_class),
                cands)))
        target = args.mixed_per_class * len(cands)
        entries = [e for c in cands for e in chosen[c]]
        short = {c: args.mixed_per_class - len(v) for c, v in chosen.items()
                 if len(v) < args.mixed_per_class}
        if short:
            # Keep the set size: top up from classes with further passing shapes.
            log(f"[mixed] short classes {short}; topping up to {target}")
            for c in cands:
                if len(entries) >= target:
                    break
                extra = evaluate_mixed(
                    c, cands[c][len([r for r in rows if r['class'] == c]):], args, rows, 1)
                entries.extend(extra)
        write_recipe(recipes / "compat3d_mixed.yaml", "~/tbp/data/compat3d_mixed",
                     args.target_size, entries,
                     header + [f"3DC-Mixed: composition 0, {args.mixed_per_class} shapes per "
                               "class where available"]
                     + ([f"Classes short of {args.mixed_per_class}: {short}"] if short else []))

    log_path = recipes / ("selection_log.csv" if args.only is None or args.classes
                          else f"selection_log_{args.only}.csv")
    if args.classes and log_path.exists():
        # Replace the re-selected classes' old rows with the new ones.
        with open(log_path) as f:
            old = [r for r in csv.DictReader(f)
                   if not (r["dataset"] == "texture" and r["class"] in args.classes)]
        rows = old + rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["dataset"], r["class"], int(r["rank"]))))
    log(f"wrote {log_path}")


if __name__ == "__main__":
    main()
