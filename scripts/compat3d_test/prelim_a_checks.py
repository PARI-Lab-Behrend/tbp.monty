# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Prelim A asset-sanity checks for a built 3DCoMPaT++ dataset.

Static checks (from the build output):
    * every recipe entry built to a distinct object (no content duplicates),
    * no part uses the ``placeholder`` material,
    * every ``.glb`` loads and its max extent is ``target_size`` +- 2%.

Dynamic checks (a real 200-step surface-agent walk per object, rotation
[0, 0, 0], under each requested lighting scenario; ``prelim_a_walk.py``):
    * fraction of steps with the patch centre on the object (pass > 0.9),
    * median patch brightness (HSV value of on-object pixels, 0-1),
    * median patch texture contrast (grayscale std of on-object pixels, 0-255),
    * fraction of on-object steps that are nearly black (V < 0.05).

Writes ``prelim_a_report.csv`` and a contact sheet (``prelim_a_contact_sheet.png``:
one row per object, patch thumbnails at steps 25..200, ambient then
directional) to ``--out``.

Usage::

    python scripts/compat3d_test/prelim_a_checks.py \
        --data-path ~/tbp/data/compat3d_texture --out /tmp/prelim_a_texture
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

BASE_EXPERIMENT = "journal/phase2_pre_training_texture_p2_012_amb"
WALK_SCRIPT = Path(__file__).with_name("prelim_a_walk.py")
REPO = Path(__file__).parents[2]
LIGHTING_OVERRIDES = {
    "amb": [],  # the base experiment's environment is ambient
    "dir": ["experiment.config.environment.env_init_args.lights=null"],
}
MIN_ON_OBJECT = 0.9
BLACK_V = 0.05


def static_checks(data_path: Path) -> tuple[dict, list[str]]:
    import trimesh

    names = yaml.safe_load((data_path / "object_names.yaml").read_text())["object_names"]
    manifest = yaml.safe_load((data_path / "object_manifest.yaml").read_text())
    recipe = yaml.safe_load((data_path / "recipe.yaml").read_text())
    target = float(recipe["target_size"])
    problems = []
    if len(names) != len(recipe["entries"]):
        problems.append(
            f"{len(recipe['entries'])} recipe entries built to {len(names)} objects "
            "(content duplicates)"
        )
    per_object = {}
    for name in names:
        row = {"placeholder": "placeholder" in manifest[name]["materials"].values()}
        glb = data_path / "objects" / name / f"{name}.glb"
        try:
            extent = float(max(trimesh.load(glb, force="scene").extents))
            row["max_extent"] = round(extent, 4)
            row["size_ok"] = abs(extent - target) <= 0.02 * target
        except Exception as e:  # noqa: BLE001 -- report any load failure
            row["max_extent"] = None
            row["size_ok"] = False
            problems.append(f"{name}: failed to load ({e})")
        per_object[name] = row
    return per_object, problems


def run_walk(data_path: Path, name: str, lighting: str, out_dir: Path) -> Path:
    out = out_dir / "walks" / f"{name}_{lighting}"
    if Path(f"{out}.json").exists():
        return out
    cmd = [
        sys.executable, str(WALK_SCRIPT),
        f"experiment={BASE_EXPERIMENT}",
        f"experiment.config.environment.env_init_args.data_path={data_path}",
        f"experiment.config.train_env_interface_args.object_names=[{name}]",
        "experiment.config.train_env_interface_args.object_init_sampler.rotations=[[0,0,0]]",
        "experiment.config.n_train_epochs=1",
        "experiment.config.monty_config.monty_args.num_exploratory_steps=200",
        "experiment.config.max_total_steps=200",
        "experiment.config.max_train_steps=200",
        f"experiment.config.logging.output_dir={out_dir / 'runs' / lighting}",
        f"experiment.config.logging.run_name={name}",
        *LIGHTING_OVERRIDES[lighting],
    ]
    env = {**os.environ, "PRELIM_A_OUT": str(out), "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    with open(f"{out}.log", "w") as log:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=False)
    return out


def summarize_walk(out: Path) -> dict:
    path = Path(f"{out}.json")
    if not path.exists():
        return {"walk_ok": False}
    steps = json.loads(path.read_text())
    on = [s for s in steps if "v_mean" in s]
    v = np.array([s["v_mean"] for s in on]) if on else np.array([np.nan])
    std = np.array([s["gray_std"] for s in on]) if on else np.array([np.nan])
    return {
        "walk_ok": len(steps) > 0,
        "steps": len(steps),
        "frac_center_on_object": round(float(np.mean([s["center_on_object"] for s in steps])), 3)
        if steps else 0.0,
        "median_v": round(float(np.median(v)), 3),
        "median_gray_std": round(float(np.median(std)), 2),
        "frac_black": round(float(np.mean(v < BLACK_V)), 3),
    }


def contact_sheet(rows: list[tuple[str, list[Path]]], path: Path) -> None:
    from PIL import Image, ImageDraw

    cell, label_w = 64, 230
    n_cols = 16
    sheet = Image.new("RGB", (label_w + n_cols * cell + 8, len(rows) * (cell + 2)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (name, outs) in enumerate(rows):
        y = r * (cell + 2)
        draw.text((3, y + cell // 2 - 5), name, fill="black")
        x = label_w
        for out in outs:
            thumbs_path = Path(f"{out}_thumbs.npz")
            if thumbs_path.exists():
                for t in np.load(thumbs_path)["thumbs"][:8]:
                    sheet.paste(Image.fromarray(t), (x, y))
                    x += cell
            x = label_w + 8 * cell + 8  # second lighting starts at a fixed column
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lighting", nargs="+", default=["amb", "dir"],
                        choices=sorted(LIGHTING_OVERRIDES))
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--no-static", action="store_true",
                        help="Walks only (for datasets built without a recipe.yaml)")
    args = parser.parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    (out_dir / "walks").mkdir(parents=True, exist_ok=True)

    if args.no_static:
        names_yaml = yaml.safe_load((data_path / "object_names.yaml").read_text())
        per_object = {n: {} for n in names_yaml["object_names"]}
        problems = []
    else:
        per_object, problems = static_checks(data_path)
    names = list(per_object)
    jobs = [(n, lt) for n in names for lt in args.lighting]
    with ThreadPoolExecutor(args.parallel) as pool:
        outs = dict(zip(jobs, pool.map(
            lambda j: run_walk(data_path, j[0], j[1], out_dir), jobs)))

    rows = []
    for name in names:
        for lighting in args.lighting:
            rows.append({"object": name, "lighting": lighting, **per_object[name],
                         **summarize_walk(outs[(name, lighting)])})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "prelim_a_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    contact_sheet([(n, [outs[(n, lt)] for lt in args.lighting]) for n in names],
                  out_dir / "prelim_a_contact_sheet.png")

    print(f"{len(names)} objects, lighting {args.lighting}")
    for p in problems:
        print("STATIC:", p)
    if not args.no_static:
        print("placeholder materials:",
              sum(r["placeholder"] for r in per_object.values()))
        print("size outside +-2%:",
              [n for n, r in per_object.items() if not r["size_ok"]])
    for lighting in args.lighting:
        lr = [r for r in rows if r["lighting"] == lighting]
        failed_walks = [r["object"] for r in lr if not r.get("walk_ok")]
        off = [r["object"] for r in lr if r.get("frac_center_on_object", 0) <= MIN_ON_OBJECT]
        black = [r["object"] for r in lr if r.get("frac_black", 0) > 0]
        v = [r["median_v"] for r in lr if r.get("walk_ok")]
        print(f"[{lighting}] failed walks: {failed_walks}")
        print(f"[{lighting}] on-object <= {MIN_ON_OBJECT}: {off}")
        print(f"[{lighting}] any nearly-black steps: {black}")
        print(f"[{lighting}] median patch V: min {min(v):.3f}, median {np.median(v):.3f}")
    print(f"Report: {out_dir / 'prelim_a_report.csv'}")
    print(f"Contact sheet: {out_dir / 'prelim_a_contact_sheet.png'}")


if __name__ == "__main__":
    main()
