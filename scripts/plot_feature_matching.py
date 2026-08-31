#!/usr/bin/env python
"""Render plot_utils_analysis.plot_feature_matching_animation to a video file.

The animation helper in plot_utils_analysis was written to display an HTML5 video
inline in a Jupyter notebook. This wrapper loads a detailed-logging experiment run,
captures the matplotlib animation it builds, and saves it to disk (mp4 via ffmpeg,
falling back to gif) so it can be viewed outside a notebook.

Usage:
    python scripts/plot_feature_matching.py \
        --exp ~/tbp/results/monty/projects/monty_runs/feature_matching_demo \
        --out /tmp/feature_matching.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation  # noqa: E402

from tbp.monty.frameworks.utils.logging_utils import load_stats  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="experiment output dir")
    p.add_argument("--out", default="/tmp/feature_matching.mp4")
    p.add_argument("--episode", default=None, help="episode key; default = last eval")
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args()

    exp_path = Path(args.exp).expanduser()
    _, eval_stats, detailed_stats, lm_models = load_stats(
        exp_path, load_train=False, load_eval=True, load_detailed=True, load_models=True
    )

    print("lm_models keys:", list(lm_models.keys()))
    print("detailed episode keys:", list(detailed_stats.keys()))

    # plot_feature_matching_animation was written against an older sensor/LM schema.
    # Bridge the current schema to what it expects, without editing the shared lib:
    #   * processed_obs["features"]["on_object"]  <- morphological_features["on_object"]
    #   * lm["displacement"]                      <- lm["displacements"]
    for ep in detailed_stats.values():
        for key, val in ep.items():
            if key.startswith("SM_") and isinstance(val, dict):
                for obs in val.get("processed_observations", []) or []:
                    if isinstance(obs, dict) and "features" not in obs:
                        morph = obs.get("morphological_features") or {}
                        obs["features"] = {"on_object": morph.get("on_object", 0.0)}
            if key.startswith("LM_") and isinstance(val, dict):
                if "displacement" not in val and "displacements" in val:
                    disp = val["displacements"]
                    if isinstance(disp, dict) and "displacement" in disp:
                        disp = disp["displacement"]
                    val["displacement"] = disp

    # Pick an eval episode: the detailed logger keys episodes by total episode index.
    # Prefer the requested one, else the last episode whose LM ran in EVAL mode.
    lm_id = "LM_0"
    if args.episode is not None:
        episode = args.episode
    else:
        episode = None
        for key in detailed_stats:
            mode = detailed_stats[key][lm_id].get("mode")
            if mode == "eval":
                episode = key
        if episode is None:
            episode = list(detailed_stats.keys())[-1]
    print(f"Using episode {episode} (mode={detailed_stats[episode][lm_id].get('mode')})")

    # Objects = everything stored in the loaded model for this episode.
    from tbp.monty.frameworks.utils.plot_utils_dev import get_model_id

    epoch = detailed_stats[episode][lm_id]["train_epochs"]
    mode = detailed_stats[episode][lm_id]["mode"]
    from tbp.monty.frameworks.experiments.mode import ExperimentMode

    mode_enum = ExperimentMode.EVAL if mode == "eval" else ExperimentMode.TRAIN
    model_id = get_model_id(epoch, mode_enum)
    if model_id not in lm_models:
        numeric = sorted((k for k in lm_models if k.isnumeric()), key=int)
        model_id = numeric[-1] if numeric else next(iter(lm_models))
        print(f"  (model_id fell back to {model_id})")
    # load_models_from_dir keys models by "LM_<n>", but the animation indexes
    # lm_models[model_id][lm_num]; use the matching string key.
    model_lm_key = lm_id if lm_id in lm_models[model_id] else next(
        iter(lm_models[model_id])
    )
    objects = list(lm_models[model_id][model_lm_key].keys())
    print(f"model_id={model_id}, lm_key={model_lm_key}, objects={objects}")

    # Newer models nest each object's graph under an input channel, e.g.
    # graph_memory[obj]["patch"], but plot_feature_matching_animation expects
    # lm_models[model_id][lm_key][obj] to be the GraphObjectModel itself. Flatten
    # by picking the input channel that carries a .pos-bearing model.
    def flatten(entry):
        if hasattr(entry, "pos"):
            return entry
        if isinstance(entry, dict):
            for v in entry.values():
                if hasattr(v, "pos"):
                    return v
        return entry

    for mid in lm_models:
        for lk in lm_models[mid]:
            gm = lm_models[mid][lk]
            if isinstance(gm, dict):
                lm_models[mid][lk] = {o: flatten(gm[o]) for o in gm}

    # Capture the animation that plot_feature_matching_animation builds internally and
    # save it, instead of trying to embed HTML5 video for a notebook.
    saved = {}
    orig_html5 = animation.FuncAnimation.to_html5_video

    def capture_and_save(self, *a, **k):
        out = Path(args.out).expanduser()
        writer = "ffmpeg" if animation.writers.is_available("ffmpeg") else "pillow"
        if writer == "pillow" and out.suffix == ".mp4":
            out = out.with_suffix(".gif")
        print(f"Saving animation -> {out} (writer={writer}) ...")
        self.save(str(out), writer=writer, fps=args.fps, dpi=120)
        saved["path"] = out
        return ""

    animation.FuncAnimation.to_html5_video = capture_and_save

    # display.display is a no-op outside a notebook, but stub it to be safe.
    import tbp.monty.frameworks.utils.plot_utils_analysis as pua

    pua.display.display = lambda *a, **k: None

    # get_action_name assumes Action objects with .name; the logged action_sequence
    # holds plain dicts ({"action": <name>, "agent_id": ..., <params>}). Swap in a
    # dict-tolerant version that produces the same title strings.
    def get_action_name_dict(action_stats, step, is_match_step, obs_on_object):
        if is_match_step:
            return (
                "updating possible matches" if obs_on_object else "patch not on object"
            )
        if step == 0:
            return "not moved yet"
        actions = action_stats[step - 1][0]
        if not actions:
            return "None"
        strings = []
        for a in actions:
            if hasattr(a, "name"):  # fall back to original behaviour
                name = a.name
                params = {k: v for k, v in dict(a).items() if k not in ("action", "agent_id")}
            else:
                name = a.get("action")
                params = {k: v for k, v in a.items() if k not in ("action", "agent_id")}
            pstr = ",".join(f"{k}:{v}" for k, v in params.items())
            strings.append(f"{name} - {pstr}")
        return strings[0] if len(strings) == 1 else "[" + ", ".join(strings) + "]"

    pua.get_action_name = get_action_name_dict

    try:
        pua.plot_feature_matching_animation(
            stats=detailed_stats,
            lm_models=lm_models,
            episode=int(episode) if str(episode).isdigit() else episode,
            objects=objects,
            lm_id=lm_id,
            lm_num=model_lm_key,
            show_num_pos=5,
            rotate=False,
        )
    finally:
        animation.FuncAnimation.to_html5_video = orig_html5

    if "path" in saved:
        print(f"Done: {saved['path']}")
    else:
        print("WARNING: animation was not saved (to_html5_video not called).")


if __name__ == "__main__":
    main()
