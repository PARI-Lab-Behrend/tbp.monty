# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Analysis-only experiments that inspect a trained model without running an env.

These experiments load a pretrained Monty model and produce plots/statistics
from its stored graphs. They deliberately skip environment setup and the
sensorimotor loop, so they run quickly and need no simulator.
"""

from __future__ import annotations

import logging
from typing import Any

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.utils.ltp_distance_analysis import (
    METRICS,
    analyze_and_plot,
    extract_histograms,
)

__all__ = ["MontyLTPDistanceExperiment"]

logger = logging.getLogger(__name__)


class MontyLTPDistanceExperiment(MontyExperiment):
    """Plot LTP texture-histogram distances captured from a trained Monty model.

    Loads the pretrained model named by ``model_name_or_path`` (the real
    ``EvidenceGraphMemory``), then -- using the *actual* Monty matching distance
    functions -- measures within-object vs across-object histogram distances and
    writes the distribution plots to the run's output directory. This is the
    experiment-system counterpart of ``scripts/plot_ltp_distances.py``.

    The analysis parameters are read from ``config["ltp_distance"]`` (all
    optional)::

        ltp_distance:
          metrics: [chisq, hellinger]   # or "all"; names from METRICS
          feature: ltp                  # histogram feature to analyze
          lm_id: 0                      # which learning module's memory to read
          max_nodes_per_object: 300     # subsample cap per object
          n_queries: 400                # number of query nodes sampled
          group_regex: '(baseball|softball)$'  # optional shape-group split
          bins: 60                      # plot histogram bins
          seed: 42

    No environment is created and the sensorimotor loop is never run.
    """

    def setup_experiment(self, config: dict[str, Any]) -> None:
        """Initialize loggers and the model only -- no environment, no data loggers."""
        self.init_loggers(self.config["logging"])
        self.model = self.init_model(
            monty_config=config["monty_config"],
            model_path=self.model_path,
        )

    def run(self) -> None:
        """Extract stored histograms and write the distance-distribution plot."""
        params = dict(self.config.get("ltp_distance", {}))

        metrics = params.get("metrics", ["chisq", "hellinger"])
        if metrics == "all" or metrics == ["all"]:
            metric_names = sorted(METRICS)
        else:
            metric_names = list(metrics)
        unknown = [m for m in metric_names if m not in METRICS]
        if unknown:
            raise ValueError(
                f"Unknown metric(s) {unknown}; choose from {sorted(METRICS)} or 'all'."
            )

        feature = params.get("feature", "ltp")
        lm_id = params.get("lm_id", 0)

        graph_memory = self.model.learning_modules[lm_id].graph_memory
        models = graph_memory.get_all_models_in_memory()
        hists = extract_histograms(models, feature)
        n_bins = next(iter(hists.values())).shape[1]
        logger.info(
            "Loaded %d objects, %d nodes, feature %r with %d bins.",
            len(hists),
            sum(h.shape[0] for h in hists.values()),
            feature,
            n_bins,
        )

        out_path = self.output_dir / f"ltp_distances_{self.run_name}.png"
        analyze_and_plot(
            hists,
            metric_names,
            out_path=out_path,
            title=f"LTP ({feature}) distances — {self.run_name}",
            max_nodes_per_object=params.get("max_nodes_per_object", 300),
            n_queries=params.get("n_queries", 400),
            group_regex=params.get("group_regex"),
            bins=params.get("bins", 60),
            seed=params.get("seed", self.config["seed"]),
            xmax=params.get("xmax"),
        )

    def close(self) -> None:
        """Close python logging only (no env / data loggers were created)."""
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()
