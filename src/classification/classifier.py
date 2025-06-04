# src/classification/classifier.py  – 2025-06-04 patch
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RegionClassifier:
    """
    Classifies “inactive” channels in GroupNorm layers based on ActivityMonitor
    statistics and returns the scale-parameter names plus channel indices for
    the InterventionHandler.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "threshold_groupnorm_activity")
        self.threshold = float(config.get("threshold", 1e-3))
        self.target_metric_key = config.get("target_metric_key", "mean_abs_activation_per_channel")
        self.layers_to_classify: List[str] = config.get("layers_to_classify", [])

        # { monitor_identifier : (param_name, num_channels) }
        self._layer_to_param_map: Dict[str, Tuple[str, int]] = {}

        if model is not None:
            self._build_groupnorm_map(model)
        else:
            logger.warning("RegionClassifier initialised without a model – "
                           "parameter mapping will be heuristic only.")

        logger.info(
            f"RegionClassifier initialised (method={self.method}, thr={self.threshold}, "
            f"metric={self.target_metric_key}, map_size={len(self._layer_to_param_map)})"
        )
        if not self._layer_to_param_map:
            logger.warning("RegionClassifier: no GroupNorm layers found / mapped.")

    # --------------------------------------------------------------------- #
    # Mapping helpers
    # --------------------------------------------------------------------- #
    def _build_groupnorm_map(self, model: torch.nn.Module):
        """
        Builds a map from ActivityMonitor layer IDs to the actual GroupNorm
        scale parameter names in `model`.

        For each GN module called “encoder.…norm1”, we register **two** keys:

            encoder.…norm1.output                (plain)
            vae.encoder.…norm1.output            (prefixed)

        so the classifier works whether or not the YAML includes the `vae.` prefix.
        """
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.GroupNorm):
                continue

            monitor_key_plain = f"{mod_name}.output"
            param_name_scale = f"{mod_name}.weight"
            num_ch = mod.num_channels

            # Verify that the parameter exists and is a Parameter object
            try:
                param_obj = model
                for part in param_name_scale.split("."):
                    param_obj = getattr(param_obj, part)
                if not isinstance(param_obj, torch.nn.Parameter):
                    logger.debug(f"Skipping {mod_name}: scale param is not a Parameter.")
                    continue
            except AttributeError:
                logger.debug(f"Skipping {mod_name}: could not resolve parameter path.")
                continue

            # 1) plain key
            self._layer_to_param_map[monitor_key_plain] = (param_name_scale, num_ch)
            # 2) alias with “vae.” prefix (only if not already prefixed)
            if not mod_name.startswith("vae."):
                self._layer_to_param_map[f"vae.{monitor_key_plain}"] = (param_name_scale, num_ch)

            logger.debug(f"Mapped GN '{monitor_key_plain}' → '{param_name_scale}' (C={num_ch})")

    def _lookup_param_info(self, layer_id: str) -> Optional[Tuple[str, int]]:
        """
        Returns (param_name, num_channels) or None.  Tries exact match, then
        strips the first token before dot (e.g. “vae.”) and retries once.
        """
        info = self._layer_to_param_map.get(layer_id)
        if info is not None:
            return info
        # fallback: drop leading scope prefix
        if "." in layer_id:
            stripped = layer_id.split(".", 1)[1]
            return self._layer_to_param_map.get(stripped)
        return None

    # --------------------------------------------------------------------- #
    # Main public API
    # --------------------------------------------------------------------- #
    def classify(self, tracked_data_for_step: Dict[str, Any], global_step: int) -> Dict[str, Any]:
        if not self.config.get("enabled", False):
            return {}

        logger.info(f"RegionClassifier.classify() step {global_step}")
        results: Dict[str, Any] = {}

        if self.method != "threshold_groupnorm_activity":
            logger.warning(f"Unknown classification method: {self.method}")
            return results

        if not tracked_data_for_step:
            return results

        for layer_id, metrics in tracked_data_for_step.items():
            # respect layers_to_classify filter if provided
            if self.layers_to_classify and layer_id not in self.layers_to_classify:
                continue

            per_channel_vals = metrics.get(self.target_metric_key)
            if per_channel_vals is None:
                continue
            if not (isinstance(per_channel_vals, np.ndarray) and per_channel_vals.ndim == 1):
                continue

            param_info = self._lookup_param_info(layer_id)
            if param_info is None:
                logger.debug(f"{layer_id}: no GN mapping found – skipped.")
                continue
            param_name_scale, num_ch = param_info

            if per_channel_vals.shape[0] != num_ch:
                logger.warning(f"{layer_id}: channel mismatch ({per_channel_vals.shape[0]} vs {num_ch}) – skipped.")
                continue

            inactive_idx = np.where(per_channel_vals < self.threshold)[0]
            if inactive_idx.size == 0:
                continue

            results[layer_id] = {
                "param_name_scale": param_name_scale,
                "inactive_channel_indices": inactive_idx.tolist(),
                "metric_used": self.target_metric_key,
                "threshold_value": self.threshold,
                "values_of_inactive_channels": per_channel_vals[inactive_idx].tolist(),
            }

            logger.info(f"Step {global_step}: {layer_id} → {len(inactive_idx)} inactive channels "
                        f"(param {param_name_scale})")

        logger.info(f"Classification complete — {len(results)} layer(s) flagged.")
        return results


# ------------------------------------------------------------------------- #
# Simple self-test (runs only when you execute this file directly)
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # dummy model with a couple of GroupNorms
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.GroupNorm(2, 8, affine=True)  # encoder.norm1
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.GroupNorm(4, 8, affine=True)  # decoder.norm1
            )

    m = Mini()
    cfg = {"enabled": True, "threshold": 0.5,
           "target_metric_key": "mean_abs_activation_per_channel",
           "layers_to_classify": ["vae.encoder.0.output"]}  # pretend monitor name
    clf = RegionClassifier(model=m, config=cfg)

    dummy_vals = np.array([0.1, 0.6, 0.4, 0.2, 0.8, 0.3, 0.7, 0.05])
    tracked = {"vae.encoder.0.output": {"mean_abs_activation_per_channel": dummy_vals}}
    res = clf.classify(tracked, 0)
    print("Results:", res)
