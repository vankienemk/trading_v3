"""
config.py — hmm_regime plugin YAML loading (guide §6 / requirements §9)

The sample config file ``research/configs/plugins/hmm_regime.yaml`` wraps the
HMM section under ``hmm:`` (and carries an optional ``regime_filter:`` block
consumed by the hard gate in t3).  For training / inference the plugin only
needs the ``hmm`` section; this loader extracts it so ``CausalGaussianHMM.fit``
can merge it over its defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_hmm_regime_config(path: str | Path) -> dict[str, Any]:
    """Read a hmm_regime plugin YAML and return the ``hmm`` config section.

    * present ``hmm:`` section  -> returned as-is (the ``regime_filter`` and
      ``plugin`` sections are ignored by the HMM itself);
    * flat file without ``hmm:`` -> returned verbatim (tolerant of minimal
      configs);
    * anything non-dict -> ``ValueError``.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    hmm_section = raw.get("hmm")
    if hmm_section is None:
        return raw
    if not isinstance(hmm_section, dict):
        raise ValueError(f"{path}: 'hmm' section must be a mapping")
    return dict(hmm_section)