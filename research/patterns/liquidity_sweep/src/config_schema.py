"""Declarative config schema validation.

Implements the config contract from docs/SCHEMAS.md section 2.  The validator
checks that the merged config contains every required key with an acceptable
type / value, and raises ``ConfigError`` naming the offending dotted path.

Owned by Agent 0.  Downstream agents may extend the *tables* below (e.g. when
adding a config key) but must do so together with an update of
``docs/SCHEMAS.md`` and the corresponding tests (rule 30.1).
"""

from __future__ import annotations

from typing import Any, Callable

from .config import ConfigError


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_float(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_str(v: Any) -> bool:
    return isinstance(v, str)


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_int(x) for x in v)


def _is_float_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_float(x) for x in v)


def _is_score_buckets(v: Any) -> bool:
    return isinstance(v, list) and all(
        isinstance(b, list) and len(b) == 2 and _is_int(b[0]) and _is_int(b[1])
        for b in v
    )


def _is_hour_range(v: Any) -> bool:
    """Half-open ``[start, end)`` hour range, 0..24, end may wrap midnight."""
    return (
        isinstance(v, list)
        and len(v) == 2
        and _is_int(v[0])
        and _is_int(v[1])
        and 0 <= v[0] <= 24
        and 0 <= v[1] <= 24
    )


def _is_sessions(v: Any) -> bool:
    return (
        isinstance(v, dict)
        and _is_str(v.get("timezone"))
        and _is_hour_range(v.get("asia"))
        and _is_hour_range(v.get("london"))
        and _is_hour_range(v.get("new_york"))
    )


#: (section -> {key: (description, checker)})
CONFIG_SPEC: dict[str, dict[str, tuple[str, Callable[[Any], bool]]]] = {
    "project": {
        "symbol": ("str", _is_str),
        "timeframe": ("str", _is_str),
        "timezone": ("str", _is_str),
        "random_seed": ("int", _is_int),
    },
    "data": {
        "input_path": ("str", _is_str),
        "output_path": ("str", _is_str),
        "timestamp_column": ("str", _is_str),
        "volume_column": ("str", _is_str),
        "source_format": ("str", _is_str),
        "volume_kind": ("str", _is_str),
        "utc_offset_hours": ("str/int/float", lambda v: _is_str(v) or _is_float(v)),
    },
    "sessions": {
        "timezone": ("str", _is_str),
        "asia": ("[int,int] hour range", _is_hour_range),
        "london": ("[int,int] hour range", _is_hour_range),
        "new_york": ("[int,int] hour range", _is_hour_range),
    },
    "indicators": {
        "atr_period": ("int >= 2", lambda v: _is_int(v) and v >= 2),
        "volume_zscore_window": ("int >= 2", lambda v: _is_int(v) and v >= 2),
    },
    "liquidity": {
        "rolling_lookback": ("int >= 2", lambda v: _is_int(v) and v >= 2),
        "enable_rolling": ("bool", _is_bool),
    },
    "sweep": {
        "min_penetration_atr": ("float >= 0", lambda v: _is_float(v) and v >= 0),
        "max_penetration_atr": ("float > min", lambda v: _is_float(v) and v >= 0),
        "min_wick_ratio": ("float 0..1", lambda v: _is_float(v) and 0 <= v <= 1),
        "min_reclaim_atr": ("float", _is_float),
        "cooldown_bars": ("int >= 0", lambda v: _is_int(v) and v >= 0),
        "group_rule": (
            "str in {first, deepest_penetration, strongest_reclaim}",
            lambda v: v in {"first", "deepest_penetration", "strongest_reclaim"},
        ),
    },
    "confirmation": {
        "enabled": ("bool", _is_bool),
        "max_wait_bars": ("int >= 0", lambda v: _is_int(v) and v >= 0),
        "require_break_sweep_extreme": ("bool", _is_bool),
        "min_body_ratio": ("float 0..1", lambda v: _is_float(v) and 0 <= v <= 1),
        "min_range_atr": ("float >= 0", lambda v: _is_float(v) and v >= 0),
    },
    "entry": {
        "mode": ("entry-mode", lambda v: v in {"next_open_after_confirmation", "next_open_after_sweep"}),
        "slippage_price": ("float >= 0", lambda v: _is_float(v) and v >= 0),
        "spread_price": ("float >= 0", lambda v: _is_float(v) and v >= 0),
    },
    "stop": {
        "mode": ("str in {sweep_extreme}", lambda v: v == "sweep_extreme"),
        "buffer_atr": ("float >= 0", lambda v: _is_float(v) and v >= 0),
    },
    "labeling": {
        "horizons": ("sorted int list", lambda v: _is_int_list(v) and v == sorted(v)),
        "reward_r_values": ("float list", _is_float_list),
        "same_bar_policy": ("policy", lambda v: v in {"ambiguous", "conservative", "optimistic"}),
        "time_barrier_result": ("result", lambda v: v in {"mark_to_market", "zero"}),
    },
    "costs": {
        "half_spread_price": ("float >= 0", lambda v: _is_float(v) and v >= 0),
        "slippage_price": ("float >= 0", lambda v: _is_float(v) and v >= 0),
        "commission_r": ("float >= 0", lambda v: _is_float(v) and v >= 0),
    },
    "splitting": {
        "train_fraction": ("float 0..1", lambda v: _is_float(v) and 0 <= v <= 1),
        "validation_fraction": ("float 0..1", lambda v: _is_float(v) and 0 <= v <= 1),
        "test_fraction": ("float 0..1", lambda v: _is_float(v) and 0 <= v <= 1),
        "embargo_bars": ("int >= 0", lambda v: _is_int(v) and v >= 0),
        "purge": ("bool", _is_bool),
    },
    "model": {
        "target": ("str", _is_str),
        "probability_calibration": ("str", lambda v: v in {"isotonic", "sigmoid", "none"}),
        "models": ("str list", _is_str_list),
    },
    "scoring": {
        "score_buckets": ("list[list[int]]", _is_score_buckets),
    },
}

#: Optional sections that may be absent entirely (e.g. confirmation disabled).
OPTIONAL_SECTIONS: set[str] = {"scoring", "sessions"}


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate a merged config against :data:`CONFIG_SPEC`.

    Raises
    ------
    ConfigError
        On a missing required key, a wrong type, or an out-of-range value.
        The message includes the dotted path of the offending key.
    """
    _validate_section(cfg, "", CONFIG_SPEC)


def _validate_section(
    cfg: dict[str, Any],
    prefix: str,
    spec: dict[str, dict[str, tuple[str, Callable[[Any], bool]]]],
) -> None:
    for section, keys in spec.items():
        path = f"{prefix}{section}"
        if section not in cfg:
            if section in OPTIONAL_SECTIONS:
                continue
            raise ConfigError(f"Config is missing required section '{path}'")
        value = cfg[section]
        if not isinstance(value, dict):
            raise ConfigError(f"Config section '{path}' must be a mapping, got {type(value).__name__}")
        for key, (kind, check) in keys.items():
            key_path = f"{path}.{key}"
            if key not in value:
                raise ConfigError(f"Config is missing required key '{key_path}'")
            if not check(value[key]):
                raise ConfigError(f"Config key '{key_path}' must be {kind}, got {value[key]!r}")