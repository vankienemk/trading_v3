# Order Comment Schema §9.2 — Migration Notes (Agent 6, t9)

**Spec:** REVERSAL_PATTERN_ENGINE_SPEC_v1.1 §9.2

## Standard schema

```
{pattern_short}-v{major}-{event_id_short}
```

Examples:

| pattern | short (§9.2) | event | comment |
|---------|---------------|-------|---------|
| liquidity_sweep | `LSW` | `XAUUSD-V2-000123` | `LSW-v2-0123` |
| double_bottom | `DB` | `XAUUSD-DB-000456` | `DB-v1-0456` |
| double_top | `DT` | `XAUUSD-DT-000789` | `DT-v1-0789` |

- `pattern_short` is the 2–4 char unique code registered in
  `pattern_registry.py` (§2.2) with fallback `PAT` for unknown patterns.
- `v{major}` is the detector's semantic major version (`pattern_version.split(".")[0]`).
- `event_id_short` is the last 4 chars of the full `event_id` (a stable
  reconcile key between trade ↔ event ↔ model across multi-pattern).

The formatting function `research/core/contracts.py::order_comment_for(event)`
is the single source of truth; `live/engine/signal_engine_v2.py` stamps every
`SignalCandidate` with `order_comment` via that function.

## Legacy format (what existed before)

The historical single-pattern engine produced `LSW-V2-{event_id[:16]}` in
`execution_layer_v2.send_order()` — hard-coded to Liquidity Sweep, using the
full event-id prefix (16 chars) and an uppercase `V2`.

## Migration

1. **Backward compatible:** `ExecutionLayer.send_order()` accepts an optional
   `order_comment` parameter.  When omitted (older callers), it keeps producing
   the legacy `LSW-V2-{event_id[:16]}` comment so the paper engine and existing
   tests keep working unchanged.
2. **New path:** the multi-pattern engine (`MultiPatternEngine` → `SignalCandidate`)
   always passes the standardised §9.2 comment; the execution layer uses it
   verbatim when supplied.
3. **Reconcile key:** MT5 order comments are unique per event.  Operators can
   join any opened trade back to its `PatternEvent` (Event Lake
   `events/{pattern}/{symbol}/{YYYY-MM}.parquet`) through `event_id_short` +
   `pattern_short` + version.  No order-comment collision is possible across
   patterns because `pattern_short` is unique (§9.2).
4. **Who migrates:** `signal_polling_engine_v2._auto_send_order` (or any future
   caller) should pass `cand.order_comment` through to `send_order`.  The
   polling engine currently passes only legacy fields — wiring it to the new
   field is optional and covered by the fallback; recommend forwarding as a
   follow-up enhancement once every pattern runs through MultiPatternEngine.

## Related §3.3 entry-drift guard

`MultiPatternEngine._entry_drift_ok()` implements the §3.3 rule: at entry, if
the live market price has moved more than `max_entry_drift_atr * ATR` away from
the recorded `entry_price`, the event is dropped with
`attributes.discard_reason = "stale"` (no signal, no chasing price).  Default
`max_entry_drift_atr = 1.5` (configurable per `PatternAssignment`).