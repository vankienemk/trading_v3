DebugDrawer event file format (ASCII only, one event per line, pipe-separated):

  0  pattern          (e.g. liquidity_sweep, double_bottom, double_top)
  1  direction        BUY or SELL (also accepts 'long')
  2  symbol           (e.g. XAUUSDm)
  3  detect_time      datetime format yyyy.MM.dd HH:mm
  4  confirm_time     datetime format yyyy.MM.dd HH:mm
  5  entry_time       datetime (may be empty -> fallback confirm_time)
  6  entry_price      double
  7  stop_price       double (may be 0/empty -> flat rectangle)
  8  target_price     double (may be 0/empty)
  9  rule_score       optional double 0..1
 10  model_prob       optional calibrated probability 0..1
 11  discard_reason   optional; non-empty => drawn in GRAY (rejected)
 12  level_price      optional (structure level, informational)
 13  extreme_price    optional (sweep extreme, informational)
 14  pattern_start    optional datetime -- FIRST bar of the pattern structure
 15  pattern_end      optional datetime -- LAST bar of the pattern structure
 16  pattern_low      optional double -- structure price envelope low
 17  pattern_high     optional double -- structure price envelope high

  Fields 14..17 drive the YELLOW pattern wash in DebugDrawer v1.9: the wash
  covers [pattern_start .. pattern_end] x [pattern_low .. pattern_high], i.e.
  the real pattern structure (e.g. for double_top: extreme1 -> extreme2).
  When they are empty the drawer falls back to detect->confirm + trade span.
  Generate them with docs/debug_drawer/export_debug_events.py.

Python side (trading_v3) can regenerate this file from PatternEvents:
  discard_reason -> event.attributes.get('discard_reason')
  known_at       -> event.known_at_ts (confirm/detect)
  entry          -> event.entry_price / entry_time
  stop/target    -> event.stop_price / target_price
  model_prob     -> event.model_prob
  rule_score     -> event.rule_score

How to use:
  1. drop this file at MQL5\Files\debug_events.txt
  2. compile DebugDrawer.mq5 in MetaEditor (F7) -- check Errors tab = 0 errors
  3. drag DebugDrawer onto the chart (symbol must match its timeframe)
  Objects are prefixed 'pat_' and are cleaned on every rerun.

DRAWING STYLE (v1.9):
  * pattern zone  : yellow washed rectangle over the REAL pattern structure
                    (fields 14..17; detect->confirm fallback)
  * entry / SL /TP: ONE rectangle split at the ENTRY price --
                    side toward TARGET light-green, side toward STOP
                    light-red (gray for rejected events)
  * also drawn    : entry arrow (green up / red down), text annotation
  * NO dashed horizontal lines.

NOTES:
  * Event prices must be NEAR the current chart price level; the Comment
    shows "bid=<price>" on every run + "visible price: A..B".
    If drawn>0 but nothing visible -> compare event price with bid and
    regenerate the file, or scroll the price scale ("Scale to full").
  * Event times are parsed in the terminal's local (server) timezone.
  * v1.6+ MUST NOT use StringTrimLeft/StringTrimRight on event fields:
    they are broken in this Wine build and turn fields into "0"
    (that caused the earlier "drawn=0 skipped=3"). Fields are used raw.
  * v1.7 writes a byte-level diagnostic to
    MQL5\Files\debug_events_echo.txt on every run for remote debugging.