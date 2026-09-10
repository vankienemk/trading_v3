"""
chart_widget.py — Minimal QPainter candlestick chart for the live GUI.

Used by the pending-signal inspector (``SignalInspectDialog`` in
gui_tab_live.py) to show the last ~50 M15 candles for a symbol with the
signal's entry / stop-loss / take-profit levels overlaid.

The widget uses only ``PySide6.QtGui.QPainter`` (no matplotlib / pyqtgraph /
QtCharts — the venv only ships PySide6), so it stays import-light and never
blocks the GUI thread on a plotting engine.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

# Import the theme colours from gui_components so the chart matches the app.
from live.gui.gui_components import (
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_SURFACE,
    COLOR_TEXT_SEC,
)

# Marker colours (distinct from the theme swing colours).
_MARKER_ENTRY = "#ffffff"   # white — entry
_MARKER_TP = "#00cc66"      # green — take-profit
_MARKER_SL = "#ff4444"      # red — stop-loss


class CandlestickChart(QWidget):
    """A compact candlestick chart with entry / TP / SL overlay lines.

    Call :meth:`set_data` with an OHLCV ``pandas.DataFrame`` (or a list of
    candle dicts) plus optional price levels; the widget repaints with the
    candles and dashed marker lines (white = entry, green = take-profit,
    red = stop-loss).  Empty/no-data input renders a "No data" placeholder.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._candles: list[dict[str, Any]] = []
        self._entry: float | None = None
        self._stop: float | None = None
        self._take_profit: float | None = None
        self._direction: str | None = None
        self._loading: bool = False
        self.setMinimumHeight(120)
        self.setAutoFillBackground(True)
        self.setStyleSheet(f"background-color: {COLOR_SURFACE};")

    def set_data(
        self,
        candles: Any,
        entry: float | None = None,
        stop: float | None = None,
        take_profit: float | None = None,
        direction: str | None = None,
    ) -> None:
        """Set the candles and overlay levels, then repaint.

        ``candles`` may be a ``pandas.DataFrame`` (columns open/high/low/close)
        or a list of dicts with those keys.  ``None``/empty renders a "No data"
        placeholder.  ``direction`` ("buy"/"long" = up, "sell"/"short" = down)
        drives the entry-point triangle marker colour/orientation.
        """
        self._candles = _to_candles(candles)
        self._entry = _as_float(entry)
        self._stop = _as_float(stop)
        self._take_profit = _as_float(take_profit)
        self._direction = None if direction is None else str(direction).lower()
        # Any resolved set_data() means the fetch completed (even if it returned
        # no candles) -> not "loading" any more.
        self._loading = False
        self.update()

    def set_loading(self, loading: bool) -> None:
        """Show the 'Loading chart…' placeholder while data is being fetched."""
        self._loading = bool(loading)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        painter.fillRect(rect, QColor(COLOR_SURFACE))

        if not self._candles:
            painter.setPen(QColor(COLOR_TEXT_SEC))
            # Distinguish the brief "loading" state (no candles set yet) from a
            # genuine "no data" result after a fetch returned empty.
            label = "Loading chart…" if self._loading else "No candle data"
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
            return

        # Price range spanning every candle high/low plus every valid marker
        # level (entry / stop / take-profit).  Levels <= 0 mean "missing" (e.g.
        # a signal record without an SL/TP passes 0.0) and are excluded from
        # the range AND from marker drawing — otherwise 0.0 would collapse the
        # scale.  All levels are fed to BOTH ends of the range, so SELL geometry
        # (TP lowest → entry middle → SL highest) stays just as visible as BUY.
        lows = [c["low"] for c in self._candles]
        highs = [c["high"] for c in self._candles]
        levels = [v for v in (self._entry, self._stop, self._take_profit)
                  if v is not None and v > 0]

        # --- Y-range (price window) ---
        # Price-magnitude based unit so the scale behaves the same for a
        # ~80k BTC, a ~2400 XAU and a ~1.10 EURUSD.  The OLD padding used a flat
        # 0.5 USD floor which ballooned low-priced FX candles into an
        # unreadable sliver (1 pip = 0.0001 vs a 0.5 pad): that's why EURUSD
        # "collapsed" while BTC looked fine.
        candle_lo = min(lows) if lows else (min(levels) if levels else 0.0)
        candle_hi = max(highs) if highs else (max(levels) if levels else 0.0)
        magnitude = max(abs(candle_lo), abs(candle_hi)) or 1.0
        candle_span = candle_hi - candle_lo
        if candle_span <= 0:
            candle_span = magnitude * 5e-4  # flat candles -> small relative span

        # Entry-centric window: ALWAYS centre the price window on the ENTRY point
        # so it never sits near / carved off an edge, and the surrounding SL/TP
        # band + candle pattern stay readable.  Half-width = the wider of the
        # trade band (|entry-SL| / |entry-TP|) and the candle context.  When the
        # entry has drifted a long way, price ends up centred (previously it was
        # pinned toward a boundary and looked "out of the chart").
        if self._entry is not None and self._entry > 0:
            center = self._entry
            band_half = 0.0
            for lv in (self._stop, self._take_profit):
                if lv is not None and lv > 0:
                    band_half = max(band_half, abs(center - lv))
            half = max(band_half, candle_span, magnitude * 2e-3)
            fit_lo, fit_hi = center - half, center + half
        else:
            # No entry: show the candles + any levels compactly (candle frame is
            # already sliced to start at the entry bar when entry_time is known).
            fit_lo, fit_hi = candle_lo, candle_hi
            if levels:
                fit_lo = min(fit_lo, min(levels))
                fit_hi = max(fit_hi, max(levels))

        lo, hi = fit_lo, fit_hi
        # Scale-aware vertical padding: 8% of span, floored at ~0.05% of the
        # price magnitude (NOT a flat USD value) so low-priced FX is not shoved
        # into a sliver.
        pad = max((hi - lo) * 0.08, magnitude * 5e-4)
        lo -= pad
        hi += pad
        span = hi - lo

        # Layout margins (px)
        pad_left = 6
        pad_right = 6
        pad_top = 8
        pad_bottom = 18
        plot_w = rect.width() - pad_left - pad_right
        plot_h = rect.height() - pad_top - pad_bottom
        n = len(self._candles)
        slot = plot_w / n if n else plot_w
        body_w = max(2.0, slot * 0.6)

        def y(price: float) -> float:
            return pad_top + plot_h * (1.0 - (price - lo) / span)

        for i, c in enumerate(self._candles):
            cx = pad_left + slot * i + slot / 2.0
            up = c["close"] >= c["open"]
            colour = QColor(COLOR_POSITIVE if up else COLOR_NEGATIVE)
            painter.setPen(colour)

            # Wick: high -> low
            wick_y0 = y(c["high"])
            wick_y1 = y(c["low"])
            painter.drawLine(int(cx), int(wick_y0), int(cx), int(wick_y1))

            # Body: open -> close
            open_y = y(c["open"])
            close_y = y(c["close"])
            top = min(open_y, close_y)
            h = max(1.0, abs(close_y - open_y))
            painter.setBrush(colour)
            painter.drawRect(int(cx - body_w / 2.0), int(top), int(body_w), int(h))

        # Marker lines (dashed) — entry (white), TP (green), SL (red).
        # Only levels that mapped to a real price (> 0) are drawn; the entry
        # line uses a wider pen so the entry point stands out.
        if self._entry is not None and self._entry > 0:
            self._marker(painter, self._entry, y, pad_left, plot_w,
                         _MARKER_ENTRY, "Entry", width=2.0)
        if self._take_profit is not None and self._take_profit > 0:
            self._marker(painter, self._take_profit, y, pad_left, plot_w,
                         _MARKER_TP, "Take Profit")
        if self._stop is not None and self._stop > 0:
            self._marker(painter, self._stop, y, pad_left, plot_w,
                         _MARKER_SL, "Stop Loss")

        # Entry-point triangle: a green UP triangle for a buy/long signal, a
        # red DOWN triangle for a sell/short signal, sitting on the entry
        # price line at a comfortable left-of-centre position so the entry
        # point is obvious and the surrounding pattern stays readable.
        if self._entry is not None and self._entry > 0 and self._direction:
            self._entry_triangle(painter, self._entry, y, pad_left, plot_w,
                                 pad_top, plot_h)

        # Price-range hint
        painter.setPen(QColor(COLOR_TEXT_SEC))
        painter.drawText(int(pad_left), int(pad_top + plot_h + 14), f"{lo:.4f} — {hi:.4f}")

    def _entry_triangle(self, painter: QPainter, price: float, y_fn,
                        pad_left: float, plot_w: float,
                        pad_top: float, plot_h: float) -> None:
        """Draw a filled direction triangle at the entry price.

        ``buy``/``long`` -> green triangle pointing UP (apex at the entry).
        ``sell``/``short`` -> red triangle pointing DOWN (apex at the entry).
        Horizontally centred at ~18% of the plot width so it is clearly
        visible and does not collide with the right-hand marker labels.
        """
        d = (self._direction or "").lower()
        is_buy = d in ("buy", "long", "bullish", "0")
        colour = QColor(COLOR_POSITIVE if is_buy else COLOR_NEGATIVE)
        my = max(pad_top + 4, min(pad_top + plot_h - 4, y_fn(price)))
        # Horizontal centre: ~18% across the plot (near the entry bar for a
        # frame that was sliced to start at the entry bar).
        cx = pad_left + plot_w * 0.18
        size = 8.0
        pen = QPen(colour)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(colour)
        if is_buy:
            # Apex UP at the entry price; base below.
            poly = QPolygonF([
                QPointF(cx, my - size),
                QPointF(cx - size, my + size),
                QPointF(cx + size, my + size),
            ])
        else:
            # Apex DOWN at the entry price; base above.
            poly = QPolygonF([
                QPointF(cx, my + size),
                QPointF(cx - size, my - size),
                QPointF(cx + size, my - size),
            ])
        painter.drawPolygon(poly)

    def _marker(self, painter: QPainter, price: float, y_fn, pad_left: float,
                plot_w: float, hex_colour: str, label: str,
                width: float = 1.0) -> None:
        """Draw a dashed horizontal marker line + right-hand label."""
        my = y_fn(price)
        pen = QPen(QColor(hex_colour))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidthF(width)
        painter.setPen(pen)
        painter.drawLine(int(pad_left), int(my), int(pad_left + plot_w), int(my))
        painter.setPen(QColor(hex_colour))
        # Right-align the label to the plot's right edge: a long label (e.g.
        # "Take Profit") grows LEFTWARD from the edge instead of overflowing
        # past the widget's right border.  QRect y/margins are ints so the
        # alignment rect stays pixel-aligned.
        label_rect = QRect(int(pad_left), int(my - 4), int(plot_w), 16)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, label)


def _to_candles(candles: Any) -> list[dict[str, Any]]:
    """Normalise a DataFrame / list-of-dicts into an ordered OHLC list."""
    if candles is None:
        return []

    # pandas.DataFrame with an index and open/high/low/close columns.
    if hasattr(candles, "empty") and hasattr(candles, "iterrows"):
        if candles.empty:
            return []
        out: list[dict[str, Any]] = []
        for _ts, row in candles.iterrows():
            candle = _row_to_candle(_ts, row)
            if candle is not None:
                out.append(candle)
        return out

    # List of dicts.
    if isinstance(candles, (list, tuple)):
        return [c for c in candles if _valid_candle(c)]

    return []


def _row_to_candle(ts: Any, row: Any) -> dict[str, Any] | None:
    """Extract one OHLC candle from a DataFrame row, or None on bad data."""
    try:
        return {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
    except (TypeError, ValueError, KeyError):
        return None


def _valid_candle(c: Any) -> bool:
    """True when ``c`` is a dict with numeric OHLC values."""
    if not isinstance(c, dict):
        return False
    try:
        for key in ("open", "high", "low", "close"):
            float(c.get(key))
        return True
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    """Return ``value`` as float, or ``None`` when it is missing/invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
