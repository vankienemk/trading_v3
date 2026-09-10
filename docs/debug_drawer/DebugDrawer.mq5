//+------------------------------------------------------------------+
//|                                                  DebugDrawer.mq5 |
//|  Debug visualization for trading_v3 pattern events on MT5 chart  |
//|  Reads Files\debug_events.txt (pipe-delimited, one event/line)   |
//|  v2.0 bug-fix release (A1/A2/A4 of trading_v3_bug_summary):     |
//|   A1 label colour by model_prob, A2 trade zone = 2 CHART BARS,  |
//|   A4 label right-anchored (no longer covers the trade zone)    |
//|  drawing style:                                             |
//|   - pattern zone = yellow wash over the REAL pattern structure   |
//|     (fields 14..17: structure start/end/low/high; fallback:       |
//|      detect->confirm with the trade-zone span)                    |
//|   - trade zone   = confirm -> confirm+2h, SPLIT at entry: the    |
//|     side toward TARGET is light-green, the side toward STOP is   |
//|     light-red (BUY: green top/red bottom; SELL: green bottom/    |
//|     red top). No horizontal overlap between the two rectangles.  |
//|   - entry arrow + text annotation; rejected events in gray       |
//+------------------------------------------------------------------+
#property copyright "trading_v3 debug"
#property version   "2.00"
#property script_show_inputs

input string InpFileName  = "debug_events.txt";   // event file in MQL5\Files
input string InpPrefix    = "pat_";               // object name prefix (cleaned on rerun)
input string InpEchoFile  = "debug_events_echo.txt"; // diagnostic dump (written to MQL5\Files)

int    g_drawn  = 0;
int    g_skipped = 0;
double g_firstPrice = 0.0;
string g_firstTime  = "";
string g_firstRaw   = "";

//+------------------------------------------------------------------+
//| Delete every chart object whose name starts with InpPrefix       |
//+------------------------------------------------------------------+
void DeleteOldObjects(string prefix)
  {
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
     {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, prefix, 0) == 0)
         ObjectDelete(0, name);
     }
  }

//+------------------------------------------------------------------+
//| Carve field k from a pipe-delimited line (always returns a NEW   |
//| independent string via StringSubstr -> no shared buffers)        |
//+------------------------------------------------------------------+
string Field(string line, int k)
  {
   int pos = 0;
   for(int i = 0; i < k; i++)
     {
      int p = StringFind(line, "|", pos);
      if(p < 0)
         return ("");
      pos = p + 1;
     }
   int e = StringFind(line, "|", pos);
   if(e < 0)
      e = StringLen(line);
   return (StringSubstr(line, pos, e - pos));
  }

//+------------------------------------------------------------------+
//| Hex dump of the first maxlen characters (diagnostics only)       |
//+------------------------------------------------------------------+
string HexDump(string s, int maxlen)
  {
   string h = "";
   int N = StringLen(s);
   if(maxlen > 0 && N > maxlen)
      N = maxlen;
   for(int i = 0; i < N; i++)
      h += StringFormat("%02X ", (int)StringGetCharacter(s, i));
   return (h);
  }

//+------------------------------------------------------------------+
//| One filled rectangle (behind candles)                            |
//+------------------------------------------------------------------+
void FillRect(string name, datetime t1, datetime t2, double p1, double p2, color clr)
  {
   if(ObjectCreate(0, name, OBJ_RECTANGLE, 0, t1, p1, t2, p2))
     {
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, name, OBJPROP_FILL, 1);
      ObjectSetInteger(0, name, OBJPROP_BACK, 1);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
     }
  }

//+------------------------------------------------------------------+
//| Draw one event from '|'-separated fields                         |
//| 0 pattern | 1 direction | 2 symbol | 3 detect | 4 confirm |       |
//| 5 entry_t | 6 entry_p | 7 stop_p | 8 target_p | 9 rule_score |    |
//| 10 model_prob | 11 discard_reason | 12 level_price | 13 extreme  |
//+------------------------------------------------------------------+
void DrawEvent(string line)
  {
   static int s_uid = 0;
   s_uid++;

   string pattern   = Field(line, 0);
   string direction = Field(line, 1);
   string detect_s  = Field(line, 3);
   string confirm_s = Field(line, 4);
   string entry_t   = Field(line, 5);
   string entry_p   = Field(line, 6);
   string stop_p    = Field(line, 7);
   string target_p  = Field(line, 8);
   string discard   = Field(line, 11);
   string score     = Field(line, 9);
   string prob      = Field(line, 10);

   if(confirm_s == "" || entry_p == "")
     {
      Print("DebugDrawer: malformed line: ", line);
      return;
     }

   double eprice = StringToDouble(entry_p);
   double sprice = StringToDouble(stop_p);
   double tprice = StringToDouble(target_p);
   datetime t_confirm = StringToTime(confirm_s);
   datetime t_detect  = StringToTime(detect_s);
   datetime t_entry   = (entry_t != "" ? StringToTime(entry_t) : t_confirm);
   if(t_detect <= 0) t_detect = t_confirm;
   if(t_entry <= 0)  t_entry  = t_confirm;

   if(t_confirm <= 0 || eprice <= 0.0)
     {
      Print("DebugDrawer: unparsable event (t_confirm=", (string)t_confirm,
            " eprice=", DoubleToString(eprice, 2), ") confirm='", confirm_s,
            "' entry='", entry_p, "': ", line);
      g_skipped++;
      return;
     }

   bool isBuy = (StringFind(direction, "BUY", 0) >= 0 || StringFind(direction, "long", 0) >= 0);
   bool isRej = (discard != "");
   string uid = pattern + "_" + IntegerToString(s_uid) + "_" + IntegerToString(TimeCurrent());

   // --- price span of the trade zone ---
   double sp = (sprice > 0.0 ? sprice : eprice);
   double tp = (tprice > 0.0 ? tprice : eprice);
   double lo = MathMin(sp, MathMin(eprice, tp));
   double hi = MathMax(sp, MathMax(eprice, tp));
   double pad = (hi - lo) * 0.05;
   if(pad <= 0.0)
      pad = eprice * 0.001;
   lo -= pad;
   hi += pad;

   // --- 1) pattern zone: yellow wash over the REAL pattern structure ---
   //      fields 14..17 = structure start / end / low / high (optional).
   //      Fallback: detect -> confirm with the trade-zone price span.
   string pat_start_s = Field(line, 14);
   string pat_end_s   = Field(line, 15);
   string pat_low_s   = Field(line, 16);
   string pat_high_s  = Field(line, 17);

   datetime t1p = (pat_start_s != "" ? StringToTime(pat_start_s) : t_detect);
   datetime t2p = (pat_end_s   != "" ? StringToTime(pat_end_s)   : t_confirm);
   if(t1p <= 0) t1p = t_detect;
   if(t2p <= 0) t2p = t_confirm;
   if(t2p <= t1p) t2p = t1p + 900;

   double patLo = (pat_low_s  != "" ? StringToDouble(pat_low_s)  : lo);
   double patHi = (pat_high_s != "" ? StringToDouble(pat_high_s) : hi);
   if(patHi <= patLo)
     {
      patLo = lo;
      patHi = hi;
     }
   double patPad = (patHi - patLo) * 0.06;
   if(patPad <= 0.0)
      patPad = eprice * 0.0005;
   patLo -= patPad;
   patHi += patPad;

   FillRect(InpPrefix + "pat_" + uid, t1p, t2p, patLo, patHi,
            isRej ? clrDarkGray : clrYellow);

   // --- 2) trade zone: confirm -> confirm + 2 CHART BARS (A2: period-aware)
   //      GREEN toward target / RED toward stop
   datetime ta = t_confirm;
   int periodSec = PeriodSeconds();
   if(periodSec <= 0)
      periodSec = 15 * 60;                        // fallback: M15
   datetime tb = t_confirm + 2 * periodSec;
   if(tb <= ta) tb = ta + periodSec;

   // green band = between entry and target (profit side)
   double g1 = MathMin(eprice, tp);
   double g2 = MathMax(eprice, tp);
   if(g2 - g1 < pad)
     {
      g1 -= pad;
      g2 += pad;
     }
   FillRect(InpPrefix + "grn_" + uid, ta, tb, g1, g2,
            isRej ? clrDarkGray : clrLightGreen);

   // red band = between entry and stop (loss side)
   double r1 = MathMin(eprice, sp);
   double r2 = MathMax(eprice, sp);
   if(r2 - r1 < pad)
     {
      r1 -= pad;
      r2 += pad;
     }
   FillRect(InpPrefix + "red_" + uid, ta, tb, r1, r2,
            isRej ? clrDarkGray : clrLightCoral);

   // --- 3) entry arrow (green up / red down) ---
   color arrCol = isRej ? clrGray : (isBuy ? clrLimeGreen : clrRed);
   string aname = InpPrefix + "arr_" + uid;
   if(ObjectCreate(0, aname, isBuy ? OBJ_ARROW_UP : OBJ_ARROW_DOWN, 0, t_entry, eprice))
     {
      ObjectSetInteger(0, aname, OBJPROP_COLOR, arrCol);
      ObjectSetInteger(0, aname, OBJPROP_WIDTH, 2);
      ObjectSetInteger(0, aname, OBJPROP_ANCHOR, ANCHOR_TOP);
     }

   // --- 4) annotation text at entry ---------------------------------
   //      A1: text colour encodes model_prob quality (low prob = warning)
   //      A4: right-anchored so the label never covers the trade zone
   double pNum = (prob != "" ? StringToDouble(prob) : -1.0);
   color txtCol = isRej ? clrGray
                : (pNum < 0.0 ? clrWhite
                : (pNum < 0.40 ? clrOrangeRed
                : (pNum < 0.60 ? clrYellow
                               : (isBuy ? clrLimeGreen : clrRed))));

   string label = pattern + " " + direction;
   if(prob != "")     label += " | p=" + prob;
   if(score != "")    label += " | s=" + score;
   if(isRej)          label += " | REJECTED:" + discard;
   string tname = InpPrefix + "txt_" + uid;
   if(ObjectCreate(0, tname, OBJ_TEXT, 0, t_entry, eprice))
     {
      ObjectSetString(0, tname, OBJPROP_TEXT, label);
      ObjectSetInteger(0, tname, OBJPROP_COLOR, txtCol);
      ObjectSetInteger(0, tname, OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, tname, OBJPROP_ANCHOR, ANCHOR_RIGHT_UPPER);
     }

   g_drawn++;
   if(g_firstPrice == 0.0)
     {
      g_firstPrice = eprice;
      g_firstTime  = TimeToString(t_confirm, TIME_DATE | TIME_MINUTES);
      g_firstRaw   = StringSubstr(line, 0, StringLen(line));
     }
  }

//+------------------------------------------------------------------+
//| Per-line diagnostic (fields carved with Field(), no arrays)      |
//+------------------------------------------------------------------+
string EchoLine(string line, int num)
  {
   string f4raw   = Field(line, 4);
   string f6raw   = Field(line, 6);
   string f4trimL = StringTrimLeft(f4raw);
   string f4trimR = StringTrimRight(f4trimL);
   string substr8 = StringSubstr(line, 0, 8);
   string e  = StringFormat("LINE%d len=%d hex40=%s\n", num, StringLen(line), HexDump(line, 40));
   e += "  f4_raw='" + f4raw + "' builtin_t='" + TimeToString(StringToTime(f4raw), TIME_DATE | TIME_MINUTES) + "'\n";
   e += "  f6_raw='" + f6raw + "' builtin_d='" + DoubleToString(StringToDouble(f6raw), 2) + "'\n";
   e += "  substr8='" + substr8 + "' trimL='" + f4trimL + "' trimLR='" + f4trimR + "'\n";
   return (e);
  }

//+------------------------------------------------------------------+
void WriteEcho(string content)
  {
   int h = FileOpen(InpEchoFile, FILE_WRITE | FILE_TXT | FILE_ANSI, 0, "");
   if(h != INVALID_HANDLE)
     {
      FileWriteString(h, content);
      FileClose(h);
      Print("DebugDrawer: echo -> Files\\", InpEchoFile);
     }
   else
      Print("DebugDrawer: cannot write echo (err=", GetLastError(), ")");
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   double pmin = ChartGetDouble(0, CHART_PRICE_MIN, 0);
   double pmax = ChartGetDouble(0, CHART_PRICE_MAX, 0);
   double bid  = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   int h = FileOpen(InpFileName, FILE_READ | FILE_TXT | FILE_ANSI, 0, "");
   if(h == INVALID_HANDLE)
     {
      string msg = "DebugDrawer v2.0: KHONG mo duoc file Files\\" + InpFileName;
      Print(msg);
      Comment(msg);
      Alert(msg);
      return;
     }

   DeleteOldObjects(InpPrefix);

   string echo = "DebugDrawer v2.0 diagnostic t=" + TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES) + "\n";
   echo += "BUILTIN literal '2026.09.10 00:45' -> " + TimeToString(StringToTime("2026.09.10 00:45"), TIME_DATE | TIME_MINUTES) + "\n";
   echo += "BUILTIN literal '4475.60' -> " + DoubleToString(StringToDouble("4475.60"), 2) + "\n";

   int lineNum = 0;
   while(!FileIsEnding(h))
     {
      string line = FileReadString(h);
      if(StringLen(line) > 0)
        {
         if(g_firstRaw == "")
            g_firstRaw = StringSubstr(line, 0, StringLen(line));
         lineNum++;
         echo += EchoLine(line, lineNum);
         DrawEvent(line);
        }
     }
   FileClose(h);

   WriteEcho(echo);

   ChartNavigate(0, CHART_END, 0);
   ChartRedraw(0);

   string diag = StringFormat(
      "DebugDrawer v2.0: drawn=%d skipped=%d (chart %s bid=%.2f)\n"
      "first: %s | t=%s @ %.2f\nvisible price: %.2f .. %.2f\n"
      "diag -> Files\\%s",
      g_drawn, g_skipped, _Symbol, bid,
      (g_firstRaw == "" ? "(none)" : g_firstRaw),
      (g_firstTime == "" ? "-" : g_firstTime), g_firstPrice,
      pmin, pmax, InpEchoFile);
   Print(diag);
   Comment(diag);
  }
//+------------------------------------------------------------------+