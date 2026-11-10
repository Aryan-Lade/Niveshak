# Niveshak Quant Copilot — Chrome Extension (Manifest V3)

A real-time AI trading copilot that injects live **UP / DOWN** predictions, target levels, and stop-loss HUDs directly on **Groww**, **Angel One**, **TradingView**, and **Zerodha Kite** candlestick charts.

---

## 🚀 Quick Setup (Install in Chrome in 30 Seconds)

1. Open Google Chrome and navigate to: `chrome://extensions/`
2. Toggle on **"Developer mode"** in the top right corner.
3. Click **"Load unpacked"** in the top left.
4. Select this directory: `F:\03. Placements\02. Project\Niveshak\extension`
5. Done! The Niveshak Quant Copilot extension is now active.

---

## 🎯 How It Works

1. Open any stock chart on:
   - **Groww**: `https://groww.in/stocks/reliance-industries-ltd`
   - **TradingView**: `https://www.tradingview.com/chart/`
   - **Angel One**: `https://trade.angelone.in/`
   - **Zerodha Kite**: `https://kite.zerodha.com/`
2. The extension automatically detects the active ticker from the URL/chart DOM.
3. It queries your local Niveshak engine (`http://127.0.0.1:8000/api/v1/copilot/overlay/{ticker}`).
4. A floating, draggable **Glassmorphism HUD Badge** appears right on the chart displaying:
   - 🟢 **STRONG UP** / 🔴 **STRONG DOWN**
   - Win Probability % (e.g. `74%`)
   - Target 1 (R1) & Stop-Loss (1% ATR Risk)
   - Real-time VWAP and Opening Range Breakout status.
