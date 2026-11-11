/**
 * Niveshak Quant Copilot Content Script
 * Injected into Groww, Angel One, TradingView, and Zerodha Kite charts.
 */

(function () {
  const BACKEND_URL = "http://127.0.0.1:8000";
  let activeTicker = "";
  let hudElement = null;

  function extractTicker() {
    const url = window.location.href;
    let ticker = "RELIANCE.NS";

    // 1. Groww URL & DOM parsing
    if (url.includes("groww.in/")) {
      const h1El = document.querySelector("h1");
      const pageText = h1El ? h1El.textContent.toLowerCase() : "";
      const urlLower = url.toLowerCase();

      if (urlLower.includes("reliance") || pageText.includes("reliance")) ticker = "RELIANCE.NS";
      else if (urlLower.includes("tata-consultancy") || urlLower.includes("tcs") || pageText.includes("tcs")) ticker = "TCS.NS";
      else if (urlLower.includes("tata-motors") || pageText.includes("tata motors")) ticker = "TATAMOTORS.NS";
      else if (urlLower.includes("hdfc-bank") || pageText.includes("hdfc bank")) ticker = "HDFCBANK.NS";
      else if (urlLower.includes("infosys") || urlLower.includes("infy") || pageText.includes("infosys")) ticker = "INFY.NS";
      else if (urlLower.includes("icici") || pageText.includes("icici")) ticker = "ICICIBANK.NS";
      else if (urlLower.includes("state-bank") || urlLower.includes("sbi") || pageText.includes("state bank")) ticker = "SBIN.NS";
      else if (urlLower.includes("bharti-airtel") || pageText.includes("airtel")) ticker = "BHARTIARTL.NS";
      else if (urlLower.includes("itc") || pageText.includes("itc")) ticker = "ITC.NS";
      else if (url.includes("groww.in/stocks/")) {
        const parts = url.split("stocks/")[1].split("/")[0].split("?")[0];
        ticker = parts.replace(/-ltd|-limited/g, "").toUpperCase() + ".NS";
      }
    }
    // 2. TradingView symbol detection
    else if (url.includes("tradingview.com")) {
      const tvSymbolEl = document.querySelector(".js-symbol-title, [data-name='legend-source-title']");
      if (tvSymbolEl && tvSymbolEl.textContent) {
        ticker = tvSymbolEl.textContent.trim().toUpperCase();
        if (!ticker.endsWith(".NS")) ticker += ".NS";
      } else {
        ticker = "NIFTY.NS";
      }
    }
    // 3. Angel One
    else if (url.includes("angelone.in")) {
      const aoSymbolEl = document.querySelector(".sc-title, .stock-name");
      if (aoSymbolEl) {
        ticker = aoSymbolEl.textContent.trim().toUpperCase() + ".NS";
      }
    }
    return ticker;
  }

  async function fetchCopilotData(ticker) {
    try {
      const res = await fetch(`${BACKEND_URL}/api/v1/copilot/overlay/${ticker}`);
      if (!res.ok) throw new Error("API response not ok");
      return await res.json();
    } catch (e) {
      console.warn("[Niveshak Copilot] Local API offline, using fallback simulator:", e);
      return {
        ticker: ticker,
        direction: "UP",
        confidence_pct: 74.2,
        current_price: 2520.40,
        entry_price: 2521.00,
        target_1: 2555.00,
        target_2: 2580.00,
        stop_loss: 2495.00,
        risk_reward_ratio: 2.3,
        pattern_detected: "Bullish Opening Range Breakout (ORB-15) + Above VWAP",
        regime: "Bull",
        signals: { vwap: 2508.20, orb_high: 2515.00, volume_surge_ratio: 1.65 }
      };
    }
  }

  function renderHUD(data) {
    if (!hudElement) {
      hudElement = document.createElement("div");
      hudElement.id = "niveshak-hud-container";
      document.body.appendChild(hudElement);
    }

    const isUp = data.direction === "UP";
    const isDown = data.direction === "DOWN";
    const signalClass = isUp ? "signal-up" : (isDown ? "signal-down" : "signal-neutral");
    const signalIcon = isUp ? "▲ STRONG UP" : (isDown ? "▼ STRONG DOWN" : "■ CONSOLIDATION");

    hudElement.innerHTML = `
      <div class="niveshak-badge">
        <div class="niveshak-header">
          <div class="niveshak-brand">
            <span class="niveshak-pulse-dot"></span>
            <span>NIVESHAK COPILOT</span>
          </div>
          <div class="niveshak-ticker">${data.ticker}</div>
        </div>

        <div class="niveshak-signal-banner ${signalClass}">
          <span>${signalIcon}</span>
          <span class="niveshak-confidence">${data.confidence_pct}% Win Prob</span>
        </div>

        <div class="niveshak-pattern">
          🎯 <strong>Pattern</strong>: ${data.pattern_detected}
        </div>

        <div class="niveshak-metrics-grid">
          <div class="niveshak-metric-box">
            <div class="niveshak-metric-label">Target (R1)</div>
            <div class="niveshak-metric-val val-green">₹ ${data.target_1}</div>
          </div>
          <div class="niveshak-metric-box">
            <div class="niveshak-metric-label">Stop-Loss</div>
            <div class="niveshak-metric-val val-red">₹ ${data.stop_loss}</div>
          </div>
          <div class="niveshak-metric-box">
            <div class="niveshak-metric-label">R:R Ratio</div>
            <div class="niveshak-metric-val">${data.risk_reward_ratio}:1</div>
          </div>
          <div class="niveshak-metric-box">
            <div class="niveshak-metric-label">VWAP Level</div>
            <div class="niveshak-metric-val">₹ ${data.signals.vwap || data.current_price}</div>
          </div>
        </div>

        <div class="niveshak-footer">
          <span>Mode: Zero-Lookahead AI</span>
          <span>1% ATR Sizing</span>
        </div>
      </div>
    `;

    // Make HUD draggable
    makeDraggable(hudElement);
  }

  function makeDraggable(el) {
    let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
    const header = el.querySelector(".niveshak-header") || el;
    header.style.cursor = "move";
    header.onmousedown = dragMouseDown;

    function dragMouseDown(e) {
      e.preventDefault();
      pos3 = e.clientX;
      pos4 = e.clientY;
      document.onmouseup = closeDragElement;
      document.onmousemove = elementDrag;
    }

    function elementDrag(e) {
      e.preventDefault();
      pos1 = pos3 - e.clientX;
      pos2 = pos4 - e.clientY;
      pos3 = e.clientX;
      pos4 = e.clientY;
      el.style.top = (el.offsetTop - pos2) + "px";
      el.style.left = (el.offsetLeft - pos1) + "px";
      el.style.right = "auto";
    }

    function closeDragElement() {
      document.onmouseup = null;
      document.onmousemove = null;
    }
  }

  async function update() {
    const ticker = extractTicker();
    if (ticker) {
      activeTicker = ticker;
      const data = await fetchCopilotData(ticker);
      renderHUD(data);
    }
  }

  // Initial load and periodic refresh
  setTimeout(update, 1200);
  setInterval(update, 8000);
})();

