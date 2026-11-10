document.getElementById("scan-btn").addEventListener("click", async () => {
  const input = document.getElementById("ticker-input").value.trim().toUpperCase();
  const ticker = input.endsWith(".NS") ? input : `${input}.NS`;
  const resultEl = document.getElementById("result");

  resultEl.style.display = "block";
  resultEl.innerHTML = "Scanning live order book & patterns...";

  try {
    const res = await fetch(`http://127.0.0.1:8000/api/v1/copilot/overlay/${ticker}`);
    if (!res.ok) throw new Error("API error");
    const data = await res.json();

    const color = data.direction === "UP" ? "#34d399" : (data.direction === "DOWN" ? "#f87171" : "#94a3b8");
    resultEl.innerHTML = `
      <div style="font-weight: bold; color: ${color}; margin-bottom: 4px;">
        Signal: ${data.direction} (${data.confidence_pct}% Win Prob)
      </div>
      <div><strong>Current:</strong> ₹${data.current_price}</div>
      <div><strong>Target 1:</strong> ₹${data.target_1}</div>
      <div><strong>Stop Loss:</strong> ₹${data.stop_loss}</div>
      <div style="margin-top: 4px; color: #94a3b8;"><strong>Pattern:</strong> ${data.pattern_detected}</div>
    `;
  } catch (e) {
    resultEl.innerHTML = `<span style="color:#f87171;">Engine offline or connection error. Ensure FastAPI is running on port 8000.</span>`;
  }
});
