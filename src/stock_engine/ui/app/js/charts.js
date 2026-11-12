// Charts for the terminal, drawn as SVG by hand. Colours come from CSS
// classes so both themes work; direction is always doubled with a sign,
// an arrow or a solid/hollow marker so it never rests on colour alone.

import { raw, html, esc, clamp, pct, num, inr, showTip, hideTip } from './util.js';

const f1 = (n) => Number(n).toFixed(1);

const BUILDUP_UP = new Set(['long_buildup', 'short_covering']);
const BUILDUP_FRESH = new Set(['long_buildup', 'short_buildup']);

export function labelToMinute(label) {
  const [h, m] = label.split(':').map(Number);
  return h * 60 + m - 555; // 09:15 is minute 0
}

export function minuteToLabel(minute) {
  const total = 555 + Math.round(minute);
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

function niceStep(span, target = 4) {
  const rough = span / target;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const unit = rough / pow;
  return (unit >= 5 ? 5 : unit >= 2 ? 2 : 1) * pow;
}

/** Push labels apart vertically so none overlap. Mutates and returns the list (each item has .y). */
function spreadLabels(items, minGap, top, bottom) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) {
    if (items[i].y - items[i - 1].y < minGap) items[i].y = items[i - 1].y + minGap;
  }
  const overflow = items.length ? items[items.length - 1].y - bottom : 0;
  if (overflow > 0) {
    for (let i = items.length - 1; i >= 0; i--) {
      items[i].y -= overflow;
      if (i > 0 && items[i].y - items[i - 1].y >= minGap) break;
    }
    for (let i = items.length - 2; i >= 0; i--) {
      if (items[i + 1].y - items[i].y < minGap) items[i].y = items[i + 1].y - minGap;
    }
  }
  for (const item of items) item.y = Math.max(top, item.y);
  return items;
}

// ---------------------------------------------------------------- sparkline

/**
 * Small line of a stock's % move since the previous close.
 * `band` is an optional [low, high] zone in the same % units.
 */
export function sparkline(values, { width = 96, height = 28, band = null } = {}) {
  if (!values || values.length < 2) {
    return raw(`<svg class="spark" width="${width}" height="${height}" aria-hidden="true"></svg>`);
  }
  let lo = Math.min(0, ...values);
  let hi = Math.max(0, ...values);
  if (band) { lo = Math.min(lo, band[0]); hi = Math.max(hi, band[1]); }
  if (hi - lo < 0.2) { hi += 0.1; lo -= 0.1; }
  const pad = 3;
  const x = (i) => pad + (i / (values.length - 1)) * (width - 2 * pad);
  const y = (v) => pad + (1 - (v - lo) / (hi - lo)) * (height - 2 * pad);
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${f1(x(i))},${f1(y(v))}`).join('');
  const last = values[values.length - 1];
  const dir = last > 0.05 ? 'up' : last < -0.05 ? 'down' : 'flat';
  const bandRect = band
    ? `<rect class="spark-band band-${dir}" x="${pad}" y="${f1(y(band[1]))}" width="${width - 2 * pad}" height="${f1(Math.max(1.5, y(band[0]) - y(band[1])))}"/>`
    : '';
  return raw(
    `<svg class="spark" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true">` +
    bandRect +
    `<line class="spark-zero" x1="${pad}" x2="${width - pad}" y1="${f1(y(0))}" y2="${f1(y(0))}"/>` +
    `<path class="spark-line s-${dir}" d="${d}"/>` +
    `<circle class="spark-dot f-${dir}" cx="${f1(x(values.length - 1))}" cy="${f1(y(last))}" r="2.300"/></svg>`
  );
}

// ---------------------------------------------------------------- breadth histogram

export function mountHistogram(el, bins) {
  const W = Math.max(280, el.clientWidth);
  const H = 156;
  const top = 20, bottom = 26;
  const gap = 3;
  const n = bins.length;
  const bw = (W - gap * (n - 1)) / n;
  const base = H - bottom;
  const max = Math.max(1, ...bins.map((b) => b.count));
  const edges = ['−3', '−2', '−1', '−0.5', '0', '+0.5', '+1', '+2', '+3'];

  let out = '';
  bins.forEach((bin, i) => {
    const x = i * (bw + gap);
    const h = bin.count ? Math.max(3, (bin.count / max) * (base - top)) : 0;
    const y = base - h;
    const r = Math.min(4, h, bw / 2);
    if (h > 0) {
      out += `<path class="hist-bar f-${bin.side}" d="M${f1(x)},${base}V${f1(y + r)}Q${f1(x)},${f1(y)} ${f1(x + r)},${f1(y)}H${f1(x + bw - r)}Q${f1(x + bw)},${f1(y)} ${f1(x + bw)},${f1(y + r)}V${base}Z"/>`;
    }
    out += `<text class="hist-count" x="${f1(x + bw / 2)}" y="${f1(y - 5)}" text-anchor="middle">${bin.count}</text>`;
    out += `<rect class="hit" data-i="${i}" x="${f1(x)}" y="0" width="${f1(bw + gap)}" height="${H}"/>`;
  });
  edges.forEach((label, i) => {
    const x = (i + 1) * (bw + gap) - gap / 2;
    out += `<text class="axis-label" x="${f1(x)}" y="${H - 8}" text-anchor="middle">${label}</text>`;
  });
  const zeroX = 5 * (bw + gap) - gap / 2;
  out += `<line class="axis-zero" x1="${f1(zeroX)}" x2="${f1(zeroX)}" y1="${top - 12}" y2="${base}"/>`;
  out += `<line class="axis-line" x1="0" x2="${W}" y1="${base}" y2="${base}"/>`;

  el.innerHTML = `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Stocks grouped by percent change from the previous close">${out}</svg>`;
  el.onpointermove = (event) => {
    const hit = event.target.closest('.hit');
    if (!hit) return hideTip();
    const bin = bins[Number(hit.dataset.i)];
    showTip(html`<strong>${bin.count} stocks</strong><span>${bin.label}</span>`, event.clientX, event.clientY);
  };
  el.onpointerleave = hideTip;
}

// ---------------------------------------------------------------- session ribbon

/**
 * The trading day as one strip: advancing stocks above the midline,
 * declining below, a needle at the time being shown. Drag (or arrow keys)
 * to replay the session.
 */
export function mountRibbon(el, { timeline, meta }, onScrub) {
  const W = Math.max(240, el.clientWidth);
  const H = 56;
  const padX = 8;
  const total = meta.session_minutes;
  const mid = 24;
  const amp = 19;
  const scale = meta.universe * 0.8;
  const x = (minute) => padX + (minute / total) * (W - 2 * padX);
  const minutes = timeline.times.map(labelToMinute);
  const live = meta.live_minute;

  let up = `M${f1(x(0))},${mid}`;
  let down = `M${f1(x(0))},${mid}`;
  minutes.forEach((m, i) => {
    if (i === 0) {
      up += `L${f1(x(0))},${f1(mid - (timeline.advancing[0] / scale) * amp)}`;
      down += `L${f1(x(0))},${f1(mid + (timeline.declining[0] / scale) * amp)}`;
    }
    up += `L${f1(x(m))},${f1(mid - (timeline.advancing[i] / scale) * amp)}`;
    down += `L${f1(x(m))},${f1(mid + (timeline.declining[i] / scale) * amp)}`;
  });
  up += `L${f1(x(live))},${mid}Z`;
  down += `L${f1(x(live))},${mid}Z`;

  let ticks = '';
  // Hour labels thin out on narrow ribbons so they never run into each other or the end labels.
  const labelled = W >= 520 ? [10, 11, 12, 13, 14, 15] : W >= 300 ? [11, 13] : [12];
  for (let hour = 10; hour <= 15; hour++) {
    const m = hour * 60 - 555;
    ticks += `<line class="rib-tick" x1="${f1(x(m))}" x2="${f1(x(m))}" y1="${mid - amp}" y2="${mid + amp}"/>`;
    if (labelled.includes(hour)) ticks += `<text class="rib-label" x="${f1(x(m))}" y="${H - 2}" text-anchor="middle">${hour}:00</text>`;
  }
  ticks += `<text class="rib-label" x="${padX}" y="${H - 2}">09:15</text>` +
    `<text class="rib-label" x="${W - padX}" y="${H - 2}" text-anchor="end">15:30</text>`;

  const needleX = x(meta.minute);
  el.innerHTML =
    `<svg class="ribbon" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" tabindex="0" role="slider" ` +
    `aria-label="Session time. Drag or use arrow keys to replay." aria-valuemin="5" aria-valuemax="${live}" ` +
    `aria-valuenow="${meta.minute}" aria-valuetext="${esc(meta.as_of)}">` +
    `<defs>` +
    `<pattern id="rib-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line class="rib-hatch" x1="0" y1="0" x2="0" y2="6"/></pattern>` +
    `<clipPath id="rib-past"><rect x="0" y="0" width="${f1(needleX)}" height="${H}"/></clipPath>` +
    `<clipPath id="rib-ahead"><rect x="${f1(needleX)}" y="0" width="${f1(W - needleX)}" height="${H}"/></clipPath>` +
    `</defs>` +
    `<rect class="rib-track" x="${padX}" y="${mid - amp}" width="${W - 2 * padX}" height="${amp * 2}" rx="4"/>` +
    (live < total ? `<rect class="rib-future" x="${f1(x(live))}" y="${mid - amp}" width="${f1(x(total) - x(live))}" height="${amp * 2}" fill="url(#rib-hatch)"/>` : '') +
    ticks +
    `<g clip-path="url(#rib-past)"><path class="rib-area f-up" d="${up}"/><path class="rib-area f-down" d="${down}"/></g>` +
    `<g clip-path="url(#rib-ahead)" class="rib-dim"><path class="rib-area f-up" d="${up}"/><path class="rib-area f-down" d="${down}"/></g>` +
    `<line class="rib-mid" x1="${padX}" x2="${W - padX}" y1="${mid}" y2="${mid}"/>` +
    `<g class="rib-needle" transform="translate(${f1(needleX)},0)"><line x1="0" x2="0" y1="${mid - amp - 3}" y2="${mid + amp + 3}"/><circle cx="0" cy="${mid - amp - 3}" r="4"/></g>` +
    `</svg>`;

  const svg = el.firstChild;
  const needle = svg.querySelector('.rib-needle');
  const state = { dragging: false };

  const minuteAt = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const position = ((clientX - rect.left - padX) / (rect.width - 2 * padX)) * total;
    return clamp(Math.round(position / 5) * 5, 5, live);
  };
  const moveNeedle = (minute) => needle.setAttribute('transform', `translate(${f1(x(minute))},0)`);
  const commit = (minute, final) => onScrub(minute >= live - 2 ? null : minute, final);

  svg.addEventListener('pointerdown', (event) => {
    state.dragging = true;
    svg.setPointerCapture(event.pointerId);
    const minute = minuteAt(event.clientX);
    moveNeedle(minute);
    commit(minute, false);
  });
  svg.addEventListener('pointermove', (event) => {
    const minute = minuteAt(event.clientX);
    if (state.dragging) {
      moveNeedle(minute);
      commit(minute, false);
    }
    const i = Math.max(0, minutes.findIndex((m) => m >= minute));
    showTip(
      html`<strong>${minuteToLabel(minute)}</strong><span>${timeline.advancing[i]} up, ${timeline.declining[i]} down</span><span class="tip-hint">${state.dragging ? 'Release to stay here' : 'Drag to replay'}</span>`,
      event.clientX, event.clientY
    );
  });
  const end = (event) => {
    if (!state.dragging) return;
    state.dragging = false;
    commit(minuteAt(event.clientX), true);
  };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', end);
  svg.addEventListener('pointerleave', hideTip);
  let cursor = meta.minute;   // repeated key presses add up even before the next refresh lands
  svg.addEventListener('keydown', (event) => {
    const step = { ArrowLeft: -5, ArrowRight: 5, PageDown: -30, PageUp: 30 }[event.key];
    if (step) cursor = clamp(Math.round((cursor + step) / 5) * 5, 5, live);
    else if (event.key === 'Home') cursor = 5;
    else if (event.key === 'End') cursor = live;
    else return;
    event.preventDefault();
    moveNeedle(cursor);
    state.dragging = true;      // hold the ribbon in place while keys are still coming
    commit(cursor, false);
    clearTimeout(state.keyTimer);
    state.keyTimer = setTimeout(() => commit(cursor, true), 350);
  });
  return state;
}

// ---------------------------------------------------------------- sector lines

/** All sectors as context lines; the three strongest and three weakest are emphasised and labelled. */
export function mountSectorLines(el, sectors, meta) {
  const W = Math.max(300, el.clientWidth);
  const H = 236;
  const m = { top: 12, right: 132, bottom: 24, left: 40 };
  const plotW = W - m.left - m.right;
  const plotH = H - m.top - m.bottom;
  const lastIdx = sectors[0].path.length - 1;

  const all = sectors.flatMap((s) => s.path);
  let lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const padY = Math.max(0.1, (hi - lo) * 0.1);
  lo -= padY; hi += padY;
  // index 0 is the open (minute 0); index k is the close of bar k, and the last bar may be partial
  const xi = (i) => m.left + (Math.min(i * meta.bar_minutes, meta.minute) / meta.session_minutes) * plotW;
  const y = (v) => m.top + (1 - (v - lo) / (hi - lo)) * plotH;

  const focus = new Set([...sectors.slice(0, 3), ...sectors.slice(-3)].map((s) => s.name));
  const step = niceStep(hi - lo, 4);
  let grid = '';
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const value = Math.abs(v) < 1e-9 ? 0 : v;
    grid += `<line class="${value === 0 ? 'axis-zero' : 'grid-line'}" x1="${m.left}" x2="${m.left + plotW}" y1="${f1(y(value))}" y2="${f1(y(value))}"/>` +
      `<text class="axis-label" x="${m.left - 6}" y="${f1(y(value) + 4)}" text-anchor="end">${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(step < 1 ? 1 : 0)}%</text>`;
  }
  for (const label of ['09:15', '11:00', '13:00', '15:30']) {
    const minute = labelToMinute(label);
    const anchor = minute === 0 ? 'start' : minute === meta.session_minutes ? 'end' : 'middle';
    grid += `<text class="axis-label" x="${f1(m.left + (minute / meta.session_minutes) * plotW)}" y="${H - 6}" text-anchor="${anchor}">${label}</text>`;
  }

  const pathOf = (s) => s.path.map((v, i) => `${i ? 'L' : 'M'}${f1(xi(i))},${f1(y(v))}`).join('');
  let lines = '';
  for (const s of sectors) {
    if (!focus.has(s.name)) lines += `<path class="sl sl-ctx" data-name="${esc(s.name)}" d="${pathOf(s)}"/>`;
  }
  const labels = [];
  for (const s of sectors) {
    if (!focus.has(s.name)) continue;
    const dir = s.change_pct > 0.05 ? 'up' : s.change_pct < -0.05 ? 'down' : 'flat';
    lines += `<path class="sl sl-halo" d="${pathOf(s)}"/><path class="sl sl-focus s-${dir}" data-name="${esc(s.name)}" d="${pathOf(s)}"/>`;
    labels.push({ s, dir, y: y(s.path[lastIdx]), anchorY: y(s.path[lastIdx]) });
  }
  spreadLabels(labels, 15, m.top + 6, H - m.bottom - 2);
  const endX = xi(lastIdx);
  const labelX = m.left + plotW + 14;
  let labelOut = '';
  for (const { s, dir, y: ly, anchorY } of labels) {
    labelOut +=
      `<path class="sl-leader" d="M${f1(endX + 3)},${f1(anchorY)}L${f1(labelX - 6)},${f1(ly)}"/>` +
      `<circle class="f-${dir}" cx="${f1(endX)}" cy="${f1(anchorY)}" r="3.200"/>` +
      `<text class="sl-label" x="${labelX}" y="${f1(ly + 4)}">${esc(s.name)} <tspan class="sl-value">${pct(s.change_pct)}</tspan></text>`;
  }

  el.innerHTML =
    `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Sector moves through the session">` +
    grid + lines + labelOut +
    `<line class="crosshair" x1="0" x2="0" y1="${m.top}" y2="${m.top + plotH}" visibility="hidden"/>` +
    `<rect class="hit" x="${m.left}" y="${m.top}" width="${plotW}" height="${plotH}"/></svg>`;

  const svg = el.firstChild;
  const cross = svg.querySelector('.crosshair');
  const focusList = sectors.filter((s) => focus.has(s.name));
  svg.querySelector('.hit').addEventListener('pointermove', (event) => {
    const rect = svg.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (W / rect.width);
    const minute = ((px - m.left) / plotW) * meta.session_minutes;
    const i = clamp(Math.round(minute / meta.bar_minutes), 0, lastIdx);
    cross.setAttribute('x1', f1(xi(i)));
    cross.setAttribute('x2', f1(xi(i)));
    cross.setAttribute('visibility', 'visible');
    const rows = [...focusList].sort((a, b) => b.path[i] - a.path[i]);
    showTip(
      html`<strong>${i === lastIdx ? meta.as_of : minuteToLabel(i * meta.bar_minutes)}</strong>${rows.map(
        (s) => html`<span class="tip-row"><i class="key f-${s.path[i] > 0.05 ? 'up' : s.path[i] < -0.05 ? 'down' : 'flat'}"></i>${s.name}<b>${pct(s.path[i])}</b></span>`
      )}`,
      event.clientX, event.clientY
    );
  });
  svg.querySelector('.hit').addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    hideTip();
  });

  return {
    /** Emphasise one sector by name (from the lists beside the chart); null clears it. */
    highlight(name) {
      svg.classList.toggle('has-pick', Boolean(name));
      for (const path of svg.querySelectorAll('.sl[data-name]')) path.classList.toggle('pick', path.dataset.name === name);
    },
  };
}

// ---------------------------------------------------------------- market strip

/** Every stock as one dot, placed by its % change and stacked where they crowd. */
export function mountStrip(el, stocks, sectors, onOpen) {
  const W = Math.max(320, el.clientWidth);
  const H = 204;
  const m = { top: 48, bottom: 26, left: 10, right: 10 };
  const plotW = W - m.left - m.right;
  const base = H - m.bottom;
  const limit = 4.5;
  const x = (v) => m.left + ((clamp(v, -limit, limit) + limit) / (2 * limit)) * plotW;
  const pitch = 9;
  const sorted = [...stocks].sort((a, b) => a.change_pct - b.change_pct);
  const stacks = new Map();
  const placed = sorted.map((s) => {
    const bin = Math.round((x(s.change_pct) - m.left) / pitch);
    const level = stacks.get(bin) || 0;
    stacks.set(bin, level + 1);
    return { s, bin, level };
  });
  const tallest = Math.max(1, ...stacks.values());
  const pitchY = Math.min(pitch, (base - m.top - 6) / tallest);

  let dots = '';
  placed.forEach(({ s, bin, level }, i) => {
    const dir = s.change_pct > 0.05 ? 'up' : s.change_pct < -0.05 ? 'down' : 'flat';
    const fresh = BUILDUP_FRESH.has(s.buildup);
    dots += `<circle class="dot ${fresh ? 'f-' + dir : 'o-' + dir}${s.bias === 'choppy' ? ' dot-quiet' : ''}" data-i="${i}" ` +
      `cx="${f1(m.left + bin * pitch)}" cy="${f1(base - 6 - level * pitchY)}" r="3.500"/>`;
  });

  let axis = `<line class="axis-line" x1="${m.left}" x2="${W - m.right}" y1="${base}" y2="${base}"/>`;
  for (let v = -4; v <= 4; v++) {
    axis += `<line class="${v === 0 ? 'axis-zero' : 'axis-tick'}" x1="${f1(x(v))}" x2="${f1(x(v))}" y1="${v === 0 ? m.top - 4 : base}" y2="${base + 4}"/>` +
      `<text class="axis-label" x="${f1(x(v))}" y="${H - 6}" text-anchor="middle">${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)}%</text>`;
  }

  // The two flags sit on separate rows so their labels can never overlap, however close the sectors are.
  const flag = (sector, word, row) => {
    const fx = x(sector.change_pct);
    const anchor = fx > W - 190 ? 'end' : fx < 190 ? 'start' : 'middle';
    const dir = sector.change_pct > 0.05 ? 'up' : sector.change_pct < -0.05 ? 'down' : 'flat';
    const ty = 12 + row * 16;
    return `<line class="flag-stem s-${dir}" x1="${f1(fx)}" x2="${f1(fx)}" y1="${ty + 5}" y2="${base}"/>` +
      `<text class="flag-label" x="${f1(fx)}" y="${ty}" text-anchor="${anchor}">${word} sector: ${esc(sector.name)} ${pct(sector.change_pct)}</text>`;
  };
  const flags = sectors.length > 1 ? flag(sectors[0], 'Leading', 0) + flag(sectors[sectors.length - 1], 'Lagging', 1) : '';

  el.innerHTML =
    `<svg class="chart strip" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Every stock placed by its percent change">` +
    flags + axis + dots + `</svg>`;
  const svg = el.firstChild;
  svg.addEventListener('pointermove', (event) => {
    const dot = event.target.closest('.dot');
    if (!dot) return hideTip();
    const s = placed[Number(dot.dataset.i)].s;
    showTip(
      html`<strong>${s.symbol}</strong><span>${pct(s.change_pct)} at ${inr(s.ltp)}</span><span>${s.buildup_label}${s.bias === 'choppy' ? ', choppy' : ''}</span>`,
      event.clientX, event.clientY
    );
  });
  svg.addEventListener('pointerleave', hideTip);
  svg.addEventListener('click', (event) => {
    const dot = event.target.closest('.dot');
    if (dot) onOpen(placed[Number(dot.dataset.i)].s.symbol);
  });
}

// ---------------------------------------------------------------- flow map (Sankey)

/**
 * Three columns: sector -> build-up class -> trend state. Band thickness
 * is the number of stocks. Click a node to filter the table below.
 */
export function mountFlowMap(el, snapshot, selected, onSelect) {
  const W = Math.max(780, el.clientWidth);
  const H = 620;
  const nodeW = 10;
  const gapY = 6;
  const stocks = snapshot.stocks;

  const classOrder = ['long_buildup', 'short_covering', 'flat', 'long_unwinding', 'short_buildup'];
  const classNames = { long_buildup: 'Long build-up', short_covering: 'Short covering', flat: 'Flat', long_unwinding: 'Long unwinding', short_buildup: 'Short build-up' };
  const classNotes = { long_buildup: 'price up, OI up', short_covering: 'price up, OI down', flat: 'unchanged', long_unwinding: 'price down, OI down', short_buildup: 'price down, OI up' };
  const biasOrder = ['bullish', 'choppy', 'bearish'];
  const biasNames = { bullish: 'Trending up', choppy: 'Choppy', bearish: 'Trending down' };
  const biasNotes = { bullish: 'ranked on the bullish board', choppy: 'left off the boards', bearish: 'ranked on the bearish board' };
  const dirOfClass = (c) => (c === 'flat' ? 'flat' : BUILDUP_UP.has(c) ? 'up' : 'down');

  const count = (test) => stocks.reduce((n, s) => n + (test(s) ? 1 : 0), 0);
  const columns = [
    snapshot.sectors.map((s) => ({
      id: 'sector:' + s.name, type: 'sector', key: s.name, label: s.name, note: pct(s.change_pct), value: s.count,
      dir: s.change_pct > 0.05 ? 'up' : s.change_pct < -0.05 ? 'down' : 'flat',
    })),
    classOrder.map((c) => ({ id: 'buildup:' + c, type: 'buildup', key: c, label: classNames[c], note: classNotes[c], value: count((s) => s.buildup === c), dir: dirOfClass(c), fresh: BUILDUP_FRESH.has(c) })).filter((n) => n.value),
    biasOrder.map((b) => ({ id: 'bias:' + b, type: 'bias', key: b, label: biasNames[b], note: biasNotes[b], value: count((s) => s.bias === b), dir: b === 'bullish' ? 'up' : b === 'bearish' ? 'down' : 'flat' })).filter((n) => n.value),
  ];
  const colX = [164, Math.round(W * 0.5), W - 196];
  const scale = Math.min(...columns.map((col) => (H - 16 - gapY * (col.length - 1)) / col.reduce((n, node) => n + node.value, 0)));
  const nodes = new Map();
  columns.forEach((col, c) => {
    const height = col.reduce((n, node) => n + node.value * scale, 0) + gapY * (col.length - 1);
    let y = (H - height) / 2;
    for (const node of col) {
      Object.assign(node, { x: colX[c], y0: y, y1: y + node.value * scale, outY: y, inY: y, col: c });
      nodes.set(node.id, node);
      y = node.y1 + gapY;
    }
  });

  const tally = new Map();
  const bump = (source, target, cls) => {
    const key = source + '|' + target;
    const link = tally.get(key) || { source, target, value: 0, cls };
    link.value += 1;
    tally.set(key, link);
  };
  for (const s of stocks) {
    bump('sector:' + s.sector, 'buildup:' + s.buildup, s.buildup);
    bump('buildup:' + s.buildup, 'bias:' + s.bias, s.buildup);
  }
  const links = [...tally.values()].filter((l) => nodes.has(l.source) && nodes.has(l.target));
  const order = (id) => nodes.get(id).y0;
  links.sort((a, b) => order(a.source) - order(b.source) || order(a.target) - order(b.target));
  for (const link of links) {
    const node = nodes.get(link.source);
    link.sy = node.outY;
    node.outY += link.value * scale;
  }
  links.sort((a, b) => order(a.target) - order(b.target) || order(a.source) - order(b.source));
  for (const link of links) {
    const node = nodes.get(link.target);
    link.ty = node.inY;
    node.inY += link.value * scale;
  }

  let out = '';
  links.forEach((link, i) => {
    const s = nodes.get(link.source), t = nodes.get(link.target);
    const sx = s.x + nodeW, tx = t.x, mx = (sx + tx) / 2, h = link.value * scale;
    const dir = dirOfClass(link.cls);
    out += `<path class="fl-link ${BUILDUP_FRESH.has(link.cls) ? 'fl-fresh' : 'fl-closing'} f-${dir} s-${dir}" data-i="${i}" data-s="${esc(link.source)}" data-t="${esc(link.target)}" ` +
      `d="M${sx},${f1(link.sy)}C${f1(mx)},${f1(link.sy)} ${f1(mx)},${f1(link.ty)} ${tx},${f1(link.ty)}L${tx},${f1(link.ty + h)}C${f1(mx)},${f1(link.ty + h)} ${f1(mx)},${f1(link.sy + h)} ${sx},${f1(link.sy + h)}Z"/>`;
  });
  for (const node of nodes.values()) {
    const cy = (node.y0 + node.y1) / 2;
    const picked = selected && selected.type === node.type && selected.key === node.key;
    const hollow = node.type === 'buildup' && !node.fresh && node.dir !== 'flat';
    out += `<g class="fl-node${picked ? ' is-picked' : ''}" data-id="${esc(node.id)}" tabindex="0" role="button" aria-pressed="${picked}" aria-label="${esc(node.label)}, ${node.value} stocks">` +
      `<rect class="fl-bar ${hollow ? 'o-' + node.dir : 'f-' + node.dir}" x="${node.x}" y="${f1(node.y0)}" width="${nodeW}" height="${f1(Math.max(2, node.y1 - node.y0))}" rx="2"/>`;
    if (node.col === 0) {
      out += `<text class="fl-label" x="${node.x - 8}" y="${f1(cy + 4)}" text-anchor="end">${esc(node.label)} <tspan class="fl-note">${esc(node.note)}</tspan></text>` +
        `<rect class="hit" x="0" y="${f1(node.y0 - gapY / 2)}" width="${node.x + nodeW}" height="${f1(node.y1 - node.y0 + gapY)}"/>`;
    } else {
      const tall = node.y1 - node.y0 > 30;
      out += `<text class="fl-label fl-strong" x="${node.x + nodeW + 10}" y="${f1(cy + (tall ? -3 : 4))}">${esc(node.label)} <tspan class="fl-count">${node.value}</tspan></text>` +
        (tall ? `<text class="fl-note" x="${node.x + nodeW + 10}" y="${f1(cy + 13)}">${esc(node.note)}</text>` : '') +
        `<rect class="hit" x="${node.x - 4}" y="${f1(node.y0 - gapY / 2)}" width="${node.col === 2 ? W - node.x + 4 : 170}" height="${f1(node.y1 - node.y0 + gapY)}"/>`;
    }
    out += `</g>`;
  }

  el.innerHTML = `<svg class="chart flowmap" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="group" aria-label="Stocks flowing from sector to build-up class to trend state">${out}</svg>`;
  const svg = el.firstChild;
  const linkEls = [...svg.querySelectorAll('.fl-link')];

  const light = (test) => {
    svg.classList.add('is-hover');
    for (const path of linkEls) path.classList.toggle('on', test(path));
  };
  const clear = () => {
    svg.classList.remove('is-hover');
    hideTip();
    if (selected) applySelection();
  };
  const applySelection = () => {
    if (!selected) return;
    const id = selected.type + ':' + selected.key;
    svg.classList.add('is-hover');
    for (const path of linkEls) path.classList.toggle('on', path.dataset.s === id || path.dataset.t === id);
  };
  applySelection();

  svg.addEventListener('pointermove', (event) => {
    const nodeEl = event.target.closest('.fl-node');
    const linkEl = event.target.closest('.fl-link');
    if (nodeEl) {
      const node = nodes.get(nodeEl.dataset.id);
      light((path) => path.dataset.s === node.id || path.dataset.t === node.id);
      showTip(html`<strong>${node.label}</strong><span>${node.value} stocks, ${node.note}</span><span class="tip-hint">Click to filter the table</span>`, event.clientX, event.clientY);
    } else if (linkEl) {
      const link = links[Number(linkEl.dataset.i)];
      light((path) => path === linkEl);
      showTip(html`<strong>${link.value} ${link.value === 1 ? 'stock' : 'stocks'}</strong><span>${nodes.get(link.source).label} to ${nodes.get(link.target).label}</span>`, event.clientX, event.clientY);
    } else {
      clear();
    }
  });
  svg.addEventListener('pointerleave', clear);
  const pick = (nodeEl) => {
    const node = nodes.get(nodeEl.dataset.id);
    const same = selected && selected.type === node.type && selected.key === node.key;
    onSelect(same ? null : { type: node.type, key: node.key, label: node.label });
  };
  svg.addEventListener('click', (event) => {
    const nodeEl = event.target.closest('.fl-node');
    if (nodeEl) pick(nodeEl);
  });
  svg.addEventListener('keydown', (event) => {
    const nodeEl = event.target.closest('.fl-node');
    if (nodeEl && (event.key === 'Enter' || event.key === ' ')) {
      event.preventDefault();
      pick(nodeEl);
    }
  });
}

// ---------------------------------------------------------------- candles

/**
 * Candlestick chart with the order-block zone, pivot levels and previous close.
 * mode: "intraday" (5-minute bars on a full-session axis) or "daily".
 */
export function mountCandles(el, detail, mode) {
  const candles = detail.candles[mode];
  const W = Math.max(320, el.clientWidth);
  const H = 330;
  const m = { top: 12, right: 78, bottom: 22, left: 6 };
  const volH = 44;
  const plotW = W - m.left - m.right;
  const priceBottom = H - m.bottom - volH - 8;
  const slots = mode === 'intraday' ? 75 : candles.length;
  const cw = plotW / slots;
  const bodyW = clamp(cw * 0.62, 1.500, 12);
  const x = (i) => m.left + (i + 0.5) * cw;

  const zone = mode === 'intraday' ? detail.levels.zone : null;
  let lo = Math.min(...candles.map((c) => c.low));
  let hi = Math.max(...candles.map((c) => c.high));
  if (zone) { lo = Math.min(lo, zone.low); hi = Math.max(hi, zone.high); }
  const span = Math.max(hi - lo, hi * 0.002);
  lo -= span * 0.07; hi += span * 0.07;
  const y = (p) => m.top + (1 - (p - lo) / (hi - lo)) * (priceBottom - m.top);
  const maxVol = Math.max(1, ...candles.map((c) => c.volume));

  const step = niceStep(hi - lo, 4);
  let grid = '';
  for (let p = Math.ceil(lo / step) * step; p <= hi; p += step) {
    grid += `<line class="grid-line" x1="${m.left}" x2="${m.left + plotW}" y1="${f1(y(p))}" y2="${f1(y(p))}"/>` +
      `<text class="axis-label" x="${m.left + 4}" y="${f1(y(p) - 4)}">${num(p, step < 1 ? 2 : 0)}</text>`;
  }
  if (mode === 'intraday') {
    for (let hour = 10; hour <= 15; hour++) {
      const slot = (hour * 60 - 555) / 5;
      grid += `<text class="axis-label" x="${f1(m.left + slot * cw)}" y="${H - 6}" text-anchor="middle">${hour}:00</text>`;
    }
  } else {
    const every = Math.ceil(candles.length / 6);
    candles.forEach((c, i) => {
      if (i % every === Math.floor(every / 2)) {
        const d = new Date(c.time + 'T00:00:00');
        grid += `<text class="axis-label" x="${f1(x(i))}" y="${H - 6}" text-anchor="middle">${d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}</text>`;
      }
    });
  }

  let zoneOut = '';
  if (zone) {
    const from = Math.max(0, candles.findIndex((c) => c.time === zone.formed_at));
    const zx = m.left + from * cw;
    zoneOut = `<rect class="zone zone-${zone.side} zone-${zone.state}" x="${f1(zx)}" y="${f1(y(zone.high))}" width="${f1(m.left + plotW - zx)}" height="${f1(Math.max(2, y(zone.low) - y(zone.high)))}"/>`;
  }

  let body = '';
  candles.forEach((c, i) => {
    const up = c.close >= c.open;
    const cls = up ? 'up' : 'down';
    const top = y(Math.max(c.open, c.close));
    const bottom = y(Math.min(c.open, c.close));
    body += `<line class="wick s-${cls}" x1="${f1(x(i))}" x2="${f1(x(i))}" y1="${f1(y(c.high))}" y2="${f1(y(c.low))}"/>` +
      `<rect class="candle f-${cls}" x="${f1(x(i) - bodyW / 2)}" y="${f1(top)}" width="${f1(bodyW)}" height="${f1(Math.max(1, bottom - top))}"/>` +
      `<rect class="vol f-${cls}" x="${f1(x(i) - bodyW / 2)}" y="${f1(H - m.bottom - (c.volume / maxVol) * volH)}" width="${f1(bodyW)}" height="${f1((c.volume / maxVol) * volH)}"/>`;
  });

  // Right-hand tags: levels that fall inside the price range, the previous close and the last price
  const tags = [];
  const lv = detail.levels;
  if (mode === 'intraday') {
    for (const [key, name] of [['r2', 'R2'], ['r1', 'R1'], ['pivot', 'Pivot'], ['s1', 'S1'], ['s2', 'S2']]) {
      if (lv[key] > lo && lv[key] < hi) tags.push({ name, price: lv[key], cls: 'tag-level', line: 'level-line' });
    }
    if (detail.prev_close > lo && detail.prev_close < hi) tags.push({ name: 'Prev', price: detail.prev_close, cls: 'tag-prev', line: 'prev-line' });
  }
  tags.push({ name: 'Last', price: detail.ltp, cls: 'tag-last', line: 'last-line' });
  let lineOut = '';
  for (const tag of tags) {
    tag.y = y(tag.price);
    lineOut += `<line class="${tag.line}" x1="${m.left}" x2="${m.left + plotW}" y1="${f1(tag.y)}" y2="${f1(tag.y)}"/>`;
  }
  spreadLabels(tags, 15, m.top + 6, priceBottom);
  const tagOut = tags.map((tag) =>
    `<text class="tag ${tag.cls}" x="${m.left + plotW + 6}" y="${f1(tag.y + 4)}">${tag.name} <tspan class="tag-price">${num(tag.price)}</tspan></text>`
  ).join('');

  el.innerHTML =
    `<svg class="chart candles" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(detail.symbol)} ${mode === 'intraday' ? 'five-minute' : 'daily'} candles">` +
    grid + zoneOut + lineOut + body + tagOut +
    `<line class="crosshair" x1="0" x2="0" y1="${m.top}" y2="${H - m.bottom}" visibility="hidden"/>` +
    `<rect class="hit" x="${m.left}" y="${m.top}" width="${plotW}" height="${H - m.top - m.bottom}"/></svg>`;

  const svg = el.firstChild;
  const cross = svg.querySelector('.crosshair');
  const hit = svg.querySelector('.hit');
  hit.addEventListener('pointermove', (event) => {
    const rect = svg.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (W / rect.width);
    const i = clamp(Math.floor((px - m.left) / cw), 0, candles.length - 1);
    const c = candles[i];
    cross.setAttribute('x1', f1(x(i)));
    cross.setAttribute('x2', f1(x(i)));
    cross.setAttribute('visibility', 'visible');
    const when = mode === 'intraday' ? c.time : new Date(c.time + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    showTip(
      html`<strong>${when}</strong>
        <span class="tip-row">Open<b>${num(c.open)}</b></span><span class="tip-row">High<b>${num(c.high)}</b></span>
        <span class="tip-row">Low<b>${num(c.low)}</b></span><span class="tip-row">Close<b>${num(c.close)}</b></span>
        <span class="tip-row">Volume<b>${Number(c.volume).toLocaleString('en-IN')}</b></span>`,
      event.clientX, event.clientY
    );
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    hideTip();
  });
}

// ---------------------------------------------------------------- equity curve

/** Cumulative P&L after each closed trade. points: [{label, value, note}] */
export function mountEquity(el, points) {
  const W = Math.max(280, el.clientWidth);
  const H = 190;
  const m = { top: 14, right: 14, bottom: 22, left: 58 };
  const plotW = W - m.left - m.right;
  const plotH = H - m.top - m.bottom;
  const values = [0, ...points.map((p) => p.value)];
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  const x = (i) => m.left + (values.length === 1 ? 0 : (i / (values.length - 1)) * plotW);
  const y = (v) => m.top + (1 - (v - lo) / (hi - lo)) * plotH;
  const step = niceStep(hi - lo, 3);
  let grid = '';
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const value = Math.abs(v) < 1e-9 ? 0 : v;
    grid += `<line class="${value === 0 ? 'axis-zero' : 'grid-line'}" x1="${m.left}" x2="${m.left + plotW}" y1="${f1(y(value))}" y2="${f1(y(value))}"/>` +
      `<text class="axis-label" x="${m.left - 8}" y="${f1(y(value) + 4)}" text-anchor="end">${value < 0 ? '−' : ''}₹${Math.abs(value) >= 1e5 ? (Math.abs(value) / 1e5).toFixed(1) + 'L' : num(Math.abs(value), 0)}</text>`;
  }
  const line = values.map((v, i) => `${i ? 'L' : 'M'}${f1(x(i))},${f1(y(v))}`).join('');
  const area = `${line}L${f1(x(values.length - 1))},${f1(y(0))}L${f1(x(0))},${f1(y(0))}Z`;
  const last = values[values.length - 1];
  el.innerHTML =
    `<svg class="chart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Cumulative profit and loss after each trade">` +
    grid + `<path class="eq-area" d="${area}"/><path class="eq-line" d="${line}"/>` +
    `<circle class="eq-dot" cx="${f1(x(values.length - 1))}" cy="${f1(y(last))}" r="4"/>` +
    `<text class="axis-label" x="${m.left}" y="${H - 6}">Start</text>` +
    `<text class="axis-label" x="${m.left + plotW}" y="${H - 6}" text-anchor="end">${points.length} ${points.length === 1 ? 'trade' : 'trades'}</text>` +
    `<line class="crosshair" x1="0" x2="0" y1="${m.top}" y2="${m.top + plotH}" visibility="hidden"/>` +
    `<rect class="hit" x="${m.left}" y="${m.top}" width="${plotW}" height="${plotH}"/></svg>`;
  const svg = el.firstChild;
  const cross = svg.querySelector('.crosshair');
  const hit = svg.querySelector('.hit');
  hit.addEventListener('pointermove', (event) => {
    const rect = svg.getBoundingClientRect();
    const px = (event.clientX - rect.left) * (W / rect.width);
    const i = clamp(Math.round(((px - m.left) / plotW) * (values.length - 1)), 0, values.length - 1);
    cross.setAttribute('x1', f1(x(i)));
    cross.setAttribute('x2', f1(x(i)));
    cross.setAttribute('visibility', 'visible');
    const point = i === 0 ? null : points[i - 1];
    showTip(
      point
        ? html`<strong>${point.label}</strong><span>${point.note}</span><span class="tip-row">Running total<b>${values[i] < 0 ? '−' : ''}${inr(Math.abs(values[i]), 0)}</b></span>`
        : html`<strong>Start</strong><span>Before the first trade</span>`,
      event.clientX, event.clientY
    );
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    hideTip();
  });
}
