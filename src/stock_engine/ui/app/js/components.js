// Pieces of interface shared by more than one screen.

import { html, raw, icon, buildupMark, pct, num, inr, dirOf, setHTML, $ } from './util.js';
import { sparkline } from './charts.js';

/** Signed percent, coloured by direction, with an arrow so colour is never the only cue. */
export function change(value, { arrow = true, neutral = false } = {}) {
  const dir = neutral ? 'flat' : dirOf(value);
  const glyph = neutral ? '' : dir === 'up' ? '▲' : dir === 'down' ? '▼' : '';
  return html`<span class="chg chg-${dir}">${arrow && glyph ? html`<span class="chg-arrow" aria-hidden="true">${glyph}</span>` : ''}${pct(value)}</span>`;
}

export function buildupTag(row) {
  return html`<span class="buildup" title="${buildupHint(row.buildup)}">${buildupMark(row.buildup)}${row.buildup_label}</span>`;
}

export function buildupHint(buildup) {
  return {
    long_buildup: 'Price up, open interest up: new longs being added',
    short_covering: 'Price up, open interest down: shorts closing',
    short_buildup: 'Price down, open interest up: new shorts being added',
    long_unwinding: 'Price down, open interest down: longs closing',
    flat: 'Price is close to the previous close',
  }[buildup];
}

/** 0-100 strength score as a thin bar plus the number. Choppy stocks score 0 and say so. */
export function scoreMeter(row) {
  if (row.bias === 'choppy') return html`<span class="score score-none">Choppy</span>`;
  const dir = row.bias === 'bullish' ? 'up' : 'down';
  return html`<span class="score" title="Strength score out of 100">
    <span class="score-track"><i class="score-fill bg-${dir}" style="width:${row.score}%"></i></span><b>${row.score}</b></span>`;
}

/** One small block per stock in the sector: advancing, unchanged, declining. */
export function advDecBar(sector) {
  const flat = sector.count - sector.advancing - sector.declining;
  const cell = (n, cls) => Array.from({ length: n }, () => html`<i class="${cls}"></i>`);
  return html`<span class="adbar" role="img" aria-label="${sector.advancing} up, ${sector.declining} down of ${sector.count}">
    ${cell(sector.advancing, 'bg-up')}${cell(flat, 'bg-flat')}${cell(sector.declining, 'bg-down')}</span>`;
}

/** The order-block zone as a % band on the stock's sparkline scale. */
export function zoneBand(row) {
  const zone = row.levels && row.levels.zone;
  if (!zone || zone.state === 'broken') return null;
  const toPct = (price) => (price / row.prev_close - 1) * 100;
  return [toPct(zone.low), toPct(zone.high)];
}

// ---------------------------------------------------------------- stock table

export const TABLE_COLUMNS = [
  { key: 'symbol', label: 'Stock', sort: (r) => r.symbol, text: true },
  { key: 'sector', label: 'Sector', sort: (r) => r.sector, text: true, wide: true },
  { key: 'ltp', label: 'Last price', sort: (r) => r.ltp, num: true },
  { key: 'change_pct', label: 'Change', sort: (r) => r.change_pct, num: true },
  { key: 'oi_change_pct', label: 'OI change', sort: (r) => r.oi_change_pct, num: true },
  { key: 'buildup', label: 'Build-up', sort: (r) => r.buildup_label, text: true },
  { key: 'volume_ratio', label: 'Volume', sort: (r) => r.volume_ratio, num: true, wide: true },
  { key: 'trend_quality', label: 'Trend quality', sort: (r) => r.trend_quality, num: true, wide: true },
  { key: 'score', label: 'Score', sort: (r) => r.score, num: true },
  { key: 'spark', label: 'Session', wide: true },
];

export function sortRows(rows, sort) {
  const column = TABLE_COLUMNS.find((c) => c.key === sort.key) || TABLE_COLUMNS[8];
  const dir = sort.dir;
  return [...rows].sort((a, b) => {
    const av = column.sort(a), bv = column.sort(b);
    return (av < bv ? -1 : av > bv ? 1 : 0) * dir || b.score - a.score;
  });
}

/**
 * Sortable table of stocks. `state` = { sort: {key, dir}, limit }.
 * Header clicks fire data-action="sort-table"; row clicks fire "open-stock".
 */
export function stockTable(rows, state) {
  const sorted = sortRows(rows, state.sort);
  const shown = state.limit ? sorted.slice(0, state.limit) : sorted;
  const head = TABLE_COLUMNS.map((c) => {
    if (!c.sort) return html`<th class="${c.wide ? 'col-wide' : ''}">${c.label}</th>`;
    const active = state.sort.key === c.key;
    const aria = active ? (state.sort.dir === 1 ? 'ascending' : 'descending') : 'none';
    return html`<th class="${c.num ? 'num' : ''} ${c.wide ? 'col-wide' : ''}" aria-sort="${aria}">
      <button class="th-btn${active ? ' is-active' : ''}" data-action="sort-table" data-key="${c.key}">${c.label}<span class="th-dir" aria-hidden="true">${active ? (state.sort.dir === 1 ? '↑' : '↓') : ''}</span></button></th>`;
  });
  const body = shown.map((r) => html`
    <tr class="row-link" data-action="open-stock" data-symbol="${r.symbol}" tabindex="0">
      <td><span class="sym">${r.symbol}</span><span class="sub">${r.name}</span></td>
      <td class="col-wide muted">${r.sector}</td>
      <td class="num">${num(r.ltp)}</td>
      <td class="num">${change(r.change_pct)}</td>
      <td class="num">${change(r.oi_change_pct, { neutral: true })}</td>
      <td>${buildupTag(r)}</td>
      <td class="num col-wide">${r.volume_ratio.toFixed(1)}×</td>
      <td class="num col-wide">${r.trend_quality.toFixed(2)}</td>
      <td class="num">${scoreMeter(r)}</td>
      <td class="col-wide">${sparkline(r.spark, { width: 92, height: 26 })}</td>
    </tr>`);
  return html`
    <div class="table-wrap">
      <table class="table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body.length ? body : html`<tr><td colspan="10" class="empty-cell">No stocks match. Clear the filter or the search to see them all.</td></tr>`}</tbody>
      </table>
    </div>
    ${state.limit && sorted.length > state.limit
      ? html`<button class="btn btn-quiet table-more" data-action="table-show-all">Show all ${sorted.length} stocks</button>`
      : ''}`;
}

// ---------------------------------------------------------------- modal

export function openModal(content, { onOpen } = {}) {
  const modal = $('#modal');
  setHTML(modal, html`<button class="icon-btn modal-close" data-action="close-modal" aria-label="Close">${icon('close')}</button>${content}`);
  if (!modal.open) modal.showModal();
  if (onOpen) onOpen(modal);
  return modal;
}

export function closeModal() {
  const modal = $('#modal');
  if (modal.open) modal.close();
}

export function emptyState({ title, text, action }) {
  return html`<div class="empty"><h3>${title}</h3><p>${text}</p>${action || ''}</div>`;
}

export const errorState = (message) => html`
  <div class="empty empty-error">
    <h3>The session did not load</h3>
    <p>${message}</p>
    <button class="btn" data-action="retry">Try again</button>
  </div>`;

export { raw, inr };
