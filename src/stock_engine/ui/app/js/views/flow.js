// Flow: the whole F&O universe four ways - market strip, flow map, and the two boards.

import { $, html, setHTML, icon, buildupMark, pct, num, debounce } from '../util.js';
import { app, on, go, hooks } from '../state.js';
import { mountStrip, mountFlowMap, sparkline } from '../charts.js';
import { change, buildupTag, scoreMeter, advDecBar, zoneBand, stockTable, errorState } from '../components.js';

export const usesSnapshot = true;

const TABS = [
  { id: 'market', label: 'Market' },
  { id: 'map', label: 'Flow map' },
  { id: 'bullish', label: 'Bullish' },
  { id: 'bearish', label: 'Bearish' },
];

const view = {
  filter: null,                       // { type: 'sector' | 'buildup' | 'bias', key, label }
  query: '',
  table: { sort: { key: 'score', dir: -1 }, limit: 25 },
};

/** Set from elsewhere (Home sector rows, search palette) before navigating here. */
export function setFilter(filter) {
  view.filter = filter;
  view.table.limit = 25;
}

function filteredStocks(snap) {
  const q = view.query.trim().toLowerCase();
  return snap.stocks.filter((s) => {
    if (view.filter) {
      const { type, key } = view.filter;
      if (type === 'sector' && s.sector !== key) return false;
      if (type === 'buildup' && s.buildup !== key) return false;
      if (type === 'bias' && s.bias !== key) return false;
    }
    return !q || s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q);
  });
}

// ---------------------------------------------------------------- pieces

function tableCard(snap) {
  const rows = filteredStocks(snap);
  return html`
    <section class="card" id="stock-table" aria-labelledby="table-title">
      <header class="card-head">
        <h2 id="table-title">${view.filter ? view.filter.label : 'All stocks'}</h2>
        <p class="card-note">${rows.length} of ${snap.stocks.length} stocks${view.query ? html`, matching “${view.query}”` : ''}</p>
        ${view.filter ? html`<button class="chip chip-clear" data-action="clear-filter">${icon('close', { size: 13 })}Clear filter</button>` : ''}
      </header>
      ${stockTable(rows, view.table)}
    </section>`;
}

function legend() {
  return html`
    <ul class="legend" aria-label="How to read the markers">
      <li>${buildupMark('long_buildup')}Up, fresh positions</li>
      <li>${buildupMark('short_covering')}Up, positions closing</li>
      <li>${buildupMark('short_buildup')}Down, fresh positions</li>
      <li>${buildupMark('long_unwinding')}Down, positions closing</li>
      <li><span class="legend-quiet" aria-hidden="true"></span>Faded: choppy</li>
    </ul>`;
}

function setupChip(r) {
  const dir = r.bias === 'bullish' ? 'up' : 'down';
  return html`
    <button class="setup setup-${dir}" data-action="open-stock" data-symbol="${r.symbol}">
      <span class="setup-rank">${r.rank}</span>
      <span class="setup-id"><span class="sym">${r.symbol}</span><span class="sub">${r.buildup_label}</span></span>
      <span class="setup-num">${change(r.change_pct)}<span class="sub">Score ${r.score}</span></span>
    </button>`;
}

function marketTab(snap) {
  return html`
    <section class="card" aria-labelledby="strip-title">
      <header class="card-head">
        <h2 id="strip-title">Where every stock stands</h2>
        <p class="card-note">One dot per stock, placed by its change from the previous close. Click a dot for detail.</p>
      </header>
      <div id="flow-strip"></div>
      ${legend()}
    </section>

    <section class="card" aria-labelledby="setups-title">
      <header class="card-head">
        <h2 id="setups-title">Top setups</h2>
        <p class="card-note">${snap.meta.settling ? 'Rankings settle a few bars after the open.' : 'The three strongest on each side'}</p>
      </header>
      <div class="setups">
        ${snap.boards.bullish.slice(0, 3).map(setupChip)}
        ${snap.boards.bearish.slice(0, 3).map(setupChip)}
      </div>
    </section>

    <section class="card" aria-labelledby="sectors-grid-title">
      <header class="card-head">
        <h2 id="sectors-grid-title">Sectors</h2>
        <p class="card-note">Each block is one stock. Pick a sector to filter the table below.</p>
      </header>
      <div class="sector-grid">
        ${snap.sectors.map((s) => html`
          <button class="sector-tile${view.filter && view.filter.type === 'sector' && view.filter.key === s.name ? ' is-picked' : ''}" data-action="pick-sector" data-sector="${s.name}" aria-pressed="${Boolean(view.filter && view.filter.key === s.name)}">
            <span class="sector-tile-head"><b>${s.name}</b>${change(s.change_pct)}</span>
            ${advDecBar(s)}
            <span class="sub">Leader ${s.leader}, laggard ${s.laggard}</span>
          </button>`)}
      </div>
    </section>
    ${tableCard(snap)}`;
}

function mapTab(snap) {
  return html`
    <section class="card" aria-labelledby="map-title">
      <header class="card-head">
        <h2 id="map-title">Flow map</h2>
        <p class="card-note">Sector, then open-interest build-up, then whether the move is trending or choppy. Band thickness is the number of stocks. Click any bar to filter the table.</p>
      </header>
      <div class="scroll-x"><div id="flow-map"></div></div>
      ${legend()}
    </section>
    ${tableCard(snap)}`;
}

function levelCell(label, value) {
  return html`<div class="lv"><dt>${label}</dt><dd>${value}</dd></div>`;
}

function boardCard(r) {
  const lv = r.levels;
  const zone = lv.zone;
  const plan = lv.plan;
  const bull = r.bias === 'bullish';
  return html`
    <li class="setup-card" data-action="open-stock" data-symbol="${r.symbol}" tabindex="0">
      <div class="setup-card-id">
        <span class="board-rank">${r.rank}</span>
        <div>
          <p class="sym sym-lg">${r.symbol}</p>
          <p class="sub">${r.name}, ${r.sector}</p>
        </div>
      </div>
      <div class="setup-card-price">
        <span class="price">${num(r.ltp)}</span>
        ${change(r.change_pct)}
      </div>
      <div class="setup-card-read">
        ${buildupTag(r)}
        <span class="sub">OI ${pct(r.oi_change_pct)}, volume ${r.volume_ratio.toFixed(1)}×, trend quality ${r.trend_quality.toFixed(2)}</span>
        ${scoreMeter(r)}
      </div>
      <div class="setup-card-spark">${sparkline(r.spark, { width: 180, height: 56, band: zoneBand(r) })}</div>
      <dl class="levels">
        ${levelCell(bull ? 'S1' : 'R1', num(bull ? lv.s1 : lv.r1))}
        ${levelCell(zone ? `${zone.side === 'demand' ? 'Demand' : 'Supply'} zone` : 'Zone', zone ? `${num(zone.low)} to ${num(zone.high)}` : 'None yet')}
        ${levelCell(bull ? 'R1' : 'S1', num(bull ? lv.r1 : lv.s1))}
        ${levelCell(bull ? 'R2' : 'S2', num(bull ? lv.r2 : lv.s2))}
        ${plan
          ? levelCell('Plan from the zone', html`Entry ${num(plan.entry)}, stop ${num(plan.stop)}, target ${num(plan.targets[0])} <b class="rr">R:R ${plan.reward_to_risk.toFixed(1)}</b>`)
          : levelCell('Plan', zone && zone.state === 'broken' ? 'Zone broken, no plan' : 'No zone yet')}
      </dl>
    </li>`;
}

function boardTab(snap, side) {
  const q = view.query.trim().toLowerCase();
  const rows = snap.boards[side].filter((r) => !q || r.symbol.toLowerCase().includes(q) || r.name.toLowerCase().includes(q));
  return html`
    <section aria-labelledby="board-title">
      <header class="card-head card-head-bare">
        <h2 id="board-title">${side === 'bullish' ? 'Bullish board' : 'Bearish board'}</h2>
        <p class="card-note">Top ${snap.boards[side].length} by strength score. Levels are from the previous session's pivots and today's order-block zone. Plans are worked examples, not calls.</p>
      </header>
      ${snap.meta.settling ? html`<p class="notice">The session has just opened. Scores and zones settle after about fifteen minutes.</p>` : ''}
      <ol class="setup-cards">${rows.map(boardCard)}</ol>
      ${rows.length ? '' : html`<p class="muted">Nothing on this board matches “${view.query}”.</p>`}
    </section>`;
}

// ---------------------------------------------------------------- render

export function render(root, route) {
  const snap = app.snapshot;
  if (!snap) {
    setHTML(root, app.error ? errorState(app.error) : html`<p class="boot">Loading the session…</p>`);
    return;
  }
  const tab = TABS.some((t) => t.id === route.arg) ? route.arg : 'market';
  const keepFocus = document.activeElement && document.activeElement.id === 'flow-query';
  const caret = keepFocus ? document.activeElement.selectionStart : null;

  setHTML(root, html`
    <header class="page-head page-head-row">
      <div>
        <h1>Flow</h1>
        <p class="lede">${snap.meta.universe} F&amp;O stocks as of ${snap.meta.as_of}${snap.meta.is_replay ? ' (replay)' : ''}. ${snap.breadth.choppy} are choppy and sit out of the boards.</p>
      </div>
      <div class="page-tools">
        <div class="tabs" role="tablist" aria-label="Flow views">
          ${TABS.map((t) => html`<a class="tab${t.id === tab ? ' is-active' : ''}" role="tab" aria-selected="${t.id === tab}" href="#/flow/${t.id}">${t.label}</a>`)}
        </div>
        <label class="field field-search">${icon('filter', { size: 16 })}
          <input id="flow-query" type="search" placeholder="Filter by symbol or name" value="${view.query}" aria-label="Filter stocks by symbol or name" autocomplete="off">
        </label>
      </div>
    </header>
    <div class="stack">
      ${tab === 'market' ? marketTab(snap) : tab === 'map' ? mapTab(snap) : boardTab(snap, tab)}
    </div>`);

  if (tab === 'market') mountStrip($('#flow-strip'), snap.stocks, snap.sectors, (symbol) => hooks.openStock(symbol));
  if (tab === 'map') mountFlowMap($('#flow-map'), snap, view.filter, (filter) => { setFilter(filter); render(root, route); });

  const input = $('#flow-query');
  input.addEventListener('input', debounce(() => {
    view.query = input.value;
    view.table.limit = 25;
    render(root, route);
  }, 160));
  if (keepFocus) {
    input.focus();
    input.setSelectionRange(caret, caret);
  }
}

on('pick-sector', (el) => {
  const name = el.dataset.sector;
  const same = view.filter && view.filter.type === 'sector' && view.filter.key === name;
  setFilter(same ? null : { type: 'sector', key: name, label: name });
  if (app.route.name === 'flow' && (app.route.arg === 'market' || !app.route.arg)) {
    render($('#view'), app.route);
    if (!same) $('#stock-table').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    go('flow/market');
  }
});

on('clear-filter', () => {
  setFilter(null);
  render($('#view'), app.route);
});

on('sort-table', (el) => {
  const key = el.dataset.key;
  const sort = view.table.sort;
  view.table.sort = sort.key === key ? { key, dir: -sort.dir } : { key, dir: ['symbol', 'sector', 'buildup'].includes(key) ? 1 : -1 };
  render($('#view'), app.route);
});

on('table-show-all', () => {
  view.table.limit = 0;
  render($('#view'), app.route);
});
