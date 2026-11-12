// Stock detail drawer: chart, reading, score breakdown, levels and plan.

import { $, html, setHTML, icon, pct, num, toast, hideTip } from '../util.js';
import { app, on, go, hooks } from '../state.js';
import { api } from '../api.js';
import { mountCandles } from '../charts.js';
import { change, buildupTag, scoreMeter } from '../components.js';
import { watchlist } from '../local.js';

const view = { symbol: null, detail: null, mode: 'intraday', error: null, returnFocus: null };

const COMPONENT_LABELS = {
  momentum: ['Momentum', 'Size of the move against the stock’s usual daily range'],
  trend: ['Trend quality', 'How one-directional the move has been'],
  open_interest: ['Open interest', 'Whether positions are being added or closed'],
  volume: ['Volume', 'Volume against the usual pace for this time of day'],
};

function levelRows(d) {
  const lv = d.levels;
  const rows = [
    { name: 'R2', price: lv.r2 }, { name: 'R1', price: lv.r1 }, { name: 'Pivot', price: lv.pivot },
    { name: 'S1', price: lv.s1 }, { name: 'S2', price: lv.s2 },
    { name: 'Previous close', price: d.prev_close },
  ];
  if (lv.zone) {
    rows.push({ name: `${lv.zone.side === 'demand' ? 'Demand' : 'Supply'} zone top`, price: lv.zone.high, zone: true });
    rows.push({ name: `${lv.zone.side === 'demand' ? 'Demand' : 'Supply'} zone bottom`, price: lv.zone.low, zone: true });
  }
  rows.push({ name: 'Last price', price: d.ltp, last: true });
  return rows.sort((a, b) => b.price - a.price);
}

function planBlock(d) {
  const { zone, plan } = d.levels;
  if (!zone) return html`<p class="muted">No order-block zone yet. One appears once the session has a clear three-bar push.</p>`;
  if (!plan) return html`<p class="muted">The ${zone.side} zone at ${num(zone.low)} to ${num(zone.high)} has been traded through, so there is no plan from it.</p>`;
  return html`
    <dl class="plan">
      <div><dt>Direction</dt><dd>${plan.direction === 'long' ? 'Long on a pullback' : 'Short on a bounce'}</dd></div>
      <div><dt>Entry</dt><dd>${num(plan.entry)}</dd></div>
      <div><dt>Stop</dt><dd>${num(plan.stop)}</dd></div>
      <div><dt>Targets</dt><dd>${plan.targets.map((t) => num(t)).join(', ')}</dd></div>
      <div><dt>Risk per share</dt><dd>${num(plan.risk_per_share)}</dd></div>
      <div><dt>Reward to risk</dt><dd>${plan.reward_to_risk.toFixed(1)}</dd></div>
    </dl>
    <p class="muted">A worked example of how the levels fit together, assuming price returns to the zone. It is not a recommendation; chasing the current price gives a different reward to risk.</p>`;
}

function body(d) {
  const starred = watchlist.has(d.symbol);
  const dir = d.change_pct > 0.05 ? 'up' : d.change_pct < -0.05 ? 'down' : 'flat';
  return html`
    <header class="drawer-head">
      <div>
        <h2 id="drawer-title">${d.symbol}</h2>
        <p class="sub">${d.name}, ${d.sector}</p>
      </div>
      <div class="drawer-actions">
        <button class="icon-btn${starred ? ' is-on' : ''}" data-action="toggle-watch" aria-pressed="${starred}" aria-label="${starred ? 'Remove from watchlist' : 'Add to watchlist'}" title="${starred ? 'Remove from watchlist' : 'Add to watchlist'}">${icon('star')}</button>
        <button class="icon-btn" data-action="close-stock" aria-label="Close">${icon('close')}</button>
      </div>
    </header>

    <div class="drawer-price">
      <span class="price price-xl">${num(d.ltp)}</span>
      ${change(d.change_pct)}
      <span class="sub">as of ${d.meta.as_of}${d.meta.is_replay ? ' (replay)' : ''}${d.meta.data_mode === 'simulated' ? ', simulated' : ''}</span>
    </div>
    <div class="drawer-tags">${buildupTag(d)}${scoreMeter(d)}</div>

    <div class="tabs tabs-small" role="tablist" aria-label="Chart interval">
      <button class="tab${view.mode === 'intraday' ? ' is-active' : ''}" role="tab" aria-selected="${view.mode === 'intraday'}" data-action="chart-mode" data-mode="intraday">5 minute</button>
      <button class="tab${view.mode === 'daily' ? ' is-active' : ''}" role="tab" aria-selected="${view.mode === 'daily'}" data-action="chart-mode" data-mode="daily">Daily</button>
    </div>
    <div class="drawer-chart" id="drawer-chart"></div>
    ${view.mode === 'intraday' && d.levels.zone
      ? html`<p class="chart-key"><i class="key key-zone zone-${d.levels.zone.side}"></i>${d.levels.zone.side === 'demand' ? 'Demand' : 'Supply'} zone formed at ${d.levels.zone.formed_at}, ${d.levels.zone.state}</p>`
      : ''}

    <section class="drawer-section">
      <h3>What the numbers show</h3>
      <ul class="reading">${d.reading.map((line) => html`<li>${line}</li>`)}</ul>
    </section>

    <section class="drawer-section">
      <h3>Strength score ${d.bias === 'choppy' ? html`<span class="muted">set to 0 because the stock is choppy</span>` : html`<span class="muted">${d.score} of 100</span>`}</h3>
      <ul class="components">
        ${Object.entries(d.score_components).map(([key, value]) => html`
          <li title="${COMPONENT_LABELS[key][1]}">
            <span>${COMPONENT_LABELS[key][0]}</span>
            <span class="score-track"><i class="score-fill bg-${dir === 'flat' ? 'flat' : dir}" style="width:${value * 100}%"></i></span>
            <span class="num">${Math.round(value * 100)}</span>
            <span class="muted num">× ${d.score_weights[key].toFixed(2)}</span>
          </li>`)}
      </ul>
    </section>

    <section class="drawer-section">
      <h3>Levels</h3>
      <table class="ladder">
        <tbody>
          ${levelRows(d).map((row) => html`
            <tr class="${row.last ? 'is-last' : row.zone ? 'is-zone' : ''}">
              <th scope="row">${row.name}</th>
              <td class="num">${num(row.price)}</td>
              <td class="num muted">${row.last ? '' : pct((row.price / d.ltp - 1) * 100)}</td>
            </tr>`)}
        </tbody>
      </table>
    </section>

    <section class="drawer-section">
      <h3>Plan from the zone</h3>
      ${planBlock(d)}
    </section>

    <section class="drawer-section">
      <h3>Session</h3>
      <dl class="plan">
        <div><dt>Open</dt><dd>${num(d.open)}</dd></div>
        <div><dt>High</dt><dd>${num(d.day_high)}</dd></div>
        <div><dt>Low</dt><dd>${num(d.day_low)}</dd></div>
        <div><dt>OI change</dt><dd>${pct(d.oi_change_pct)}</dd></div>
        <div><dt>Volume</dt><dd>${d.volume_ratio.toFixed(1)}× usual</dd></div>
        <div><dt>Turnover</dt><dd>₹${num(d.turnover_cr, 0)} Cr</dd></div>
        <div><dt>${d.sector}</dt><dd>${pct(d.sector_change_pct)}, rank ${d.sector_rank} of ${d.sector_count}</dd></div>
      </dl>
    </section>

    <footer class="drawer-foot">
      <button class="btn" data-action="log-trade" data-symbol="${d.symbol}" data-entry="${d.ltp}" data-side="${d.bias === 'bearish' ? 'short' : 'long'}">${icon('journal', { size: 16 })}Log a trade</button>
      <button class="btn btn-quiet" data-action="share-setup" data-symbol="${d.symbol}">${icon('community', { size: 16 })}Share to community</button>
    </footer>`;
}

function paint() {
  const drawer = $('#drawer');
  if (view.error) {
    setHTML(drawer, html`
      <header class="drawer-head"><h2 id="drawer-title">${view.symbol}</h2>
        <button class="icon-btn" data-action="close-stock" aria-label="Close">${icon('close')}</button></header>
      <div class="empty empty-error"><h3>Could not load ${view.symbol}</h3><p>${view.error}</p>
        <button class="btn" data-action="reload-stock">Try again</button></div>`);
    return;
  }
  if (!view.detail) {
    setHTML(drawer, html`<header class="drawer-head"><h2 id="drawer-title">${view.symbol}</h2>
      <button class="icon-btn" data-action="close-stock" aria-label="Close">${icon('close')}</button></header><p class="boot">Loading ${view.symbol}…</p>`);
    return;
  }
  const scroll = drawer.scrollTop;
  setHTML(drawer, body(view.detail));
  mountCandles($('#drawer-chart'), view.detail, view.mode);
  drawer.scrollTop = scroll;
}

async function load() {
  try {
    view.detail = await api.stock(view.symbol, app.at);
    view.error = null;
  } catch (error) {
    view.error = error.message;
  }
  if (view.symbol) paint();
}

export function open(symbol) {
  const drawer = $('#drawer');
  if (drawer.hidden) view.returnFocus = document.activeElement;
  view.symbol = symbol;
  view.detail = null;
  view.error = null;
  drawer.hidden = false;
  $('#drawer-scrim').hidden = false;
  drawer.scrollTop = 0;
  hideTip();
  paint();
  const closeButton = drawer.querySelector('[data-action="close-stock"]');
  if (closeButton) closeButton.focus();
  load();
}

export function close() {
  if (!view.symbol) return;
  view.symbol = null;
  $('#drawer').hidden = true;
  $('#drawer-scrim').hidden = true;
  hideTip();
  if (view.returnFocus && document.contains(view.returnFocus)) view.returnFocus.focus();
}

/** Called after every snapshot refresh so an open drawer follows the session (or the replay). */
export function onSnapshot() {
  if (view.symbol) load();
}

export function onResize() {
  if (view.symbol && view.detail) paint();
}

on('chart-mode', (el) => {
  view.mode = el.dataset.mode;
  paint();
});
on('reload-stock', () => { view.error = null; paint(); load(); });
on('toggle-watch', () => {
  const added = watchlist.toggle(view.symbol);
  toast(added ? `${view.symbol} added to your watchlist` : `${view.symbol} removed from your watchlist`);
  paint();
  if (app.route.name === 'home') hooks.rerender();
});
on('share-setup', (el) => {
  close();
  sessionStorage.setItem('niveshak.draft-symbol', el.dataset.symbol);
  if (app.route.name === 'community') hooks.rerender();
  else go('community');
});
