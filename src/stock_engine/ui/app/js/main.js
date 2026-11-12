// Niveshak terminal: shell (sidebar, session bar, footer), router, data refresh.

import { $, html, setHTML, icon, store, debounce, hideTip, shortDate } from './util.js';
import { app, actions, on, hooks, saveSettings, go } from './state.js';
import { api } from './api.js';
import { mountRibbon } from './charts.js';
import { openModal, closeModal } from './components.js';
import * as home from './views/home.js';
import * as flow from './views/flow.js';
import * as stock from './views/stock.js';
import * as journal from './views/journal.js';
import * as learn from './views/learn.js';
import * as community from './views/community.js';
import * as feedback from './views/feedback.js';
import * as settings from './views/settings.js';

const VIEWS = { home, flow, journal, learn, community, feedback, settings };

const NAV = [
  { group: 'Markets', items: [
    { name: 'home', label: 'Home', icon: 'home' },
    { name: 'flow', label: 'Flow', icon: 'flow', live: true },
  ] },
  { group: 'Practice', items: [
    { name: 'journal', label: 'Journal', icon: 'journal' },
    { name: 'learn', label: 'Learn', icon: 'learn' },
  ] },
  { group: 'People', items: [
    { name: 'community', label: 'Community', icon: 'community' },
    { name: 'feedback', label: 'Feedback', icon: 'feedback' },
  ] },
];

const ui = { collapsed: store.load('ui', {}).collapsed === true, ribbon: null, timer: null, ticks: 0, marketState: null };

// ---------------------------------------------------------------- theme

function applyTheme() {
  const choice = app.settings.theme;
  const theme = choice === 'system'
    ? (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
    : choice;
  document.documentElement.dataset.theme = theme;
}

// ---------------------------------------------------------------- sidebar

function engineCard() {
  const engine = app.engine;
  if (!engine) return html`<div class="engine"><span class="engine-row">${icon('shield', { size: 16 })}<span class="side-text">Checking the engine…</span></span></div>`;
  if (engine.error) return html`<div class="engine engine-warn" title="${engine.error}"><span class="engine-row">${icon('shield', { size: 16 })}<span class="side-text">Engine status unavailable</span></span></div>`;
  const tripped = engine.risk.is_tripped;
  return html`
    <a class="engine${tripped ? ' engine-warn' : ''}" href="#/settings" title="Open engine status in Settings">
      <span class="engine-row">${icon('shield', { size: 16 })}<span class="side-text"><b>${engine.health.mode === 'PAPER' ? 'Paper mode' : engine.health.mode}</b></span></span>
      <span class="engine-sub side-text">${tripped ? 'Kill switch tripped. New entries are halted.' : 'Kill switch armed. No live orders.'}</span>
    </a>`;
}

function renderSide() {
  const current = app.route.name;
  const item = (it) => html`
    <a class="nav-item${current === it.name ? ' is-active' : ''}" href="#/${it.name}" ${current === it.name ? html`aria-current="page"` : ''} title="${it.label}">
      ${icon(it.icon)}<span class="side-text">${it.label}</span>
      ${it.live && app.snapshot && app.snapshot.meta.market_state === 'open' ? html`<span class="nav-live side-text">Live</span>` : ''}
    </a>`;
  const theme = document.documentElement.dataset.theme;
  setHTML($('#side'), html`
    <div class="side-head">
      <a class="brand" href="#/home" aria-label="Niveshak home">
        <span class="brand-mark" lang="hi">${ui.collapsed ? 'नि' : 'निवेशक'}</span>
        <span class="brand-sub side-text">Niveshak market terminal</span>
      </a>
      <button class="icon-btn side-toggle" data-action="toggle-side" aria-label="${ui.collapsed ? 'Expand navigation' : 'Collapse navigation'}" aria-expanded="${!ui.collapsed}">
        ${icon(ui.collapsed ? 'expand' : 'collapse')}
      </button>
    </div>
    <nav class="nav">
      ${NAV.map((group) => html`
        <div class="nav-group">
          <p class="nav-label side-text">${group.group}</p>
          ${group.items.map(item)}
        </div>`)}
    </nav>
    <div class="side-foot">
      ${item({ name: 'settings', label: 'Settings', icon: 'settings' })}
      ${engineCard()}
      <div class="theme-switch" role="group" aria-label="Theme">
        <button class="theme-btn${theme === 'light' ? ' is-active' : ''}" data-action="set-theme" data-theme="light" aria-pressed="${theme === 'light'}" title="Light theme">${icon('sun', { size: 16 })}<span class="side-text">Light</span></button>
        <button class="theme-btn${theme === 'dark' ? ' is-active' : ''}" data-action="set-theme" data-theme="dark" aria-pressed="${theme === 'dark'}" title="Dark theme">${icon('moon', { size: 16 })}<span class="side-text">Dark</span></button>
      </div>
      <a class="user" href="#/settings" title="${app.settings.name}">
        <span class="avatar" aria-hidden="true">${(app.settings.name || 'N').trim().charAt(0).toUpperCase()}</span>
        <span class="side-text"><b>${app.settings.name}</b><span>This device only</span></span>
      </a>
    </div>`);
  $('#app').classList.toggle('is-collapsed', ui.collapsed);
}

// ---------------------------------------------------------------- session bar

const STATE_TEXT = {
  open: { title: 'Market open', dot: 'is-open' },
  closed: { title: 'Market closed', dot: 'is-closed' },
  pre_open: { title: 'Opens at 09:15', dot: 'is-closed' },
};

function renderTop() {
  const snap = app.snapshot;
  const top = $('#top');
  if (!snap) {
    setHTML(top, html`
      <button class="icon-btn top-menu" data-action="open-nav" aria-label="Open navigation">${icon('menu')}</button>
      <div class="session session-empty">${app.error ? 'Session unavailable' : 'Loading the session…'}</div>`);
    return;
  }
  const { meta, breadth } = snap;
  const state = STATE_TEXT[meta.market_state];
  const sub = meta.market_state === 'open' ? shortDate(meta.session_date) : `Showing ${shortDate(meta.session_date)}`;

  if (!top.querySelector('#ribbon')) {
    setHTML(top, html`
      <button class="icon-btn top-menu" data-action="open-nav" aria-label="Open navigation">${icon('menu')}</button>
      <div class="session">
        <div class="session-state" id="session-state"></div>
        <div class="session-ribbon" id="ribbon"></div>
        <div class="session-now" id="session-now"></div>
      </div>
      <div class="top-tools">
        <button class="search-btn" data-action="open-palette">${icon('search', { size: 16 })}<span>Search</span><kbd>Ctrl K</kbd></button>
        <button class="chip chip-note" data-action="about-data" id="data-chip"></button>
      </div>`);
  }
  setHTML($('#session-state'), html`
    <span class="pulse ${state.dot}" aria-hidden="true"></span>
    <span class="session-text"><b>${state.title}</b><span>${sub}</span></span>`);
  setHTML($('#session-now'), html`
    <span class="now-time">${meta.as_of}</span>
    <span class="now-sub">
      ${meta.is_replay
        ? html`<button class="link-btn" data-action="go-live">${icon('live', { size: 14 })}Back to ${meta.market_state === 'open' ? 'live' : 'the close'}</button>`
        : html`<span class="now-counts"><span class="chg-up">▲ ${breadth.advancing}</span><span class="chg-down">▼ ${breadth.declining}</span></span>`}
    </span>`);
  setHTML($('#data-chip'), html`${meta.data_mode === 'simulated' ? 'Simulated data' : 'Live data'}`);
  $('#top').classList.toggle('is-replay', meta.is_replay);

  // Do not rebuild the ribbon in the middle of a drag: it would drop the pointer capture.
  if (!(ui.ribbon && ui.ribbon.dragging)) {
    ui.ribbon = mountRibbon($('#ribbon'), snap, scrubTo);
  }
}

const refreshSoon = debounce(() => refresh(), 90);

function scrubTo(minute, final) {
  const total = 555 + (minute || 0);
  app.at = minute === null ? null : `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
  if (final) {
    if (ui.ribbon) ui.ribbon.dragging = false;
    refresh();
  } else {
    refreshSoon();
  }
}

function renderFoot() {
  setHTML($('#foot'), html`
    <p>Niveshak is a research and learning tool. It describes what price, volume and open interest did; nothing here is investment advice or a recommendation to buy or sell. Trading derivatives carries a high risk of loss.</p>
    <p><button class="link-btn" data-action="about-data">Where the data comes from</button></p>`);
}

// ---------------------------------------------------------------- data

async function refresh() {
  try {
    app.snapshot = await api.snapshot(app.at);
    app.error = null;
  } catch (error) {
    app.error = error.message;
  }
  renderTop();
  const view = VIEWS[app.route.name];
  if (view && (view.usesSnapshot || !app.snapshot)) renderView();
  if (app.snapshot) {
    stock.onSnapshot();
    if (ui.marketState !== app.snapshot.meta.market_state) {   // the Live tag in the sidebar follows the market
      ui.marketState = app.snapshot.meta.market_state;
      renderSide();
    }
  }
}

async function loadEngine() {
  try {
    const [health, risk] = await Promise.all([api.health(), api.risk()]);
    app.engine = { health, risk };
  } catch (error) {
    app.engine = { error: error.message };
  }
  renderSide();
  if (app.route.name === 'settings') renderView();
}

function schedule() {
  clearInterval(ui.timer);
  const seconds = Number(app.settings.refreshSec);
  if (!seconds) return;
  ui.timer = setInterval(() => {
    if (document.hidden || app.at) return;            // paused in the background and during replay
    ui.ticks += 1;
    const open = !app.snapshot || app.snapshot.meta.market_state === 'open';
    const everyNth = Math.max(1, Math.round(60 / seconds));       // closed: about once a minute is plenty
    if (open || ui.ticks % everyNth === 0) refresh();
  }, seconds * 1000);
}

// ---------------------------------------------------------------- router

function parseRoute() {
  const parts = window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  const name = VIEWS[parts[0]] ? parts[0] : null;
  return name ? { name, arg: parts[1] ? decodeURIComponent(parts[1]) : null } : null;
}

function renderView() {
  const root = $('#view');
  const view = VIEWS[app.route.name];
  hideTip();
  view.render(root, app.route);
}

function navigate() {
  const route = parseRoute();
  if (!route) { go(VIEWS[app.settings.startPage] ? app.settings.startPage : 'home'); return; }
  const changedView = route.name !== app.route.name;
  app.route = route;
  $('#app').classList.remove('nav-open');
  renderSide();
  renderView();
  document.title = `${route.name.charAt(0).toUpperCase() + route.name.slice(1)} | Niveshak`;
  if (changedView) $('.main').scrollTo({ top: 0 });
}

// ---------------------------------------------------------------- search palette

const palette = { items: [], index: 0 };

function paletteItems(query) {
  const q = query.trim().toLowerCase();
  const pages = [...NAV.flatMap((g) => g.items), { name: 'settings', label: 'Settings', icon: 'settings' }]
    .map((p) => ({ kind: 'Page', title: p.label, sub: '', run: () => go(p.name) }));
  const stocks = (app.snapshot ? app.snapshot.stocks : [])
    .map((s) => ({ kind: 'Stock', title: s.symbol, sub: `${s.name}, ${s.sector}`, run: () => hooks.openStock(s.symbol) }));
  const sectors = (app.snapshot ? app.snapshot.sectors : [])
    .map((s) => ({ kind: 'Sector', title: s.name, sub: `${s.count} stocks`, run: () => { flow.setFilter({ type: 'sector', key: s.name, label: s.name }); go('flow/market'); } }));
  if (!q) return [...pages, ...stocks.slice(0, 6)];
  const score = (item) => {
    const title = item.title.toLowerCase();
    if (title === q) return 0;
    if (title.startsWith(q)) return 1;
    if (title.includes(q)) return 2;
    if (item.sub.toLowerCase().includes(q)) return 3;
    return 9;
  };
  return [...stocks, ...sectors, ...pages]
    .map((item) => ({ item, rank: score(item) }))
    .filter((x) => x.rank < 9)
    .sort((a, b) => a.rank - b.rank)
    .slice(0, 12)
    .map((x) => x.item);
}

function renderPaletteList() {
  setHTML($('#palette-list'), palette.items.length
    ? palette.items.map((item, i) => html`
        <li class="pal-item${i === palette.index ? ' is-active' : ''}" role="option" aria-selected="${i === palette.index}" data-action="palette-pick" data-i="${i}">
          <span class="pal-kind">${item.kind}</span><span class="pal-title">${item.title}</span><span class="pal-sub">${item.sub}</span>
        </li>`)
    : html`<li class="pal-empty">Nothing matches. Try a symbol like TCS or a sector like Auto.</li>`);
  const active = $('#palette-list .is-active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function openPalette() {
  const dialog = $('#palette');
  setHTML(dialog, html`
    <div class="pal-head">${icon('search')}<input id="palette-input" type="text" placeholder="Search stocks, sectors and pages" autocomplete="off" spellcheck="false" aria-label="Search" aria-controls="palette-list"></div>
    <ul class="pal-list" id="palette-list" role="listbox"></ul>
    <p class="pal-foot"><kbd>↑</kbd><kbd>↓</kbd> to move, <kbd>Enter</kbd> to open, <kbd>Esc</kbd> to close</p>`);
  palette.items = paletteItems('');
  palette.index = 0;
  renderPaletteList();
  if (!dialog.open) dialog.showModal();
  const input = $('#palette-input');
  input.focus();
  input.addEventListener('input', () => {
    palette.items = paletteItems(input.value);
    palette.index = 0;
    renderPaletteList();
  });
  input.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      palette.index = (palette.index + step + palette.items.length) % Math.max(1, palette.items.length);
      renderPaletteList();
    } else if (event.key === 'Enter') {
      event.preventDefault();
      runPalette(palette.index);
    }
  });
}

function runPalette(index) {
  const item = palette.items[index];
  $('#palette').close();
  if (item) item.run();
}

// ---------------------------------------------------------------- actions

on('toggle-side', () => {
  ui.collapsed = !ui.collapsed;
  store.save('ui', { collapsed: ui.collapsed });
  renderSide();
  setTimeout(() => { renderTop(); renderView(); }, 220);   // charts re-measure after the width transition
});
on('open-nav', () => $('#app').classList.add('nav-open'));
on('close-nav', () => $('#app').classList.remove('nav-open'));
on('set-theme', (el) => {
  saveSettings({ theme: el.dataset.theme });
  applyTheme();
  renderSide();
  if (app.route.name === 'settings') renderView();
});
on('open-palette', openPalette);
on('palette-pick', (el) => runPalette(Number(el.dataset.i)));
on('go-live', () => { app.at = null; refresh(); });
on('retry', () => refresh());
on('close-modal', closeModal);
on('open-stock', (el) => stock.open(el.dataset.symbol));
on('close-stock', () => stock.close());
on('about-data', () => {
  const meta = app.snapshot ? app.snapshot.meta : null;
  openModal(html`
    <h2>Where the data comes from</h2>
    <p>Prices, volume and open interest on these screens are <strong>simulated</strong>. The backend generates one deterministic session per calendar date for ${meta ? meta.universe : 'about 200'} NSE F&amp;O stocks, so the terminal works without a paid market feed. The symbols are real; the numbers are not quotes.</p>
    <p>Everything derived from that feed is computed the same way it would be on real data: breadth, sector averages, the four build-up classes, trend quality, the strength score, pivot levels and order-block zones. The Learn section explains each formula.</p>
    <p>To go live, connect a broker or licensed vendor feed in <code>analytics/market_engine.py</code> (replace <code>simulate_day</code>). Showing exchange data to other people needs a data licence from the exchange or an authorised vendor.</p>
    <p class="muted">Not investment advice. Session hours follow NSE: 09:15 to 15:30 IST, Monday to Friday. Exchange holidays are not modelled yet.</p>`);
});

hooks.refresh = refresh;
hooks.rerender = renderView;
hooks.openStock = (symbol) => stock.open(symbol);
hooks.renderSide = renderSide;
hooks.applyTheme = applyTheme;
hooks.reschedule = schedule;

// ---------------------------------------------------------------- boot

document.addEventListener('click', (event) => {
  const el = event.target.closest('[data-action]');
  if (!el) return;
  const handler = actions[el.dataset.action];
  if (handler) handler(el, event);
});

document.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    openPalette();
    return;
  }
  if (event.key === 'Escape' && !$('#drawer').hidden && !$('#modal').open && !$('#palette').open) {
    stock.close();
    return;
  }
  // Rows and tiles that act like buttons respond to Enter and Space.
  if ((event.key === 'Enter' || event.key === ' ') && event.target.matches('[data-action][tabindex]')) {
    event.preventDefault();
    event.target.click();
  }
});

// Clicking the backdrop of a dialog closes it.
for (const id of ['palette', 'modal']) {
  document.getElementById(id).addEventListener('click', (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
}

window.addEventListener('hashchange', navigate);
window.addEventListener('resize', debounce(() => { renderTop(); renderView(); stock.onResize(); }, 180));
window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => { applyTheme(); renderSide(); });
document.addEventListener('visibilitychange', () => { if (!document.hidden && !app.at) refresh(); });

applyTheme();
renderSide();
renderTop();
renderFoot();
navigate();
refresh();
loadEngine();
schedule();
