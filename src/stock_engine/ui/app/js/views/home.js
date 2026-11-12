// Home: the session at a glance, then your own practice (journal, lessons, community).

import { $, html, setHTML, icon, buildupMark, pct, num, signedInr, longDate, timeAgo } from '../util.js';
import { app, on } from '../state.js';
import { mountHistogram, mountSectorLines, sparkline } from '../charts.js';
import { change, advDecBar, zoneBand, errorState } from '../components.js';
import { journal, journalStats, filterPeriod, posts, POST_TAGS, progress, watchlist } from '../local.js';
import { api } from '../api.js';

export const usesSnapshot = true;

const view = { allSectors: false, lines: null };

function greeting() {
  const hour = new Date().getHours();
  const part = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
  return `${part}, ${app.settings.name}`;
}

function moodLine(snap) {
  const { breadth, meta } = snap;
  const when = meta.is_replay ? `At ${meta.as_of}` : meta.market_state === 'open' ? 'Right now' : 'At the close';
  const tone = { 'Broadly higher': 'so the market is broadly higher', 'Broadly lower': 'so the market is broadly lower', Mixed: 'so the market is mixed' }[breadth.mood];
  return `${longDate(meta.session_date)}. ${when}, ${breadth.advancing} stocks are up and ${breadth.declining} are down, ${tone}.`;
}

// ---------------------------------------------------------------- cards

function breadthCard(snap) {
  const b = snap.breadth;
  const mix = [
    ['long_buildup', 'Long build-up', 'bg-up'],
    ['short_covering', 'Short covering', 'ol-up'],
    ['long_unwinding', 'Long unwinding', 'ol-down'],
    ['short_buildup', 'Short build-up', 'bg-down'],
  ];
  const total = mix.reduce((n, [key]) => n + b.buildup[key], 0) || 1;
  return html`
    <section class="card" aria-labelledby="breadth-title">
      <header class="card-head">
        <h2 id="breadth-title">Breadth</h2>
        <p class="card-note">${snap.meta.universe} F&amp;O stocks, change from the previous close</p>
      </header>
      <div class="breadth">
        <dl class="figures">
          <div><dt>Advancing</dt><dd class="fig chg-up"><span class="chg-arrow" aria-hidden="true">▲</span>${b.advancing}</dd></div>
          <div><dt>Declining</dt><dd class="fig chg-down"><span class="chg-arrow" aria-hidden="true">▼</span>${b.declining}</dd></div>
          <div><dt>Unchanged</dt><dd class="fig">${b.unchanged}</dd></div>
          <div><dt>Average move</dt><dd class="fig">${pct(b.average_change_pct)}</dd></div>
        </dl>
        <div class="breadth-hist" id="home-hist"></div>
      </div>
      <div class="mix">
        <p class="mix-title">Open-interest build-up <span class="muted">solid is fresh positions, outlined is positions closing</span></p>
        <div class="mix-bar" role="img" aria-label="Build-up mix">
          ${mix.map(([key, , cls]) => (b.buildup[key] ? html`<i class="${cls}" style="flex:${b.buildup[key] / total}"></i>` : ''))}
        </div>
        <ul class="mix-legend">
          ${mix.map(([key, label]) => html`<li>${buildupMark(key)}<span>${label}</span><b>${b.buildup[key]}</b></li>`)}
        </ul>
      </div>
    </section>`;
}

function boardColumn(title, rows, side) {
  return html`
    <div class="board-col">
      <h3>${title}</h3>
      <ol class="board-list">
        ${rows.slice(0, 5).map((r) => html`
          <li class="board-row" data-action="open-stock" data-symbol="${r.symbol}" tabindex="0">
            <span class="board-rank">${r.rank}</span>
            <span class="board-id"><span class="sym">${r.symbol}</span><span class="sub sub-mark">${buildupMark(r.buildup, 9)}${r.buildup_label}</span></span>
            ${sparkline(r.spark, { width: 88, height: 32, band: zoneBand(r) })}
            <span class="board-num">${change(r.change_pct)}</span>
            <span class="board-score" title="Strength score out of 100"><b>${r.score}</b><span class="score-track"><i class="score-fill bg-${side === 'bullish' ? 'up' : 'down'}" style="width:${r.score}%"></i></span><span class="visually-hidden">strength score</span></span>
          </li>`)}
      </ol>
      ${rows.length ? '' : html`<p class="muted">No ${side} setups have cleared the choppy filter yet.</p>`}
    </div>`;
}

function boardsCard(snap) {
  return html`
    <section class="card" aria-labelledby="boards-title">
      <header class="card-head">
        <h2 id="boards-title">Strongest setups</h2>
        <p class="card-note">${snap.meta.settling ? 'The session has just opened; rankings settle after a few bars.' : 'Ranked by strength score. The shaded band on each line is its order-block zone.'}</p>
        <a class="card-link" href="#/flow/bullish">All boards${icon('arrowRight', { size: 15 })}</a>
      </header>
      <div class="boards">
        ${boardColumn('Bullish', snap.boards.bullish, 'bullish')}
        ${boardColumn('Bearish', snap.boards.bearish, 'bearish')}
      </div>
    </section>`;
}

function sectorRow(s) {
  return html`
    <li class="sector-row" data-action="pick-sector" data-sector="${s.name}" tabindex="0">
      <span class="sector-name">${s.name}</span>
      ${advDecBar(s)}
      <span class="sector-count">${s.advancing}/${s.count} up</span>
      ${change(s.change_pct)}
    </li>`;
}

function sectorsCard(snap) {
  const sectors = snap.sectors;
  const middle = sectors.slice(3, -3);
  return html`
    <section class="card" aria-labelledby="sectors-title">
      <header class="card-head">
        <h2 id="sectors-title">Sectors now</h2>
        <p class="card-note">Equal-weighted average of each sector's stocks</p>
        <a class="card-link" href="#/flow/market">Open Flow${icon('arrowRight', { size: 15 })}</a>
      </header>
      <div class="sector-chart" id="home-sectors"></div>
      <div class="sector-lists">
        <div><h3>Strongest</h3><ul class="sector-list">${sectors.slice(0, 3).map(sectorRow)}</ul></div>
        <div><h3>Weakest</h3><ul class="sector-list">${sectors.slice(-3).reverse().map(sectorRow)}</ul></div>
      </div>
      ${view.allSectors ? html`<ul class="sector-list sector-list-all">${middle.map(sectorRow)}</ul>` : ''}
      <button class="btn btn-quiet" data-action="toggle-sectors" aria-expanded="${view.allSectors}">
        ${view.allSectors ? 'Show fewer sectors' : `Show ${middle.length} more sectors`}
      </button>
    </section>`;
}

function journalCard() {
  const month = filterPeriod(journal.all(), 'month');
  if (!journal.all().length) {
    return html`
      <section class="card card-side">
        <h2>Journal</h2>
        <p>Log your first trade. Profit and loss, win rate and an equity curve build up from what you record.</p>
        <button class="btn" data-action="log-trade">${icon('plus', { size: 16 })}Log a trade</button>
      </section>`;
  }
  const stats = journalStats(month);
  return html`
    <section class="card card-side">
      <header class="card-head"><h2>Journal</h2><a class="card-link" href="#/journal">Open${icon('arrowRight', { size: 15 })}</a></header>
      <dl class="figures figures-tight">
        <div><dt>This month</dt><dd class="fig ${stats.net > 0 ? 'chg-up' : stats.net < 0 ? 'chg-down' : ''}">${signedInr(stats.net)}</dd></div>
        <div><dt>Win rate</dt><dd class="fig">${stats.winRate === null ? 'No trades' : Math.round(stats.winRate * 100) + '%'}</dd></div>
        <div><dt>Trades</dt><dd class="fig">${stats.count}</dd></div>
      </dl>
      <button class="btn btn-quiet" data-action="log-trade">${icon('plus', { size: 16 })}Log a trade</button>
    </section>`;
}

function learnCard() {
  const course = app.lessons;
  if (!course) return html`<section class="card card-side"><h2>Learn</h2><p class="muted">Loading lessons…</p></section>`;
  const done = progress.done();
  const all = course.modules.flatMap((m) => m.lessons.map((l) => ({ ...l, module: m.title })));
  const next = all.find((l) => !done.has(l.id));
  const finished = all.filter((l) => done.has(l.id)).length;
  return html`
    <section class="card card-side">
      <header class="card-head"><h2>Learn</h2><span class="card-note">${finished} of ${all.length} done</span></header>
      <div class="meter" role="img" aria-label="${finished} of ${all.length} lessons done"><i style="width:${(finished / all.length) * 100}%"></i></div>
      ${next
        ? html`<a class="next-lesson" href="#/learn/${next.id}">
            <span class="sub">${finished ? 'Up next' : 'Start here'}, ${next.module}</span>
            <b>${next.title}</b>
            <span class="sub">${icon('clock', { size: 14 })}${next.minutes} min read</span>
          </a>`
        : html`<p>You have finished every lesson. Revisit any of them from the Learn page.</p>`}
    </section>`;
}

function watchCard(snap) {
  const symbols = watchlist.all();
  if (!symbols.length) return '';
  const rows = symbols.map((s) => snap.stocks.find((x) => x.symbol === s)).filter(Boolean);
  return html`
    <section class="card card-side">
      <h2>Watchlist</h2>
      <ul class="watch-list">
        ${rows.map((r) => html`
          <li class="watch-row" data-action="open-stock" data-symbol="${r.symbol}" tabindex="0">
            <span class="sym">${r.symbol}</span><span class="num">${num(r.ltp)}</span>${change(r.change_pct)}
          </li>`)}
      </ul>
    </section>`;
}

function shortcutsCard() {
  const links = [
    ['log-trade', 'plus', 'Log a trade', 'Add it to your journal', null],
    [null, 'community', 'Share a setup', 'Post to the community', '#/community'],
    [null, 'book', 'Browse lessons', 'Twelve short reads', '#/learn'],
    [null, 'feedback', 'Send feedback', 'Ideas, bugs, data issues', '#/feedback'],
  ];
  return html`
    <section class="card card-side">
      <h2>Shortcuts</h2>
      <ul class="shortcuts">
        ${links.map(([action, ic, title, sub, href]) => html`
          <li>${href
            ? html`<a class="shortcut" href="${href}">${icon(ic)}<span><b>${title}</b><span class="sub">${sub}</span></span></a>`
            : html`<button class="shortcut" data-action="${action}">${icon(ic)}<span><b>${title}</b><span class="sub">${sub}</span></span></button>`}</li>`)}
      </ul>
    </section>`;
}

function communityCard() {
  const feed = posts.all().slice(0, 4);
  return html`
    <section class="card" aria-labelledby="community-title">
      <header class="card-head">
        <h2 id="community-title">Community this week</h2>
        <a class="card-link" href="#/community">Open feed${icon('arrowRight', { size: 15 })}</a>
      </header>
      <ul class="feed feed-compact">
        ${feed.map((p) => html`
          <li class="post">
            <span class="avatar" aria-hidden="true">${p.author.charAt(0).toUpperCase()}</span>
            <div class="post-body">
              <p class="post-meta"><b>${p.author}</b><span class="chip">${POST_TAGS[p.tag] || p.tag}</span>${p.sample ? html`<span class="chip chip-note">Sample</span>` : ''}<span class="muted">${timeAgo(p.created)}</span></p>
              <p class="post-text clamp-2">${p.text}</p>
            </div>
          </li>`)}
      </ul>
    </section>`;
}

// ---------------------------------------------------------------- render

export function render(root) {
  const snap = app.snapshot;
  if (!snap) {
    setHTML(root, app.error ? errorState(app.error) : html`<p class="boot">Loading the session…</p>`);
    return;
  }
  if (!app.lessons) {
    api.lessons().then((course) => { app.lessons = course; if (app.route.name === 'home') render(root); }).catch(() => {});
  }
  setHTML(root, html`
    <header class="page-head">
      <h1>${greeting()}</h1>
      <p class="lede">${moodLine(snap)}</p>
    </header>
    <div class="home-grid">
      <div class="home-main">
        ${breadthCard(snap)}
        ${boardsCard(snap)}
      </div>
      <aside class="home-side" aria-label="Your practice">
        ${journalCard()}
        ${learnCard()}
        ${watchCard(snap)}
        ${shortcutsCard()}
      </aside>
      <div class="home-wide">
        ${sectorsCard(snap)}
        ${communityCard()}
      </div>
    </div>`);

  mountHistogram($('#home-hist'), snap.breadth.histogram);
  view.lines = mountSectorLines($('#home-sectors'), snap.sectors, snap.meta);
  for (const row of root.querySelectorAll('.sector-row')) {
    row.addEventListener('pointerenter', () => view.lines.highlight(row.dataset.sector));
    row.addEventListener('pointerleave', () => view.lines.highlight(null));
    row.addEventListener('focus', () => view.lines.highlight(row.dataset.sector));
    row.addEventListener('blur', () => view.lines.highlight(null));
  }
}

on('toggle-sectors', () => {
  view.allSectors = !view.allSectors;
  render($('#view'));
});
