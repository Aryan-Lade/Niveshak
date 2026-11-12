// Journal: log closed trades, then read your own numbers.

import { $, html, setHTML, icon, num, inr, signedInr, shortDate, todayISO, toast, downloadText } from '../util.js';
import { app, on, hooks } from '../state.js';
import { mountEquity } from '../charts.js';
import { openModal, closeModal, emptyState } from '../components.js';
import { journal, journalStats, filterPeriod, tradePnl, tradeR } from '../local.js';
import { store } from '../util.js';

const PERIODS = [['month', 'This month'], ['30d', 'Last 30 days'], ['fy', 'This financial year'], ['all', 'All time']];
const SETUPS = ['Order-block retest', 'Breakout', 'Pullback', 'Reversal', 'Other'];

const view = { period: 'fy' };

const tone = (value) => (value > 0 ? 'chg-up' : value < 0 ? 'chg-down' : '');
const orDash = (value, format) => (value === null || value === undefined ? html`<span class="muted">None yet</span>` : format(value));

// ---------------------------------------------------------------- log-a-trade form

function tradeForm(prefill) {
  const symbols = app.snapshot ? app.snapshot.stocks.map((s) => s.symbol) : [];
  return html`
    <h2>Log a trade</h2>
    <p class="muted">Record a closed trade. It is saved on this device.</p>
    <form id="trade-form" class="form" novalidate>
      <div class="form-row">
        <label class="field"><span>Date</span><input name="date" type="date" value="${todayISO()}" max="${todayISO()}" required></label>
        <label class="field"><span>Stock</span><input name="symbol" type="text" list="trade-symbols" value="${prefill.symbol || ''}" placeholder="TCS" autocomplete="off" required></label>
        <datalist id="trade-symbols">${symbols.map((s) => html`<option value="${s}"></option>`)}</datalist>
      </div>
      <fieldset class="field">
        <legend>Side</legend>
        <div class="segmented">
          <label><input type="radio" name="side" value="long" ${prefill.side === 'short' ? '' : 'checked'}><span>Long</span></label>
          <label><input type="radio" name="side" value="short" ${prefill.side === 'short' ? 'checked' : ''}><span>Short</span></label>
        </div>
      </fieldset>
      <div class="form-row form-row-3">
        <label class="field"><span>Quantity</span><input name="qty" type="number" min="1" step="1" inputmode="numeric" required></label>
        <label class="field"><span>Entry price</span><input name="entry" type="number" min="0" step="0.05" inputmode="decimal" value="${prefill.entry || ''}" required></label>
        <label class="field"><span>Exit price</span><input name="exit" type="number" min="0" step="0.05" inputmode="decimal" required></label>
      </div>
      <div class="form-row">
        <label class="field"><span>Planned stop <em>optional, gives you R</em></span><input name="stop" type="number" min="0" step="0.05" inputmode="decimal"></label>
        <label class="field"><span>Charges <em>optional, in rupees</em></span><input name="charges" type="number" min="0" step="0.01" inputmode="decimal"></label>
      </div>
      <label class="field"><span>Setup</span>
        <select name="setup">${SETUPS.map((s) => html`<option>${s}</option>`)}</select>
      </label>
      <label class="field"><span>Notes <em>optional</em></span><textarea name="notes" rows="3" maxlength="600" placeholder="What did you see, and did you follow the plan?"></textarea></label>
      <p class="form-error" id="trade-error" role="alert" hidden></p>
      <div class="form-actions">
        <button type="button" class="btn btn-quiet" data-action="close-modal">Cancel</button>
        <button type="submit" class="btn">Save trade</button>
      </div>
    </form>`;
}

function readTrade(form) {
  const data = new FormData(form);
  const number = (name) => (data.get(name) === '' ? null : Number(data.get(name)));
  const trade = {
    date: data.get('date'),
    symbol: String(data.get('symbol')).trim().toUpperCase(),
    side: data.get('side'),
    qty: number('qty'),
    entry: number('entry'),
    exit: number('exit'),
    stop: number('stop'),
    charges: number('charges') || 0,
    setup: data.get('setup'),
    notes: String(data.get('notes')).trim(),
  };
  if (!trade.date) return { error: 'Pick the date of the trade.', field: 'date' };
  if (trade.date > todayISO()) return { error: 'The date cannot be in the future.', field: 'date' };
  if (!trade.symbol) return { error: 'Enter the stock symbol, for example TCS.', field: 'symbol' };
  if (!Number.isInteger(trade.qty) || trade.qty < 1) return { error: 'Quantity must be a whole number of 1 or more.', field: 'qty' };
  if (!(trade.entry > 0)) return { error: 'Enter the entry price.', field: 'entry' };
  if (!(trade.exit > 0)) return { error: 'Enter the exit price.', field: 'exit' };
  if (trade.stop !== null) {
    if (!(trade.stop > 0)) return { error: 'The stop must be a price above zero, or leave it empty.', field: 'stop' };
    if (trade.side === 'long' && trade.stop >= trade.entry) return { error: 'For a long, the stop sits below the entry price.', field: 'stop' };
    if (trade.side === 'short' && trade.stop <= trade.entry) return { error: 'For a short, the stop sits above the entry price.', field: 'stop' };
  }
  return { trade };
}

function openTradeForm(prefill = {}) {
  openModal(tradeForm(prefill), {
    onOpen(modal) {
      const form = $('#trade-form', modal);
      const errorEl = $('#trade-error', modal);
      (prefill.symbol ? form.elements.qty : form.elements.symbol).focus();
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        for (const el of form.querySelectorAll('[aria-invalid]')) el.removeAttribute('aria-invalid');
        const { trade, error, field } = readTrade(form);
        if (error) {
          errorEl.textContent = error;
          errorEl.hidden = false;
          form.elements[field].setAttribute('aria-invalid', 'true');
          form.elements[field].focus();
          return;
        }
        journal.add(trade);
        closeModal();
        const pnl = tradePnl(trade);
        toast(`Trade saved: ${trade.symbol} ${signedInr(pnl)}`);
        if (['journal', 'home'].includes(app.route.name)) hooks.rerender();
      });
    },
  });
}

// ---------------------------------------------------------------- page

function statTiles(stats) {
  return html`
    <dl class="tiles">
      <div class="tile"><dt>Net profit and loss</dt><dd class="fig fig-lg ${tone(stats.net)}">${signedInr(stats.net)}</dd><p class="sub">${stats.count} ${stats.count === 1 ? 'trade' : 'trades'}, after any charges you logged</p></div>
      <div class="tile"><dt>Win rate</dt><dd class="fig fig-lg">${orDash(stats.winRate, (v) => Math.round(v * 100) + '%')}</dd><p class="sub">Read it with the next two figures, never alone</p></div>
      <div class="tile"><dt>Average win</dt><dd class="fig fig-lg">${orDash(stats.avgWin, (v) => inr(v, 0))}</dd><p class="sub">Average loss ${orDash(stats.avgLoss, (v) => inr(Math.abs(v), 0))}</p></div>
      <div class="tile"><dt>Profit factor</dt><dd class="fig fig-lg">${orDash(stats.profitFactor, (v) => v.toFixed(2))}</dd><p class="sub">Gross wins divided by gross losses. Above 1 is profitable</p></div>
      <div class="tile"><dt>Average R</dt><dd class="fig fig-lg">${orDash(stats.avgR, (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2) + 'R')}</dd><p class="sub">Result against planned risk, for trades with a stop</p></div>
    </dl>`;
}

function setupBreakdown(trades) {
  const groups = new Map();
  for (const t of trades) {
    const g = groups.get(t.setup) || [];
    g.push(t);
    groups.set(t.setup, g);
  }
  const rows = [...groups.entries()].map(([setup, list]) => ({ setup, stats: journalStats(list) })).sort((a, b) => b.stats.net - a.stats.net);
  return html`
    <section class="card">
      <header class="card-head"><h2>By setup</h2><p class="card-note">Which kinds of trade are paying</p></header>
      <div class="table-wrap">
        <table class="table table-plain">
          <thead><tr><th>Setup</th><th class="num">Trades</th><th class="num">Win rate</th><th class="num">Net</th></tr></thead>
          <tbody>${rows.map((r) => html`<tr><td>${r.setup}</td><td class="num">${r.stats.count}</td><td class="num">${Math.round(r.stats.winRate * 100)}%</td><td class="num ${tone(r.stats.net)}">${signedInr(r.stats.net)}</td></tr>`)}</tbody>
        </table>
      </div>
    </section>`;
}

function tradesTable(trades) {
  const rows = [...trades].reverse();
  return html`
    <section class="card">
      <header class="card-head"><h2>Trades</h2><p class="card-note">Newest first</p></header>
      <div class="table-wrap">
        <table class="table table-plain">
          <thead><tr><th>Date</th><th>Stock</th><th>Side</th><th class="num">Qty</th><th class="num">Entry</th><th class="num">Exit</th><th class="num">Result</th><th class="num">R</th><th>Setup</th><th>Notes</th><th><span class="visually-hidden">Delete</span></th></tr></thead>
          <tbody>
            ${rows.map((t) => {
              const pnl = tradePnl(t);
              const r = tradeR(t);
              return html`<tr>
                <td>${shortDate(t.date)}</td>
                <td><span class="sym">${t.symbol}</span></td>
                <td>${t.side === 'short' ? 'Short' : 'Long'}</td>
                <td class="num">${t.qty}</td>
                <td class="num">${num(t.entry)}</td>
                <td class="num">${num(t.exit)}</td>
                <td class="num ${tone(pnl)}">${signedInr(pnl)}</td>
                <td class="num">${r === null ? html`<span class="muted">No stop</span>` : (r > 0 ? '+' : r < 0 ? '−' : '') + Math.abs(r).toFixed(2)}</td>
                <td>${t.setup}</td>
                <td class="note-cell" title="${t.notes}">${t.notes}</td>
                <td><button class="icon-btn" data-action="delete-trade" data-id="${t.id}" aria-label="Delete ${t.symbol} trade of ${shortDate(t.date)}">${icon('trash', { size: 16 })}</button></td>
              </tr>`;
            })}
          </tbody>
        </table>
      </div>
    </section>`;
}

export function render(root) {
  const all = journal.all();
  const trades = filterPeriod(all, view.period);
  const stats = journalStats(trades);

  setHTML(root, html`
    <header class="page-head page-head-row">
      <div>
        <h1>Journal</h1>
        <p class="lede">Your closed trades and what they add up to. Stored on this device only.</p>
      </div>
      <div class="page-tools">
        <div class="tabs" role="group" aria-label="Period">
          ${PERIODS.map(([id, label]) => html`<button class="tab${view.period === id ? ' is-active' : ''}" aria-pressed="${view.period === id}" data-action="journal-period" data-period="${id}">${label}</button>`)}
        </div>
        ${all.length ? html`<button class="btn btn-quiet" data-action="export-journal">${icon('download', { size: 16 })}Export CSV</button>` : ''}
        <button class="btn" data-action="log-trade">${icon('plus', { size: 16 })}Log a trade</button>
      </div>
    </header>
    ${!all.length
      ? emptyState({
          title: 'No trades logged yet',
          text: 'Add each trade after you close it: the stock, your entry and exit, the stop you planned and the setup you were trading. The numbers on this page build up from there.',
          action: html`<button class="btn" data-action="log-trade">${icon('plus', { size: 16 })}Log your first trade</button>`,
        })
      : !trades.length
        ? emptyState({ title: 'No trades in this period', text: 'Pick a longer period, or log a trade dated inside this one.', action: html`<button class="btn btn-quiet" data-action="journal-period" data-period="all">Show all time</button>` })
        : html`
          <div class="stack">
            ${statTiles(stats)}
            <div class="split">
              <section class="card">
                <header class="card-head"><h2>Equity curve</h2><p class="card-note">Running total after each trade</p></header>
                <div id="journal-equity"></div>
              </section>
              ${setupBreakdown(trades)}
            </div>
            ${tradesTable(trades)}
          </div>`}`);

  if (trades.length) {
    let running = 0;
    const points = trades.map((t) => {
      const pnl = tradePnl(t);
      running += pnl;
      return { label: `${t.symbol}, ${shortDate(t.date)}`, note: `${t.side === 'short' ? 'Short' : 'Long'} ${t.qty}: ${signedInr(pnl)}`, value: running };
    });
    mountEquity($('#journal-equity'), points);
  }
}

on('log-trade', (el) => openTradeForm({ symbol: el.dataset.symbol, entry: el.dataset.entry, side: el.dataset.side }));
on('journal-period', (el) => { view.period = el.dataset.period; render($('#view')); });
on('delete-trade', (el) => {
  const removed = journal.all().find((t) => t.id === el.dataset.id);
  journal.remove(el.dataset.id);
  render($('#view'));
  toast('Trade deleted', {
    action: 'Undo',
    onAction: () => {
      const trades = store.load('journal', []);
      trades.push(removed);
      store.save('journal', trades);
      if (app.route.name === 'journal') render($('#view'));
    },
  });
});
on('export-journal', () => {
  const header = ['date', 'symbol', 'side', 'qty', 'entry', 'exit', 'stop', 'charges', 'pnl', 'r', 'setup', 'notes'];
  const cell = (value) => `"${String(value ?? '').replace(/"/g, '""')}"`;
  // Free text that starts like a formula is prefixed so a spreadsheet shows it as text.
  const text = (value) => cell(/^[=+\-@]/.test(String(value)) ? "'" + value : value);
  const lines = journal.all().map((t) => [
    cell(t.date), text(t.symbol), cell(t.side), cell(t.qty), cell(t.entry), cell(t.exit), cell(t.stop ?? ''), cell(t.charges || 0),
    cell(tradePnl(t).toFixed(2)), cell(tradeR(t) === null ? '' : tradeR(t).toFixed(2)), text(t.setup), text(t.notes),
  ].join(','));
  downloadText(`niveshak-journal-${todayISO()}.csv`, [header.join(','), ...lines].join('\n'), 'text/csv');
  toast('Journal exported');
});
