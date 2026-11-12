// Settings: profile, appearance, refresh, engine status and local data.

import { $, html, setHTML, icon, toast, downloadText, todayISO } from '../util.js';
import { app, on, hooks, saveSettings, DEFAULT_SETTINGS } from '../state.js';
import { exportLocalData, clearLocalData } from '../local.js';

const view = { confirmClear: false };

const pctText = (value) => `${Math.abs(value * 100).toFixed(0)}%`;

function engineSection() {
  const engine = app.engine;
  if (!engine) return html`<p class="muted">Checking the engine…</p>`;
  if (engine.error) return html`<p class="form-error">Engine status is unavailable: ${engine.error}</p>`;
  const { health, risk } = engine;
  return html`
    <dl class="plan">
      <div><dt>Mode</dt><dd>${health.mode === 'PAPER' ? 'Paper trading. No live orders are ever placed.' : health.mode}</dd></div>
      <div><dt>Kill switch</dt><dd>${risk.is_tripped ? `Tripped: ${risk.reason || 'no reason recorded'}` : 'Armed, not tripped'}</dd></div>
      <div><dt>Open positions allowed</dt><dd>${risk.max_positions}</dd></div>
      <div><dt>Largest single stock</dt><dd>${pctText(risk.max_weight_per_stock)} of the portfolio</dd></div>
      <div><dt>Daily loss limit</dt><dd>${pctText(risk.daily_loss_limit_pct)}</dd></div>
      <div><dt>Drawdown circuit breaker</dt><dd>${pctText(risk.drawdown_circuit_breaker_pct)} from the peak</dd></div>
      <div><dt>API version</dt><dd>${health.version}</dd></div>
    </dl>
    <p class="muted">These limits come from configs/risk.yaml and are enforced by the risk engine, not by this page. Tripping or resetting the kill switch needs the API key and is done through the API.</p>`;
}

export function render(root) {
  const s = app.settings;
  const meta = app.snapshot ? app.snapshot.meta : null;
  setHTML(root, html`
    <header class="page-head">
      <h1>Settings</h1>
      <p class="lede">Preferences are saved on this device.</p>
    </header>
    <div class="settings">
      <section class="card" aria-labelledby="set-profile">
        <h2 id="set-profile">Profile</h2>
        <form class="form" id="profile-form" novalidate>
          <label class="field"><span>Display name</span><input name="name" type="text" maxlength="30" value="${s.name}" autocomplete="given-name"></label>
          <div class="form-actions"><button class="btn" type="submit">Save name</button></div>
        </form>
      </section>

      <section class="card" aria-labelledby="set-look">
        <h2 id="set-look">Appearance</h2>
        <fieldset class="field">
          <legend>Theme</legend>
          <div class="segmented">
            ${[['dark', 'Dark'], ['light', 'Light'], ['system', 'Match my device']].map(([id, label]) => html`
              <label><input type="radio" name="theme" value="${id}" ${s.theme === id ? 'checked' : ''} data-setting="theme"><span>${label}</span></label>`)}
          </div>
        </fieldset>
        <label class="field"><span>Open Niveshak on</span>
          <select data-setting="startPage">
            ${[['home', 'Home'], ['flow', 'Flow'], ['journal', 'Journal']].map(([id, label]) => html`<option value="${id}" ${s.startPage === id ? 'selected' : ''}>${label}</option>`)}
          </select>
        </label>
      </section>

      <section class="card" aria-labelledby="set-data">
        <h2 id="set-data">Session data</h2>
        <label class="field"><span>Refresh the session</span>
          <select data-setting="refreshSec">
            ${[[10, 'Every 10 seconds'], [20, 'Every 20 seconds'], [60, 'Every minute'], [0, 'Only when I reload']].map(([v, label]) => html`<option value="${v}" ${Number(s.refreshSec) === v ? 'selected' : ''}>${label}</option>`)}
          </select>
        </label>
        <dl class="plan">
          <div><dt>Feed</dt><dd>${meta ? (meta.data_mode === 'simulated' ? 'Simulated session, one per calendar date' : 'Live') : 'Not loaded'}</dd></div>
          <div><dt>Universe</dt><dd>${meta ? `${meta.universe} NSE F&O stocks` : 'Not loaded'}</dd></div>
          <div><dt>Session shown</dt><dd>${meta ? `${meta.session_date}, as of ${meta.as_of} IST` : 'Not loaded'}</dd></div>
        </dl>
        <button class="btn btn-quiet" data-action="about-data">Where the data comes from</button>
      </section>

      <section class="card" aria-labelledby="set-engine">
        <h2 id="set-engine">Engine</h2>
        ${engineSection()}
      </section>

      <section class="card" aria-labelledby="set-local">
        <h2 id="set-local">Your data on this device</h2>
        <p>Journal trades, community posts, feedback notes, lesson progress and the watchlist are stored in this browser. Export a copy before clearing browser data or switching devices.</p>
        <div class="form-actions form-actions-start">
          <button class="btn btn-quiet" data-action="export-all">${icon('download', { size: 16 })}Export everything</button>
          ${view.confirmClear
            ? html`<span class="confirm">Delete all of it from this device? <button class="btn btn-danger" data-action="clear-all-yes">Delete everything</button><button class="btn btn-quiet" data-action="clear-all-no">Keep it</button></span>`
            : html`<button class="btn btn-quiet" data-action="clear-all">${icon('trash', { size: 16 })}Delete everything</button>`}
        </div>
      </section>
    </div>`);

  $('#profile-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const name = String(new FormData(event.target).get('name')).trim() || DEFAULT_SETTINGS.name;
    saveSettings({ name });
    hooks.renderSide();
    toast('Name saved');
    render(root);
  });
  for (const control of root.querySelectorAll('[data-setting]')) {
    control.addEventListener('change', () => {
      const key = control.dataset.setting;
      const value = key === 'refreshSec' ? Number(control.value) : control.value;
      saveSettings({ [key]: value });
      if (key === 'theme') { hooks.applyTheme(); hooks.renderSide(); }
      if (key === 'refreshSec') hooks.reschedule();
      toast('Saved');
    });
  }
}

on('export-all', () => {
  downloadText(`niveshak-data-${todayISO()}.json`, JSON.stringify(exportLocalData(), null, 2), 'application/json');
  toast('Export downloaded');
});
on('clear-all', () => { view.confirmClear = true; render($('#view')); });
on('clear-all-no', () => { view.confirmClear = false; render($('#view')); });
on('clear-all-yes', () => {
  clearLocalData();
  view.confirmClear = false;
  render($('#view'));
  toast('Everything on this device was deleted');
});
