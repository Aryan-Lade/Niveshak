// Feedback: ideas, bugs and data issues. Saved on this device and exportable,
// because no feedback inbox is connected yet - and the page says so.

import { $, html, setHTML, icon, timeAgo, toast, downloadText, todayISO } from '../util.js';
import { on } from '../state.js';
import { feedback, FEEDBACK_KINDS } from '../local.js';

export function render(root) {
  const notes = feedback.all();
  setHTML(root, html`
    <header class="page-head">
      <h1>Feedback</h1>
      <p class="lede">Tell us what to build, what broke or where a number looks wrong. Notes are saved on this device; no inbox is connected yet, so use Export to send them on.</p>
    </header>
    <div class="feedback">
      <form class="card form" id="feedback-form" novalidate>
        <h2>New note</h2>
        <fieldset class="field">
          <legend>Kind</legend>
          <div class="segmented">
            ${Object.entries(FEEDBACK_KINDS).map(([id, label], i) => html`<label><input type="radio" name="kind" value="${id}" ${i === 0 ? 'checked' : ''}><span>${label}</span></label>`)}
          </div>
        </fieldset>
        <label class="field"><span>Which screen <em>optional</em></span>
          <select name="screen"><option value="">Not about one screen</option><option>Home</option><option>Flow</option><option>Stock detail</option><option>Journal</option><option>Learn</option><option>Community</option><option>Settings</option></select>
        </label>
        <label class="field"><span>What happened, or what would help</span>
          <textarea name="text" rows="5" maxlength="1000" placeholder="For a bug: what you did, what you expected and what you saw instead."></textarea>
        </label>
        <p class="form-error" id="feedback-error" role="alert" hidden></p>
        <div class="form-actions"><button class="btn" type="submit">Save note</button></div>
      </form>

      <section class="card" aria-labelledby="notes-title">
        <header class="card-head">
          <h2 id="notes-title">Your notes</h2>
          <p class="card-note">${notes.length ? `${notes.length} saved on this device` : 'Nothing saved yet'}</p>
          ${notes.length ? html`<button class="btn btn-quiet" data-action="export-feedback">${icon('download', { size: 16 })}Export</button>` : ''}
        </header>
        ${notes.length
          ? html`<ul class="notes">${notes.map((n) => html`
              <li class="note">
                <p class="post-meta"><span class="chip">${FEEDBACK_KINDS[n.kind] || n.kind}</span>${n.screen ? html`<span class="chip">${n.screen}</span>` : ''}<span class="muted">${timeAgo(n.created)}</span>
                  <button class="icon-btn note-delete" data-action="delete-feedback" data-id="${n.id}" aria-label="Delete this note">${icon('trash', { size: 15 })}</button></p>
                <p class="post-text">${n.text}</p>
              </li>`)}</ul>`
          : html`<p class="muted">Notes you save appear here so you can review and export them.</p>`}
      </section>
    </div>`);

  const form = $('#feedback-form');
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const text = String(data.get('text')).trim();
    const errorEl = $('#feedback-error');
    if (text.length < 10) {
      errorEl.textContent = 'Add a little more detail so the note can be acted on.';
      errorEl.hidden = false;
      form.elements.text.setAttribute('aria-invalid', 'true');
      form.elements.text.focus();
      return;
    }
    feedback.add({ kind: data.get('kind'), screen: data.get('screen'), text });
    toast('Note saved');
    render(root);
  });
}

on('delete-feedback', (el) => { feedback.remove(el.dataset.id); render($('#view')); toast('Note deleted'); });
on('export-feedback', () => {
  const text = feedback.all().map((n) =>
    `[${new Date(n.created).toISOString().slice(0, 16).replace('T', ' ')}] ${FEEDBACK_KINDS[n.kind] || n.kind}${n.screen ? ' / ' + n.screen : ''}\n${n.text}\n`
  ).join('\n');
  downloadText(`niveshak-feedback-${todayISO()}.txt`, text);
  toast('Feedback exported');
});
