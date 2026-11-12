// Community: a feed of setups, analysis and lessons learned. Posts live on this
// device until a community backend exists, and the feed says so.

import { $, html, setHTML, icon, timeAgo, toast } from '../util.js';
import { app, on, hooks } from '../state.js';
import { posts, POST_TAGS } from '../local.js';
import { store } from '../util.js';

const view = { tag: 'all' };

function composer() {
  const draftSymbol = sessionStorage.getItem('niveshak.draft-symbol') || '';
  sessionStorage.removeItem('niveshak.draft-symbol');
  const symbols = app.snapshot ? app.snapshot.stocks.map((s) => s.symbol) : [];
  return html`
    <form class="card composer" id="post-form" novalidate>
      <h2>Share a note</h2>
      <label class="field"><span class="visually-hidden">Your note</span>
        <textarea name="text" rows="3" maxlength="500" placeholder="What did you trade or notice today? Say what you saw, what you did and how it went.">${draftSymbol ? `${draftSymbol}: ` : ''}</textarea>
      </label>
      <div class="composer-row">
        <label class="field field-inline"><span>Kind</span>
          <select name="tag">${Object.entries(POST_TAGS).map(([id, label]) => html`<option value="${id}" ${draftSymbol && id === 'setup' ? 'selected' : ''}>${label}</option>`)}</select>
        </label>
        <label class="field field-inline"><span>Stock <em>optional</em></span>
          <input name="symbol" type="text" list="post-symbols" value="${draftSymbol}" placeholder="TCS" autocomplete="off">
        </label>
        <datalist id="post-symbols">${symbols.map((s) => html`<option value="${s}"></option>`)}</datalist>
        <button class="btn" type="submit">Post</button>
      </div>
      <p class="form-error" id="post-error" role="alert" hidden></p>
    </form>`;
}

function post(p) {
  return html`
    <li class="post card">
      <span class="avatar" aria-hidden="true">${p.author.charAt(0).toUpperCase()}</span>
      <div class="post-body">
        <p class="post-meta">
          <b>${p.author}</b><span class="chip">${POST_TAGS[p.tag] || p.tag}</span>
          ${p.sample ? html`<span class="chip chip-note">Sample</span>` : ''}
          <span class="muted">${timeAgo(p.created)}</span>
        </p>
        <p class="post-text">${p.text}</p>
        <p class="post-actions">
          ${p.symbol ? html`<button class="chip chip-link" data-action="open-stock" data-symbol="${p.symbol}">${p.symbol}</button>` : ''}
          <button class="link-btn${p.liked ? ' is-on' : ''}" data-action="like-post" data-id="${p.id}" aria-pressed="${p.liked}">${icon('heart', { size: 15 })}${p.likes} <span class="visually-hidden">likes</span></button>
          ${p.mine ? html`<button class="link-btn" data-action="delete-post" data-id="${p.id}">${icon('trash', { size: 15 })}Delete</button>` : ''}
        </p>
      </div>
    </li>`;
}

export function render(root) {
  const all = posts.all();
  const feed = view.tag === 'all' ? all : all.filter((p) => p.tag === view.tag);
  const hasSamples = all.some((p) => p.sample);
  setHTML(root, html`
    <header class="page-head page-head-row">
      <div>
        <h1>Community</h1>
        <p class="lede">Setups, analysis and lessons learned. Posts are kept on this device for now; there is no shared feed until a community backend is connected.</p>
      </div>
      <div class="page-tools">
        <div class="tabs" role="group" aria-label="Filter posts">
          <button class="tab${view.tag === 'all' ? ' is-active' : ''}" aria-pressed="${view.tag === 'all'}" data-action="post-filter" data-tag="all">All</button>
          ${Object.entries(POST_TAGS).map(([id, label]) => html`<button class="tab${view.tag === id ? ' is-active' : ''}" aria-pressed="${view.tag === id}" data-action="post-filter" data-tag="${id}">${label}</button>`)}
        </div>
      </div>
    </header>
    <div class="community">
      ${composer()}
      ${hasSamples ? html`<p class="notice">Posts marked Sample are examples that show how the feed looks. <button class="link-btn" data-action="hide-samples">Hide the samples</button></p>` : ''}
      ${feed.length
        ? html`<ul class="feed">${feed.map(post)}</ul>`
        : html`<div class="empty"><h3>${all.length ? 'No posts of this kind yet' : 'No posts yet'}</h3><p>${all.length ? 'Pick another filter, or write the first one above.' : 'Write the first note above.'}</p></div>`}
    </div>`);

  const form = $('#post-form');
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const data = new FormData(form);
    const text = String(data.get('text')).trim();
    const errorEl = $('#post-error');
    if (text.length < 10) {
      errorEl.textContent = 'Write at least a sentence so the note is useful to someone else.';
      errorEl.hidden = false;
      form.elements.text.focus();
      return;
    }
    posts.add({ author: app.settings.name || 'You', tag: data.get('tag'), symbol: String(data.get('symbol')).trim().toUpperCase(), text });
    toast('Posted');
    view.tag = 'all';
    render(root);
  });
  if (form.elements.text.value) {
    form.elements.text.focus();
    form.elements.text.setSelectionRange(form.elements.text.value.length, form.elements.text.value.length);
  }
}

on('post-filter', (el) => { view.tag = el.dataset.tag; render($('#view')); });
on('like-post', (el) => { posts.toggleLike(el.dataset.id); hooks.rerender(); });
on('hide-samples', () => { posts.hideSamples(); render($('#view')); toast('Samples hidden'); });
on('delete-post', (el) => {
  const removed = store.load('posts', []).find((p) => p.id === el.dataset.id);
  posts.remove(el.dataset.id);
  render($('#view'));
  toast('Post deleted', {
    action: 'Undo',
    onAction: () => {
      const mine = store.load('posts', []);
      mine.push(removed);
      store.save('posts', mine);
      if (app.route.name === 'community') render($('#view'));
    },
  });
});
