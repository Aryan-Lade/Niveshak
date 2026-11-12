// Everything the terminal keeps on this device: journal trades, community
// posts, feedback notes, lesson progress and the watchlist. There is no
// account system yet, so nothing here leaves the browser.

import { store, uid } from './util.js';

// ---------------------------------------------------------------- journal

export const journal = {
  all() {
    return store.load('journal', []).sort((a, b) => (a.date === b.date ? a.created - b.created : a.date < b.date ? -1 : 1));
  },
  add(trade) {
    const trades = store.load('journal', []);
    trades.push({ id: uid(), created: Date.now(), ...trade });
    store.save('journal', trades);
  },
  remove(id) {
    store.save('journal', store.load('journal', []).filter((t) => t.id !== id));
  },
};

/** Gross P&L of one closed trade, in rupees, before charges. */
export function tradePnl(trade) {
  const direction = trade.side === 'short' ? -1 : 1;
  return (trade.exit - trade.entry) * trade.qty * direction - (trade.charges || 0);
}

/** R multiple: P&L divided by the amount risked to the planned stop. Null when no stop was logged. */
export function tradeR(trade) {
  if (!trade.stop || trade.stop === trade.entry) return null;
  return tradePnl(trade) / (Math.abs(trade.entry - trade.stop) * trade.qty);
}

export function journalStats(trades) {
  const results = trades.map(tradePnl);
  const wins = results.filter((p) => p > 0);
  const losses = results.filter((p) => p < 0);
  const sum = (list) => list.reduce((a, b) => a + b, 0);
  const rs = trades.map(tradeR).filter((r) => r !== null);
  return {
    count: trades.length,
    net: sum(results),
    winRate: trades.length ? wins.length / trades.length : null,
    avgWin: wins.length ? sum(wins) / wins.length : null,
    avgLoss: losses.length ? sum(losses) / losses.length : null,
    profitFactor: losses.length ? sum(wins) / Math.abs(sum(losses)) : null,
    avgR: rs.length ? sum(rs) / rs.length : null,
    best: results.length ? Math.max(...results) : null,
    worst: results.length ? Math.min(...results) : null,
  };
}

/** Trades inside a named period. Indian financial year runs 1 April to 31 March. */
export function filterPeriod(trades, period) {
  const now = new Date();
  const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  let from = '0000-00-00';
  if (period === 'month') from = iso(new Date(now.getFullYear(), now.getMonth(), 1));
  else if (period === '30d') from = iso(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 30));
  else if (period === 'fy') from = iso(new Date(now.getMonth() >= 3 ? now.getFullYear() : now.getFullYear() - 1, 3, 1));
  return trades.filter((t) => t.date >= from);
}

// ---------------------------------------------------------------- community

const hoursAgo = (h) => Date.now() - h * 3600 * 1000;

/** Shown until real posts exist, and labelled as samples in the feed. */
const SAMPLE_POSTS = [
  { id: 'sample-1', sample: true, author: 'Meera', tag: 'setup', symbol: 'HDFCBANK', created: hoursAgo(3), likes: 4,
    text: 'Waited for the retest of the morning demand zone instead of chasing the open. Entry near the top of the zone, stop under it. Took half off at R1.' },
  { id: 'sample-2', sample: true, author: 'Rohan', tag: 'analysis', symbol: '', created: hoursAgo(9), likes: 7,
    text: 'Breadth peaked before 11 and kept slipping while the index held. That divergence kept me out of fresh longs after lunch.' },
  { id: 'sample-3', sample: true, author: 'Farah', tag: 'learning', symbol: '', created: hoursAgo(30), likes: 11,
    text: 'Thirty trades into journaling. Win rate is 47% but average win is 1.8x average loss, so the account is up. I would not have believed that without the numbers.' },
  { id: 'sample-4', sample: true, author: 'Dev', tag: 'analysis', symbol: 'TATASTEEL', created: hoursAgo(52), likes: 2,
    text: 'Up 2% on falling OI. That reads as short covering, not fresh buying, so I passed. It gave most of it back by the close.' },
];

export const posts = {
  all() {
    const mine = store.load('posts', []);
    const hidden = new Set(store.load('posts.hidden', []));
    const likes = store.load('posts.likes', {});
    return [...mine, ...SAMPLE_POSTS.filter((p) => !hidden.has(p.id))]
      .map((p) => ({ ...p, liked: Boolean(likes[p.id]), likes: (p.likes || 0) + (likes[p.id] ? 1 : 0) }))
      .sort((a, b) => b.created - a.created);
  },
  add(post) {
    const mine = store.load('posts', []);
    mine.push({ id: uid(), created: Date.now(), likes: 0, mine: true, ...post });
    store.save('posts', mine);
  },
  remove(id) {
    store.save('posts', store.load('posts', []).filter((p) => p.id !== id));
  },
  hideSamples() {
    store.save('posts.hidden', SAMPLE_POSTS.map((p) => p.id));
  },
  toggleLike(id) {
    const likes = store.load('posts.likes', {});
    if (likes[id]) delete likes[id]; else likes[id] = true;
    store.save('posts.likes', likes);
  },
};

export const POST_TAGS = { setup: 'Trade setup', analysis: 'Analysis', learning: 'Learning', question: 'Question' };

// ---------------------------------------------------------------- feedback

export const feedback = {
  all: () => store.load('feedback', []).sort((a, b) => b.created - a.created),
  add(note) {
    const notes = store.load('feedback', []);
    notes.push({ id: uid(), created: Date.now(), ...note });
    store.save('feedback', notes);
  },
  remove(id) {
    store.save('feedback', store.load('feedback', []).filter((n) => n.id !== id));
  },
};

export const FEEDBACK_KINDS = { idea: 'Idea', bug: 'Bug', data: 'Data issue' };

// ---------------------------------------------------------------- lessons & watchlist

export const progress = {
  done: () => new Set(store.load('progress', [])),
  toggle(lessonId) {
    const done = progress.done();
    if (done.has(lessonId)) done.delete(lessonId); else done.add(lessonId);
    store.save('progress', [...done]);
    return done.has(lessonId);
  },
};

export const watchlist = {
  all: () => store.load('watchlist', []),
  has: (symbol) => store.load('watchlist', []).includes(symbol),
  toggle(symbol) {
    const list = store.load('watchlist', []);
    const next = list.includes(symbol) ? list.filter((s) => s !== symbol) : [...list, symbol];
    store.save('watchlist', next);
    return next.includes(symbol);
  },
};

export const LOCAL_KEYS = ['journal', 'posts', 'posts.hidden', 'posts.likes', 'feedback', 'progress', 'watchlist'];

export function exportLocalData() {
  const out = {};
  for (const key of [...LOCAL_KEYS, 'settings']) out[key] = store.load(key, null);
  return out;
}

export function clearLocalData() {
  for (const key of LOCAL_KEYS) store.remove(key);
}
