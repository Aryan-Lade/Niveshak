// Small shared helpers: safe HTML templating, formatting, icons, storage,
// tooltip and toast. No dependencies.

// ---------------------------------------------------------------- templating

class Safe {
  constructor(text) { this.text = text; }
  toString() { return this.text; }
}

export function esc(value) {
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function render(value) {
  if (value === null || value === undefined || value === false) return '';
  if (value instanceof Safe) return value.text;
  if (Array.isArray(value)) return value.map(render).join('');
  return esc(value);
}

/** Tagged template: every interpolated value is escaped unless it is itself html`` or raw(). */
export function html(strings, ...values) {
  let out = strings[0];
  for (let i = 0; i < values.length; i++) out += render(values[i]) + strings[i + 1];
  return new Safe(out);
}

/** Mark a string that is already safe markup (icons, SVG built by this app). */
export const raw = (text) => new Safe(String(text));

export function setHTML(el, content) {
  el.innerHTML = render(content);
  return el;
}

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

// ---------------------------------------------------------------- formatting

const MINUS = '−';

export function num(value, digits = 2) {
  return Number(value).toLocaleString('en-IN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export const inr = (value, digits = 2) => '₹' + num(value, digits);

/** Signed percent with a real minus sign: +1.25% / −0.40% */
export function pct(value, digits = 2) {
  const n = Number(value);
  const sign = n > 0 ? '+' : n < 0 ? MINUS : '';
  return `${sign}${Math.abs(n).toFixed(digits)}%`;
}

export function signedInr(value) {
  const n = Number(value);
  const sign = n > 0 ? '+' : n < 0 ? MINUS : '';
  return `${sign}₹${num(Math.abs(n), 0)}`;
}

/** Rupees in lakh / crore for large amounts. */
export function compactInr(value) {
  const n = Math.abs(Number(value));
  const sign = value < 0 ? MINUS : '';
  if (n >= 1e7) return `${sign}₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `${sign}₹${(n / 1e5).toFixed(2)} L`;
  return `${sign}₹${num(n, 0)}`;
}

export const dirOf = (value) => (value > 0.05 ? 'up' : value < -0.05 ? 'down' : 'flat');

export function timeAgo(timestamp) {
  const seconds = Math.max(0, (Date.now() - timestamp) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  const days = Math.floor(seconds / 86400);
  return days === 1 ? 'yesterday' : `${days} days ago`;
}

export function longDate(iso) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-IN', { weekday: 'long', day: 'numeric', month: 'long' });
}

export function shortDate(iso) {
  const d = new Date(iso + (iso.length === 10 ? 'T00:00:00' : ''));
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short' });
}

export const todayISO = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

export function debounce(fn, wait) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), wait); };
}

export const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

// ---------------------------------------------------------------- icons

const ICON_PATHS = {
  home: '<path d="M4 11.5 12 5l8 6.5"/><path d="M6.5 10v9h11v-9"/><path d="M10 19v-5h4v5"/>',
  flow: '<path d="M3 6h4c4 0 4 6 8 6h6"/><path d="M3 18h4c4 0 4-6 8-6"/><path d="m18 9 3 3-3 3"/>',
  journal: '<path d="M6 4h11a1 1 0 0 1 1 1v15H7a2 2 0 0 1-2-2V5a1 1 0 0 1 1-1Z"/><path d="M5 17.5A1.5 1.5 0 0 1 6.5 16H18"/><path d="M9 8h6M9 11h4"/>',
  learn: '<path d="m3 9 9-4 9 4-9 4-9-4Z"/><path d="M7 11v4.5c1.4 1.3 3.1 2 5 2s3.6-.7 5-2V11"/><path d="M21 9v5"/>',
  community: '<path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h9A1.5 1.5 0 0 1 16 6.5v6a1.5 1.5 0 0 1-1.5 1.5H9l-3.5 3V14A1.5 1.5 0 0 1 4 12.5v-6Z"/><path d="M18 9h.5A1.5 1.5 0 0 1 20 10.5v8l-2.8-2.5H12"/>',
  feedback: '<path d="M5 19 6 15 16.5 4.5a2.1 2.1 0 0 1 3 3L9 18l-4 1Z"/><path d="m14.5 6.5 3 3"/>',
  settings: '<path d="M5 7h7M16 7h3M5 17h3M12 17h7"/><circle cx="14" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  search: '<circle cx="11" cy="11" r="6"/><path d="m16 16 4 4"/>',
  sun: '<circle cx="12" cy="12" r="3.5"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4"/>',
  moon: '<path d="M19 14.5A7.5 7.5 0 0 1 9.5 5a7.5 7.5 0 1 0 9.5 9.5Z"/>',
  star: '<path d="m12 4 2.4 5 5.6.7-4.1 3.8 1.1 5.5L12 16.3 7 19l1.1-5.5L4 9.700l5.600-.7L12 4Z"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  collapse: '<path d="m13 7-5 5 5 5"/><path d="M19 5v14"/>',
  expand: '<path d="m11 7 5 5-5 5"/><path d="M5 5v14"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/>',
  download: '<path d="M12 4v11M7.500 10.500 12 15l4.500-4.500M5 19h14"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.500"/>',
  clock: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4.500l3 2"/>',
  live: '<circle cx="12" cy="12" r="3"/><path d="M6.500 6.500a8 8 0 0 0 0 11M17.500 6.500a8 8 0 0 1 0 11"/>',
  shield: '<path d="M12 4 5 6.500V12c0 4 3 6.800 7 8 4-1.200 7-4 7-8V6.500L12 4Z"/>',
  arrowRight: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  filter: '<path d="M4 6h16l-6 7v5l-4 2v-7L4 6Z"/>',
  heart: '<path d="M12 19s-7-4.300-7-9.500A3.500 3.500 0 0 1 12 7.600 3.500 3.500 0 0 1 19 9.500C19 14.700 12 19 12 19Z"/>',
  book: '<path d="M4 5.500C6.500 4.500 9.500 4.500 12 6c2.500-1.500 5.500-1.500 8-.500V18c-2.500-1-5.500-1-8 .500-2.500-1.500-5.500-1.500-8-.500V5.500Z"/><path d="M12 6v12.500"/>',
};

/** Inline stroke icon. `label` makes it meaningful to screen readers; otherwise it is decorative. */
export function icon(name, { size = 18, label = '' } = {}) {
  const a11y = label ? `role="img" aria-label="${esc(label)}"` : 'aria-hidden="true"';
  return raw(
    `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" ` +
    `stroke-width="1.700" stroke-linecap="round" stroke-linejoin="round" ${a11y}>${ICON_PATHS[name] || ''}</svg>`
  );
}

/**
 * Build-up marker. Direction is the triangle (up/down) and its colour;
 * fresh positions are solid, positions closing are hollow.
 */
export function buildupMark(buildup, size = 11) {
  const up = buildup === 'long_buildup' || buildup === 'short_covering';
  const solid = buildup === 'long_buildup' || buildup === 'short_buildup';
  if (buildup === 'flat') {
    return raw(`<svg class="mark mark-flat" width="${size}" height="${size}" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="5" width="8" height="2" rx="1" fill="currentColor"/></svg>`);
  }
  const points = up ? '6,1.500 11,10.500 1,10.500' : '6,10.500 11,1.500 1,1.500';
  return raw(
    `<svg class="mark mark-${up ? 'up' : 'down'}" width="${size}" height="${size}" viewBox="0 0 12 12" aria-hidden="true">` +
    `<polygon points="${points}" fill="${solid ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="1.500" stroke-linejoin="round"/></svg>`
  );
}

// ---------------------------------------------------------------- storage

export const store = {
  load(key, fallback) {
    try {
      const text = localStorage.getItem('niveshak.' + key);
      return text === null ? fallback : JSON.parse(text);
    } catch (error) {
      return fallback;
    }
  },
  save(key, value) {
    try {
      localStorage.setItem('niveshak.' + key, JSON.stringify(value));
      return true;
    } catch (error) {
      return false;
    }
  },
  remove(key) {
    try { localStorage.removeItem('niveshak.' + key); } catch (error) { /* nothing to remove */ }
  },
};

export const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

// ---------------------------------------------------------------- tooltip & toast

const tipEl = () => document.getElementById('tip');

export function showTip(content, x, y) {
  const tip = tipEl();
  setHTML(tip, content);
  tip.hidden = false;
  const { width, height } = tip.getBoundingClientRect();
  const left = clamp(x + 14, 8, window.innerWidth - width - 8);
  const top = y + 18 + height > window.innerHeight ? y - height - 12 : y + 18;
  tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(Math.max(8, top))}px)`;
}

export function hideTip() {
  const tip = tipEl();
  if (tip) tip.hidden = true;
}

/** Brief confirmation. Pass { action, onAction } to offer one follow-up such as Undo. */
export function toast(message, { action = '', onAction = null } = {}) {
  const host = document.getElementById('toasts');
  const item = document.createElement('div');
  item.className = 'toast';
  const text = document.createElement('span');
  text.textContent = message;
  item.appendChild(text);
  if (action && onAction) {
    const button = document.createElement('button');
    button.className = 'toast-action';
    button.textContent = action;
    button.addEventListener('click', () => { onAction(); item.remove(); });
    item.appendChild(button);
  }
  host.appendChild(item);
  const life = action ? 6000 : 2600;
  setTimeout(() => item.classList.add('toast-out'), life);
  setTimeout(() => item.remove(), life + 400);
}

export function downloadText(filename, text, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
