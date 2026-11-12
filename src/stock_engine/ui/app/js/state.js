// Shared application state and the action registry. Views import from here
// so they never need to import main.js (no import cycles).

import { store } from './util.js';

export const DEFAULT_SETTINGS = {
  name: 'Aryan',
  theme: 'dark',        // 'dark' | 'light' | 'system'
  refreshSec: 20,       // 0 turns auto-refresh off
  startPage: 'home',
};

export const app = {
  settings: { ...DEFAULT_SETTINGS, ...store.load('settings', {}) },
  snapshot: null,       // latest /snapshot payload
  error: null,          // message when the last refresh failed
  at: null,             // 'HH:MM' while replaying, null when live
  lessons: null,        // /lessons payload, loaded on demand
  engine: null,         // { health, risk } from the backend, for the sidebar and Settings
  route: { name: 'home', arg: null },
};

/** data-action handlers, called as handler(element, event). */
export const actions = {};
export const on = (name, handler) => { actions[name] = handler; };

/** Hooks main.js fills in so views can ask for app-level behaviour. */
export const hooks = {
  refresh: async () => {},
  rerender: () => {},
  openStock: () => {},
  renderSide: () => {},
  applyTheme: () => {},
  reschedule: () => {},
};

export function saveSettings(patch) {
  Object.assign(app.settings, patch);
  store.save('settings', app.settings);
}

export const go = (path) => { window.location.hash = '#/' + path; };
