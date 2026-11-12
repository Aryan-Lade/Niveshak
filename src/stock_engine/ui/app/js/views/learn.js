// Learn: twelve short lessons that explain the terms and formulas the terminal uses.

import { $, html, setHTML, icon, toast } from '../util.js';
import { app, on } from '../state.js';
import { api } from '../api.js';
import { progress } from '../local.js';

const view = { error: null, revealed: new Set() };

const flat = (course) => course.modules.flatMap((m) => m.lessons.map((l) => ({ ...l, moduleTitle: m.title })));

function outline(course, currentId, done) {
  return html`
    <nav class="outline" aria-label="Course outline">
      ${course.modules.map((m) => {
        const finished = m.lessons.filter((l) => done.has(l.id)).length;
        return html`
          <section class="outline-module">
            <h2>${m.title}<span class="muted">${finished} of ${m.lessons.length}</span></h2>
            <p class="sub">${m.summary}</p>
            <ol>
              ${m.lessons.map((l) => html`
                <li><a class="outline-item${l.id === currentId ? ' is-active' : ''}" href="#/learn/${l.id}" ${l.id === currentId ? html`aria-current="page"` : ''}>
                  <span class="tick${done.has(l.id) ? ' is-done' : ''}" aria-hidden="true">${done.has(l.id) ? icon('check', { size: 13 }) : ''}</span>
                  <span>${l.title}${done.has(l.id) ? html`<span class="visually-hidden"> (done)</span>` : ''}</span>
                  <span class="sub">${l.minutes} min</span>
                </a></li>`)}
            </ol>
          </section>`;
      })}
    </nav>`;
}

function reader(lesson, lessons, done) {
  const index = lessons.findIndex((l) => l.id === lesson.id);
  const next = lessons[index + 1];
  const isDone = done.has(lesson.id);
  const shown = view.revealed.has(lesson.id);
  return html`
    <article class="lesson" aria-labelledby="lesson-title">
      <p class="sub">${lesson.moduleTitle}, lesson ${index + 1} of ${lessons.length}, ${lesson.minutes} min read</p>
      <h1 id="lesson-title">${lesson.title}</h1>
      ${lesson.body.map((p) => html`<p>${p}</p>`)}
      <section class="takeaways">
        <h2>Keep these</h2>
        <ul>${lesson.takeaways.map((t) => html`<li>${t}</li>`)}</ul>
      </section>
      <section class="check">
        <h2>Check yourself</h2>
        <p>${lesson.check.question}</p>
        ${shown
          ? html`<p class="check-answer"><b>Answer.</b> ${lesson.check.answer}</p>`
          : html`<button class="btn btn-quiet" data-action="reveal-answer" data-id="${lesson.id}">Show the answer</button>`}
      </section>
      <footer class="lesson-foot">
        <button class="btn${isDone ? ' btn-quiet' : ''}" data-action="toggle-lesson" data-id="${lesson.id}" aria-pressed="${isDone}">
          ${isDone ? html`${icon('check', { size: 16 })}Done. Mark as not done` : 'Mark as done'}
        </button>
        ${next ? html`<a class="btn btn-quiet" href="#/learn/${next.id}">Next: ${next.title}${icon('arrowRight', { size: 15 })}</a>` : ''}
      </footer>
    </article>`;
}

export function render(root, route) {
  if (!app.lessons) {
    setHTML(root, view.error
      ? html`<div class="empty empty-error"><h3>The lessons did not load</h3><p>${view.error}</p><button class="btn" data-action="reload-lessons">Try again</button></div>`
      : html`<p class="boot">Loading lessons…</p>`);
    if (!view.error) {
      api.lessons()
        .then((course) => { app.lessons = course; })
        .catch((error) => { view.error = error.message; })
        .finally(() => { if (app.route.name === 'learn') render(root, app.route); });
    }
    return;
  }
  const course = app.lessons;
  const lessons = flat(course);
  const done = progress.done();
  const lesson = lessons.find((l) => l.id === route.arg) || lessons.find((l) => !done.has(l.id)) || lessons[0];
  const finished = lessons.filter((l) => done.has(l.id)).length;

  setHTML(root, html`
    <header class="page-head page-head-row">
      <div>
        <h1 class="page-title-sm">Learn</h1>
        <p class="lede">${course.lesson_count} short lessons, about ${course.total_minutes} minutes in all. Each one explains a term or formula used on the Home and Flow screens.</p>
      </div>
      <div class="page-tools">
        <span class="sub">${finished} of ${lessons.length} done</span>
        <div class="meter meter-wide" role="img" aria-label="${finished} of ${lessons.length} lessons done"><i style="width:${(finished / lessons.length) * 100}%"></i></div>
      </div>
    </header>
    <div class="learn">
      ${outline(course, lesson.id, done)}
      ${reader(lesson, lessons, done)}
    </div>`);
}

on('reload-lessons', () => { view.error = null; render($('#view'), app.route); });
on('reveal-answer', (el) => { view.revealed.add(el.dataset.id); render($('#view'), app.route); });
on('toggle-lesson', (el) => {
  const nowDone = progress.toggle(el.dataset.id);
  if (nowDone) toast('Lesson marked as done');
  render($('#view'), app.route);
});
