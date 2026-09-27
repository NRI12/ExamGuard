"use strict";
// Realtime replay of the cheating detector. Everything shown at time T is computed only from the model windows
// decided at or before T (a window at time t covers the video [t - 3.2 s, t]), exactly like the live app.

const WIN_SPAN = 3.0;          // eval-videos: an alert up to 75 frames (3 s) after an event still counts for it
const TRACK_GAP = 1.6;         // a track is drawn while its last decision is at most this old
const COLORS = {
  normal: "#3fb68b", looking: "#e3a33b", interaction: "#a78bfa", phone_cheatsheet: "#f0524f", abnormal: "#f0524f",
};
const NICE = { normal: "normal", looking: "looking", interaction: "interaction", phone_cheatsheet: "phone / cheat sheet", abnormal: "abnormal" };
const SHORT = { normal: "normal", looking: "looking", interaction: "interact", phone_cheatsheet: "phone/sheet", abnormal: "abnormal" };
const $ = (id) => document.getElementById(id);

const S = {
  manifest: null, session: null, model: null, gt: null, rule: null,
  tracks: [], episodes: [],
  view: "follow", seen: new Set(), feedKey: "", lastPanel: 0, cache: {},
};

// ---------- helpers ----------
const fmt = (t) => {
  t = Math.max(0, t);
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};
const fmt1 = (t) => `${fmt(t)}.${Math.floor((Math.max(0, t) % 1) * 10)}`;
function lastLE(arr, x) { // index of the last element <= x, or -1
  let lo = 0, hi = arr.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
}
async function getJSON(url) {
  if (!S.cache[url]) S.cache[url] = fetch(url).then((r) => { if (!r.ok) throw new Error(url); return r.json(); });
  return S.cache[url];
}
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

// ---------- replay of the app's decision rule (detector.py) ----------
function replay(pred, classes, rule) {
  const tracks = [], episodes = [];
  for (const [tid, tr] of Object.entries(pred.tracks)) {
    const n = tr.t.length, q = [];
    const out = { id: tid, pid: tr.pid, t: tr.t, box: tr.box, p: tr.p, score: new Float32Array(n),
                  sm: new Array(n), alert: new Uint8Array(n), label: new Array(n) };
    let alert = false, ep = null;
    const close = () => {
      let best = null, bc = -1;
      for (const [k, c] of Object.entries(ep.types)) if (c > bc) { bc = c; best = k; }
      ep.type = best;
      if (ep.windows >= rule.min_windows) episodes.push(ep);
      alert = false; ep = null;
    };
    for (let i = 0; i < n; i++) {
      q.push(tr.p[i]);
      while (q.length > rule.smooth) q.shift();
      const sm = classes.map((_, c) => q.reduce((a, v) => a + v[c], 0) / q.length);
      const score = 1 - sm[0];
      let lab = 1;
      for (let c = 2; c < classes.length; c++) if (sm[c] > sm[lab]) lab = c;
      const t = tr.t[i];
      if (!alert && score >= rule.thr_on) {
        alert = true;
        ep = { track: tid, pid: tr.pid, start: t, end: t, types: {}, peak: 0, windows: 0, confirm: null };
      } else if (alert && score < rule.thr_off) close();
      if (alert) {
        ep.end = t; ep.windows++; ep.peak = Math.max(ep.peak, score);
        ep.types[classes[lab]] = (ep.types[classes[lab]] || 0) + 1;
        if (ep.windows === rule.min_windows) ep.confirm = t;
      }
      out.score[i] = score; out.sm[i] = sm; out.alert[i] = alert ? 1 : 0;
      out.label[i] = alert ? classes[lab] : classes[0];
    }
    if (alert) close();
    tracks.push(out);
  }
  episodes.sort((a, b) => a.confirm - b.confirm);
  return { tracks, episodes };
}

// ---------- scoring (tune_thresholds.py rules), optionally only up to time `now` ----------
// An interaction counts for both people. An event is caught when an alert episode [start - 3 s, end] overlaps
// [event start, event end + 3 s] of that person; an episode overlapping no cheating event of its person is a false alarm.
function personEvents(gt) {
  if (gt._pev) return gt._pev;
  const out = [];
  for (const e of gt.events) {
    if (!e.cheating) continue;
    out.push({ ...e, src: e, passive: false });
    if (e.related) out.push({ ...e, pid: e.related, related: e.pid, src: e, passive: true });
  }
  return (gt._pev = out);
}
function evaluate(gt, rep, now = Infinity) {
  const pev = personEvents(gt);
  const eps = rep.episodes.filter((p) => p.confirm <= now).map((p) => {
    const end = Math.min(p.end, now);
    const match = p.pid ? pev.find((e) => e.pid === p.pid && e.start <= end && e.end >= p.start - WIN_SPAN) : null;
    return { ...p, shownEnd: end, ongoing: p.end > now, match };
  });
  const status = pev.map((e) => {
    let at = Infinity;
    for (const p of eps) if (p.pid === e.pid && p.start - WIN_SPAN <= e.end + WIN_SPAN && p.shownEnd >= e.start) at = Math.min(at, Math.max(p.confirm, e.start));
    if (at !== Infinity) return { e, state: "hit", at };
    if (e.end + WIN_SPAN < now) return { e, state: "miss" };
    if (e.start <= now) return { e, state: "live" };
    return { e, state: "wait" };
  });
  const fa = eps.filter((p) => !p.match).length;
  const span = Math.min(now, gt.duration);
  return {
    status, eps, fa,
    caught: status.filter((s) => s.state === "hit").length,
    missed: status.filter((s) => s.state === "miss").length,
    done: status.filter((s) => s.state === "hit" || s.state === "miss").length,
    total: pev.length,
    faph: span > 0 ? fa / (Object.keys(gt.persons).length * span / 3600) : 0,
    prec: eps.length ? (eps.length - fa) / eps.length : null,
  };
}

// ---------- loading ----------
async function init() {
  S.manifest = await getJSON("data/manifest.json");
  const ss = $("sessionSel");
  for (const s of S.manifest.sessions) ss.add(new Option(`${s.id}  ${fmt(s.duration)}, ${s.persons} people`, s.id));
  ss.onchange = () => loadSession(ss.value);
  $("modelSel").onchange = () => loadModel($("modelSel").value);
  bindUI();
  await loadSession(S.manifest.sessions[0].id);
  requestAnimationFrame(frame);
}

async function loadSession(id) {
  $("loading").classList.remove("hidden");
  S.session = S.manifest.sessions.find((s) => s.id === id);
  S.gt = await getJSON(`data/gt_${id}.json`);
  const ms = $("modelSel"), keep = S.model && S.model.id;
  ms.innerHTML = "";
  const avail = S.manifest.models.filter((m) => m.sessions.includes(id));
  for (const m of avail) ms.add(new Option(m.name + (m.recommended ? "  *" : ""), m.id));
  ms.value = avail.some((m) => m.id === keep) ? keep : avail[0].id;
  const v = $("video");
  v.src = S.session.video;
  await new Promise((res) => { v.onloadedmetadata = res; v.onerror = res; });
  v.playbackRate = Number(document.querySelector("#speedSeg .on").dataset.speed);
  $("timeTotal").textContent = fmt(S.gt.duration);
  await loadModel(ms.value, keep !== ms.value);
  buildPersons();
  compareModels();
  $("loading").classList.add("hidden");
}

async function loadModel(id, resetRule = true) {
  S.model = S.manifest.models.find((m) => m.id === id);
  S.pred = await getJSON(`data/${id}_${S.session.id}.json`);
  $("modelNote").textContent = S.model.note;
  if (resetRule || !S.rule) setRule(S.model);
  else recompute();
  buildPersons();
  compareModels();
}

function setRule(r) {
  S.rule = { thr_on: r.thr_on, thr_off: r.thr_off, smooth: r.smooth, min_windows: r.min_windows };
  $("thrOn").value = r.thr_on; $("thrOff").value = r.thr_off; $("smooth").value = r.smooth; $("minWin").value = r.min_windows;
  recompute();
}

function recompute() {
  $("thrOnVal").textContent = S.rule.thr_on.toFixed(2);
  $("thrOffVal").textContent = S.rule.thr_off.toFixed(2);
  $("smoothVal").textContent = `${S.rule.smooth}  (${S.rule.smooth} s)`;
  $("minWinVal").textContent = S.rule.min_windows;
  const rep = replay(S.pred, S.model.classes, S.rule);
  Object.assign(S, rep);
  S.feedKey = ""; S.seen = new Set(S.episodes.filter((p) => p.confirm <= $("video").currentTime).map(epKey));
  compareModels();
  render(true);
}

// ---------- UI bindings ----------
function bindUI() {
  const v = $("video");
  $("playBtn").onclick = togglePlay;
  $("restartBtn").onclick = () => seek(0);
  v.onplay = v.onpause = v.onended = () => { $("playBtn").classList.toggle("playing", !v.paused); render(true); };
  for (const b of document.querySelectorAll("#speedSeg button")) b.onclick = () => {
    document.querySelectorAll("#speedSeg button").forEach((x) => x.classList.toggle("on", x === b));
    v.playbackRate = Number(b.dataset.speed);
  };
  for (const b of document.querySelectorAll("#viewSeg button")) b.onclick = () => {
    document.querySelectorAll("#viewSeg button").forEach((x) => x.classList.toggle("on", x === b));
    S.view = b.dataset.view; render(true);
  };
  const slide = (id, key, parse) => $(id).oninput = () => {
    S.rule[key] = parse($(id).value);
    if (key === "thr_on" && S.rule.thr_off > S.rule.thr_on) { S.rule.thr_off = S.rule.thr_on; $("thrOff").value = S.rule.thr_on; }
    if (key === "thr_off" && S.rule.thr_off > S.rule.thr_on) { S.rule.thr_on = S.rule.thr_off; $("thrOn").value = S.rule.thr_off; }
    recompute();
  };
  slide("thrOn", "thr_on", Number); slide("thrOff", "thr_off", Number);
  slide("smooth", "smooth", (x) => parseInt(x, 10)); slide("minWin", "min_windows", (x) => parseInt(x, 10));
  $("resetRule").onclick = () => setRule(S.model);
  for (const id of ["showBoxes", "showScores", "showGT", "focusAlerts", "revealFuture"]) $(id).onchange = () => render(true);
  for (const t of document.querySelectorAll(".tab")) t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
    $("alertFeed").classList.toggle("hidden", t.dataset.tab !== "alerts");
    $("eventFeed").classList.toggle("hidden", t.dataset.tab !== "events");
  };
  $("prevEvent").onclick = () => jumpEvent(-1);
  $("nextEvent").onclick = () => jumpEvent(1);
  $("nextAlert").onclick = jumpAlert;
  $("exportCsv").onclick = exportCsv;
  $("timeline").onclick = (ev) => {
    const r = ev.currentTarget.getBoundingClientRect(), [t0, t1, x0] = tlRange(r.width);
    const x = ev.clientX - r.left;
    if (x >= x0) seek(t0 + ((x - x0) / (r.width - x0)) * (t1 - t0));
  };
  window.addEventListener("resize", () => render(true));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
    const k = e.key;
    if (k === " ") { e.preventDefault(); togglePlay(); }
    else if (k === "ArrowLeft") { e.preventDefault(); seek(v.currentTime - 5); }
    else if (k === "ArrowRight") { e.preventDefault(); seek(v.currentTime + 5); }
    else if (k === ",") jumpEvent(-1);
    else if (k === ".") jumpEvent(1);
    else if (k === "a" || k === "A") jumpAlert();
    else if (k === "Home") seek(0);
  });
}

function togglePlay() { const v = $("video"); if (v.paused) v.play(); else v.pause(); }
function seek(t) { const v = $("video"); v.currentTime = Math.max(0, Math.min(S.gt.duration - 0.05, t)); render(true); }
function jumpEvent(dir) {
  const now = $("video").currentTime, ev = S.gt.events.filter((e) => e.cheating);
  const target = dir > 0 ? ev.find((e) => e.start - 2 > now + 0.5) : [...ev].reverse().find((e) => e.start - 2 < now - 1.5);
  if (target) seek(target.start - 2);
}
function jumpAlert() {
  const now = $("video").currentTime, p = S.episodes.find((p) => p.start - 2 > now + 0.5);
  if (p) seek(p.start - 2);
}
const epKey = (p) => `${p.track}@${p.start}`;

// ---------- per-frame rendering ----------
function frame() {
  render(false);
  requestAnimationFrame(frame);
}

function render(force) {
  if (!S.gt || !S.tracks) return;
  const v = $("video"), now = v.currentTime;
  drawOverlay(now);
  const t = performance.now();
  if (!force && t - S.lastPanel < 200) return;
  S.lastPanel = t;
  $("timeNow").textContent = fmt(now);
  const badge = $("liveBadge");
  badge.dataset.state = v.ended ? "ended" : v.paused ? "paused" : "live";
  $("liveText").textContent = v.ended ? "END OF STREAM" : v.paused ? "PAUSED" : `LIVE  ${v.playbackRate}x`;
  const res = evaluate(S.gt, S, now);
  drawTimeline(now);
  updatePersons(now);
  updateMetrics(res, now);
  updateFeeds(res, now);
}

function trackAt(tr, now) { // index of the decision in force at `now`, or -1 when the track is not visible
  const i = lastLE(tr.t, now);
  if (i < 0 || now - tr.t[i] > TRACK_GAP) return -1;
  return i;
}

function drawOverlay(now) {
  const c = $("overlay"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const sx = W / S.gt.width, sy = H / S.gt.height;
  const showGT = $("showGT").checked, showBoxes = $("showBoxes").checked, showScores = $("showScores").checked, focus = $("focusAlerts").checked;
  g.font = "600 12px ui-monospace, Consolas, monospace";
  g.textBaseline = "top";

  if (showGT) {
    const k = Math.round(now / 0.2);
    for (const [pid, rows] of Object.entries(S.gt.persons)) {
      let r = rows[Math.min(k, rows.length - 1)];
      if (!r || Math.abs(r[0] - now) > 0.3) r = rows.find((x) => Math.abs(x[0] - now) <= 0.3);
      if (!r) continue;
      const ev = S.gt.events.find((e) => e.pid === pid && e.start <= now && now <= e.end);
      const col = ev ? (ev.cheating ? COLORS[ev.cls] : "#9aa4b1") : "rgba(255,255,255,.55)";
      g.save(); g.setLineDash([6, 5]); g.lineWidth = 1.5; g.strokeStyle = col;
      g.strokeRect(r[1] * sx, r[2] * sy, (r[3] - r[1]) * sx, (r[4] - r[2]) * sy); g.restore();
      const txt = `GT ${pid}` + (ev ? `  ${ev.cheating ? NICE[ev.cls] : "hard negative"}` : "");
      pill(g, txt, r[1] * sx, r[4] * sy + 3, col, true);
    }
  }
  if (!showBoxes) return;
  for (const tr of S.tracks) {
    const i = trackAt(tr, now);
    if (i < 0) continue;
    let b = tr.box[i];
    const j = i + 1;
    if (j < tr.t.length && tr.t[j] - tr.t[i] <= TRACK_GAP) { // smooth box motion between two detections
      const a = (now - tr.t[i]) / (tr.t[j] - tr.t[i]);
      b = b.map((x, k) => x + (tr.box[j][k] - x) * a);
    }
    const on = tr.alert[i] === 1, lab = tr.label[i], score = tr.score[i];
    const col = on ? COLORS[lab] : COLORS.normal;
    const x = b[0] * sx, y = b[1] * sy, w = (b[2] - b[0]) * sx, h = (b[3] - b[1]) * sy;
    if (focus && !on) {
      g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,.25)"; g.strokeRect(x, y, w, h);
      continue;
    }
    g.lineWidth = on ? 3 : 1.5; g.strokeStyle = col;
    if (on) { g.fillStyle = hexA(col, 0.12); g.fillRect(x, y, w, h); }
    g.strokeRect(x, y, w, h);
    const who = tr.pid ? `${tr.pid} ` : "";
    // normal: confidence P(normal); alert: confidence that the person is not normal (cheat score)
    const conf = on ? score : tr.sm[i][0];
    pill(g, `${who}#${tr.id}  ${on ? NICE[lab] : "normal"}  ${conf.toFixed(2)}`, x, y - 20, col, false, on);
    if (showScores) {
      const cls = S.model.classes, bw = Math.min(90, w - 8), bh = 4;
      for (let c = 0; c < cls.length; c++) {
        const yy = y + 5 + c * (bh + 3);
        g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(x + 4, yy, bw, bh);
        g.fillStyle = COLORS[cls[c]]; g.fillRect(x + 4, yy, bw * tr.sm[i][c], bh);
      }
    }
  }
}

function pill(g, text, x, y, col, dashed, solid) {
  const w = g.measureText(text).width + 12, h = 18;
  y = Math.max(0, y);
  g.fillStyle = solid ? col : "rgba(8,11,15,.8)";
  g.fillRect(x, y, w, h);
  if (!solid) { g.strokeStyle = col; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
  g.fillStyle = solid ? "#0b0f14" : col;
  g.fillText(text, x + 6, y + 3);
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

// ---------- timeline ----------
function tlRange(width) {
  const x0 = 40, D = S.gt.duration, now = $("video").currentTime;
  if (S.view === "whole") return [0, D, x0];
  const span = 120, t0 = Math.max(0, Math.min(D - span, now - span * 0.75));
  return [t0, t0 + span, x0];
}

function drawTimeline(now) {
  const c = $("timeline"), dpr = window.devicePixelRatio || 1;
  const pids = Object.keys(S.gt.persons), rowH = 30, top = 4, axis = 20;
  const W = c.clientWidth, H = top + pids.length * rowH + axis;
  c.style.height = H + "px";
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const [t0, t1, x0] = tlRange(W), X = (t) => x0 + ((t - t0) / (t1 - t0)) * (W - x0);
  const future = $("revealFuture").checked;
  g.font = "12px Segoe UI, system-ui, sans-serif"; g.textBaseline = "middle";

  const tlEvents = [...personEvents(S.gt), ...S.gt.events.filter((e) => !e.cheating)];
  pids.forEach((pid, r) => {
    const y = top + r * rowH;
    g.fillStyle = r % 2 ? "#161d26" : "#19212b"; g.fillRect(x0, y, W - x0, rowH - 2);
    g.fillStyle = "#8b96a3"; g.fillText(pid, 8, y + rowH / 2);
    for (const e of tlEvents) {
      if (e.pid !== pid || e.end < t0 || e.start > t1) continue;
      const a = X(Math.max(t0, e.start)), b = Math.max(a + 2, X(Math.min(t1, e.end)));
      if (e.cheating) { g.fillStyle = COLORS[e.cls]; g.globalAlpha = e.passive ? 0.45 : 0.85; g.fillRect(a, y + 3, b - a, 9); g.globalAlpha = 1; }
      else hatch(g, a, y + 3, b - a, 9);
    }
    // cheat score curve (max over this person's tracks), causal unless "show future" is on
    const lim = future ? t1 : Math.min(t1, now);
    g.beginPath(); let started = false;
    for (const tr of S.tracks) {
      if (tr.pid !== pid) continue;
      started = false;
      for (let i = 0; i < tr.t.length; i++) {
        const t = tr.t[i];
        if (t < t0 - 1 || t > lim) continue;
        const px = X(t), py = y + rowH - 3 - tr.score[i] * 14;
        if (!started || (i > 0 && t - tr.t[i - 1] > TRACK_GAP)) { g.moveTo(px, py); started = true; } else g.lineTo(px, py);
      }
    }
    g.strokeStyle = "rgba(78,161,255,.75)"; g.lineWidth = 1; g.stroke();
    for (const p of S.episodes) {
      if (p.pid !== pid) continue;
      if (!future && p.confirm > now) continue;
      const end = future ? p.end : Math.min(p.end, now);
      if (end < t0 || p.start > t1) continue;
      const a = X(Math.max(t0, p.start)), b = Math.max(a + 3, X(Math.min(t1, end)));
      g.strokeStyle = COLORS[p.type] || "#f0524f"; g.lineWidth = 2; g.strokeRect(a, y + 15, b - a, 10);
      g.fillStyle = hexA(COLORS[p.type] || "#f0524f", 0.25); g.fillRect(a, y + 15, b - a, 10);
    }
  });
  const yb = top + pids.length * rowH;
  if (!future && now < t1) { g.fillStyle = "rgba(13,17,23,.55)"; g.fillRect(Math.max(x0, X(now)), top, W, yb - top); }
  // axis
  g.fillStyle = "#8b96a3"; g.textBaseline = "top";
  const step = niceStep((t1 - t0) / Math.max(2, (W - x0) / 90));
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    const x = X(t); g.fillRect(x, yb, 1, 4); g.fillText(fmt(t), x + 3, yb + 4);
  }
  const px = X(now);
  if (px >= x0 && px <= W) { g.fillStyle = "#4ea1ff"; g.fillRect(px - 1, 0, 2, yb + 2); }
}
function hatch(g, x, y, w, h) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.strokeStyle = "#5b6573"; g.lineWidth = 1;
  for (let i = -h; i < w; i += 5) { g.beginPath(); g.moveTo(x + i, y + h); g.lineTo(x + i + h, y); g.stroke(); }
  g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); g.restore();
}
const niceStep = (s) => [5, 10, 15, 30, 60, 120, 300, 600].find((v) => v >= s) || 600;

// ---------- person cards ----------
function buildPersons() {
  const box = $("persons"); box.innerHTML = "";
  if (!S.model) return;
  for (const pid of Object.keys(S.gt.persons)) {
    const card = el("div", "person", `
      <div class="row"><span class="pid">${pid}</span><span class="state">-</span></div>
      <div class="gt"></div>
      <div class="meter"><i></i><span class="thr on"></span><span class="thr off"></span></div>
      <div class="row"><span class="score">cheat score -</span><span class="score trk"></span></div>
      <div class="probs">${S.model.classes.map((c) => `<div class="p"><span>${SHORT[c]}</span><span class="bar"><i style="background:${COLORS[c]};width:0"></i></span><span class="v">-</span></div>`).join("")}</div>`);
    card.dataset.pid = pid;
    card.title = "Jump to this person's next ground-truth event";
    card.onclick = () => {
      const now = $("video").currentTime, e = S.gt.events.find((e) => e.pid === pid && e.cheating && e.start - 2 > now + 0.5);
      if (e) seek(e.start - 2);
    };
    box.appendChild(card);
  }
}

function updatePersons(now) {
  const showGT = $("showGT").checked;
  for (const card of $("persons").children) {
    const pid = card.dataset.pid;
    let best = null, bi = -1;
    for (const tr of S.tracks) {
      if (tr.pid !== pid) continue;
      const i = trackAt(tr, now);
      if (i >= 0 && (!best || tr.t[i] > best.t[bi])) { best = tr; bi = i; }
    }
    const st = card.querySelector(".state"), on = best && best.alert[bi] === 1;
    card.classList.toggle("alert", !!on);
    card.classList.toggle("idle", !best);
    st.textContent = !best ? "NOT TRACKED" : on ? `ALERT  ${NICE[best.label[bi]]}` : "NORMAL";
    const score = best ? best.score[bi] : 0;
    const m = card.querySelector(".meter i");
    m.style.width = `${score * 100}%`;
    m.style.background = on ? COLORS[best.label[bi]] : score >= S.rule.thr_off ? "#e3a33b" : "#3fb68b";
    card.querySelector(".thr.on").style.left = `${S.rule.thr_on * 100}%`;
    card.querySelector(".thr.off").style.left = `${S.rule.thr_off * 100}%`;
    card.querySelector(".score").textContent = best ? `cheat score ${score.toFixed(2)}` : "cheat score -";
    card.querySelector(".trk").textContent = best ? `track #${best.id}` : "";
    const rows = card.querySelectorAll(".probs .p");
    S.model.classes.forEach((c, k) => {
      const val = best ? best.sm[bi][k] : 0;
      rows[k].querySelector("i").style.width = `${val * 100}%`;
      rows[k].querySelector(".v").textContent = best ? val.toFixed(2) : "-";
    });
    const ev = S.gt.events.find((e) => e.pid === pid && e.start <= now && now <= e.end);
    card.querySelector(".gt").innerHTML = !showGT ? "" : ev
      ? `ground truth: <b>${ev.cheating ? NICE[ev.cls] : "hard negative"}</b> ${ev.subtype.replace(/_/g, " ")}`
      : "ground truth: normal";
  }
}

// ---------- metrics and feeds ----------
function updateMetrics(r, now) {
  $("mCaught").textContent = `${r.caught} / ${r.done}`;
  $("mCaught").className = "value" + (r.done ? (r.caught / r.done >= 0.7 ? " good" : "") : "");
  $("mMissed").textContent = r.missed;
  $("mAlerts").textContent = r.eps.length;
  $("mFA").textContent = r.fa;
  $("mPrec").textContent = r.prec == null ? "-" : `${Math.round(r.prec * 100)} %`;
  $("mFAh").textContent = now < 60 ? "-" : r.faph.toFixed(1);
  $("metricsNote").textContent = `first ${fmt(now)} of ${fmt(S.gt.duration)}, ${r.total} events in total`;
  const active = S.tracks.filter((tr) => { const i = trackAt(tr, now); return i >= 0 && tr.alert[i]; });
  const decided = S.tracks.reduce((a, tr) => a + lastLE(tr.t, now) + 1, 0);
  const visible = S.tracks.filter((tr) => trackAt(tr, now) >= 0).length;
  $("hudLeft").textContent = `${S.session.id}  ${S.model.name.split("  ")[0]}\n${fmt1(now)}\ntracks ${visible}   decisions ${decided}`;
  $("hudRight").innerHTML = active.length
    ? `<span class="alert-on">ACTIVE ALERTS ${active.length}</span>\n` + active.map((tr) => {
        const i = trackAt(tr, now); return `${tr.pid || "#" + tr.id}  ${NICE[tr.label[i]]}  ${tr.score[i].toFixed(2)}`;
      }).join("\n")
    : "no active alert";
}

function updateFeeds(r, now) {
  const key = [S.model.id, S.session.id, r.eps.map((p) => `${epKey(p)}:${p.ongoing}`).join(","),
               r.status.map((s) => s.state[0]).join(""), $("showGT").checked].join("|");
  if (key === S.feedKey) return;
  S.feedKey = key;
  const af = $("alertFeed"); af.innerHTML = "";
  $("alertCount").textContent = r.eps.length;
  if (!r.eps.length) af.appendChild(el("li", "empty", "No alert raised yet. Press play."));
  for (const p of [...r.eps].reverse()) {
    const k = epKey(p), fresh = !S.seen.has(k);
    S.seen.add(k);
    const tag = p.ongoing ? `<span class="tag live">ONGOING</span>` : p.match ? `<span class="tag hit">MATCHES EVENT</span>` : `<span class="tag fa">FALSE ALARM</span>`;
    const li = el("li", fresh ? "fresh" : "", `<span class="t">${fmt(p.confirm)}</span>
      <span class="what">${p.pid || "unmatched"}  ${NICE[p.type] || ""}</span>${tag}
      <span class="sub">track #${p.track}, peak ${p.peak.toFixed(2)}, ${(p.shownEnd - p.start + 1).toFixed(0)} s${p.match ? `, ground truth ${NICE[p.match.cls]} (${p.match.subtype.replace(/_/g, " ")})` : ""}</span>`);
    li.style.borderLeftColor = COLORS[p.type] || "#f0524f";
    li.onclick = () => seek(p.start - 2);
    af.appendChild(li);
  }
  const ef = $("eventFeed"); ef.innerHTML = "";
  $("eventCount").textContent = r.total;
  const items = [...r.status, ...S.gt.events.filter((e) => !e.cheating).map((e) => ({ e, state: "hn" }))]
    .sort((a, b) => a.e.start - b.e.start || (a.e.passive ? 1 : 0) - (b.e.passive ? 1 : 0));
  for (const s of items) {
    const e = s.e;
    const tag = {
      hn: `<span class="tag wait">HARD NEGATIVE</span>`,
      hit: `<span class="tag hit">CAUGHT +${(s.at - e.start).toFixed(1)} s</span>`,
      miss: `<span class="tag miss">MISSED</span>`,
      live: `<span class="tag live">IN PROGRESS</span>`,
      wait: `<span class="tag wait">UPCOMING</span>`,
    }[s.state];
    const li = el("li", "", `<span class="t">${fmt(e.start)}</span>
      <span class="what">${e.pid}  ${e.cheating ? NICE[e.cls] : "hard negative"}</span>${tag}
      <span class="sub">${e.subtype.replace(/_/g, " ") || e.label}, ${(e.end - e.start).toFixed(1)} s${e.related ? `, ${e.passive ? "partner of" : "with"} ${e.related}` : ""}</span>`);
    li.style.borderLeftColor = e.cheating ? COLORS[e.cls] : "#5b6573";
    li.onclick = () => seek(e.start - 2);
    ef.appendChild(li);
  }
}

function exportCsv() {
  const now = $("video").currentTime, r = evaluate(S.gt, S, now);
  const rows = [["session", "model", "person", "track", "start_s", "confirmed_s", "end_s", "type", "peak_score", "windows",
                 "verdict", "gt_event", "thr_on", "thr_off", "smooth", "min_windows"]];
  for (const p of r.eps) rows.push([S.session.id, S.model.id, p.pid || "", p.track, p.start, p.confirm, p.shownEnd, p.type,
    p.peak.toFixed(3), p.windows, p.ongoing ? "ongoing" : p.match ? "matches_event" : "false_alarm",
    p.match ? `${p.match.cls}:${p.match.start}-${p.match.end}` : "", S.rule.thr_on, S.rule.thr_off, S.rule.smooth, S.rule.min_windows]);
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = `alerts_${S.session.id}_${S.model.id}_${fmt(now).replace(":", "m")}s.csv`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- whole-session comparison of all models ----------
async function compareModels() {
  if (!S.session || !S.model) return;
  const sid = S.session.id, token = (S.cmpToken = {});
  const rows = [];
  for (const m of S.manifest.models.filter((m) => m.sessions.includes(sid))) {
    const pred = await getJSON(`data/${m.id}_${sid}.json`);
    if (token !== S.cmpToken) return;
    const rule = m.id === S.model.id ? S.rule : m;
    const r = evaluate(S.gt, replay(pred, m.classes, rule));
    rows.push({ m, r, rule });
  }
  const t = $("compare");
  t.innerHTML = `<tr><th>Model</th><th>Rule on / off / smooth / min</th><th>Events caught</th><th>Recall</th><th>False alarms</th><th>FA / person-hour</th><th>Alert precision</th></tr>` +
    rows.map(({ m, r, rule }) => `<tr class="${m.id === S.model.id ? "cur" : ""}"><td>${m.name.split("  ").join(" ")}</td>
      <td>${rule.thr_on.toFixed(2)} / ${rule.thr_off.toFixed(2)} / ${rule.smooth} / ${rule.min_windows}</td>
      <td>${r.caught} / ${r.total}</td><td>${Math.round((r.caught / r.total) * 100)} %</td><td>${r.fa}</td>
      <td>${r.faph.toFixed(1)}</td><td>${r.prec == null ? "-" : Math.round(r.prec * 100) + " %"}</td></tr>`).join("");
}

init().catch((e) => { $("loading").textContent = "Could not load demo data: " + e.message + ". Run python demo_web/build_demo.py"; });
