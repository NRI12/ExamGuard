"use strict";
// Live version of demo_web. The page plays the video; the GPU server analyses the same video in step with the
// player (YOLO11m + ByteTrack every 0.2 s, every classifier once per second on the last 3.2 s of each person) and
// streams detector boxes and class probabilities over a WebSocket. Everything shown at time T comes from frames at
// or before T. The decision rule (smoothing + hysteresis), scoring, timeline and feeds run here, as in demo_web.

const WIN_SPAN = 3.0;          // eval-videos: an alert up to 75 frames (3 s) after an event still counts for it
const TRACK_GAP = 1.6;         // a decision stays in force while it is at most this old
const BOX_GAP = 0.6;           // a detector box is drawn while it is at most this old
const COLORS = {
  normal: "#3fb68b", looking: "#e3a33b", interaction: "#a78bfa", phone_cheatsheet: "#f0524f", abnormal: "#f0524f",
};
const NICE = { normal: "normal", looking: "looking", interaction: "interaction", phone_cheatsheet: "phone / cheat sheet", abnormal: "abnormal" };
const SHORT = { normal: "normal", looking: "looking", interaction: "interact", phone_cheatsheet: "phone/sheet", abnormal: "abnormal" };
const $ = (id) => document.getElementById(id);

const S = {
  manifest: null, session: null, model: null, gt: null, hasGT: false, gtIdx: {}, rule: null,
  data: new Map(), samples: [], tracks: [], episodes: [], byId: new Map(), keys: [], keysSig: "",
  view: "follow", seen: new Set(), feedKey: "", lastPanel: 0, lastRecompute: 0, lastCompare: 0, dirty: false,
  ws: null, ready: false, stat: null, cache: {}, loadToken: 0, running: [], cover: [],
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
function lastLEt(arr, x) { // same on objects with a .t
  let lo = 0, hi = arr.length - 1, r = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m].t <= x) { r = m; lo = m + 1; } else hi = m - 1; }
  return r;
}
async function getJSON(url) {
  if (!S.cache[url]) S.cache[url] = fetch(url).then((r) => { if (!r.ok) throw new Error(url); return r.json(); });
  return S.cache[url];
}
function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- the app's decision rule (detector.py), identical to demo_web ----------
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
    if (alert) { // still open at the latest decision: shown as ongoing, kept if already confirmed
      let best = null, bc = -1;
      for (const [k, c] of Object.entries(ep.types)) if (c > bc) { bc = c; best = k; }
      ep.type = best; ep.open = true;
      if (ep.windows >= rule.min_windows) episodes.push(ep);
    }
    tracks.push(out);
  }
  episodes.sort((a, b) => a.confirm - b.confirm);
  return { tracks, episodes };
}

// ---------- scoring (tune_thresholds.py rules) up to time `now` ----------
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
    return { ...p, shownEnd: end, ongoing: p.open || p.end > now, match };
  });
  const status = pev.map((e) => {
    if (e.start <= now && !covered(e.start, e.end + WIN_SPAN, now)) return { e, state: "skip" }; // skipped by a seek
    let at = Infinity;
    for (const p of eps) if (p.pid === e.pid && p.start - WIN_SPAN <= e.end + WIN_SPAN && p.shownEnd >= e.start) at = Math.min(at, Math.max(p.confirm, e.start));
    if (at !== Infinity) return { e, state: "hit", at };
    if (e.end + WIN_SPAN < now) return { e, state: "miss" };
    if (e.start <= now) return { e, state: "live" };
    return { e, state: "wait" };
  });
  const fa = eps.filter((p) => !p.match).length;
  const span = Math.min(now, gt.duration);
  const people = Math.max(1, Object.keys(gt.persons).length);
  return {
    status, eps, fa,
    caught: status.filter((s) => s.state === "hit").length,
    missed: status.filter((s) => s.state === "miss").length,
    skipped: status.filter((s) => s.state === "skip").length,
    done: status.filter((s) => s.state === "hit" || s.state === "miss").length,
    total: pev.length,
    faph: span > 0 ? fa / (people * span / 3600) : 0,
    prec: eps.length ? (eps.length - fa) / eps.length : null,
  };
}

// ---------- live data from the server ----------
function gtIndex(gt) { // pid -> {t: [...], rows: [...]} for the causal track -> participant vote
  const idx = {};
  for (const [pid, rows] of Object.entries(gt.persons)) idx[pid] = { t: rows.map((r) => r[0]), rows };
  return idx;
}
function vote(tr, t, box) { // same rule as eval-videos match_tracks (IoU >= 0.3, majority), using only the past
  let best = 0, who = null;
  for (const [pid, g] of Object.entries(S.gtIdx)) {
    const i = lastLE(g.t, t + 0.05);
    if (i < 0 || t - g.t[i] > 0.15) continue;
    const r = g.rows[i], v = iou(box, [r[1], r[2], r[3], r[4]]);
    if (v > best) { best = v; who = pid; }
  }
  if (!who || best < 0.3) return;
  tr.votes[who] = (tr.votes[who] || 0) + 1;
  let top = -1;
  for (const [pid, c] of Object.entries(tr.votes)) if (c > top) { top = c; tr.pid = pid; }
}
// time ranges the server analysed (a seek starts a new one): events outside them are "not analysed", not missed
function covered(a, b, now) { // the range being analysed reaches the present
  return S.cover.some(([x, y], i) => x <= b && (i === S.cover.length - 1 ? Math.max(y, now) : y) >= a);
}
function coverFrom(t) {
  S.cover = S.cover.filter(([x]) => x < t).map(([x, y]) => [x, Math.min(y, t)]);
  S.cover.push([t, t]);
}
function onWin(rows) {
  for (const r of rows) {
    let tr = S.data.get(r.track);
    if (!tr) {
      tr = { id: r.track, t: [], box: [], p: {}, votes: {}, pid: S.hasGT ? null : `#${r.track}` };
      S.data.set(r.track, tr);
    }
    if (tr.t.length && r.t <= tr.t[tr.t.length - 1]) continue;
    const last = S.cover[S.cover.length - 1];
    if (last && r.t > last[1]) last[1] = r.t;
    tr.t.push(r.t); tr.box.push(r.box);
    for (const m of S.manifest.models) (tr.p[m.id] ||= []).push(r.p[m.id] || null); // null: model not running then
    if (S.hasGT) vote(tr, r.t, r.box);
  }
  S.dirty = true;
}
function onDet(msg) {
  while (S.samples.length && S.samples[S.samples.length - 1].t >= msg.t) S.samples.pop();
  S.samples.push({ t: msg.t, b: msg.b });
}
function truncate(t) { // drop everything decided at or after t (the server recomputes it after a seek)
  for (const [id, tr] of S.data) {
    const n = lastLE(tr.t, t - 1e-6) + 1;
    if (n === tr.t.length) continue;
    if (n === 0) { S.data.delete(id); continue; }
    tr.t.length = n; tr.box.length = n;
    for (const k of Object.keys(tr.p)) tr.p[k].length = n;
    if (S.hasGT) {
      tr.votes = {}; tr.pid = null;
      for (let i = 0; i < n; i++) vote(tr, tr.t[i], tr.box[i]);
    }
  }
  S.samples.length = lastLEt(S.samples, t - 1e-6) + 1;
  S.dirty = true;
}
function predFor(mid) { // the windows this model decided (it may have been switched on / off meanwhile)
  const tracks = {};
  for (const [id, tr] of S.data) {
    const p = tr.p[mid];
    if (!p) continue;
    const keep = [];
    for (let i = 0; i < p.length; i++) if (p[i]) keep.push(i);
    if (keep.length) tracks[id] = { pid: tr.pid, t: keep.map((i) => tr.t[i]), box: keep.map((i) => tr.box[i]), p: keep.map((i) => p[i]) };
  }
  return { tracks };
}
// the model on screen always runs; with it, the models of the same family when they fit on the GPU in real time
// (RTX 3060: the three RGB X3D-L together are real time, the flow models only alone), or all with "Run all"
function runIds() {
  if ($("runAll").checked || S.manifest.default_run === "all") return S.manifest.models.map((m) => m.id);
  if (S.manifest.default_run === "family") return S.manifest.models.filter((m) => m.arch === S.model.arch).map((m) => m.id);
  return [S.model.id];
}
function sendModels() { if (S.ready) send({ type: "models", ids: runIds() }); }

// ---------- server connection ----------
function wsUrl() {
  const token = new URLSearchParams(location.search).get("token");
  const base = location.pathname.replace(/[^/]*$/, "");
  return `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${base}ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;
}
function connect() {
  const ws = new WebSocket(wsUrl());
  S.ws = ws;
  ws.onopen = () => { setConn("busy", "connected, opening video"); if (S.session && S.model) openStream(); };
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => {
    S.ready = false;
    setConn("down", "connection lost, reconnecting");
    setTimeout(connect, 2000);
  };
}
function send(msg) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(msg)); }
function clockMsg(type = "clock") {
  const v = $("video");
  return { type, t: v.currentTime, playing: !v.paused && !v.ended, rate: v.playbackRate };
}
function openStream() {
  S.ready = false;
  send({ type: "open", video: S.session.id, models: runIds() });
}
function onMessage(m) {
  if (m.type === "win") onWin(m.rows);
  else if (m.type === "det") onDet(m);
  else if (m.type === "stat") S.stat = m;
  else if (m.type === "reset") {
    truncate(m.t);
    coverFrom(m.t);
    if (m.models) S.running = m.models;
    if (m.reason === "lag") S.lagJumps = (S.lagJumps || 0) + 1;
  }
  else if (m.type === "ready") {
    if (!S.session || m.video !== S.session.id) return;
    S.ready = true;
    truncate($("video").currentTime);
    send(clockMsg("seek"));
    $("loading").classList.add("hidden");
    setConn("ok", `${S.manifest.gpu}${S.manifest.bf16 ? " (bf16)" : ""}`);
  } else if (m.type === "error") {
    setConn("down", m.message);
    if (!S.ready) { $("loading").textContent = m.message; $("loading").classList.remove("hidden"); }
  }
}
function setConn(state, text) { $("gpuNote").dataset.state = state; $("gpuText").textContent = text; }

// ---------- loading ----------
async function init() {
  S.manifest = await fetch("api/manifest", { cache: "no-store" }).then((r) => r.json());
  $("runAll").checked = S.manifest.default_run === "all";
  if (S.manifest.default_run !== "family") $("runAll").parentElement.title = "Every model on the same tracks, at the GPU's cost";
  fillSessions();
  $("sessionSel").onchange = () => loadSession($("sessionSel").value);
  $("modelSel").onchange = () => loadModel($("modelSel").value);
  bindUI();
  connect();
  const first = S.manifest.sessions.find((s) => s.status === "ready");
  if (!first) { $("loading").textContent = "No video on the server yet: upload one (Source card)"; }
  else await loadSession(first.id);
  requestAnimationFrame(frame);
}

function sessionLabel(s) {
  if (s.gt) return `${s.id}  ${fmt(s.duration)}, ${s.persons} people`;
  const state = s.status === "ready" ? fmt(s.duration) : s.status === "processing" ? "preparing" : "error";
  return `${s.name}  ${state}`;
}
function fillSessions(selected) {
  const ss = $("sessionSel");
  ss.innerHTML = "";
  for (const s of S.manifest.sessions) {
    const o = new Option(sessionLabel(s), s.id);
    o.disabled = s.status !== "ready";
    ss.add(o);
  }
  if (selected) ss.value = selected;
  else if (S.session) ss.value = S.session.id;
}

async function loadSession(id) {
  const token = ++S.loadToken;
  $("loading").textContent = "Loading session";
  $("loading").classList.remove("hidden");
  S.session = S.manifest.sessions.find((s) => s.id === id);
  S.hasGT = !!S.session.gt;
  S.gt = S.hasGT ? await getJSON(S.session.gt)
    : { duration: S.session.duration, width: S.session.width, height: S.session.height, persons: {}, events: [] };
  if (token !== S.loadToken) return;
  S.gtIdx = S.hasGT ? gtIndex(S.gt) : {};
  S.data = new Map(); S.samples = []; S.stat = null; S.lagJumps = 0; S.keysSig = null; S.cover = [];
  S.seen = new Set(); S.feedKey = "";
  $("removeUpload").classList.toggle("hidden", S.hasGT);
  for (const id of ["showGT", "prevEvent", "nextEvent"]) $(id).disabled = !S.hasGT;
  const ms = $("modelSel"), keep = S.model && S.model.id;
  ms.innerHTML = "";
  for (const m of S.manifest.models) ms.add(new Option(m.name + (m.recommended ? "  *" : ""), m.id));
  ms.value = S.manifest.models.some((m) => m.id === keep) ? keep : S.manifest.models[0].id;
  const v = $("video");
  v.pause();
  v.src = S.session.media;
  await new Promise((res) => { v.onloadedmetadata = res; v.onerror = res; });
  if (token !== S.loadToken) return;
  v.playbackRate = Number(document.querySelector("#speedSeg .on").dataset.speed);
  $("timeTotal").textContent = fmt(S.gt.duration);
  $("loading").textContent = S.session.note || "Starting the analysis on the GPU";
  await loadModel(ms.value, keep !== ms.value);
  if (S.ws && S.ws.readyState === 1) openStream();
}

async function loadModel(id, resetRule = true) {
  S.model = S.manifest.models.find((m) => m.id === id);
  $("modelNote").textContent = S.model.note;
  if (resetRule || !S.rule) setRule(S.model);
  else applyRule();
  S.keysSig = null;
  sendModels();
  compareModels(true);
}

function setRule(r) {
  S.rule = { thr_on: r.thr_on, thr_off: r.thr_off, smooth: r.smooth, min_windows: r.min_windows };
  $("thrOn").value = r.thr_on; $("thrOff").value = r.thr_off; $("smooth").value = r.smooth; $("minWin").value = r.min_windows;
  applyRule();
}

function applyRule() { // rule changed: recompute everything, alerts already shown do not flash again
  $("thrOnVal").textContent = S.rule.thr_on.toFixed(2);
  $("thrOffVal").textContent = S.rule.thr_off.toFixed(2);
  $("smoothVal").textContent = `${S.rule.smooth}  (${S.rule.smooth} s)`;
  $("minWinVal").textContent = S.rule.min_windows;
  recompute();
  S.feedKey = ""; S.seen = new Set(S.episodes.filter((p) => p.confirm <= $("video").currentTime).map(epKey));
  compareModels(true);
  render(true);
}

function recompute() {
  if (!S.model) return;
  const rep = replay(predFor(S.model.id), S.model.classes, S.rule);
  Object.assign(S, rep);
  S.byId = new Map(S.tracks.map((tr) => [Number(tr.id), tr]));
  S.dirty = false;
  S.lastRecompute = performance.now();
  const keys = rowKeys();
  const sig = keys.join(",");
  if (sig !== S.keysSig) { S.keys = keys; S.keysSig = sig; buildPersons(); }
}

function rowKeys() { // timeline rows / person cards: participants with ground truth, else the longest-lived tracks
  if (S.hasGT) return Object.keys(S.gt.persons);
  return [...S.data.values()].filter((tr) => tr.t.length >= 1)
    .sort((a, b) => b.t.length - a.t.length).slice(0, 12)
    .sort((a, b) => a.id - b.id).map((tr) => tr.pid);
}

// ---------- uploads ----------
function uploadFile(f) {
  const note = $("uploadNote");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `api/upload?name=${encodeURIComponent(f.name)}`);
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) note.textContent = `uploading ${Math.round((e.loaded / e.total) * 100)} %`; };
  xhr.onerror = () => { note.textContent = "upload failed"; };
  xhr.onload = async () => {
    if (xhr.status !== 200) { note.textContent = `upload failed (${xhr.status})`; return; }
    const v = JSON.parse(xhr.responseText);
    note.textContent = "preparing video";
    for (;;) {
      await new Promise((r) => setTimeout(r, 1500));
      const s = await fetch(`api/videos/${v.id}`, { cache: "no-store" }).then((r) => r.json());
      if (s.status === "ready") break;
      if (s.status === "error") { note.textContent = `cannot use this video: ${s.error}`; return; }
    }
    note.textContent = "";
    S.manifest = await fetch("api/manifest", { cache: "no-store" }).then((r) => r.json());
    fillSessions(v.id);
    loadSession(v.id);
  };
  note.textContent = "uploading";
  xhr.send(f);
}
async function removeUpload() {
  if (!S.session || S.hasGT || !confirm(`Delete "${S.session.name}" from the server?`)) return;
  await fetch(`api/videos/${S.session.id}`, { method: "DELETE" });
  S.manifest = await fetch("api/manifest", { cache: "no-store" }).then((r) => r.json());
  const first = S.manifest.sessions.find((s) => s.status === "ready");
  fillSessions(first && first.id);
  if (first) loadSession(first.id);
}

// ---------- UI bindings ----------
function bindUI() {
  const v = $("video");
  $("playBtn").onclick = togglePlay;
  $("restartBtn").onclick = () => seek(0);
  v.onplay = v.onpause = v.onended = v.onratechange = () => {
    $("playBtn").classList.toggle("playing", !v.paused);
    send(clockMsg());
    render(true);
  };
  setInterval(() => { if (S.ready && !v.paused) send(clockMsg()); }, 250);
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
    applyRule();
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
  $("uploadInput").onchange = () => { const f = $("uploadInput").files[0]; $("uploadInput").value = ""; if (f) uploadFile(f); };
  $("removeUpload").onclick = removeUpload;
  $("runAll").onchange = () => { sendModels(); compareModels(true); };
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
function seek(t) {
  const v = $("video");
  t = Math.max(0, Math.min(S.gt.duration - 0.05, t));
  v.currentTime = t;
  truncate(t);
  coverFrom(t);
  recompute();
  S.feedKey = "";
  send({ ...clockMsg("seek"), t });
  render(true);
}
function jumpEvent(dir) {
  if (!S.hasGT) return;
  const now = $("video").currentTime, ev = S.gt.events.filter((e) => e.cheating);
  const target = dir > 0 ? ev.find((e) => e.start - 2 > now + 0.5) : [...ev].reverse().find((e) => e.start - 2 < now - 1.5);
  if (target) seek(target.start - 2);
}
function jumpAlert() { // alerts only exist up to the analysed time: next one already decided, else the previous one
  const now = $("video").currentTime;
  const p = S.episodes.find((p) => p.start - 2 > now + 0.5) || [...S.episodes].reverse().find((p) => p.start - 2 < now - 1.5);
  if (p) seek(p.start - 2);
}
const epKey = (p) => `${p.track}@${p.start}`;

// ---------- per-frame rendering ----------
function frame() {
  render(false);
  requestAnimationFrame(frame);
}

function render(force) {
  if (!S.gt || !S.rule) return;
  const v = $("video"), now = v.currentTime;
  const t = performance.now();
  if (S.dirty && t - S.lastRecompute > 250) recompute();
  drawOverlay(now);
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
  if (force || t - S.lastCompare > 1000) compareModels();
}

function trackAt(tr, now) { // index of the decision in force at `now`, or -1 when the track has none
  const i = lastLE(tr.t, now);
  if (i < 0 || i >= tr.score.length || now - tr.t[i] > TRACK_GAP) return -1;
  return i;
}

function drawOverlay(now) {
  const c = $("overlay"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  // the video keeps its aspect ratio inside the stage (object-fit: contain)
  const ar = S.gt.width / S.gt.height, fitW = Math.min(W, H * ar), fitH = fitW / ar;
  g.translate((W - fitW) / 2, (H - fitH) / 2);
  const sx = fitW / S.gt.width, sy = fitH / S.gt.height;
  const showGT = S.hasGT && $("showGT").checked, showBoxes = $("showBoxes").checked;
  g.font = "600 12px ui-monospace, Consolas, monospace";
  g.textBaseline = "top";

  if (showGT) {
    for (const [pid, gi] of Object.entries(S.gtIdx)) {
      const i = lastLE(gi.t, now + 0.1);
      if (i < 0 || Math.abs(gi.t[i] - now) > 0.3) continue;
      const r = gi.rows[i];
      const ev = S.gt.events.find((e) => e.pid === pid && e.start <= now && now <= e.end);
      const col = ev ? (ev.cheating ? COLORS[ev.cls] : "#9aa4b1") : "rgba(255,255,255,.55)";
      g.save(); g.setLineDash([6, 5]); g.lineWidth = 1.5; g.strokeStyle = col;
      g.strokeRect(r[1] * sx, r[2] * sy, (r[3] - r[1]) * sx, (r[4] - r[2]) * sy); g.restore();
      const txt = `GT ${pid}` + (ev ? `  ${ev.cheating ? NICE[ev.cls] : "hard negative"}` : "");
      pill(g, txt, r[1] * sx, r[4] * sy + 3, col, true);
    }
  }
  if (!showBoxes) return;
  const j = lastLEt(S.samples, now);
  if (j >= 0 && now - S.samples[j].t <= BOX_GAP) { // live detector boxes (every 0.2 s), labelled by the decisions
    const a = S.samples[j], b = S.samples[j + 1];
    const next = b && b.t - a.t <= 0.45 ? new Map(b.b.map((x) => [x[0], x])) : null;
    for (const bx of a.b) {
      let box = bx.slice(1);
      const q = next && next.get(bx[0]);
      if (q) { const f = (now - a.t) / (b.t - a.t); box = box.map((x, k) => x + (q[k + 1] - x) * f); }
      drawTrack(g, bx[0], box, now, sx, sy);
    }
  } else { // no detector sample here (e.g. far behind the analysis): boxes of the decisions, as demo_web
    for (const tr of S.tracks) {
      const i = trackAt(tr, now);
      if (i >= 0) drawTrack(g, Number(tr.id), tr.box[i], now, sx, sy);
    }
  }
}

function drawTrack(g, id, b, now, sx, sy) {
  const tr = S.byId.get(id), i = tr ? trackAt(tr, now) : -1;
  const x = b[0] * sx, y = b[1] * sy, w = (b[2] - b[0]) * sx, h = (b[3] - b[1]) * sy;
  const focus = $("focusAlerts").checked;
  if (i < 0) { // tracked, no decision yet (needs 3.2 s of this person)
    g.lineWidth = 1; g.strokeStyle = focus ? "rgba(255,255,255,.25)" : "rgba(160,160,160,.8)"; g.strokeRect(x, y, w, h);
    if (!focus) pill(g, `#${id}  analysing`, x, y - 20, "#9aa4b1");
    return;
  }
  const on = tr.alert[i] === 1, lab = tr.label[i], score = tr.score[i];
  const col = on ? COLORS[lab] : COLORS.normal;
  if (focus && !on) {
    g.lineWidth = 1; g.strokeStyle = "rgba(255,255,255,.25)"; g.strokeRect(x, y, w, h);
    return;
  }
  g.lineWidth = on ? 3 : 1.5; g.strokeStyle = col;
  if (on) { g.fillStyle = hexA(col, 0.12); g.fillRect(x, y, w, h); }
  g.strokeRect(x, y, w, h);
  const who = S.hasGT && tr.pid ? `${tr.pid} ` : "";
  const conf = on ? score : tr.sm[i][0];
  pill(g, `${who}#${tr.id}  ${on ? NICE[lab] : "normal"}  ${conf.toFixed(2)}`, x, y - 20, col, false, on);
  if ($("showScores").checked) {
    const cls = S.model.classes, bw = Math.min(90, w - 8), bh = 4;
    for (let c = 0; c < cls.length; c++) {
      const yy = y + 5 + c * (bh + 3);
      g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(x + 4, yy, bw, bh);
      g.fillStyle = COLORS[cls[c]]; g.fillRect(x + 4, yy, bw * tr.sm[i][c], bh);
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
  const pids = S.keys, rowH = 30, top = 4, axis = 20;
  const W = c.clientWidth, H = top + Math.max(1, pids.length) * rowH + axis;
  c.style.height = H + "px";
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const [t0, t1, x0] = tlRange(W), X = (t) => x0 + ((t - t0) / (t1 - t0)) * (W - x0);
  const future = $("revealFuture").checked;
  g.font = "12px Segoe UI, system-ui, sans-serif"; g.textBaseline = "middle";
  if (!pids.length) { g.fillStyle = "#8b96a3"; g.fillText("waiting for the first decisions (3.2 s per person)", x0, top + rowH / 2); }

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
    const lim = future ? t1 : Math.min(t1, now);
    g.beginPath(); let started = false;
    for (const tr of S.tracks) {
      if (tr.pid !== pid) continue;
      started = false;
      for (let i = 0; i < tr.score.length; i++) {
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
  const yb = top + Math.max(1, pids.length) * rowH;
  if (!future && now < t1) { g.fillStyle = "rgba(13,17,23,.55)"; g.fillRect(Math.max(x0, X(now)), top, W, yb - top); }
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
  for (const pid of S.keys) {
    const card = el("div", "person", `
      <div class="row"><span class="pid">${esc(pid)}</span><span class="state">-</span></div>
      <div class="gt"></div>
      <div class="meter"><i></i><span class="thr on"></span><span class="thr off"></span></div>
      <div class="row"><span class="score">cheat score -</span><span class="score trk"></span></div>
      <div class="probs">${S.model.classes.map((c) => `<div class="p"><span>${SHORT[c]}</span><span class="bar"><i style="background:${COLORS[c]};width:0"></i></span><span class="v">-</span></div>`).join("")}</div>`);
    card.dataset.pid = pid;
    card.title = S.hasGT ? "Jump to this person's next ground-truth event" : "Jump to this person's latest alert";
    card.onclick = () => {
      const now = $("video").currentTime;
      if (S.hasGT) {
        const e = S.gt.events.find((e) => e.pid === pid && e.cheating && e.start - 2 > now + 0.5);
        if (e) seek(e.start - 2);
      } else {
        const p = [...S.episodes].reverse().find((p) => p.pid === pid);
        if (p) seek(p.start - 2);
      }
    };
    box.appendChild(card);
  }
}

function updatePersons(now) {
  const showGT = S.hasGT && $("showGT").checked;
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
  if (S.hasGT) {
    $("mCaught").textContent = `${r.caught} / ${r.done}`;
    $("mCaught").className = "value" + (r.done ? (r.caught / r.done >= 0.7 ? " good" : "") : "");
    $("mMissed").textContent = r.missed;
    $("mFA").textContent = r.fa;
    $("mPrec").textContent = r.prec == null ? "-" : `${Math.round(r.prec * 100)} %`;
    $("mFAh").textContent = now < 60 ? "-" : r.faph.toFixed(1);
    $("metricsNote").textContent = `first ${fmt(now)} of ${fmt(S.gt.duration)}, ${r.total} events in total` +
      (r.skipped ? `, ${r.skipped} not analysed (seek)` : "");
  } else {
    for (const id of ["mCaught", "mMissed", "mFA", "mPrec", "mFAh"]) { $(id).textContent = "-"; $(id).className = "value"; }
    $("metricsNote").textContent = "no ground truth for this video";
  }
  $("mAlerts").textContent = r.eps.length;
  const active = S.tracks.filter((tr) => { const i = trackAt(tr, now); return i >= 0 && tr.alert[i]; });
  const decided = S.tracks.reduce((a, tr) => a + Math.min(lastLE(tr.t, now) + 1, tr.score.length), 0);
  const visible = S.tracks.filter((tr) => trackAt(tr, now) >= 0).length;
  const st = S.stat;
  const gpu = !S.ready ? "GPU starting" : !st ? "GPU -"
    : `GPU ${Math.round(st.load * 100)} %  ${st.ms.toFixed(0)} ms/sample  ${st.lag > 1 ? `<span class="lag">lag ${st.lag.toFixed(1)} s</span>` : "in step"}`;
  $("hudLeft").innerHTML = `${esc(S.session.id)}  ${esc(S.model.name.split("  ")[0])}\n${fmt1(now)}\ntracks ${visible}   decisions ${decided}\n${gpu}`;
  $("hudRight").innerHTML = active.length
    ? `<span class="alert-on">ACTIVE ALERTS ${active.length}</span>\n` + active.map((tr) => {
        const i = trackAt(tr, now); return `${esc(tr.pid || "#" + tr.id)}  ${NICE[tr.label[i]]}  ${tr.score[i].toFixed(2)}`;
      }).join("\n")
    : "no active alert";
}

function updateFeeds(r, now) {
  const key = [S.model.id, S.session.id, r.eps.map((p) => `${epKey(p)}:${p.ongoing}:${p.pid}`).join(","),
               r.status.map((s) => s.state[0]).join(""), S.hasGT && $("showGT").checked].join("|");
  if (key === S.feedKey) return;
  S.feedKey = key;
  const af = $("alertFeed"); af.innerHTML = "";
  $("alertCount").textContent = r.eps.length;
  if (!r.eps.length) af.appendChild(el("li", "empty", "No alert raised yet. Press play."));
  for (const p of [...r.eps].reverse()) {
    const k = epKey(p), fresh = !S.seen.has(k);
    S.seen.add(k);
    const tag = p.ongoing ? `<span class="tag live">ONGOING</span>` : !S.hasGT ? `<span class="tag fa">ALERT</span>`
      : p.match ? `<span class="tag hit">MATCHES EVENT</span>` : `<span class="tag fa">FALSE ALARM</span>`;
    const li = el("li", fresh ? "fresh" : "", `<span class="t">${fmt(p.confirm)}</span>
      <span class="what">${esc(S.hasGT ? p.pid || "unmatched" : "#" + p.track)}  ${NICE[p.type] || ""}</span>${tag}
      <span class="sub">track #${p.track}, peak ${p.peak.toFixed(2)}, ${(p.shownEnd - p.start + 1).toFixed(0)} s${p.match ? `, ground truth ${NICE[p.match.cls]} (${p.match.subtype.replace(/_/g, " ")})` : ""}</span>`);
    li.style.borderLeftColor = COLORS[p.type] || "#f0524f";
    li.onclick = () => seek(p.start - 2);
    af.appendChild(li);
  }
  const ef = $("eventFeed"); ef.innerHTML = "";
  $("eventCount").textContent = r.total;
  if (!S.hasGT) ef.appendChild(el("li", "empty", "No ground truth for uploaded videos."));
  const items = [...r.status, ...S.gt.events.filter((e) => !e.cheating).map((e) => ({ e, state: "hn" }))]
    .sort((a, b) => a.e.start - b.e.start || (a.e.passive ? 1 : 0) - (b.e.passive ? 1 : 0));
  for (const s of items) {
    const e = s.e;
    const tag = {
      hn: `<span class="tag wait">HARD NEGATIVE</span>`,
      hit: `<span class="tag hit">CAUGHT +${(s.at - e.start).toFixed(1)} s</span>`,
      miss: `<span class="tag miss">MISSED</span>`,
      skip: `<span class="tag wait">NOT ANALYSED</span>`,
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
  for (const p of r.eps) rows.push([S.session.id, S.model.id, S.hasGT ? p.pid || "" : "", p.track, p.start, p.confirm, p.shownEnd, p.type,
    p.peak.toFixed(3), p.windows, p.ongoing ? "ongoing" : !S.hasGT ? "alert" : p.match ? "matches_event" : "false_alarm",
    p.match ? `${p.match.cls}:${p.match.start}-${p.match.end}` : "", S.rule.thr_on, S.rule.thr_off, S.rule.smooth, S.rule.min_windows]);
  const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
  const a = el("a"); a.href = URL.createObjectURL(blob); a.download = `alerts_${S.session.id}_${S.model.id}_${fmt(now).replace(":", "m")}s.csv`;
  a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- comparison of all models on the same live tracks, up to now ----------
function compareModels(force) {
  if (!S.session || !S.model || !S.rule) return;
  S.lastCompare = performance.now();
  const now = $("video").currentTime, rows = [];
  for (const m of S.manifest.models) {
    const rule = m.id === S.model.id ? S.rule : m;
    rows.push({ m, rule, r: evaluate(S.gt, replay(predFor(m.id), m.classes, rule), now) });
  }
  const pct = (a, b) => (b ? `${Math.round((a / b) * 100)} %` : "-");
  $("compare").innerHTML = `<tr><th>Model</th><th>Rule on / off / smooth / min</th><th>Events caught</th><th>Recall</th><th>Alerts</th><th>False alarms</th><th>FA / person-hour</th><th>Alert precision</th></tr>` +
    rows.map(({ m, r, rule }) => `<tr class="${m.id === S.model.id ? "cur" : ""}"><td>${m.name.split("  ").join(" ")}${S.running.includes(m.id) ? "" : ` <span class="muted small">(not running)</span>`}</td>
      <td>${rule.thr_on.toFixed(2)} / ${rule.thr_off.toFixed(2)} / ${rule.smooth} / ${rule.min_windows}</td>
      <td>${S.hasGT ? `${r.caught} / ${r.done}` : "-"}</td><td>${S.hasGT ? pct(r.caught, r.done) : "-"}</td><td>${r.eps.length}</td>
      <td>${S.hasGT ? r.fa : "-"}</td><td>${S.hasGT && now >= 60 ? r.faph.toFixed(1) : "-"}</td>
      <td>${S.hasGT && r.prec != null ? Math.round(r.prec * 100) + " %" : "-"}</td></tr>`).join("");
}

init().catch((e) => { $("loading").textContent = "Could not reach the server: " + e.message; });
