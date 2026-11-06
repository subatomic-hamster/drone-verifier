// drone-verifier dashboard. Static, no build step, no CDN.
//   ?live=8801            one live session (?a=8801 too)         big dvsim GUI + log, meters, ladder, flight plot
//   ?live=8801,8802       two live sessions (?a=8801&b=8802)     versus
//   ?replay=URL[,URL]&speed=4[&video=URL]                        a recorded events.jsonl (or two); stream.mp4 beside it plays if present
//   ?replay=1  or  ?board=1                                      the leaderboard (../results/leaderboard.json, ?board=URL for another)
//   &host=127.0.0.1      where the session ports are (harness/bridge.py on the laptop)
"use strict";

const P = new URLSearchParams(location.search);
const HOST = P.get("host") || "127.0.0.1";
const app = document.getElementById("app");
const el = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
};
const esc = (s) =>
  String(s ?? "").replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );
const GATES = ["G1", "G2", "G3", "G4", "G5", "G6"];
const LAYERS = [
  { id: "L0", name: "static", gates: ["G1"] },
  { id: "L1", name: "analytic", gates: ["G2"] },
  { id: "L2", name: "sim", gates: ["G3", "G4", "G5"] },
  { id: "L3", name: "perturbed", gates: ["G6"] },
];
const ANALYTIC = /^(tw|hover|flight_time|peak)/; // constraint names that belong to L1; the rest are L0
const LEVELS = ["none", "passfail", "diag", "whatif"];
const AGENT_SAW = {
  none: "nothing, only that it was recorded",
  passfail: "pass/fail only",
  diag: "rules, margins, telemetry",
  whatif: "rules, margins, telemetry, what-if",
};
const TARGET = 3.0,
  BAND = 0.3;

// ---- one session's state, built only from its events ------------------------------------------------------------------
class Feed {
  constructor(label) {
    this.label = label;
    this.seq = -1;
    this.log = [];
    this.meters = null;
    this.gates = {};
    this.score = null;
    this.budget = null;
    this.budgetAt = 0;
    this.flights = [];
    this.code = null;
    this.codeRevs = 0;
    this.diff = null;
    this.final = null;
    this.last = null;
    this.tokens = 0;
    this.cost = 0;
    this.model = null;
    this.verifies = 0;
    this.feedback = null;
    this.broken = {}; // constraint name -> the failing constraint event
    this.flown = {}; // scenario -> {pass, reason}, for the current design and controller
    this.flying = 0;
    this.flyingWhat = null;
    this.finishing = false;
    this.dirty = true;
  }

  // Before the first verify of a design the rungs come from these; a design or controller change voids what they said.
  voidFlights(gates) {
    for (const id of gates) delete this.gates[id];
    this.flown = {};
  }

  add(kind, cls, text) {
    this.log.push({ cls, text });
    if (this.log.length > 200) this.log.shift();
  }

  event(e) {
    if (e.seq !== undefined) {
      if (e.seq <= this.seq) return;
      this.seq = e.seq;
    }
    this.dirty = true;
    if (e.feedback) this.feedback = e.feedback;
    switch (e.kind) {
      case "agent_text":
        this.add(e.kind, "k-agent", String(e.text).trim().slice(0, 240));
        break;
      case "agent_usage":
        this.tokens += (e.input_tokens || 0) + (e.output_tokens || 0);
        this.cost += e.cost_usd || 0;
        break;
      case "tool_call":
        this.add(e.kind, "k-call", `> ${e.tool}(${argText(e.args)})`);
        if (e.tool === "finish") this.finishing = true;
        break;
      case "tool_result":
        if (e.tool === "write_controller" && e.ok)
          this.voidFlights(["G3", "G4", "G5", "G6"]);
        if (e.tool === "fly" || e.tool === "verify") break; // flight_end and verify already said it, in bold
        this.add(
          e.kind,
          e.ok ? "k-ok" : "k-err",
          `${e.ok ? "" : "ERROR "}${e.tool}: ${String(e.summary ?? "").slice(0, 200)}`,
        );
        break;
      case "meters":
        this.meters = e;
        this.voidFlights(GATES); // emitted once per design
        break;
      case "constraint":
        if (e.ok) delete this.broken[e.name];
        else this.broken[e.name] = e;
        this.add(
          e.kind,
          e.ok ? "k-good" : "k-err",
          e.message || `${e.name} ${fmt(e.value)} (limit ${fmt(e.limit)})`,
        );
        break;
      case "drc":
        for (const x of (e.errors || []).slice(0, 4))
          this.add(
            e.kind,
            "k-err",
            `DRC ${x.rule}: ${String(x.message).slice(0, 160)}`,
          );
        break;
      case "code": {
        const before = this.code;
        this.code = e.text ?? "";
        this.codeRevs++;
        this.diff = before === null ? null : lineDiff(before, this.code);
        const d = this.diff ? `  +${this.diff[0]} -${this.diff[1]}` : "";
        this.add(
          e.kind,
          e.ok ? "k-code" : "k-err",
          `controller v${this.codeRevs}: ${e.lines} lines${d}${e.ok ? "" : "  REJECTED"}`,
        );
        for (const x of (e.errors || []).slice(0, 3))
          this.add(e.kind, "k-err", `line ${x.line}: ${x.message}`);
        break;
      }
      case "flight_start":
        this.flights.push({ scenario: e.scenario, pts: [], pass: null });
        this.flying++;
        this.flyingWhat = e.scenario;
        this.add(e.kind, "k-flight", `FLIGHT ${e.scenario} seed ${e.seed}`);
        break;
      case "flight_tick": {
        if (!this.flights.length)
          this.flights.push({ scenario: e.scenario, pts: [], pass: null });
        this.flights[this.flights.length - 1].pts.push([e.t, e.z]);
        break;
      }
      case "flight_end": {
        const f = this.flights[this.flights.length - 1];
        if (f) f.pass = !!e.pass;
        this.flying = Math.max(0, this.flying - 1);
        if (!this.finishing && e.scenario !== "robust")
          this.flown[e.scenario] = { pass: !!e.pass, reason: e.reason };
        this.add(
          e.kind,
          e.pass ? "k-verify good" : "k-verify bad",
          `${e.scenario}: ${e.pass ? "PASS" : "FAIL"}${e.reason ? ", " + String(e.reason).slice(0, 160) : ""}`,
        );
        this.last = {
          pass: !!e.pass,
          text: e.pass ? "FLIGHT PASS" : "CRASH",
          why: e.reason,
          of: e.scenario,
        };
        break;
      }
      case "verify": {
        this.verifies = e.attempt ?? this.verifies + 1;
        this.score = e.score;
        this.gates = {};
        this.flying = 0;
        for (const g of e.gates) this.gates[g.id] = g;
        const bad = e.gates.find((g) => !g.pass);
        const layer = bad && LAYERS.find((l) => l.gates.includes(bad.id));
        const line =
          `VERIFIER #${e.attempt}: ${e.score} points` +
          (bad
            ? `, stops at ${layer ? layer.id + " " : ""}${bad.id}: ${String(bad.reason ?? "").slice(0, 150)}`
            : ", all gates pass");
        this.add(e.kind, bad ? "k-verify bad" : "k-verify good", line);
        this.last = {
          pass: !bad,
          text: bad ? `${layer ? layer.id : bad.id} FAILS` : "ALL GATES PASS",
          why: bad ? bad.reason : "",
          of: `attempt ${e.attempt}, ${e.score}/100`,
        };
        break;
      }
      case "budget":
        this.budget = e;
        this.budgetAt = e.t || Date.now() / 1000;
        break;
      case "final":
        this.final = e;
        this.score = e.score;
        this.flying = 0;
        this.finishing = false;
        if (e.gates) {
          this.gates = {};
          for (const g of e.gates) this.gates[g.id] = g;
        }
        this.add(
          e.kind,
          e.score >= 100 ? "k-verify good" : "k-verify",
          `FINAL (held-out seeds): ${e.score}/100, ${e.verify_calls} verify calls, ${e.flights} flights`,
        );
        this.last = {
          pass: e.score >= 100,
          text: `${e.score}/100`,
          why: "final score, held-out seeds",
          of: "",
        };
        break;
    }
  }
}

function argText(a) {
  if (a === undefined || a === null) return "";
  const s =
    typeof a === "string"
      ? a
      : Object.entries(a)
          .map(
            ([k, v]) =>
              `${k}=${k === "code" ? "..." : typeof v === "object" ? JSON.stringify(v) : v}`,
          )
          .join(", ");
  return s.slice(0, 160);
}

function fmt(v) {
  return typeof v === "number"
    ? Math.abs(v) >= 100
      ? v.toFixed(0)
      : v.toFixed(2)
    : Array.isArray(v)
      ? v.map(fmt).join("x")
      : String(v ?? "-");
}

function lineDiff(a, b) {
  // lines added, removed (as multisets; the order of a moved line does not count)
  const n = new Map();
  for (const l of a.split("\n")) n.set(l, (n.get(l) || 0) + 1);
  let added = 0;
  for (const l of b.split("\n")) {
    const c = n.get(l) || 0;
    if (c) n.set(l, c - 1);
    else added++;
  }
  let removed = 0;
  for (const c of n.values()) removed += c;
  return [added, removed];
}

// ---- panels -----------------------------------------------------------------------------------------------------------
// One rung: its gates if a verify ran on this design, else what the tools have said since the last change.
function rung(f, l) {
  const rows = l.gates
    .map((id) => f.gates[id])
    .filter((g) => g && !String(g.reason ?? "").startsWith("not run"));
  const held = f.final ? `held-out ${f.final.score}/100; ` : "";
  const bad = rows.find((g) => !g.pass);
  if (bad) return { state: "fail", text: bad.reason };
  if (rows.length === l.gates.length)
    return {
      state: "pass",
      text: (l.id === "L3" ? held : "") + rows[rows.length - 1].reason,
    };
  const heldOut = f.finishing || f.flyingWhat === "robust";
  if (f.flying && l.id === (heldOut ? "L3" : "L2"))
    return {
      state: "run",
      text: f.finishing ? "held-out seeds" : `flying ${f.flyingWhat}`,
    };
  if (l.id === "L2") {
    const done = Object.entries(f.flown);
    const no = done.find(([, r]) => !r.pass);
    if (no) return { state: "fail", text: `${no[0]}: ${no[1].reason}` };
    if (done.length)
      return {
        state: done.length === l.gates.length ? "pass" : "",
        text: done.map(([s]) => s).join(", ") + " pass",
      };
  }
  if (l.id === "L0" || l.id === "L1") {
    const broken = Object.values(f.broken).filter(
      (c) => ANALYTIC.test(c.name) === (l.id === "L1"),
    );
    if (broken.length)
      return { state: "fail", text: broken.map((c) => c.message).join("; ") };
    const tw = f.meters ? f.meters.tw : null;
    if (tw !== null && tw !== undefined)
      return {
        state: "pass",
        text:
          l.id === "L0"
            ? "cost, mass, envelope, rules within limits"
            : `thrust to weight ${tw.toFixed(2)}`,
      };
  }
  return { state: "", text: "" };
}

// Stops at the first failure: what comes after it was never run, and looks it.
function layersOf(f) {
  let stop = null;
  return LAYERS.map((l) => {
    const r = { id: l.id, name: l.name, ...rung(f, l) };
    if (stop) {
      r.state = "skip";
      r.text = `not run: stopped at ${stop}`;
    } else if (r.state === "fail") stop = l.id;
    return r;
  });
}

const RUNG_WORD = { pass: "PASS", fail: "FAIL", run: "RUN", skip: "-", "": "" };

function ladderHtml(f) {
  const rows = layersOf(f)
    .map(
      (r) =>
        `<div class="rung ${r.state}"><b>${r.id}</b><span class="nm">${r.name}</span><span class="st">${RUNG_WORD[r.state]}</span><span class="tx">${esc(String(r.text ?? ""))}</span></div>`,
    )
    .join("");
  const saw = f.feedback
    ? `feedback <b>${esc(f.feedback)}</b>: agent saw ${esc(AGENT_SAW[f.feedback] ?? "?")}`
    : "feedback level not reported";
  return `${rows}<div class="fb"><span>${saw}</span><span>L4 real: not built</span></div>`;
}

// Later gates do not run once one fails: the first failure is red, what comes after it was never run.
function gateClass(g, id, firstBad) {
  if (!g) return "";
  if (g.pass) return "pass";
  return id === firstBad ? "fail" : "";
}

function metersHtml(f) {
  const m = f.meters;
  if (!m) return "";
  const l = m.limits || {};
  const env = Array.isArray(m.envelope_mm) ? m.envelope_mm : [];
  const lim = Array.isArray(l.envelope_mm) ? l.envelope_mm : [];
  const envFrac =
    env.length && lim.length ? Math.max(...env.map((x, i) => x / lim[i])) : 0;
  const row = (name, val, frac, bad) =>
    `<div class="meter ${bad ? "bad" : ""}"><div class="row"><span class="dim">${name}</span><span class="val">${val}</span></div><div class="bar"><i style="width:${Math.min(100, Math.max(0, frac * 100))}%"></i></div></div>`;
  return [
    row(
      "cost",
      `$${(m.cost_usd ?? 0).toFixed(0)} / ${l.cost_usd ?? "?"}`,
      (m.cost_usd ?? 0) / (l.cost_usd || 1),
      m.cost_usd > l.cost_usd,
    ),
    row(
      "mass",
      `${(m.mass_kg ?? 0).toFixed(2)} / ${l.mass_kg ?? "?"} kg`,
      (m.mass_kg ?? 0) / (l.mass_kg || 1),
      m.mass_kg > l.mass_kg,
    ),
    row(
      "envelope",
      env.length ? env.map((x) => x.toFixed(0)).join("x") : "-",
      envFrac,
      envFrac > 1,
    ),
    row(
      "T:W",
      `${(m.tw ?? 0).toFixed(2)} min ${l.tw ?? "?"}`,
      (m.tw ?? 0) / (2 * (l.tw || 1)),
      m.tw !== undefined && m.tw !== null && m.tw < l.tw,
    ),
  ].join("");
}

function factsHtml(f, now) {
  const b = f.budget;
  const wall = b
    ? Math.max(0, b.wall_left_s - (f.final ? 0 : Math.max(0, now - f.budgetAt)))
    : null;
  const mmss = (s) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const diff = f.diff ? `+${f.diff[0]} -${f.diff[1]}` : "-";
  const drc = f.meters ? (f.meters.drc_errors ?? 0) : 0;
  return (
    `<span class="dim">verify left <b class="${b && b.verify_left <= 2 ? "red" : ""}" style="color:${b && b.verify_left <= 2 ? "" : "var(--fg)"}">${b ? b.verify_left : "-"}</b></span>` +
    `<span class="dim">time <b style="color:var(--fg)">${wall === null ? "-" : mmss(wall)}</b></span>` +
    `<span class="dim">code v${f.codeRevs} <b style="color:var(--fg)">${diff}</b></span>` +
    `<span class="dim">DRC <b class="${drc ? "red" : ""}" style="${drc ? "" : "color:var(--fg)"}">${drc}</b></span>`
  );
}

function drawPlot(cv, f) {
  const r = window.devicePixelRatio || 1,
    W = cv.clientWidth,
    H = cv.clientHeight;
  if (cv.width !== W * r || cv.height !== H * r) {
    cv.width = W * r;
    cv.height = H * r;
  }
  const c = cv.getContext("2d");
  c.setTransform(r, 0, 0, r, 0, 0);
  c.clearRect(0, 0, W, H);
  const zMax = 4.5,
    fs = Math.max(13, H / 7);
  const cur = f.flights[f.flights.length - 1];
  const tMax = Math.max(15, ...(cur ? cur.pts.map((p) => p[0]) : [0])) * 1.02;
  const X = (t) => 4 + (t / tMax) * (W - 8),
    Y = (z) => H - 4 - (Math.max(-0.3, Math.min(zMax, z)) / zMax) * (H - 8);
  c.fillStyle = "#ffc85022";
  c.fillRect(0, Y(TARGET + BAND), W, Y(TARGET - BAND) - Y(TARGET + BAND));
  c.strokeStyle = "#ffc850";
  c.lineWidth = 1;
  c.setLineDash([6, 5]);
  c.beginPath();
  c.moveTo(0, Y(TARGET));
  c.lineTo(W, Y(TARGET));
  c.stroke();
  c.setLineDash([]);
  c.fillStyle = "#ffc850";
  c.font = `${fs}px system-ui`;
  c.fillText(`${TARGET} m`, W - fs * 3, Y(TARGET + BAND) - 3);
  f.flights.slice(-4).forEach((fl, i, all) => {
    const latest = i === all.length - 1;
    c.strokeStyle = latest
      ? fl.pass === false
        ? "#ff5a50"
        : fl.pass
          ? "#5fe38a"
          : "#ffffff"
      : "#ffffff30";
    c.lineWidth = latest ? 3 : 1.5;
    c.beginPath();
    fl.pts.forEach((p, k) =>
      k ? c.lineTo(X(p[0]), Y(p[1])) : c.moveTo(X(p[0]), Y(p[1])),
    );
    c.stroke();
  });
  if (cur && cur.pts.length) {
    c.fillStyle = "#98a2b0";
    const p = cur.pts[cur.pts.length - 1];
    c.fillText(
      `${cur.scenario}  t ${p[0].toFixed(1)} s  z ${p[1].toFixed(2)} m`,
      8,
      fs + 2,
    );
  } else {
    c.fillStyle = "#98a2b0";
    c.fillText("height vs time: no flight yet", 8, fs + 2);
  }
}

function logHtml(f, n) {
  return f.log
    .slice(-n)
    .map((l) => `<div class="${l.cls}">${esc(l.text)}</div>`)
    .join("");
}

// ---- the video --------------------------------------------------------------------------------------------------------
const CATCH_UP_S = 1.5; // behind the live edge by more than this: seek to it (MSE: a seek never re-requests the stream)
const STALL_S = 10;

function codecOf(init) {
  // avc1.PPCCLL from the init segment's avcC box (3 bytes after its version byte)
  for (let i = 0; i + 8 < init.length; i++)
    if (
      init[i] === 0x61 &&
      init[i + 1] === 0x76 &&
      init[i + 2] === 0x63 &&
      init[i + 3] === 0x43
    ) {
      const hex = (b) => b.toString(16).padStart(2, "0");
      return `avc1.${hex(init[i + 5])}${hex(init[i + 6])}${hex(init[i + 7])}`;
    }
  return null;
}

function hasMoof(b) {
  for (let i = 0; i + 4 < b.length; i++)
    if (
      b[i] === 0x6d &&
      b[i + 1] === 0x6f &&
      b[i + 2] === 0x6f &&
      b[i + 3] === 0x66
    )
      return true;
  return false;
}

// The session's fragmented MP4 (GET /stream) through Media Source. Each attempt is one MediaSource; a dropped link or a stall starts a new one.
function liveVideo(box, port) {
  const url = `http://${HOST}:${port}/stream`;
  const note = el(`<span class="nolink">connecting to the window...</span>`);
  box.append(note);
  const stat = { bytes: 0, reconnects: 0, seeks: 0, video: null };
  (window.__streams = window.__streams || {})[port] = stat;

  async function attempt() {
    const ctl = new AbortController();
    const v = document.createElement("video");
    v.muted = true;
    v.autoplay = true;
    v.playsInline = true;
    stat.video = v;
    box.querySelectorAll("video").forEach((x) => x.remove());
    box.prepend(v);
    let last = Date.now();
    const timer = setInterval(() => {
      if (Date.now() - last > STALL_S * 1000) ctl.abort();
    }, 1000);
    try {
      const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const reader = r.body.getReader();
      let head = new Uint8Array(0);
      while (!hasMoof(head)) {
        const { value, done } = await reader.read();
        if (done) throw new Error("stream ended");
        const b = new Uint8Array(head.length + value.length);
        b.set(head);
        b.set(value, head.length);
        head = b;
        last = Date.now();
      }
      const ms = new MediaSource();
      v.src = URL.createObjectURL(ms);
      await new Promise((ok) =>
        ms.addEventListener("sourceopen", ok, { once: true }),
      );
      const sb = ms.addSourceBuffer(`video/mp4; codecs="${codecOf(head)}"`);
      const pending = [head];
      const pump = () => {
        if (sb.updating || !pending.length) return;
        try {
          sb.appendBuffer(pending[0]);
          pending.shift();
        } catch (e) {
          if (e.name !== "QuotaExceededError") throw e;
          sb.remove(0, v.currentTime - 2); // full: free the past, retry on updateend
        }
      };
      sb.addEventListener("updateend", pump);
      const keeper = setInterval(() => {
        if (!sb.buffered.length) return;
        const end = sb.buffered.end(sb.buffered.length - 1);
        if (
          end - v.currentTime > CATCH_UP_S ||
          v.currentTime < sb.buffered.start(0)
        ) {
          v.currentTime = Math.max(
            end - 0.3,
            sb.buffered.start(sb.buffered.length - 1),
          );
          stat.seeks++;
        }
        if (v.paused) v.play().catch(() => {});
        if (!sb.updating && v.currentTime - sb.buffered.start(0) > 30)
          sb.remove(0, v.currentTime - 15);
        note.style.display = "none";
      }, 500);
      try {
        pump();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          last = Date.now();
          stat.bytes += value.length;
          pending.push(value);
          pump();
        }
      } finally {
        clearInterval(keeper);
      }
    } finally {
      clearInterval(timer);
      ctl.abort();
    }
  }

  (async function forever() {
    for (;;) {
      try {
        await attempt();
      } catch (e) {
        console.warn("stream", port, e);
      }
      stat.reconnects++;
      note.textContent = "link down, reconnecting...";
      note.style.display = "";
      await new Promise((ok) => setTimeout(ok, 2000));
    }
  })();
}

// ---- views ------------------------------------------------------------------------------------------------------------
function singleView(f, mount) {
  const root = el(`<div class="single">
    <div class="col"><div class="head"><span class="model"></span><span class="score"></span></div><div class="video"><span class="badge"></span></div>
      <div class="under"><div class="meters"></div><canvas class="plot"></canvas></div></div>
    <div class="col"><div class="ladder"></div><div class="facts"></div><div class="log"></div></div></div>`);
  app.append(root);
  mount(root.querySelector(".video"), root.querySelector(".badge"));
  const q = (s) => root.querySelector(s);
  return (now) => {
    q(".model").textContent = f.label;
    q(".score").textContent = f.score === null ? "" : `${f.score}/100`;
    q(".facts").innerHTML = factsHtml(f, now);
    if (!f.dirty) return;
    f.dirty = false;
    q(".ladder").innerHTML = ladderHtml(f);
    q(".meters").innerHTML = metersHtml(f);
    q(".log").innerHTML = logHtml(f, 40);
    drawPlot(q(".plot"), f);
  };
}

function versusView(fs, mounts) {
  const root = el(`<div class="versus"></div>`);
  app.append(root);
  const parts = fs.map((f, i) => {
    const c =
      el(`<div class="col"><div class="head"><span class="model"></span><span class="score"></span></div><div class="video"><span class="badge"></span></div>
      <div class="vsinfo"><div class="ladder"></div><div class="vsside"><div class="verdict"></div><canvas class="plot"></canvas></div></div><div class="log"></div></div>`);
    root.append(c);
    mounts[i](c.querySelector(".video"), c.querySelector(".badge"));
    return c;
  });
  return () =>
    fs.forEach((f, i) => {
      if (!f.dirty) return;
      f.dirty = false;
      const q = (s) => parts[i].querySelector(s);
      q(".model").textContent = f.label;
      q(".score").textContent = f.score === null ? "" : `${f.score}/100`;
      q(".ladder").innerHTML = ladderHtml(f);
      q(".log").innerHTML = logHtml(f, 8);
      drawPlot(q(".plot"), f);
      const v = q(".verdict");
      v.className =
        "verdict " + (f.last ? (f.last.pass ? "pass" : "fail") : "");
      v.innerHTML = f.last
        ? `${esc(f.last.text)}<small>${esc(String(f.last.why ?? "").slice(0, 110))}${f.last.of ? " &middot; " + esc(f.last.of) : ""}</small>`
        : "";
    });
}

function loop(draw) {
  const tick = () => {
    draw(Date.now() / 1000);
  };
  tick();
  setInterval(tick, 150);
}

// ---- live -------------------------------------------------------------------------------------------------------------
function liveSession(port) {
  const f = new Feed(`session ${port}`);
  const events = () => {
    const es = new EventSource(`http://${HOST}:${port}/events`); // the server replays from seq 0, Feed drops what it has seen
    es.onmessage = (m) => {
      try {
        f.event(JSON.parse(m.data));
      } catch (e) {
        console.error(e, m.data);
      }
    };
    // a dropped connection reconnects by itself, an HTTP error (the bridge's link not up yet: 503) ends it for good
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) setTimeout(events, 2000);
    };
  };
  const state = () =>
    fetch(`http://${HOST}:${port}/state`)
      .then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      })
      .then((s) => {
        f.label = s.model || f.label;
        f.feedback = s.feedback || f.feedback;
        f.dirty = true;
      })
      .catch(() => setTimeout(state, 2000));
  events();
  state();
  const mount = (box, badge) => {
    badge.textContent = "LIVE";
    badge.classList.add("live");
    liveVideo(box, port);
  };
  return { f, mount };
}

function startLive(ports) {
  const ss = ports.map(liveSession);
  loop(
    ss.length === 1
      ? singleView(ss[0].f, ss[0].mount)
      : versusView(
          ss.map((s) => s.f),
          ss.map((s) => s.mount),
        ),
  );
}

// ---- replay -----------------------------------------------------------------------------------------------------------
const MAX_GAP_S = 2; // no video: long silences are cut to this at playback

async function replaySession(url, video, speed) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const ev = (await r.text())
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l));
  const name = url.split("/").slice(-2, -1)[0] || url;
  const f = new Feed(name);
  const abs = new URL(url, location.href);
  try {
    f.feedback = (
      await (await fetch(new URL("results.json", abs))).json()
    ).feedback;
  } catch (e) {
    /* no results beside the events: the events may carry it */
  }
  const vurl = video || new URL("stream.mp4", abs).href;
  let hasVideo = false;
  try {
    hasVideo = (await fetch(vurl, { method: "HEAD" })).ok;
  } catch (e) {
    /* no video: events only */
  }
  let t0 = ev.length ? ev[0].t : 0;
  if (hasVideo)
    // when the recording started (dv/gui.py), so its clock lines up with the events'
    try {
      t0 = (await (await fetch(new URL("stream.json", vurl))).json()).t0;
    } catch (e) {
      /* no sidecar: from the first event */
    }
  let i = 0,
    clock = 0,
    v = null,
    last = performance.now();
  const step = () => {
    // the video is the clock when there is one, so picture and log stay together
    const now = performance.now();
    if (v) clock = v.currentTime;
    else {
      clock += ((now - last) / 1000) * speed;
      if (i < ev.length && ev[i].t - t0 > clock + MAX_GAP_S)
        clock = ev[i].t - t0 - MAX_GAP_S;
    }
    last = now;
    while (i < ev.length && ev[i].t - t0 <= clock) f.event(ev[i++]);
  };
  const mount = (box, badge) => {
    badge.textContent = `RECORDED ${speed}x`;
    badge.classList.add("replay");
    if (hasVideo) {
      v = document.createElement("video");
      v.src = vurl;
      v.muted = true;
      v.autoplay = true;
      v.playbackRate = speed;
      v.playsInline = true;
      v.onerror = () => {
        v.remove();
        v = null;
        box.prepend(
          el(
            `<span class="nolink">recording will not play here; log only</span>`,
          ),
        );
      };
      box.prepend(v);
    } else
      box.prepend(el(`<span class="nolink">no recording: events only</span>`));
  };
  setInterval(step, 50);
  return { f, mount };
}

async function startReplay(urls) {
  const speed = parseFloat(P.get("speed") || "4");
  const ss = await Promise.all(
    urls.map((u) =>
      replaySession(u, urls.length === 1 ? P.get("video") : null, speed),
    ),
  );
  loop(
    ss.length === 1
      ? singleView(ss[0].f, ss[0].mount)
      : versusView(
          ss.map((s) => s.f),
          ss.map((s) => s.mount),
        ),
  );
}

// ---- leaderboard ------------------------------------------------------------------------------------------------------
const NO_RUNS = `<div class="norun">no runs yet</div>`;
const PALETTE = [
  "#5cd6ff",
  "#ffc850",
  "#5fe38a",
  "#ff8ad8",
  "#b79cff",
  "#ff9d5c",
];
const or = (x, show) => (x === null || x === undefined ? "-" : show(x));

function curveSvg(curve) {
  const models = Object.keys(curve || {});
  if (!models.length) return NO_RUNS;
  const W = 620,
    H = 330,
    L = 62,
    B = 34,
    T = 66;
  const x = (i, k) => L + 30 + (i * (W - L - 90)) / 3 + k * 7, // a little apart: several models often share a rate
    y = (r) => H - B - r * (H - B - T);
  let s = `<svg viewBox="0 0 ${W} ${H}">`;
  for (const r of [0, 0.5, 1])
    s += `<line x1="${L}" x2="${W}" y1="${y(r)}" y2="${y(r)}" class="grid"/><text x="${L - 8}" y="${y(r) + 7}" text-anchor="end">${r * 100}%</text>`;
  LEVELS.forEach((l, i) => {
    s += `<text x="${x(i, 0)}" y="${H - 8}" text-anchor="middle">${l}</text>`;
  });
  const stack = {};
  models.forEach((m, k) => {
    const c = PALETTE[k % PALETTE.length];
    const pts = LEVELS.map((l, i) => [curve[m][l], i]).filter(([p]) => p);
    s += `<polyline fill="none" stroke="${c}" stroke-width="3" points="${pts.map(([p, i]) => `${x(i, k)},${y(p.rate)}`).join(" ")}"/>`;
    for (const [p, i] of pts) {
      const key = `${i}:${p.rate}`;
      stack[key] = (stack[key] || 0) + 1;
      s += `<circle cx="${x(i, k)}" cy="${y(p.rate)}" r="6" fill="${c}"/>`;
      s += `<text x="${x(i, 0) + 10}" y="${y(p.rate) - 10 - (stack[key] - 1) * 19}" fill="${c}">${p.passed}/${p.runs}</text>`;
    }
    s += `<text x="${L + k * 110}" y="22" fill="${c}" font-weight="700">${esc(m)}</text>`;
  });
  return s + "</svg>";
}

function paretoSvg(es) {
  const withFront = es.filter((e) => (e.pareto || []).length);
  if (!withFront.length) return NO_RUNS;
  const W = 620,
    H = 330,
    L = 62,
    B = 34,
    T = 66;
  const pts = withFront.flatMap((e) => e.pareto);
  const span = (v) => {
    const lo = Math.min(...v),
      hi = Math.max(...v),
      pad = (hi - lo) * 0.1 || lo * 0.05 || 1;
    return [lo - pad, hi + pad];
  };
  const [c0, c1] = span(pts.map((p) => p.cost_usd)),
    [m0, m1] = span(pts.map((p) => p.mass_kg));
  const x = (c) => L + ((c - c0) / (c1 - c0)) * (W - L - 10),
    y = (m) => H - B - ((m - m0) / (m1 - m0)) * (H - B - T);
  let s = `<svg viewBox="0 0 ${W} ${H}"><line x1="${L}" x2="${L}" y1="${T}" y2="${H - B}" class="grid"/><line x1="${L}" x2="${W}" y1="${H - B}" y2="${H - B}" class="grid"/>
    <text x="${L}" y="${H - 8}">$${c0.toFixed(0)}</text><text x="${W - 10}" y="${H - 8}" text-anchor="end">$${c1.toFixed(0)} cost</text>
    <text x="${L - 6}" y="${H - B}" text-anchor="end">${m0.toFixed(2)}</text><text x="${L - 6}" y="${T + 14}" text-anchor="end">${m1.toFixed(2)}</text><text x="${L - 6}" y="${T + 34}" text-anchor="end">kg</text>`;
  let legend = L;
  withFront.forEach((e, k) => {
    const c = PALETTE[k % PALETTE.length];
    const base = e.baseline;
    if (e.pareto.length > 1)
      s += `<polyline fill="none" stroke="${c}" stroke-width="2.5" ${base ? 'stroke-dasharray="6 5"' : ""} points="${e.pareto.map((p) => `${x(p.cost_usd)},${y(p.mass_kg)}`).join(" ")}"/>`;
    for (const p of e.pareto)
      s += `<circle cx="${x(p.cost_usd)}" cy="${y(p.mass_kg)}" r="6" ${base ? `fill="none" stroke="${c}" stroke-width="2.5"` : `fill="${c}"`}/>`;
    const name = base ? `${e.model} (baseline)` : e.model;
    s += `<text x="${legend}" y="22" fill="${c}" font-weight="700">${esc(name)}</text>`;
    legend += name.length * 11 + 16;
  });
  return s + "</svg>";
}

function trustHtml(p) {
  if (!p) return NO_RUNS;
  const fp = Array.isArray(p.false_positives)
    ? p.false_positives.length
    : p.false_positives;
  const rows = (p.rows || [])
    .map(
      (r) =>
        `<div class="${r.flagged ? "" : "red"}"><span>${esc(r.flaw)}</span><b>${r.flagged ? esc(r.layer ?? "flagged") : "MISSED"}</b></div>`,
    )
    .join("");
  return `<div class="trustnum"><span><b>${p.flagged}/${p.total}</b> planted flaws flagged</span><span><b class="${fp ? "red" : ""}">${fp}</b> false positives</span></div><div class="trustrows">${rows}</div>`;
}

async function fetchJson(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch (e) {
    return null;
  }
}

async function startBoard(url) {
  const results = new URL(".", new URL(url, location.href)).href;
  const [lb, planted] = await Promise.all([
    fetchJson(url),
    fetchJson(results + "planted.json"),
  ]);
  const es = [...((lb && lb.entries) || [])].sort(
    (a, b) =>
      !!a.baseline - !!b.baseline ||
      b.score - a.score ||
      a.verify_calls - b.verify_calls ||
      a.wall_s - b.wall_s ||
      a.tokens - b.tokens,
  );
  const events = (m) => `${results}${encodeURIComponent(m)}-0/events.jsonl`;
  const here = location.pathname;
  const ranked = es.filter((e) => !e.baseline);
  const low = ranked[ranked.length - 1],
    top = ranked[0];
  const rows = es
    .map((e) => {
      const byId = Object.fromEntries((e.gates || []).map((g) => [g.id, g]));
      const firstBad = GATES.find((id) => byId[id] && !byId[id].pass);
      const chips = GATES.map(
        (id) =>
          `<span class="chip ${gateClass(byId[id], id, firstBad)}">${id}</span>`,
      ).join("");
      const two = ranked.length > 1;
      const tag = e.baseline
        ? `<span class="tag">BASELINE</span>`
        : two && e === top
          ? `<span class="tag">HIGHEST</span>`
          : two && e === low
            ? `<span class="tag">LOWEST</span>`
            : "";
      const cls = e.baseline
        ? "base"
        : two && e === top
          ? "top"
          : two && e === low
            ? "low"
            : "";
      const why = (e.failure_reasons || []).slice(-1)[0];
      const name = e.baseline
        ? `<b>${esc(e.model)}</b>`
        : `<a style="color:inherit;text-decoration:none" href="${here}?replay=${encodeURIComponent(events(e.model))}&speed=4"><b>${esc(e.model)}</b></a>`;
      const p = e.pass_within_budget,
        fv = e.first_viable,
        pe = e.prediction_error || {};
      return `<tr class="${cls}" title="${esc(String(why ?? ""))}"><td class="name">${name}${tag}</td>
      <td class="score"><b>${e.score}</b></td><td><div class="chips">${chips}</div></td>
      <td>${p ? `${p.passed}/${p.runs}` : "-"}</td>
      <td>${fv && fv.reached ? `${fv.reached}/${fv.runs} ${or(fv.mean_cost_usd, (c) => "$" + c.toFixed(0))}` : "-"}</td>
      <td>${or(e.robustness_gap, (g) => g.toFixed(1))}</td>
      <td>${or(pe.hover_throttle_pct, (x) => x.toFixed(1))} / ${or(pe.flight_time_pct, (x) => x.toFixed(1))}</td>
      <td>${or(e.verify_calls, (x) => x)}</td><td>${or(e.flights, (x) => x)}</td>
      <td>${or(e.wall_s, (w) => `${Math.floor(w / 60)}:${String(Math.round(w % 60)).padStart(2, "0")}`)}</td>
      <td>${e.baseline ? "no LLM" : or(e.cost_usd, (c) => "$" + c.toFixed(2))}</td><td>${e.baseline ? "-" : or(e.tokens, (t) => (t / 1000).toFixed(0) + "k")}</td></tr>`;
    })
    .join("");
  const vs =
    ranked.length > 1
      ? `<a href="${here}?replay=${encodeURIComponent(events(low.model))},${encodeURIComponent(events(top.model))}&speed=4">${esc(low.model)} vs ${esc(top.model)}, recorded</a>`
      : "";
  const n = Math.max(0, ...es.map((e) => e.trials || 0));
  const table = es.length
    ? `<table><tr><th>model</th><th>score</th><th>gates</th><th>pass in budget</th><th>first viable</th><th>robust gap</th><th>L1 vs sim err % hover / time</th><th>verify</th><th>flights</th><th>time</th><th>LLM cost</th><th>tokens</th></tr>${rows}</table>
    <div class="actions dim"><span>up to ${n} run${n === 1 ? "" : "s"} per model: a demonstration, not a statistic. Score on held-out seeds. Prediction error is model against sim, not real flight.</span>${vs}</div>`
    : NO_RUNS;
  app.append(
    el(`<div class="board"><h1>Leaderboard</h1>${table}
    <div class="panels">
      <div class="panel"><h2>Does verification help? Pass rate by feedback level</h2>${curveSvg(lb && lb.feedback_curve)}<div class="dim small">passed/runs at each level. The level is what the agent saw; this page always shows the full result.</div></div>
      <div class="panel"><h2>Cost vs mass, best viable designs</h2>${paretoSvg(es)}<div class="dim small">Non-dominated designs per model; lower left is better. Hollow: baseline.</div></div>
      <div class="panel"><h2>Can the verifier be trusted?</h2>${trustHtml(planted)}</div>
    </div></div>`),
  );
}

// ---- entry ------------------------------------------------------------------------------------------------------------
function fail(e) {
  const hint =
    location.protocol === "file:"
      ? " (replay and the leaderboard need http: python -m http.server at the repo root, open /web/index.html)"
      : "";
  app.append(el(`<div class="err">${esc(e.message || e)}${hint}</div>`));
  console.error(e);
}

(function main() {
  const live = P.get("live")
    ? P.get("live").split(",")
    : [P.get("a"), P.get("b")].filter(Boolean);
  const rep = P.get("replay");
  try {
    if (live.length) startLive(live);
    else if (rep && rep !== "1") startReplay(rep.split(",")).catch(fail);
    else if (rep === "1" || P.get("board"))
      startBoard(
        P.get("board") && P.get("board") !== "1"
          ? P.get("board")
          : "../results/leaderboard.json",
      ).catch(fail);
    else
      app.append(
        el(
          `<div class="err" style="color:var(--fg)">Say which: ?live=8801 | ?a=8801&amp;b=8802 | ?replay=&lt;events.jsonl&gt;&amp;speed=4 | ?replay=1 (leaderboard)</div>`,
        ),
      );
  } catch (e) {
    fail(e);
  }
})();

// A key flips to the leaderboard and back during the talk.
addEventListener("keydown", (e) => {
  if (e.key === "l" || e.key === "L") {
    const q = new URLSearchParams(location.search);
    if (q.get("board")) {
      history.back();
      return;
    }
    location.search =
      "?board=1" + (P.get("host") ? "&host=" + P.get("host") : "");
  }
});

// A check for the one bit of logic that can quietly lie on stage: open the page with ?selftest=1 and read the console / title.
if (P.get("selftest")) {
  const f = new Feed("t");
  f.event({
    seq: 0,
    kind: "verify",
    attempt: 1,
    score: 20,
    gates: [
      { id: "G1", pass: true, points: 10, max_points: 10 },
      { id: "G3", pass: false, points: 0, max_points: 30, reason: "tilt" },
    ],
  });
  f.event({ seq: 0, kind: "verify", attempt: 2, score: 99, gates: [] }); // a replayed seq is dropped
  const d = lineDiff("a\nb\nc", "a\nc\nd\ne");
  // the rungs: early exit never lights a later layer, a new design voids a verify
  const NR = "not run: an earlier gate failed";
  const g = new Feed("g");
  g.event({
    kind: "verify",
    attempt: 1,
    score: 0,
    gates: [
      { id: "G1", pass: false, reason: "cost $700 over $620" },
      ...GATES.slice(1).map((id) => ({ id, pass: false, reason: NR })),
    ],
  });
  const stopped = layersOf(g);
  const h = new Feed("h");
  h.event({ kind: "meters", tw: 2.1 });
  h.event({ kind: "constraint", name: "tw", ok: false, message: "low" });
  const lowTw = layersOf(h);
  h.event({ kind: "constraint", name: "tw", ok: true, message: "ok" });
  h.event({ kind: "flight_start", scenario: "wind", seed: 1 });
  const flying = layersOf(h);
  h.event({
    kind: "flight_end",
    scenario: "wind",
    pass: false,
    reason: "tilt",
  });
  const crashed = layersOf(h);
  h.event({ kind: "meters", tw: 2.1 });
  const ok =
    f.score === 20 &&
    f.last.text === "L2 FAILS" &&
    f.gates.G1.pass &&
    d[0] === 2 &&
    d[1] === 1 &&
    stopped.map((r) => r.state).join() === "fail,skip,skip,skip" &&
    stopped[2].text === "not run: stopped at L0" &&
    lowTw.map((r) => r.state).join() === "pass,fail,skip,skip" &&
    flying[2].state === "run" &&
    crashed[2].state === "fail" &&
    crashed[3].state === "skip" &&
    layersOf(h)[2].state === "";
  document.title = ok ? "SELFTEST OK" : "SELFTEST FAIL";
}
