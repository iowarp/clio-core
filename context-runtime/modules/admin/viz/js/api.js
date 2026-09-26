// Shared helpers for the Clio dashboard pages.
//
// Every page is plain static HTML served by the daemon and driven by fetch()
// against the routes the admin ChiMod registered (see admin_viz.cc). Keeping the
// API surface in one place means a ChiMod that adds its own page can reuse it.

const API = {
  /** GET a dashboard route and parse its JSON body. Throws on HTTP errors with
   *  the server's own {"error": ...} message when there is one. */
  async get(path) {
    const resp = await fetch(path, { headers: { Accept: 'application/json' } });
    let body = null;
    try {
      body = await resp.json();
    } catch (e) {
      throw new Error(`${path}: ${resp.status} ${resp.statusText}`);
    }
    if (!resp.ok) {
      throw new Error(body && body.error ? body.error
                                        : `${path}: ${resp.status}`);
    }
    return body;
  },

  /** POST a urlencoded form (the shape every dashboard action route accepts).
   *  Throws with the server's error message or joined field errors. */
  async post(path, fields) {
    const resp = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams(fields || {}).toString(),
    });
    let body = {};
    try { body = await resp.json(); } catch (e) { /* non-JSON error */ }
    if (!resp.ok || body.ok === false) {
      const msg = body.error
        || (body.errors
          ? Object.entries(body.errors).map(([k, v]) => `${k}: ${v}`).join('; ')
          : `${resp.status}`);
      throw new Error(msg);
    }
    return body;
  },
};

/** The ?pool= a pool card navigated here with, or null. */
function poolParam() {
  return new URLSearchParams(window.location.search).get('pool');
}

/** Render every scalar entry of a monitor-stats object as a key/value table —
 *  the "all monitorable statistics" view. Arrays/objects are skipped (pages
 *  render those with purpose-built tables); `skip` drops noisy keys. */
function kvTable(obj, skip = []) {
  const rows = Object.entries(obj || {})
    .filter(([k, v]) => !skip.includes(k) && (typeof v !== 'object' || v === null))
    .map(([k, v]) => {
      let shown = v;
      if (/bytes|capacity|_space/.test(k) && typeof v === 'number') shown = bytes(v);
      else if (typeof v === 'number' && !Number.isInteger(v)) shown = num(v, 3);
      else shown = esc(v);
      return `<tr><td>${esc(k)}</td><td class="num">${shown}</td></tr>`;
    }).join('');
  if (!rows) return '<div class="empty">No statistics reported.</div>';
  return `<div class="table-wrap"><table><tbody>${rows}</tbody></table></div>`;
}

/**
 * Render the pool's learned task-cost model into `elId`: per-method CPU/wall
 * coefficients and their MAPE (mean absolute percentage error — how far the
 * prediction is off on average; lower is more accurate), plus the averages
 * across methods that have actually learned something.
 */
async function renderPredictions(elId, poolId) {
  const host = document.getElementById(elId);
  if (!host || !poolId) return;
  const data = await API.get('/api/nodes/local/containers');
  const entry = (data.containers || []).find((c) => c.pool_id === poolId);
  if (!entry) {
    host.innerHTML = '<div class="empty">No model for this pool on this node.</div>';
    return;
  }
  const methods = (entry.methods || []).filter((m) => m.name);
  // A method whose MAPE is still 0 has never had a real completion reinforce
  // it: its coefficient is the seed and an error of "0.0%" would read as
  // "perfectly accurate". Show untrained columns as em-dashes instead, and
  // list trained methods first so the meaningful rows lead.
  const trained = (m) => m.mape > 0 || m.wall_mape > 0;
  methods.sort((a, b) => Number(trained(b)) - Number(trained(a))
    || a.name.localeCompare(b.name));
  // First cell of each group carries the group rule, matching the header.
  const cell = (coef, mape) => (mape > 0
    ? `<td class="num pred-group">${num(coef, 3)}</td>
       <td class="num">${num(mape * 100, 1)}%</td>`
    : `<td class="num pred-group untrained">${num(coef, 3)}</td>
       <td class="num untrained">&mdash;</td>`);
  const rows = methods.map((m) => `<tr${trained(m) ? '' : ' class="untrained"'}>
      <td>${esc(m.name)}</td>
      ${cell(m.coefficient, m.mape)}
      ${cell(m.wall_coefficient, m.wall_mape)}
    </tr>`).join('');
  // Average error over the methods the model has trained on (mape > 0);
  // untouched methods sit at their seed and would dilute the answer to
  // "how accurate is this model in practice".
  const trainedCpu = methods.filter((m) => m.mape > 0);
  const trainedWall = methods.filter((m) => m.wall_mape > 0);
  const avg = (list, key) => (list.length
    ? num(list.reduce((a, m) => a + m[key], 0) / list.length * 100, 1) + '%'
    : 'no samples yet');
  // Two-tier header: "CPU time" / "Wall time" group their coefficient+error
  // pairs, with a rule between groups, so every number sits under an
  // unambiguous column.
  host.innerHTML = `
    <div class="sub">learning rate ${num(entry.learning_rate, 2)} ·
      avg CPU error ${avg(trainedCpu, 'mape')}
      (${trainedCpu.length} trained method${trainedCpu.length === 1 ? '' : 's'}) ·
      avg wall error ${avg(trainedWall, 'wall_mape')} ·
      &mdash; = not trained yet</div>
    <div class="table-wrap"><table class="pred-table">
      <thead>
        <tr><th rowspan="2" class="pred-method">method</th>
            <th colspan="2" class="pred-group">CPU time</th>
            <th colspan="2" class="pred-group">Wall time</th></tr>
        <tr><th class="num pred-group">coefficient</th>
            <th class="num">avg error</th>
            <th class="num pred-group">coefficient</th>
            <th class="num">avg error</th></tr>
      </thead>
      <tbody>${rows}</tbody></table></div>`;
}

/** Reflect connection health in the navbar. */
function setStatus(text, cls) {
  const el = document.getElementById('conn-status');
  if (!el) return;
  el.textContent = text;
  el.className = 'nav-status' + (cls ? ' ' + cls : '');
}

/** Render an error banner into `containerId`, replacing any previous one. */
function showError(containerId, message) {
  const host = document.getElementById(containerId);
  if (!host) return;
  host.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'error';
  div.textContent = message;
  host.appendChild(div);
}

/** Escape text for innerHTML use. */
function esc(value) {
  const div = document.createElement('div');
  div.textContent = value === undefined || value === null ? '' : String(value);
  return div.innerHTML;
}

/** Human-readable byte count. */
function bytes(n) {
  if (n === undefined || n === null || isNaN(n)) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let v = Number(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/** Round to at most `digits` decimals, tolerating null. */
function num(v, digits = 1) {
  if (v === undefined || v === null || isNaN(v)) return '-';
  return Number(v).toFixed(digits);
}

/** A labelled percentage meter, coloured by how hot the value is. */
function meter(label, pct) {
  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  const cls = value > 85 ? 'hot' : value > 60 ? 'warm' : '';
  return `<div class="meter">
    <div class="meter-label"><span>${esc(label)}</span><span>${num(value)}%</span></div>
    <div class="meter-bar"><div class="meter-fill ${cls}" style="width:${value}%"></div></div>
  </div>`;
}

/** Build a table from an array of objects, using `columns` = [[key, label, fmt]].
 *  Alignment is decided PER COLUMN (numeric anywhere -> the whole column,
 *  header included, right-aligns): deciding it per cell left the header and
 *  even some cells of one column aligned differently, so a value could sit
 *  visually under its neighbour's heading. */
function table(rows, columns) {
  if (!rows || rows.length === 0) {
    return '<div class="empty">Nothing to show.</div>';
  }
  const numeric = columns.map(([key]) =>
    rows.some((row) => typeof row[key] === 'number'));
  const cls = (i) => (numeric[i] ? ' class="num"' : '');
  const head = columns.map(([, label], i) =>
    `<th${cls(i)}>${esc(label)}</th>`).join('');
  const body = rows.map((row) => {
    const cells = columns.map(([key, , fmt], i) => {
      const raw = row[key];
      const text = fmt ? fmt(raw, row) : esc(raw);
      return `<td${cls(i)}>${text}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead>
          <tbody>${body}</tbody></table></div>`;
}

/** A minimal inline-SVG sparkline: no charting library, so pages render on a
 *  node with no network access. */
function sparkline(values) {
  const vals = (values || []).filter((v) => v !== null && !isNaN(v));
  if (vals.length < 2) return '<div class="empty">Not enough samples yet.</div>';
  const w = 300;
  const h = 40;
  const max = Math.max(...vals, 1);
  const step = w / (vals.length - 1);
  const points = vals.map((v, i) => {
    const x = (i * step).toFixed(1);
    const y = (h - (v / max) * (h - 4) - 2).toFixed(1);
    return `${i === 0 ? 'M' : 'L'}${x},${y}`;
  }).join(' ');
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
            <rect width="${w}" height="${h}" rx="3"/><path d="${points}"/>
          </svg>`;
}

/** Mark the current page's nav link active. */
function markNav() {
  const here = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-links a').forEach((a) => {
    if (a.getAttribute('href') === here) a.classList.add('active');
  });
}

/** Run `fn` now and every `ms`, keeping the navbar status in sync. A failing
 *  tick also raises a visible banner in #errors (a tiny navbar note is easy to
 *  miss on a page that just stays blank) and clears it when a later tick
 *  succeeds. */
function poll(fn, ms) {
  const tick = async () => {
    const err = document.getElementById('errors');
    try {
      await fn();
      setStatus('connected', 'ok');
      if (err && err.dataset.pollError) {
        err.innerHTML = '';
        delete err.dataset.pollError;
      }
    } catch (e) {
      setStatus(String(e.message || e), 'bad');
      if (err) {
        showError('errors', `refresh failed: ${e.message || e}`);
        err.dataset.pollError = '1';
      }
    }
  };
  tick();
  return setInterval(tick, ms);
}

document.addEventListener('DOMContentLoaded', markNav);
