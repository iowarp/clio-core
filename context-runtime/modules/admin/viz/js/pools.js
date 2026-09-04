// Pools composed on this node, the Add Pool flow, and a generic Monitor()
// explorer.
//
// Add Pool: the module list comes from /api/chimods. A module that registered
// the create convention (GET /api/mod/<mod>/create) supplies its own form spec,
// which we render field-by-field; validation and creation POST back to that
// same route, so the module owns what is acceptable. Modules without a form get
// the generic compose editor (identity fields + raw YAML), which reaches the
// same code path as `clio_run compose`.

let CHIMODS = [];
let CUR_SPEC = null;   // the selected module's form spec, or null for generic
let PAGED_MODS = {};   // mod_name -> url_prefix, for modules that ship a page

async function post(path, fields) {
  try {
    const body = await API.post(path, fields);
    return { ok: true, body, status: 200 };
  } catch (e) {
    return { ok: false, body: { error: String(e.message || e) }, status: 0 };
  }
}

/**
 * One-line statistics summary for a pool card.
 *
 * Prefers the well-known keys from the pool's own Monitor("stats") payload
 * (capacity, latency, array shape, CTE holdings), then pads with whatever
 * other scalars the module reported; a pool whose module answers no stats
 * falls back to its task-model training state so every card says something.
 */
function statsSummary(s, modelEntry) {
  const parts = [];
  if (s) {
    if (s.total_capacity !== undefined && s.remaining_capacity !== undefined) {
      parts.push(`${bytes(s.remaining_capacity)} free of ${bytes(s.total_capacity)}`);
    }
    if (s.read_latency_us !== undefined) parts.push(`read ${num(s.read_latency_us)} µs`);
    if (s.write_latency_us !== undefined) parts.push(`write ${num(s.write_latency_us)} µs`);
    if (s.data_count !== undefined) {
      parts.push(`${s.data_count} data + ${s.parity_level ?? 0} parity`);
    }
    if (Number(s.recovery_active) === 1) parts.push('RECOVERING');
    if (s.num_targets !== undefined) {
      parts.push(`${s.num_targets} targets · ${s.num_tags} tags · ${s.num_blobs} blobs`);
    }
    if (parts.length === 0) {
      Object.entries(s)
        .filter(([, v]) => typeof v === 'number')
        .slice(0, 3)
        .forEach(([k, v]) => parts.push(`${k} ${Number.isInteger(v) ? v : num(v, 2)}`));
    }
  }
  if (parts.length === 0 && modelEntry) {
    const trained = (modelEntry.methods || [])
      .filter((m) => m.name && (m.mape > 0 || m.wall_mape > 0)).length;
    const total = (modelEntry.methods || []).filter((m) => m.name).length;
    parts.push(`model: ${trained}/${total} methods trained`);
  }
  return parts.slice(0, 3).join(' · ') || 'no statistics reported';
}

// ---- Progressive card rendering ---------------------------------------------
//
// The pool list itself is one cheap node-local call, so the cards paint
// IMMEDIATELY from it. Everything slower streams in afterwards and updates
// cards in place as it lands: each pool's Monitor("stats") fills that card's
// summary (and its expanded table) the moment it answers, the model fetch
// upgrades the fallback summaries, and the route table upgrades card targets.
// The DOM is rebuilt only when the pool SET changes, so open "more" panels,
// scroll position and hover survive the refresh ticks.

let POOL_SIG = '';           // structure signature of the last build
let POOLS = [];              // last pool list
const EXPANDED = new Set();  // pool ids with their stats panel open
const LAST_STATS = {};       // pool_id -> last Monitor("stats") payload
const LAST_MODELS = {};      // pool_id -> containers entry (task model)

/** Where a pool card navigates: the module's own website (with the pool
 *  preselected via ?pool=), or the generic pool page when it ships none. The
 *  admin module's "page" is the dashboard shell itself, so the admin pool
 *  uses the generic pool page too. */
function poolHref(pool) {
  const prefix = pool.chimod_name === 'clio_admin'
    ? null : PAGED_MODS[pool.chimod_name];
  const page = prefix ? `${prefix}/index.html` : '/viz/clio_admin/pool.html';
  return `${page}?pool=${encodeURIComponent(pool.pool_id)}`;
}

// eslint-disable-next-line no-unused-vars
function openPool(poolId) {
  const pool = POOLS.find((p) => p.pool_id === poolId);
  if (pool) window.location.href = poolHref(pool);
}

/** Refresh one card's summary line and, when expanded, its full stats table. */
function updateCard(poolId) {
  const sum = document.getElementById(`sum-${poolId}`);
  if (sum) sum.textContent = statsSummary(LAST_STATS[poolId], LAST_MODELS[poolId]);
  const panel = document.getElementById(`stats-${poolId}`);
  const toggle = document.getElementById(`more-${poolId}`);
  if (!panel || !toggle) return;
  if (EXPANDED.has(poolId)) {
    panel.style.display = 'block';
    toggle.textContent = 'hide statistics ▴';
    panel.innerHTML = LAST_STATS[poolId]
      ? kvTable(LAST_STATS[poolId], ['pool_name'])
      : '<div class="empty">no statistics reported</div>';
  } else {
    panel.style.display = 'none';
    toggle.textContent = 'read more ▾';
  }
}

// eslint-disable-next-line no-unused-vars
function toggleStats(poolId) {
  if (EXPANDED.has(poolId)) EXPANDED.delete(poolId);
  else EXPANDED.add(poolId);
  updateCard(poolId);
}

// eslint-disable-next-line no-unused-vars
async function destroyPool(poolId, poolName) {
  if (!window.confirm(
      `Shut down pool ${poolName} (${poolId})?\n\n` +
      'Its containers are destroyed on this node and tasks addressed to it will fail.')) {
    return;
  }
  const msg = document.getElementById('add-pool-msg');
  try {
    await API.post(`/api/pools/${encodeURIComponent(poolId)}/destroy`);
    msg.textContent = `shut down ${poolName} (${poolId})`;
  } catch (e) {
    msg.textContent = `shutdown failed: ${e.message}`;
  }
  POOL_SIG = '';  // pool set changed: force a structure rebuild
  renderPools();
}

/** Build the section/card structure. Only called when the pool set changed. */
function buildCards(pools) {
  const groups = {};
  pools.forEach((p) => {
    (groups[p.chimod_name] = groups[p.chimod_name] || []).push(p);
  });
  const sections = Object.keys(groups).sort().map((mod) => {
    const cards = groups[mod]
      .sort((a, b) => a.pool_id.localeCompare(b.pool_id, undefined, { numeric: true }))
      .map((p) => {
        // The admin pool IS the runtime; the server refuses to destroy it, so
        // don't offer the control.
        const closer = p.chimod_name === 'clio_admin' ? '' :
          `<button class="card-close" title="Shut down this pool"
             onclick="event.stopPropagation();
                      destroyPool('${esc(p.pool_id)}','${esc(p.pool_name)}')">&times;</button>`;
        // title = the full name: long ones (file-backed pools are paths) crop
        // and scroll inside the fixed-size card, so hover shows it whole.
        return `<div class="card clickable pool-card" title="${esc(p.pool_name)}"
                     onclick="openPool('${esc(p.pool_id)}')">
          ${closer}
          <h3>${esc(p.pool_name)}</h3>
          <div class="sub"><span class="badge">${esc(p.pool_id)}</span>
            container ${esc(p.container_id)}</div>
          <div class="sub pool-summary" id="sum-${esc(p.pool_id)}">loading statistics…</div>
          <div class="pool-more">
            <a id="more-${esc(p.pool_id)}"
               onclick="event.stopPropagation(); toggleStats('${esc(p.pool_id)}')">read more ▾</a>
            <div class="pool-stats" id="stats-${esc(p.pool_id)}" style="display:none;"></div>
          </div>
        </div>`;
      }).join('');
    return `<h2>${esc(mod)} <span class="badge">${groups[mod].length}</span></h2>
            <div class="grid pool-grid">${cards}</div>`;
  });
  document.getElementById('pools').innerHTML = sections.length
    ? sections.join('')
    : '<div class="empty">No pools composed on this node.</div>';
  pools.forEach((p) => updateCard(p.pool_id));
}

async function renderPools() {
  // The load-bearing call: fast, node-local, no Monitor round trips.
  const data = await API.get('/api/pools');
  POOLS = data.pools || [];
  const sig = POOLS.map((p) => `${p.pool_id}/${p.chimod_name}/${p.pool_name}`)
    .sort().join('|');
  if (sig !== POOL_SIG) {
    POOL_SIG = sig;
    buildCards(POOLS);
  }

  // Everything below streams in and updates cards in place; failures leave
  // the previous values standing rather than blanking anything.
  API.get('/api/routes').then((mods) => {
    PAGED_MODS = {};
    (mods.mounts || []).forEach((m) => { PAGED_MODS[m.mod_name] = m.url_prefix; });
  }).catch(() => {});
  API.get('/api/nodes/local/containers').then((c) => {
    (c.containers || []).forEach((entry) => { LAST_MODELS[entry.pool_id] = entry; });
    POOLS.forEach((p) => updateCard(p.pool_id));
  }).catch(() => {});
  POOLS.forEach((p) => {
    API.get(`/api/pools/${encodeURIComponent(p.pool_id)}/monitor?query=stats&routing=local`)
      .then((d) => {
        const values = Object.values(d.results || {})
          .filter((v) => v && typeof v === 'object');
        if (values.length) {
          LAST_STATS[p.pool_id] = values[0];
          updateCard(p.pool_id);
        }
      }).catch(() => { /* mid-destroy or module without stats */ });
  });
}

// ---- Add Pool flow ---------------------------------------------------------

function fieldHtml(f) {
  const id = `apf-${f.name}`;
  const label = `<label for="${id}">${esc(f.label || f.name)}${f.required ? ' *' : ''}</label>`;
  let input;
  if (f.type === 'select') {
    const opts = (f.options || []).map((o) =>
      `<option value="${esc(o)}" ${o === f.default ? 'selected' : ''}>${esc(o)}</option>`).join('');
    input = `<select id="${id}" data-field="${esc(f.name)}">${opts}</select>`;
  } else if (f.type === 'textarea') {
    input = `<textarea id="${id}" data-field="${esc(f.name)}" rows="6" cols="48"
      placeholder="${esc(f.placeholder || '')}">${esc(f.default || '')}</textarea>`;
  } else {
    input = `<input id="${id}" data-field="${esc(f.name)}" size="28"
      value="${esc(f.default || '')}" placeholder="${esc(f.placeholder || '')}">`;
  }
  const help = f.help ? `<span class="sub">${esc(f.help)}</span>` : '';
  return `<div class="row" style="margin:0.35rem 0;">${label}${input}${help}</div>`;
}

/** The generic compose editor, shown for modules without a create form. */
const GENERIC_FIELDS = [
  { name: 'pool_name', label: 'Pool name', type: 'text', required: true },
  { name: 'pool_id', label: 'Pool ID', type: 'text', required: true, help: 'major.minor' },
  { name: 'pool_query', label: 'Routing', type: 'select', options: ['local', 'dynamic', 'broadcast'], default: 'local' },
  { name: 'config', label: 'Module params (YAML)', type: 'textarea', placeholder: 'key: value' },
];

async function renderModuleForm() {
  const mod = document.getElementById('ap-module').value;
  const host = document.getElementById('ap-form');
  const info = CHIMODS.find((c) => c.name === mod);
  document.getElementById('ap-result').textContent = '';
  if (!mod) { host.innerHTML = ''; CUR_SPEC = null; return; }
  if (info && info.has_create) {
    try {
      CUR_SPEC = await API.get(`/api/mod/${encodeURIComponent(mod)}/create`);
      host.innerHTML =
        `<div class="sub">${esc(CUR_SPEC.title || mod)}${CUR_SPEC.note ? ' — ' + esc(CUR_SPEC.note) : ''}</div>`
        + (CUR_SPEC.fields || []).map(fieldHtml).join('');
      return;
    } catch (e) { /* fall through to generic */ }
  }
  CUR_SPEC = null;
  host.innerHTML =
    '<div class="sub">This module has no form; parameters go in as compose YAML.</div>'
    + GENERIC_FIELDS.map(fieldHtml).join('');
}

function collectFields() {
  const fields = {};
  document.querySelectorAll('#ap-form [data-field]').forEach((el) => {
    fields[el.dataset.field] = el.value;
  });
  return fields;
}

async function submitAddPool(action) {
  const mod = document.getElementById('ap-module').value;
  const out = document.getElementById('ap-result');
  if (!mod) { out.textContent = 'pick a module first'; return; }
  const fields = collectFields();
  fields.action = action;
  const path = CUR_SPEC
    ? `/api/mod/${encodeURIComponent(mod)}/create`
    : '/api/pools/compose';
  if (!CUR_SPEC) fields.mod_name = mod;
  out.textContent = action === 'validate' ? 'validating…' : 'creating…';
  const { ok, body, status } = await post(path, fields);
  if (!ok) {
    out.textContent = body.errors
      ? Object.entries(body.errors).map(([k, v]) => `${k}: ${v}`).join(' · ')
      : (body.error || `failed (${status})`);
    return;
  }
  if (action === 'validate') {
    out.textContent = 'valid ✓';
    return;
  }
  out.textContent = `created ${body.pool_name} (${body.pool_id})`;
  renderPools();
}

async function openAddPool() {
  const panel = document.getElementById('add-pool-panel');
  panel.style.display = 'block';
  const data = await API.get('/api/chimods');
  CHIMODS = data.chimods || [];
  fillModuleSelect('');
  renderModuleForm();
}

function fillModuleSelect(filter) {
  const sel = document.getElementById('ap-module');
  const prev = sel.value;
  sel.innerHTML = '';
  CHIMODS
    .filter((c) => c.name.includes(filter))
    .forEach((c) => {
      const o = document.createElement('option');
      o.value = c.name;
      o.textContent = c.name + (c.has_create ? ' (form)' : '');
      sel.appendChild(o);
    });
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

document.getElementById('add-pool-btn').onclick = openAddPool;
document.getElementById('ap-cancel').onclick = () => {
  document.getElementById('add-pool-panel').style.display = 'none';
};
document.getElementById('ap-search').oninput = (e) => {
  fillModuleSelect(e.target.value.trim());
  renderModuleForm();
};
document.getElementById('ap-module').onchange = renderModuleForm;
document.getElementById('ap-validate').onclick = () => submitAddPool('validate');
document.getElementById('ap-create').onclick = () => submitAddPool('create');

// ---- Monitor explorer -------------------------------------------------------

async function runQuery() {
  const pool = document.getElementById('pool').value.trim();
  const query = document.getElementById('query').value.trim();
  const routing = document.getElementById('routing').value;
  const out = document.getElementById('result');
  if (!pool || !query) {
    out.textContent = 'Need both a pool id and a query.';
    return;
  }
  out.textContent = 'querying…';
  try {
    const data = await API.get(
      `/api/pools/${encodeURIComponent(pool)}/monitor` +
      `?query=${encodeURIComponent(query)}&routing=${encodeURIComponent(routing)}`);
    out.textContent = JSON.stringify(data.results, null, 2);
  } catch (e) {
    out.textContent = String(e.message || e);
  }
}

document.getElementById('run').onclick = runQuery;
poll(renderPools, 5000);
