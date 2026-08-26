// Safe-bdev page. Reads come through the admin's generic Monitor forward
// (query=stats, answered by this module's own Monitor handler); the writes are
// this module's add_member / remove_member routes.

const MOD = 'clio_safe_bdev';
// Preselect the array a Pools-tab card navigated here with.
let CUR_POOL = poolParam();

const post = (path, fields) => API.post(path, fields);

async function loadPools() {
  const sel = document.getElementById('pool-select');
  const data = await API.get(`/api/mod/${MOD}/pools`);
  const pools = data.pools || [];
  const prev = CUR_POOL;
  sel.innerHTML = '';
  pools.forEach((p) => {
    const o = document.createElement('option');
    o.value = p.pool_id;
    o.textContent = `${p.pool_name} (${p.pool_id})`;
    sel.appendChild(o);
  });
  if (pools.length === 0) {
    CUR_POOL = null;
    document.getElementById('recovery').innerHTML =
      '<div class="empty">No safe-bdev arrays on this node.</div>';
    document.getElementById('members').innerHTML = '';
    return;
  }
  CUR_POOL = pools.some((p) => p.pool_id === prev) ? prev : pools[0].pool_id;
  sel.value = CUR_POOL;
}

async function stats() {
  const data = await API.get(
    `/api/pools/${encodeURIComponent(CUR_POOL)}/monitor?query=stats&routing=local`);
  const values = Object.values(data.results || {}).filter((v) => v);
  return values.length ? values[0] : {};
}

function renderRecovery(s) {
  const total = Number(s.recovery_ops_total || 0);
  const done = Number(s.recovery_ops_completed || 0);
  const active = Number(s.recovery_active || 0) === 1;
  const pct = total > 0 ? (done / total) * 100 : 0;
  const state = active ? '<span class="badge warn">RECOVERING</span>'
    : (total > 0 && done >= total) ? '<span class="badge alive">recovered</span>'
      : '<span class="badge">idle</span>';
  document.getElementById('pool-meta').textContent =
    `tolerates ${s.max_failures ?? '?'} failure(s) · parity level ${s.parity_level ?? '?'}`;
  document.getElementById('recovery').innerHTML = `
    <div class="card">
      <h3>Status ${state}</h3>
      ${meter('rebuilt', pct)}
      <div class="sub">${done} / ${total} ops · ${esc(s.recovery_ops_in_flight ?? 0)} in flight
        · ${esc(s.recovery_ops_remaining ?? 0)} remaining</div>
    </div>
    <div class="card">
      <h3>Array</h3>
      <div class="sub">data members ${esc(s.data_count ?? '?')} ·
        written slots ${esc(s.written_slots ?? 0)} / ${esc(s.total_slots ?? 0)} ·
        dirty ${esc(s.dirty_slots ?? 0)}</div>
      <div class="sub">alloc-log records ${esc(s.alloc_log_records ?? 0)} ·
        reattached ${esc(s.reattached_members ?? 0)}</div>
    </div>`;
}

function memberBadge(state) {
  const cls = state === 'active' ? 'alive'
    : state === 'recovering' ? 'warn' : 'dead';
  return `<span class="badge ${cls}">${esc(state)}</span>`;
}

function renderMembers(members) {
  const rows = (members || []).map((m) => Object.assign({}, m, {
    // safe-bdev packs pool_id as a raw u64 (major<<32|minor); show major.minor
    pool_id_str: `${Math.floor(Number(m.pool_id) / 4294967296)}.${Number(m.pool_id) % 4294967296}`,
  }));
  document.getElementById('members').innerHTML = table(rows, [
    ['role', 'role'],
    ['index', 'index'],
    ['pool_name', 'member'],
    ['pool_id_str', 'pool id'],
    ['state', 'state', (v) => memberBadge(v)],
    ['pool_id_str', '', (v, row) => (row.state === 'active'
      ? `<button onclick="removeMember('${esc(v)}','${esc(row.role)}')">Remove</button>`
      : `<button onclick="replaceMember('${esc(v)}')">Replace + recover</button>`)],
  ]);
}

async function refresh() {
  await loadPools();
  if (!CUR_POOL) return;
  const s = await stats();
  renderRecovery(s);
  // Every scalar the module's Monitor("stats") reports -- capacity (slots),
  // parity level, WAL depth, recovery counters -- in one table.
  document.getElementById('all-stats').innerHTML =
    kvTable(s, ['pool_name']);
  renderMembers(s.members);
  await renderPredictions('predictions', CUR_POOL);
}

// eslint-disable-next-line no-unused-vars
async function removeMember(poolId, role) {
  if (!window.confirm(`Remove ${role} member ${poolId}? The array degrades until it is replaced.`)) return;
  const msg = document.getElementById('am-msg');
  try {
    await post(`/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/remove_member`,
      { member_pool_id: poolId, was_faulty: '1' });
    msg.textContent = `removed ${poolId}`;
  } catch (e) {
    msg.textContent = `remove failed: ${e.message}`;
  }
  refresh();
}

// eslint-disable-next-line no-unused-vars
async function replaceMember(failedPoolId) {
  const path = window.prompt('Backing path for the replacement bdev:', '/mnt/replacement.dat');
  if (!path) return;
  const msg = document.getElementById('am-msg');
  try {
    const body = await post(
      `/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/replace_member`, {
        failed_pool_id: failedPoolId,
        member_name: path,
        capacity: '256MB',
        bdev_type: 'file',
      });
    msg.textContent = `recovering onto ${body.member_name} (${body.member_pool_id})`;
  } catch (e) {
    msg.textContent = `replace failed: ${e.message}`;
  }
  refresh();
}

document.getElementById('am-submit').onclick = async () => {
  const msg = document.getElementById('am-msg');
  if (!CUR_POOL) { msg.textContent = 'no array selected'; return; }
  try {
    const body = await post(`/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/add_member`, {
      member_name: document.getElementById('am-name').value.trim(),
      capacity: document.getElementById('am-capacity').value.trim(),
      bdev_type: document.getElementById('am-type').value,
      as_parity: document.getElementById('am-parity').checked ? '1' : '0',
    });
    msg.textContent = `added ${body.member_name} (${body.member_pool_id})`;
  } catch (e) {
    msg.textContent = `add failed: ${e.message}`;
  }
  refresh();
};

document.getElementById('pool-select').onchange = (e) => {
  CUR_POOL = e.target.value;
  refresh();
};

poll(refresh, 2000);
