// CTE page: the pool's storage-target roster, plus register / unregister.
// All endpoints are the clio_cte_core module's own (see core_viz.cc).

const MOD = 'clio_cte_core';
// Preselect the pool a Pools-tab card navigated here with.
let CUR_POOL = poolParam();

const post = (path, fields) => API.post(path, fields);

/**
 * A target's name IS its backing bdev pool's name, so the device-level stats
 * (latency, bandwidth, total capacity) live in that pool's Monitor("stats").
 * Join them in so the roster shows every monitorable statistic, not just what
 * the CTE's own bookkeeping carries.
 */
async function bdevStatsByName() {
  const byName = {};
  try {
    const index = await API.get('/api/mod/clio_bdev/pools');
    await Promise.all((index.pools || []).map(async (p) => {
      try {
        const data = await API.get(
          `/api/pools/${encodeURIComponent(p.pool_id)}/monitor?query=stats&routing=local`);
        const values = Object.values(data.results || {}).filter((v) => v);
        if (values.length) byName[p.pool_name] = values[0];
      } catch (e) { /* device gone mid-poll; leave its columns blank */ }
    }));
  } catch (e) { /* no bdev module routes yet */ }
  return byName;
}

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
    document.getElementById('targets').innerHTML =
      '<div class="empty">No CTE pools on this node.</div>';
    return;
  }
  CUR_POOL = pools.some((p) => p.pool_id === prev) ? prev : pools[0].pool_id;
  sel.value = CUR_POOL;
}

async function renderTargets() {
  if (!CUR_POOL) return;
  const [data, devices] = await Promise.all([
    API.get(`/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/targets`),
    bdevStatsByName(),
  ]);
  const rows = (data.targets || []).map((t) => {
    const d = devices[t.name] || {};
    return Object.assign({
      total_capacity: d.total_capacity,
      read_latency_us: d.read_latency_us,
      write_latency_us: d.write_latency_us,
      read_bandwidth_mbps: d.read_bandwidth_mbps,
      write_bandwidth_mbps: d.write_bandwidth_mbps,
    }, t);
  });
  document.getElementById('targets').innerHTML = table(rows, [
    ['name', 'target'],
    ['score', 'score', (v) => num(v, 3)],
    ['remaining_space', 'free', (v) => bytes(v)],
    ['total_capacity', 'capacity', (v) => bytes(v)],
    ['read_latency_us', 'read lat', (v) => (v === undefined ? '-' : `${num(v)} µs`)],
    ['write_latency_us', 'write lat', (v) => (v === undefined ? '-' : `${num(v)} µs`)],
    ['read_bandwidth_mbps', 'read MB/s', (v) => num(v)],
    ['write_bandwidth_mbps', 'write MB/s', (v) => num(v)],
    ['bytes_read', 'bytes read', (v) => bytes(v)],
    ['bytes_written', 'bytes written', (v) => bytes(v)],
    ['ops_read', 'reads'],
    ['ops_written', 'writes'],
    ['name', '', (v) =>
      `<button onclick="unregisterTarget('${esc(v)}')">Unregister</button>`],
  ]);
  await renderPredictions('predictions', CUR_POOL);
}

// eslint-disable-next-line no-unused-vars
async function unregisterTarget(name) {
  if (!window.confirm(`Unregister target ${name}? Placement stops using it.`)) return;
  const msg = document.getElementById('rt-msg');
  try {
    await post(`/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/unregister_target`,
      { name });
    msg.textContent = `unregistered ${name}`;
  } catch (e) {
    msg.textContent = `unregister failed: ${e.message}`;
  }
  renderTargets();
}

document.getElementById('rt-submit').onclick = async () => {
  const msg = document.getElementById('rt-msg');
  if (!CUR_POOL) { msg.textContent = 'no pool selected'; return; }
  const fields = { name: document.getElementById('rt-name').value.trim() };
  const attach = document.getElementById('rt-attach').value.trim();
  if (attach) {
    fields.attach_pool_id = attach;
  } else {
    fields.bdev_type = document.getElementById('rt-type').value;
    fields.capacity = document.getElementById('rt-capacity').value.trim();
  }
  try {
    const body = await post(
      `/api/mod/${MOD}/${encodeURIComponent(CUR_POOL)}/register_target`, fields);
    msg.textContent = `registered ${body.name} (bdev ${body.bdev_pool_id})`;
  } catch (e) {
    msg.textContent = `register failed: ${e.message}`;
  }
  renderTargets();
};

document.getElementById('pool-select').onchange = (e) => {
  CUR_POOL = e.target.value;
  renderTargets();
};

poll(async () => {
  await loadPools();
  await renderTargets();
}, 3000);
