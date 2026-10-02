// Block-device pool page: one pool at a time, like every other pool page.
// A Pools-tab card lands here with ?pool= preselected; the selector switches
// between this node's bdev pools. Data comes from the module's own endpoints:
//   /api/mod/clio_bdev/pools            -- registered by bdev's RegisterViz()
//   /api/pools/<id>/monitor?query=stats -- reaches bdev's Monitor("stats")

const MOD = 'clio_bdev';
const BDEV_TYPES = ['file', 'ram', 'hbm', 'pinned', 'noop', 's3', 'gcs'];
let CUR_POOL = poolParam();

function typeName(v) {
  return BDEV_TYPES[v] !== undefined ? BDEV_TYPES[v] : `type ${v}`;
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
    document.getElementById('device').innerHTML =
      '<div class="empty">No block-device pools on this node.</div>';
    return;
  }
  CUR_POOL = pools.some((p) => p.pool_id === prev) ? prev : pools[0].pool_id;
  sel.value = CUR_POOL;
  document.getElementById('title').textContent =
    pools.find((p) => p.pool_id === CUR_POOL).pool_name;
}

async function renderDevice() {
  if (!CUR_POOL) return;
  const data = await API.get(
    `/api/pools/${encodeURIComponent(CUR_POOL)}/monitor?query=stats&routing=local`);
  const values = Object.values(data.results || {}).filter((v) => v);
  const s = values.length ? values[0] : null;
  if (!s) {
    document.getElementById('device').innerHTML =
      '<div class="empty">No statistics reported.</div>';
    return;
  }
  const used = s.total_capacity
    ? ((s.total_capacity - s.remaining_capacity) / s.total_capacity) * 100
    : 0;
  // Everything Monitor("stats") reports, minus the two device-health JSON
  // blobs (nested documents, not stats) and the fields the header and meter
  // already show.
  document.getElementById('device').innerHTML = `
    <div class="card">
      <h3>Capacity <span class="badge">${esc(typeName(s.bdev_type))}</span></h3>
      ${meter('used', used)}
      <div class="sub">${bytes(s.remaining_capacity)} free of ${bytes(s.total_capacity)}</div>
    </div>
    <div class="card">
      <h3>All statistics</h3>
      ${kvTable(s, ['pool_name', 'bdev_type', 'device_health', 'failure_prediction'])}
    </div>`;
  await renderPredictions('predictions', CUR_POOL);
}

document.getElementById('pool-select').onchange = (e) => {
  CUR_POOL = e.target.value;
  renderDevice();
};

poll(async () => {
  await loadPools();
  await renderDevice();
}, 3000);
