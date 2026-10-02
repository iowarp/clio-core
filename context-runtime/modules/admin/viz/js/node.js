// Per-node detail. Everything here is one Monitor query per panel, addressed at
// ?node= (a node id, or "local"); the admin ChiMod forwards each one and hands
// back the decoded payload.

const nodeParam = new URLSearchParams(window.location.search).get('node') || 'local';
const nodePath = `/api/nodes/${encodeURIComponent(nodeParam)}`;
let lastEventId = 0;
const cpuHistory = [];

document.getElementById('title').textContent =
  nodeParam === 'local' ? 'This node' : `Node ${nodeParam}`;

async function renderSystem() {
  const stats = await API.get(`${nodePath}/system_stats?min_event_id=${lastEventId}`);
  (stats.entries || []).forEach((e) => {
    if (e.event_id !== undefined) lastEventId = Math.max(lastEventId, e.event_id + 1);
    cpuHistory.push(e.cpu_usage_pct);
  });
  while (cpuHistory.length > 120) cpuHistory.shift();

  const entries = stats.entries || [];
  const latest = entries.length ? entries[entries.length - 1] : null;
  if (!latest && cpuHistory.length === 0) {
    document.getElementById('system').innerHTML =
      '<div class="empty">No samples yet.</div>';
    return;
  }
  const host = latest
    ? `<div class="sub">${esc(latest.hostname)} · ${esc(latest.ip_address)}
       ${latest.is_leader ? '<span class="badge leader">leader</span>' : ''}</div>`
    : '';
  document.getElementById('system').innerHTML = `
    <div class="card">
      <h3>CPU</h3>
      ${meter('utilization', latest ? latest.cpu_usage_pct : cpuHistory[cpuHistory.length - 1])}
      ${sparkline(cpuHistory)}
    </div>
    <div class="card">
      <h3>Memory</h3>
      ${latest ? meter('used', latest.ram_usage_pct) : ''}
      ${latest ? `<div class="sub">${bytes(latest.ram_available_bytes)} free of
                  ${bytes(latest.ram_total_bytes)}</div>` : ''}
    </div>
    <div class="card">
      <h3>Identity</h3>
      ${host}
      <div class="sub">node ${esc(stats.node_id)}</div>
    </div>`;
}

async function renderWorkers() {
  const data = await API.get(`${nodePath}/workers`);
  document.getElementById('workers').innerHTML = table(data.workers, [
    ['worker_id', 'worker'],
    ['is_active', 'active', (v) => (v ? '<span class="badge alive">yes</span>' : '<span class="badge">idle</span>')],
    ['num_queued_tasks', 'queued'],
    ['num_blocked_tasks', 'blocked'],
    ['num_periodic_tasks', 'periodic'],
    ['num_retry_tasks', 'retry'],
    ['num_tasks_processed', 'processed'],
    ['load', 'load', (v) => num(v, 2)],
    ['suspend_period_us', 'suspend µs'],
  ]);
}

// Block-device and container detail moved to where they belong: devices live
// on the bdev module's website and per-pool models on each pool's page (via
// the Pools tab cards). This page is the node's utilization + worker view.

poll(async () => {
  await renderSystem();
  await renderWorkers();
}, 2000);
