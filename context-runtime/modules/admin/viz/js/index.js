// Cluster overview: membership from the local host table plus this node's own
// utilization sample. Deliberately no cross-node fan-out on load -- /api/topology
// answers from SWIM state, so a dead peer costs nothing to display.

let lastEventId = 0;
const cpuHistory = [];
const ramHistory = [];

async function renderTopology() {
  const topo = await API.get('/api/topology');
  document.getElementById('errors').innerHTML = '';

  const self = topo.self_node_id;
  document.getElementById('self-line').textContent =
    `${topo.hostname} · node ${self} · leader is node ${topo.leader_node_id}`;

  const cards = (topo.nodes || []).map((node) => {
    const badges = [
      `<span class="badge ${node.alive ? 'alive' : 'dead'}">${esc(node.state)}</span>`,
      node.is_leader ? '<span class="badge leader">leader</span>' : '',
      node.is_self ? '<span class="badge">this node</span>' : '',
    ].join(' ');
    return `<div class="card clickable ${node.alive ? '' : 'dead'}"
                 onclick="location.href='node.html?node=${encodeURIComponent(node.node_id)}'">
      <h3>node ${esc(node.node_id)} ${badges}</h3>
      <div class="sub">${esc(node.ip_address)}</div>
    </div>`;
  });

  document.getElementById('nodes').innerHTML = cards.length
    ? cards.join('')
    : '<div class="empty">No hosts registered.</div>';
}

async function renderLocal() {
  const stats = await API.get(`/api/nodes/local/system_stats?min_event_id=${lastEventId}`);
  const entries = stats.entries || [];
  entries.forEach((e) => {
    if (e.event_id !== undefined) lastEventId = Math.max(lastEventId, e.event_id + 1);
    cpuHistory.push(e.cpu_usage_pct);
    ramHistory.push(e.ram_usage_pct);
  });
  while (cpuHistory.length > 120) cpuHistory.shift();
  while (ramHistory.length > 120) ramHistory.shift();

  const latest = entries.length ? entries[entries.length - 1] : null;
  const cpu = latest ? latest.cpu_usage_pct : cpuHistory[cpuHistory.length - 1];
  const ram = latest ? latest.ram_usage_pct : ramHistory[ramHistory.length - 1];

  const workers = await API.get('/api/nodes/local/workers');
  const list = workers.workers || [];
  const queued = list.reduce((a, w) => a + (w.num_queued_tasks || 0), 0);
  const blocked = list.reduce((a, w) => a + (w.num_blocked_tasks || 0), 0);
  const processed = list.reduce((a, w) => a + (w.num_tasks_processed || 0), 0);

  document.getElementById('local-summary').innerHTML = `
    <div class="card">
      <h3>CPU</h3>
      ${meter('utilization', cpu)}
      ${sparkline(cpuHistory)}
    </div>
    <div class="card">
      <h3>Memory</h3>
      ${meter('used', ram)}
      ${latest ? `<div class="sub">${bytes(latest.ram_available_bytes)} free of
                  ${bytes(latest.ram_total_bytes)}</div>` : ''}
    </div>
    <div class="card clickable" onclick="location.href='node.html?node=local'">
      <h3>Workers <span class="badge">${list.length}</span></h3>
      <div class="sub">queued ${queued} · blocked ${blocked} · processed ${processed}</div>
    </div>`;
}

poll(async () => {
  await renderTopology();
  await renderLocal();
}, 2000);
