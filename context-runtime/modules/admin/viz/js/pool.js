// Generic pool page: the click target for pools whose ChiMod ships no website
// of its own. Identity, the learned task-cost model, and a Monitor() prompt.

const POOL = poolParam();

async function render() {
  if (!POOL) {
    showError('errors', 'No ?pool= given — open this page from a pool card.');
    return;
  }
  const data = await API.get('/api/pools');
  const pool = (data.pools || []).find((p) => p.pool_id === POOL);
  if (!pool) {
    showError('errors', `Pool ${POOL} is not composed on this node (shut down?).`);
    return;
  }
  document.getElementById('errors').innerHTML = '';
  document.getElementById('title').textContent = pool.pool_name;
  document.getElementById('identity').innerHTML = `
    <div class="card">
      <h3>${esc(pool.pool_name)}</h3>
      <div class="sub"><span class="badge">${esc(pool.pool_id)}</span>
        ${esc(pool.chimod_name)} · container ${esc(pool.container_id)}</div>
    </div>`;
  await renderPredictions('predictions', POOL);
}

document.getElementById('run').onclick = async () => {
  const query = document.getElementById('query').value.trim();
  const out = document.getElementById('result');
  if (!query) { out.textContent = 'need a query'; return; }
  out.textContent = 'querying…';
  try {
    const data = await API.get(
      `/api/pools/${encodeURIComponent(POOL)}/monitor?query=${encodeURIComponent(query)}`);
    out.textContent = JSON.stringify(data.results, null, 2);
  } catch (e) {
    out.textContent = String(e.message || e);
  }
};

poll(render, 3000);
