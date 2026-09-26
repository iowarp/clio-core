// Effective runtime configuration, plus what the dashboard itself is made of:
// the pages each ChiMod mounted and the routes each one registered.

async function renderConfig() {
  const cfg = await API.get('/api/config');
  const card = (title, rows) => `<div class="card"><h3>${esc(title)}</h3>
    <div class="table-wrap"><table><tbody>${rows.map(
      ([k, v]) => `<tr><td>${esc(k)}</td><td>${v}</td></tr>`).join('')}
    </tbody></table></div></div>`;

  document.getElementById('config').innerHTML = [
    card('Networking', [
      ['rpc port', esc(cfg.port)],
      ['server addr', esc(cfg.server_addr)],
      ['hostfile', esc(cfg.hostfile || '(none)')],
      ['SWIM', cfg.swim_enabled ? '<span class="badge alive">on</span>'
                                : '<span class="badge">off</span>'],
    ]),
    card('Scheduling', [
      ['configured threads', esc(cfg.num_threads)],
      ['live workers', esc(cfg.worker_count)],
      ['local scheduler', esc(cfg.local_sched)],
      ['learning rate', num(cfg.learning_rate, 2)],
    ]),
    card('Dashboard', [
      ['bind', `${esc(cfg.viz_bind)}:${esc(cfg.viz_port)}`],
      ['conf dir', esc(cfg.conf_dir)],
      ['ephemeral', cfg.ephemeral ? 'yes' : 'no'],
    ]),
    card('Loaded ChiMods', (cfg.chimods || []).map((m) => [m, ''])),
  ].join('');
}

async function renderRoutes() {
  // Module pages themselves live on the Pools tab now (each pool card links to
  // its module's website); this page keeps the raw route table.
  const data = await API.get('/api/routes');
  document.getElementById('routes').innerHTML = table(data.routes, [
    ['method', 'method'],
    ['path', 'path'],
    ['mod_name', 'chimod'],
    ['doc', 'description'],
  ]);
}

poll(async () => {
  await renderConfig();
  await renderRoutes();
}, 5000);
