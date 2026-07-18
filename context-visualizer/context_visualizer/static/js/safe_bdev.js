// Safe-bdev dashboard: poll recovery progress + member roster; drive the
// add / remove / replace controls. Read path is live today; the write controls
// return HTTP 501 until the runtime python bindings expose safe_bdev member
// management (Phase 3b) -- the UI surfaces that gracefully.

const POLL_MS = 2000;
let CUR_POOL = null;
let pollTimer = null;

function toast(msg, isErr) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " toast-err" : "");
  t.style.display = "block";
  setTimeout(() => { t.style.display = "none"; }, 4000);
}

function fmtInt(v) { return (v === undefined || v === null) ? "--" : v; }

async function loadPools() {
  const sel = document.getElementById("poolSelect");
  try {
    const r = await fetch("/api/safe_bdev/pools");
    const data = await r.json();
    const pools = data.pools || [];
    sel.innerHTML = "";
    if (pools.length === 0) {
      sel.innerHTML = "<option value=''>(no safe_bdev pools in config)</option>";
      return;
    }
    pools.forEach(p => {
      const o = document.createElement("option");
      o.value = p.pool_id;
      o.textContent = `${p.pool_name} (${p.pool_id})`;
      o.dataset.maxFailures = p.max_failures;
      sel.appendChild(o);
    });
    CUR_POOL = pools[0].pool_id;
    document.getElementById("poolMeta").textContent =
      `tolerates ${pools[0].max_failures} failure(s)`;
  } catch (e) {
    sel.innerHTML = "<option value=''>(error loading pools)</option>";
  }
}

function renderRecovery(s) {
  const active = Number(s.recovery_active || 0) === 1;
  const total = Number(s.recovery_ops_total || 0);
  const done = Number(s.recovery_ops_completed || 0);
  const inflight = Number(s.recovery_ops_in_flight || 0);
  const remaining = Number(s.recovery_ops_remaining || 0);

  const statusEl = document.getElementById("rec-status");
  if (active) {
    statusEl.textContent = "RECOVERING";
    statusEl.className = "card-value status-recovering";
  } else if (total > 0 && done >= total) {
    statusEl.textContent = "recovered";
    statusEl.className = "card-value status-ok";
  } else {
    statusEl.textContent = "idle";
    statusEl.className = "card-value";
  }
  document.getElementById("rec-inflight").textContent = fmtInt(inflight);
  document.getElementById("rec-remaining").textContent = fmtInt(remaining);
  document.getElementById("rec-completed").textContent = fmtInt(done);
  document.getElementById("rec-total").textContent = fmtInt(total);

  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const bar = document.getElementById("rec-progress");
  bar.style.width = pct + "%";
  bar.className = "progress-inner" + (active ? " progress-active" : "");
  document.getElementById("rec-progress-label").textContent =
    total > 0 ? `${done} / ${total} rows (${pct}%)` : "";
}

function renderMembers(members) {
  const body = document.getElementById("memberBody");
  body.innerHTML = "";
  (members || []).forEach(m => {
    const tr = document.createElement("tr");
    const stateClass = "state-" + (m.state || "active");
    const canReplace = (m.state === "faulty" || m.state === "removed");
    tr.innerHTML =
      `<td>${m.role || ""}</td>` +
      `<td>${fmtInt(m.index)}</td>` +
      `<td class="mono">${m.pool_name || ""}</td>` +
      `<td class="mono">${m.pool_id || ""}</td>` +
      `<td><span class="member-state ${stateClass}">${m.state || ""}</span></td>`;
    const actions = document.createElement("td");
    if (m.state === "active") {
      const rm = document.createElement("button");
      rm.className = "btn btn-danger";
      rm.textContent = "Remove";
      rm.onclick = () => removeMember(m.pool_id, m.role);
      actions.appendChild(rm);
    }
    if (canReplace) {
      const rep = document.createElement("button");
      rep.className = "btn btn-warn";
      rep.textContent = "Replace + recover";
      rep.onclick = () => replaceMember(m.pool_id);
      actions.appendChild(rep);
    }
    tr.appendChild(actions);
    body.appendChild(tr);
  });
}

async function pollStats() {
  if (!CUR_POOL) return;
  try {
    const r = await fetch(`/api/safe_bdev/${CUR_POOL}/stats`);
    const data = await r.json();
    const s = data.stats || {};
    renderRecovery(s);
    renderMembers(s.members);
  } catch (e) { /* transient; keep last render */ }
}

async function removeMember(targetPoolId, role) {
  if (!confirm(`Remove ${role} member ${targetPoolId}?`)) return;
  const r = await fetch(`/api/safe_bdev/${CUR_POOL}/remove_member`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_pool_id: targetPoolId, was_faulty: 1 }),
  });
  const d = await r.json();
  toast(d.success ? "Member removed" : ("Remove failed: " + (d.error || r.status)),
        !d.success);
  pollStats();
}

async function replaceMember(failedPoolId) {
  const path = prompt("Backing path for the replacement bdev:", "/mnt/replacement.dat");
  if (!path) return;
  const r = await fetch(`/api/safe_bdev/${CUR_POOL}/replace_member`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ failed_pool_id: failedPoolId, member_path: path,
                           capacity: "256MB", node_id: 0 }),
  });
  const d = await r.json();
  toast(d.success ? "Recovery started" : ("Replace failed: " + (d.error || r.status)),
        !d.success);
  pollStats();
}

async function addMember() {
  const path = document.getElementById("am-path").value;
  if (!path) { document.getElementById("am-msg").textContent = "path required"; return; }
  const r = await fetch(`/api/safe_bdev/${CUR_POOL}/add_member`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      member_path: path,
      capacity: document.getElementById("am-capacity").value,
      node_id: Number(document.getElementById("am-node").value),
      as_parity: document.getElementById("am-parity").checked ? 1 : 0,
    }),
  });
  const d = await r.json();
  toast(d.success ? "Member added" : ("Add failed: " + (d.error || r.status)),
        !d.success);
  document.getElementById("addMemberForm").style.display = "none";
  pollStats();
}

function wireControls() {
  document.getElementById("poolSelect").addEventListener("change", (e) => {
    CUR_POOL = e.target.value;
    const opt = e.target.selectedOptions[0];
    document.getElementById("poolMeta").textContent =
      opt ? `tolerates ${opt.dataset.maxFailures} failure(s)` : "";
    pollStats();
  });
  document.getElementById("addMemberBtn").onclick = () => {
    document.getElementById("addMemberForm").style.display = "block";
  };
  document.getElementById("am-cancel").onclick = () => {
    document.getElementById("addMemberForm").style.display = "none";
  };
  document.getElementById("am-submit").onclick = addMember;
}

async function main() {
  wireControls();
  await loadPools();
  await pollStats();
  pollTimer = setInterval(pollStats, POLL_MS);
}

document.addEventListener("DOMContentLoaded", main);
