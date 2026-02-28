/* Topology page -- cluster node grid with search/filter */

(function () {
    "use strict";

    var POLL_MS = 2000;

    // Track which nodes the user has shut down or restarted so we can
    // update their state immediately without waiting for the next poll.
    // Maps node_id -> "down" | "restarting"
    var localOverrides = {};

    function setStatus(ok) {
        var el = document.getElementById("conn-status");
        if (ok) {
            el.textContent = "Connected";
            el.className = "nav-status connected";
        } else {
            el.textContent = "Disconnected";
            el.className = "nav-status error";
        }
    }

    function parseFilter(text) {
        text = text.trim();
        if (!text) return null; // null = show all

        // Try range: "1-20"
        var rangeMatch = text.match(/^(\d+)\s*-\s*(\d+)$/);
        if (rangeMatch) {
            var lo = parseInt(rangeMatch[1], 10);
            var hi = parseInt(rangeMatch[2], 10);
            return function (node) {
                var nid = node.node_id;
                return nid >= lo && nid <= hi;
            };
        }

        // Try comma-separated: "1,3,5"
        if (/^\d+(,\s*\d+)+$/.test(text)) {
            var ids = text.split(",").map(function (s) { return parseInt(s.trim(), 10); });
            return function (node) {
                return ids.indexOf(node.node_id) !== -1;
            };
        }

        // Single number
        if (/^\d+$/.test(text)) {
            var singleId = parseInt(text, 10);
            return function (node) {
                return node.node_id === singleId;
            };
        }

        // Keyword search (hostname or IP)
        var lower = text.toLowerCase();
        return function (node) {
            return (node.hostname && node.hostname.toLowerCase().indexOf(lower) !== -1) ||
                   (node.ip_address && node.ip_address.toLowerCase().indexOf(lower) !== -1);
        };
    }

    function utilizationColor(pct) {
        if (pct < 60) return "var(--success)";
        if (pct < 80) return "var(--warning)";
        return "var(--accent)";
    }

    function makeBar(label, pct) {
        var color = utilizationColor(pct);
        return '<div class="utilization-row">' +
            '<span class="utilization-label">' + label + '</span>' +
            '<div class="utilization-bar">' +
            '<div class="utilization-fill" style="width:' + pct.toFixed(1) + '%;background:' + color + '"></div>' +
            '</div>' +
            '<span class="utilization-pct">' + pct.toFixed(1) + '%</span>' +
            '</div>';
    }

    function renderNodes(nodes) {
        var filterText = document.getElementById("nodeSearch").value;
        var filterFn = parseFilter(filterText);

        var filtered = filterFn ? nodes.filter(filterFn) : nodes;
        // Sort by node_id
        filtered.sort(function (a, b) { return a.node_id - b.node_id; });

        var grid = document.getElementById("nodeGrid");
        grid.innerHTML = "";

        filtered.forEach(function (node) {
            // Apply local overrides (user just clicked shutdown/restart)
            var override = localOverrides[node.node_id];
            var alive = node.alive;
            if (override === "down") {
                alive = false;
            } else if (override === "restarting") {
                // Keep showing as down until the server confirms it's back
                if (node.alive) {
                    // Server says it's alive again — clear the override
                    delete localOverrides[node.node_id];
                } else {
                    alive = false;
                }
            }

            var card = document.createElement("div");
            card.className = "node-card" + (alive ? "" : " node-card-down");
            card.onclick = function () {
                window.location = "/node/" + node.node_id;
            };

            var badgeClass = alive ? "alive-badge" : "dead-badge";
            var badgeText = alive ? "alive" : "down";

            var html = '<div class="node-card-header">' +
                '<span class="node-hostname">' + (node.hostname || "node-" + node.node_id) + '</span>' +
                '<span class="' + badgeClass + '">' + badgeText + '</span>' +
                '</div>' +
                '<div class="node-card-meta">' + (node.ip_address || "") + ' &middot; ID ' + node.node_id + '</div>';

            if (alive) {
                html += makeBar("CPU", node.cpu_usage_pct || 0);
                html += makeBar("RAM", node.ram_usage_pct || 0);
                if (node.gpu_count > 0) {
                    html += makeBar("GPU", node.gpu_usage_pct || 0);
                }
            }

            // Actions: Shutdown only when alive, Restart only when down
            html += '<div class="node-card-actions">';
            if (alive) {
                html += '<button class="btn-action btn-shutdown" onclick="event.stopPropagation(); nodeAction(this, ' + node.node_id + ', \'shutdown\')">Shutdown</button>';
            } else {
                html += '<button class="btn-action btn-restart" onclick="event.stopPropagation(); nodeAction(this, ' + node.node_id + ', \'restart\')">Restart</button>';
            }
            html += '</div>';

            card.innerHTML = html;
            grid.appendChild(card);
        });

        if (filtered.length === 0) {
            grid.innerHTML = '<div class="empty-state">No nodes match filter</div>';
        }
    }

    // Expose nodeAction globally so inline onclick handlers can call it
    window.nodeAction = function (btn, nodeId, action) {
        if (!confirm("Are you sure you want to " + action + " node " + nodeId + "?")) {
            return;
        }
        btn.disabled = true;
        btn.textContent = action === "shutdown" ? "Stopping..." : "Restarting...";

        // Restart waits for the runtime to accept connections (up to ~10s)
        var controller = new AbortController();
        var timeoutId = setTimeout(function () { controller.abort(); }, 30000);

        fetch("/api/topology/node/" + nodeId + "/" + action,
              { method: "POST", signal: controller.signal })
            .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
            .then(function (res) {
                clearTimeout(timeoutId);
                if (res.ok && res.data.success) {
                    if (action === "shutdown") {
                        // Mark as down immediately so the UI updates on next render
                        localOverrides[nodeId] = "down";
                        renderNodes(lastNodes);
                    } else {
                        btn.textContent = "Restarted";
                        btn.classList.add("btn-success-done");
                        // Clear any "down" override so the poll picks it up
                        delete localOverrides[nodeId];
                    }
                } else {
                    var errMsg = (res.data && res.data.stderr) ? res.data.stderr : "";
                    btn.textContent = "Failed";
                    btn.title = errMsg || "Restart failed";
                    btn.classList.add("btn-failed");
                    if (errMsg) {
                        console.error("Restart node " + nodeId + " failed:\n" + errMsg);
                    }
                }
            })
            .catch(function () {
                clearTimeout(timeoutId);
                btn.textContent = "Failed";
                btn.classList.add("btn-failed");
            });
    };

    var lastNodes = [];

    function poll() {
        fetch("/api/topology")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { setStatus(false); return; }
                setStatus(true);
                lastNodes = data.nodes || [];
                renderNodes(lastNodes);
            })
            .catch(function () { setStatus(false); });
    }

    document.addEventListener("DOMContentLoaded", function () {
        // Re-render on search input
        document.getElementById("nodeSearch").addEventListener("input", function () {
            renderNodes(lastNodes);
        });

        poll();
        setInterval(poll, POLL_MS);
    });
})();
