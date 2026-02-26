/* Topology page -- cluster node grid with search/filter */

(function () {
    "use strict";

    var POLL_MS = 2000;

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
            var card = document.createElement("div");
            card.className = "node-card";
            card.onclick = function () {
                window.location = "/node/" + node.node_id;
            };

            var html = '<div class="node-card-header">' +
                '<span class="node-hostname">' + (node.hostname || "node-" + node.node_id) + '</span>' +
                '<span class="alive-badge">alive</span>' +
                '</div>' +
                '<div class="node-card-meta">' + (node.ip_address || "") + ' &middot; ID ' + node.node_id + '</div>';

            html += makeBar("CPU", node.cpu_usage_pct || 0);
            html += makeBar("RAM", node.ram_usage_pct || 0);

            if (node.gpu_count > 0) {
                html += makeBar("GPU", node.gpu_usage_pct || 0);
            }

            card.innerHTML = html;
            grid.appendChild(card);
        });

        if (filtered.length === 0) {
            grid.innerHTML = '<div class="empty-state">No nodes match filter</div>';
        }
    }

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
