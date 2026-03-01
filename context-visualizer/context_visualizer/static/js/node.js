/* Node overview page -- per-node stats with time-series charts */

(function () {
    "use strict";

    var POLL_MS = 2000;
    var MAX_POINTS = 60;

    // Extract node_id from URL path: /node/<id>
    var pathParts = window.location.pathname.split("/");
    var NODE_ID = pathParts[pathParts.length - 1];

    var timestamps = [];
    var cpuHistory = [];
    var ramHistory = [];
    var gpuHistory = [];
    var hbmHistory = [];

    // Worker ring buffers
    var processedHistory = {};
    var queueHistory = {};
    var workerTimestamps = [];

    var lastEventId = 0;
    var headerSet = false;
    var nodeDown = false;

    var cpuChart, ramChart, gpuChart, hbmChart;
    var processedChart, queueChart;

    var COLORS = [
        "rgba(83, 216, 251, 0.9)",
        "rgba(233, 69, 96, 0.9)",
        "rgba(76, 175, 80, 0.9)",
        "rgba(255, 152, 0, 0.9)",
        "rgba(156, 39, 176, 0.9)",
        "rgba(255, 235, 59, 0.9)",
        "rgba(0, 188, 212, 0.9)",
        "rgba(255, 87, 34, 0.9)",
    ];

    function makeChart(canvasId) {
        var ctx = document.getElementById(canvasId).getContext("2d");
        return new Chart(ctx, {
            type: "line",
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        ticks: { color: "#aaa", maxTicksLimit: 10 },
                        grid: { color: "rgba(255,255,255,0.05)" },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { color: "#aaa" },
                        grid: { color: "rgba(255,255,255,0.05)" },
                    },
                },
                plugins: {
                    legend: { labels: { color: "#eee" } },
                },
            },
        });
    }

    function pushRing(arr, value) {
        arr.push(value);
        if (arr.length > MAX_POINTS) arr.shift();
    }

    function pushRingMap(ring, key, value) {
        if (!ring[key]) ring[key] = [];
        ring[key].push(value);
        if (ring[key].length > MAX_POINTS) ring[key].shift();
    }

    function buildDatasets(ring) {
        var keys = Object.keys(ring).sort();
        return keys.map(function (k, i) {
            return {
                label: k,
                data: ring[k].slice(),
                borderColor: COLORS[i % COLORS.length],
                backgroundColor: "transparent",
                tension: 0.3,
                pointRadius: 0,
            };
        });
    }

    function updateSingleChart(chart, labels, data, label, color) {
        chart.data.labels = labels.slice();
        chart.data.datasets = [{
            label: label,
            data: data.slice(),
            borderColor: color,
            backgroundColor: "transparent",
            tension: 0.3,
            pointRadius: 0,
        }];
        chart.update("none");
    }

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

    function setNodeDown(down) {
        nodeDown = down;
        var banner = document.getElementById("nodeDownBanner");
        if (!banner) {
            // Create banner on first use
            banner = document.createElement("div");
            banner.id = "nodeDownBanner";
            banner.className = "error-banner";
            banner.textContent = "Node " + NODE_ID + " is down. Monitoring paused until the node comes back online.";
            var content = document.querySelector(".content");
            if (content) content.insertBefore(banner, content.firstChild.nextSibling);
        }
        banner.style.display = down ? "" : "none";
    }

    function utilizationColor(pct) {
        if (pct < 60) return "var(--success)";
        if (pct < 80) return "var(--warning)";
        return "var(--accent)";
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0 B";
        var units = ["B", "KB", "MB", "GB", "TB"];
        var i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(1) + " " + units[i];
    }

    function updateWorkerTable(workers) {
        var tbody = document.querySelector("#workerTable tbody");
        tbody.innerHTML = "";
        workers.forEach(function (w, i) {
            var tr = document.createElement("tr");
            var q = w.num_queued_tasks || 0;
            var b = w.num_blocked_tasks || 0;
            tr.className = (q + b > 0) ? "row-busy" : "row-idle";
            tr.innerHTML =
                "<td>Worker " + i + "</td>" +
                "<td>" + q + "</td>" +
                "<td>" + b + "</td>" +
                "<td>" + (w.num_tasks_processed || 0) + "</td>";
            tbody.appendChild(tr);
        });
    }

    // ---- Poll Workers ----
    function pollWorkers() {
        fetch("/api/node/" + NODE_ID + "/workers")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) return;

                var workers = data.workers || [];
                var summary = data.summary || {};

                // Update summary cards
                document.getElementById("card-workers").textContent = summary.count || workers.length;
                document.getElementById("card-queued").textContent = summary.queued || 0;
                document.getElementById("card-blocked").textContent = summary.blocked || 0;
                document.getElementById("card-processed").textContent = summary.processed || 0;

                // Update worker table
                updateWorkerTable(workers);

                // Update time-series charts
                var now = new Date().toLocaleTimeString();
                workerTimestamps.push(now);
                if (workerTimestamps.length > MAX_POINTS) workerTimestamps.shift();

                workers.forEach(function (w, i) {
                    var key = "W" + i;
                    pushRingMap(processedHistory, key, w.num_tasks_processed || 0);
                    pushRingMap(queueHistory, key, w.num_queued_tasks || 0);
                });

                processedChart.data.labels = workerTimestamps.slice();
                processedChart.data.datasets = buildDatasets(processedHistory);
                processedChart.update("none");

                queueChart.data.labels = workerTimestamps.slice();
                queueChart.data.datasets = buildDatasets(queueHistory);
                queueChart.update("none");
            })
            .catch(function () { /* ignore */ });
    }

    // ---- Poll System Stats ----
    function pollSystemStats() {
        fetch("/api/node/" + NODE_ID + "/system_stats?min_event_id=" + lastEventId)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error === "node_down") { setNodeDown(true); setStatus(false); return; }
                if (data.error) { setStatus(false); return; }
                setNodeDown(false);
                setStatus(true);

                var entries = data.entries || [];
                if (entries.length === 0) return;

                entries.forEach(function (e) {
                    if (!e || typeof e !== "object") return;

                    // Set header from first response
                    if (!headerSet && e.hostname) {
                        document.getElementById("nodeHeader").textContent =
                            e.hostname + " (" + (e.ip_address || "") + ") - Node " + NODE_ID;
                        headerSet = true;
                    }

                    var now = new Date().toLocaleTimeString();
                    pushRing(timestamps, now);
                    pushRing(cpuHistory, e.cpu_usage_pct || 0);
                    pushRing(ramHistory, e.ram_usage_pct || 0);

                    // Update summary cards
                    document.getElementById("card-cpu").textContent = (e.cpu_usage_pct || 0).toFixed(1) + "%";
                    document.getElementById("card-ram").textContent = (e.ram_usage_pct || 0).toFixed(1) + "%";

                    // GPU sections
                    if (e.gpu_count > 0) {
                        pushRing(gpuHistory, e.gpu_usage_pct || 0);
                        pushRing(hbmHistory, e.hbm_usage_pct || 0);

                        document.getElementById("card-gpu-wrap").style.display = "";
                        document.getElementById("card-hbm-wrap").style.display = "";
                        document.getElementById("gpuChartWrap").style.display = "";
                        document.getElementById("hbmChartWrap").style.display = "";

                        document.getElementById("card-gpu").textContent = (e.gpu_usage_pct || 0).toFixed(1) + "%";
                        document.getElementById("card-hbm").textContent = (e.hbm_usage_pct || 0).toFixed(1) + "%";
                    }

                    // Track last event_id for incremental fetch
                    if (e.event_id !== undefined) {
                        var eid = e.event_id + 1;
                        if (eid > lastEventId) lastEventId = eid;
                    }
                });

                // Update charts
                updateSingleChart(cpuChart, timestamps, cpuHistory, "CPU %", "rgba(83, 216, 251, 0.9)");
                updateSingleChart(ramChart, timestamps, ramHistory, "RAM %", "rgba(76, 175, 80, 0.9)");

                if (gpuHistory.length > 0) {
                    updateSingleChart(gpuChart, timestamps, gpuHistory, "GPU %", "rgba(156, 39, 176, 0.9)");
                    updateSingleChart(hbmChart, timestamps, hbmHistory, "HBM %", "rgba(255, 152, 0, 0.9)");
                }
            })
            .catch(function () { setStatus(false); });
    }

    // ---- Poll BDev Stats ----
    function pollBdevStats() {
        fetch("/api/node/" + NODE_ID + "/bdev_stats")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) return;

                var devices = data.devices || [];
                // Sort by write_bandwidth_mbps descending
                devices.sort(function (a, b) {
                    return (b.write_bandwidth_mbps || 0) - (a.write_bandwidth_mbps || 0);
                });

                var grid = document.getElementById("bdevGrid");
                grid.innerHTML = "";

                if (devices.length === 0) {
                    grid.innerHTML = '<div class="empty-state">No storage devices</div>';
                    return;
                }

                devices.forEach(function (dev) {
                    var card = document.createElement("div");
                    card.className = "bdev-card";

                    var total = dev.total_capacity || 1;
                    var remaining = dev.remaining_capacity || 0;
                    var used = total - remaining;
                    var usedPct = (used / total * 100);
                    var color = utilizationColor(usedPct);
                    var typeLabel = dev.bdev_type === 1 ? "RAM" : "File";

                    card.innerHTML =
                        '<div class="bdev-card-header">' +
                        '<span class="bdev-name">' + (dev.pool_name || dev.pool_id || "bdev") + '</span>' +
                        '<span class="bdev-type-badge bdev-type-' + typeLabel.toLowerCase() + '">' + typeLabel + '</span>' +
                        '</div>' +
                        '<div class="utilization-row">' +
                        '<span class="utilization-label">Capacity</span>' +
                        '<div class="utilization-bar">' +
                        '<div class="utilization-fill" style="width:' + usedPct.toFixed(1) + '%;background:' + color + '"></div>' +
                        '</div>' +
                        '<span class="utilization-pct">' + formatBytes(used) + ' / ' + formatBytes(total) + '</span>' +
                        '</div>' +
                        '<div class="bdev-stats">' +
                        '<div class="bdev-stat"><span>Read BW</span><span>' + (dev.read_bandwidth_mbps || 0).toFixed(1) + ' MB/s</span></div>' +
                        '<div class="bdev-stat"><span>Write BW</span><span>' + (dev.write_bandwidth_mbps || 0).toFixed(1) + ' MB/s</span></div>' +
                        '<div class="bdev-stat"><span>Read Lat</span><span>' + (dev.read_latency_us || 0).toFixed(1) + ' us</span></div>' +
                        '<div class="bdev-stat"><span>Write Lat</span><span>' + (dev.write_latency_us || 0).toFixed(1) + ' us</span></div>' +
                        '<div class="bdev-stat"><span>IOPS</span><span>' + (dev.iops || 0).toFixed(0) + '</span></div>' +
                        '<div class="bdev-stat"><span>Reads</span><span>' + (dev.total_reads || 0) + '</span></div>' +
                        '<div class="bdev-stat"><span>Writes</span><span>' + (dev.total_writes || 0) + '</span></div>' +
                        '</div>';

                    grid.appendChild(card);
                });
            })
            .catch(function () { /* ignore */ });
    }

    function pollAll() {
        pollWorkers();
        pollSystemStats();
        pollBdevStats();
    }

    document.addEventListener("DOMContentLoaded", function () {
        processedChart = makeChart("processedChart");
        queueChart = makeChart("queueChart");
        cpuChart = makeChart("cpuChart");
        ramChart = makeChart("ramChart");
        gpuChart = makeChart("gpuChart");
        hbmChart = makeChart("hbmChart");

        pollAll();
        setInterval(pollAll, POLL_MS);
    });
})();
