/* Workers page -- time-series ring buffer for processed + queue depth */

(function () {
    "use strict";

    var POLL_MS = 2000;
    var MAX_POINTS = 60;  // 2 minutes of data at 2s intervals
    var timestamps = [];

    // Per-worker ring buffers: { "W0": [v1, v2, ...], ... }
    var processedHistory = {};
    var queueHistory = {};

    var processedChart = null;
    var queueChart = null;

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

    function makeChart(canvasId, label) {
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

    function pushRing(ring, key, value) {
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

    function updateTable(workers) {
        var tbody = document.querySelector("#workerTable tbody");
        tbody.innerHTML = "";
        workers.forEach(function (w, i) {
            var tr = document.createElement("tr");
            var q = w.queued || 0;
            var b = w.blocked || 0;
            tr.className = (q + b > 0) ? "row-busy" : "row-idle";
            tr.innerHTML =
                "<td>Worker " + i + "</td>" +
                "<td>" + q + "</td>" +
                "<td>" + b + "</td>" +
                "<td>" + (w.processed || 0) + "</td>";
            tbody.appendChild(tr);
        });
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

    function poll() {
        fetch("/api/workers")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) { setStatus(false); return; }
                setStatus(true);

                var now = new Date().toLocaleTimeString();
                timestamps.push(now);
                if (timestamps.length > MAX_POINTS) timestamps.shift();

                data.workers.forEach(function (w, i) {
                    var key = "W" + i;
                    pushRing(processedHistory, key, w.processed || 0);
                    pushRing(queueHistory, key, w.queued || 0);
                });

                processedChart.data.labels = timestamps.slice();
                processedChart.data.datasets = buildDatasets(processedHistory);
                processedChart.update("none");

                queueChart.data.labels = timestamps.slice();
                queueChart.data.datasets = buildDatasets(queueHistory);
                queueChart.update("none");

                updateTable(data.workers);
            })
            .catch(function () { setStatus(false); });
    }

    document.addEventListener("DOMContentLoaded", function () {
        processedChart = makeChart("processedChart", "Processed");
        queueChart = makeChart("queueChart", "Queue Depth");
        poll();
        setInterval(poll, POLL_MS);
    });
})();
