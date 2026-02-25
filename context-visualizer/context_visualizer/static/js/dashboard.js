/* Dashboard -- auto-poll /api/workers every 2 seconds */

(function () {
    "use strict";

    var chart = null;
    var POLL_MS = 2000;

    function initChart() {
        var ctx = document.getElementById("workerChart").getContext("2d");
        chart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Queued",
                        backgroundColor: "rgba(83, 216, 251, 0.7)",
                        data: [],
                    },
                    {
                        label: "Blocked",
                        backgroundColor: "rgba(233, 69, 96, 0.7)",
                        data: [],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: true,
                        ticks: { color: "#aaa" },
                        grid: { color: "rgba(255,255,255,0.05)" },
                    },
                    y: {
                        stacked: true,
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

    function updateCards(summary) {
        document.getElementById("card-workers").textContent = summary.count;
        document.getElementById("card-queued").textContent = summary.queued;
        document.getElementById("card-blocked").textContent = summary.blocked;
        document.getElementById("card-processed").textContent = summary.processed;
    }

    function updateChart(workers) {
        var labels = workers.map(function (_, i) { return "W" + i; });
        var queued = workers.map(function (w) { return w.queued || 0; });
        var blocked = workers.map(function (w) { return w.blocked || 0; });

        chart.data.labels = labels;
        chart.data.datasets[0].data = queued;
        chart.data.datasets[1].data = blocked;
        chart.update("none");
    }

    function updateTable(workers) {
        var tbody = document.querySelector("#workerTable tbody");
        tbody.innerHTML = "";
        workers.forEach(function (w, i) {
            var tr = document.createElement("tr");
            var q = w.queued || 0;
            var b = w.blocked || 0;
            if (q + b > 0) {
                tr.className = "row-busy";
            } else {
                tr.className = "row-idle";
            }
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
                if (data.error) {
                    setStatus(false);
                    return;
                }
                setStatus(true);
                updateCards(data.summary);
                updateChart(data.workers);
                updateTable(data.workers);
            })
            .catch(function () { setStatus(false); });
    }

    document.addEventListener("DOMContentLoaded", function () {
        initChart();
        poll();
        setInterval(poll, POLL_MS);
    });
})();
