/* Config page -- fetch and display YAML config as formatted JSON */

(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", function () {
        fetch("/api/config")
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data.error) {
                    document.getElementById("config-error").textContent = data.error;
                    document.getElementById("config-error").style.display = "block";
                    document.getElementById("config-display").textContent = "";
                    return;
                }
                document.getElementById("config-display").textContent =
                    JSON.stringify(data.config, null, 2);
            })
            .catch(function (err) {
                document.getElementById("config-error").textContent =
                    "Failed to load config: " + err;
                document.getElementById("config-error").style.display = "block";
                document.getElementById("config-display").textContent = "";
            });
    });
})();
