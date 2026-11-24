// Progressive enhancements for the CLIO Core site. Every page works without
// this file; it adds theme choice, the phone menu, copy buttons, tabs, the
// "On this page" highlight, and one-time section reveals.
(function () {
  "use strict";
  var root = document.documentElement;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  /** Save a preference, ignoring storage that is blocked or unavailable. */
  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }

  // Theme toggle: dark is the default; a saved choice persists across pages.
  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.dataset.theme === "light" ? "dark" : "light";
      root.dataset.theme = next;
      store("clio-core-theme", next);
    });
  }

  // Phone menu.
  var menuBtn = document.querySelector(".menu-toggle");
  var nav = document.getElementById("primary-nav");
  if (menuBtn && nav) {
    var setOpen = function (open) {
      nav.classList.toggle("open", open);
      menuBtn.setAttribute("aria-expanded", String(open));
    };
    menuBtn.addEventListener("click", function () { setOpen(!nav.classList.contains("open")); });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && nav.classList.contains("open")) { setOpen(false); menuBtn.focus(); }
    });
  }

  // Header dropdowns (Technologies): click or keyboard toggle; hover is CSS.
  document.querySelectorAll(".nav-drop").forEach(function (drop) {
    var btn = drop.querySelector(".nav-drop-toggle");
    if (!btn) return;
    var set = function (open) {
      drop.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
    };
    btn.addEventListener("click", function (e) { e.stopPropagation(); set(!drop.classList.contains("open")); });
    document.addEventListener("click", function (e) { if (!drop.contains(e.target)) set(false); });
    drop.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && drop.classList.contains("open")) { set(false); btn.focus(); }
    });
  });

  // Collapse the section menu on phones so the article leads.
  var sideDetails = document.querySelector(".side-nav details");
  if (sideDetails && window.matchMedia("(max-width: 900px)").matches) sideDetails.open = false;

  // Copy buttons on code blocks and install commands.
  document.querySelectorAll(".copy").forEach(function (btn) {
    btn.hidden = false;
    btn.addEventListener("click", function () {
      var text = btn.dataset.copy;
      if (!text) {
        var block = btn.closest(".code, .installer-command");
        var code = block && block.querySelector("pre code, code");
        text = code ? code.innerText : "";
        // Drop shell prompts so the copied text runs as-is.
        text = text.split("\n").map(function (l) { return l.replace(/^(\$ |PS> )/, ""); }).join("\n");
      }
      var label = btn.textContent;
      var done = function (ok) {
        btn.dataset.state = ok ? "done" : "fail";
        btn.textContent = ok ? "Copied" : "Press Ctrl+C";
        btn.setAttribute("aria-live", "polite");
        clearTimeout(btn._t);
        btn._t = setTimeout(function () { btn.textContent = label === "Copied" ? "Copy" : label; delete btn.dataset.state; }, 1600);
      };
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text.trim()).then(function () { done(true); }, function () { done(false); });
      } else { done(false); }
    });
  });

  // Tabs: <div class="tabs"><section class="tab-panel" data-tab="Linux">...</section></div>
  // A .picker group shows each panel's data-note under its name, and a panel
  // with an id is opened by a link to #id (and records itself in the URL).
  var tabGroup = 0;
  var openers = [];
  document.querySelectorAll(".tabs").forEach(function (tabs) {
    var panels = Array.prototype.slice.call(tabs.querySelectorAll(":scope > .tab-panel"));
    if (panels.length < 2) return;
    tabGroup += 1;
    var list = document.createElement("div");
    list.className = "tab-list";
    list.setAttribute("role", "tablist");
    if (tabs.dataset.label) list.setAttribute("aria-label", tabs.dataset.label);
    var buttons = panels.map(function (panel, i) {
      var id = "tab-" + tabGroup + "-" + i;
      var b = document.createElement("button");
      b.type = "button";
      b.id = id;
      b.setAttribute("role", "tab");
      if (panel.dataset.note) {
        var name = document.createElement("span");
        name.className = "tab-name";
        name.textContent = panel.dataset.tab;
        var note = document.createElement("span");
        note.className = "tab-note";
        note.textContent = panel.dataset.note;
        b.appendChild(name);
        b.appendChild(note);
      } else {
        b.textContent = panel.dataset.tab;
      }
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", id);
      list.appendChild(b);
      return b;
    });
    var select = function (i, focus) {
      buttons.forEach(function (b, j) {
        b.setAttribute("aria-selected", String(i === j));
        b.tabIndex = i === j ? 0 : -1;
        panels[j].hidden = i !== j;
      });
      if (focus) buttons[i].focus();
    };
    panels.forEach(function (panel, i) {
      openers.push({ panel: panel, open: function () { select(i, false); } });
    });
    buttons.forEach(function (b, i) {
      b.addEventListener("click", function () {
        select(i, false);
        if (panels[i].id && history.replaceState) history.replaceState(null, "", "#" + panels[i].id);
      });
      b.addEventListener("keydown", function (e) {
        var n = buttons.length;
        var step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!step) return;
        // Skip tabs that are disabled (the home page's install matrix greys some out).
        for (var k = 1, j = i; k < n; k++) {
          j = (j + step + n) % n;
          if (!buttons[j].disabled) { select(j, true); return; }
        }
      });
    });
    // Open the visitor's platform first when a tab names it.
    var ua = navigator.userAgent;
    var platform = /Windows/.test(ua) ? "Windows" : /Mac OS X/.test(ua) ? "macOS" : "Linux";
    var start = tabs.dataset.platform !== undefined
      ? Math.max(0, panels.findIndex(function (p) { return p.dataset.tab.indexOf(platform) !== -1; })) : 0;
    tabs.insertBefore(list, panels[0]);
    tabs.classList.add("enhanced");
    select(start, false);
  });

  // Install feature matrix (home page). Each <tr data-feature> carries its
  // Spack variants (data-spack), CMake options (data-cmake), commands to run
  // before configuring (data-pre) and a comment for the source build
  // (data-comment); its cells say which method has it (td.y / td.n, in the
  // column order of thead th[data-method]). Ticking features greys out the
  // install tabs that miss one, and rewrites the Spack and Source commands.
  var matrix = document.querySelector(".feature-matrix");
  var installTabs = document.querySelector(".tabs.hero-install.enhanced");
  if (matrix && installTabs) setupFeatureMatrix(matrix, installTabs);

  /** Escape text for insertion as HTML. */
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  /** Split a space-separated attribute into tokens, skipping empty ones. */
  function tokens(s) { return (s || "").split(/\s+/).filter(Boolean); }

  /** Append each token to list unless it is already there. */
  function addUnique(list, items) {
    items.forEach(function (t) { if (list.indexOf(t) === -1) list.push(t); });
  }

  /**
   * Replace a code block's text with shell lines, highlighted like the build
   * does (prompts and comments), so the copy button copies the new command.
   */
  function writeShell(panel, lines) {
    var code = panel && panel.querySelector(".code pre code");
    if (!code) return;
    code.innerHTML = lines.map(function (l) {
      if (l.indexOf("# ") === 0) return '<span class="c">' + escapeHtml(l) + "</span>";
      if (l.indexOf("$ ") === 0) return '<span class="p">$ </span>' + escapeHtml(l.slice(2));
      return escapeHtml(l);
    }).join("\n");
  }

  /** Join words into "cmd a b \" continuation lines, perLine words after the first line. */
  function wrapCommand(head, words, perLine) {
    if (!words.length) return [head];
    var lines = [head + " \\"];
    for (var i = 0; i < words.length; i += perLine) {
      var last = i + perLine >= words.length;
      lines.push("    " + words.slice(i, i + perLine).join(" ") + (last ? "" : " \\"));
    }
    return lines;
  }

  /** Spack commands that install every picked feature. */
  function spackLines(picked) {
    var variants = [];
    picked.forEach(function (row) { addUnique(variants, tokens(row.dataset.spack)); });
    // Compiler constraints (%oneapi) read best at the end of the spec.
    var spec = variants.filter(function (t) { return t.charAt(0) !== "%"; })
      .concat(variants.filter(function (t) { return t.charAt(0) === "%"; }));
    var install = spec.length > 5 ? wrapCommand("$ spack install iowarp", spec, 5)
      : ["$ spack install iowarp" + (spec.length ? " " + spec.join(" ") : "")];
    return ["$ git clone --recurse-submodules https://github.com/iowarp/clio-core.git",
            "$ spack repo add clio-core/installers/spack"].concat(install);
  }

  /** vcpkg commands: install the port with the features (data-vcpkg) the picked rows need. */
  function vcpkgLines(picked) {
    var feats = [];
    picked.forEach(function (row) { addUnique(feats, tokens(row.dataset.vcpkg)); });
    var port = feats.length ? '"iowarp-core[' + feats.join(",") + ']"' : "iowarp-core";
    return ["$ git clone --recurse-submodules https://github.com/iowarp/clio-core.git",
            "$ vcpkg install " + port + " --head \\",
            "    --overlay-ports=clio-core/installers/vcpkg \\",
            "    --overlay-ports=clio-core/installers/vcpkg/overlay-ports"];
  }

  /**
   * Make each matrix section (tbody.fm-section) collapsible from its header
   * row, and keep a "n ticked" count in the header so a collapsed section
   * still shows what it contributes. Returns a function that refreshes the
   * counts.
   */
  function setupSections(matrix) {
    var sections = Array.prototype.slice.call(matrix.querySelectorAll("tbody.fm-section"));
    sections.forEach(function (sec) {
      var btn = sec.querySelector(".fm-toggle");
      var label = sec.querySelector(".fm-static");
      if (!btn) return;
      btn.hidden = false;
      if (label) label.hidden = true;
      var setOpen = function (open) {
        btn.setAttribute("aria-expanded", String(open));
        sec.classList.toggle("collapsed", !open);
      };
      btn.addEventListener("click", function () {
        setOpen(btn.getAttribute("aria-expanded") !== "true");
      });
      // Start with only the sections that have something ticked open, so the
      // install command stays near the top; the counts show the rest.
      setOpen(sec.querySelectorAll("tr[data-feature] input:checked").length > 0);
    });
    return function () {
      sections.forEach(function (sec) {
        var count = sec.querySelector(".fm-count");
        if (!count) return;
        var n = sec.querySelectorAll("tr[data-feature] input:checked").length;
        count.textContent = n ? n + " ticked" : "";
      });
    };
  }

  /** Source-build commands: clone, any extra steps, one cmake configure, build, install. */
  function sourceLines(picked) {
    var flags = [], pre = [], notes = [];
    picked.forEach(function (row) {
      addUnique(flags, tokens(row.dataset.cmake));
      if (row.dataset.pre) addUnique(pre, ["$ " + row.dataset.pre]);
      if (row.dataset.comment) addUnique(notes, [row.dataset.comment]);
    });
    return ["$ git clone --recurse-submodules https://github.com/iowarp/clio-core.git", "$ cd clio-core"]
      .concat(pre, notes,
              wrapCommand("$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release", flags, 1),
              ['$ cmake --build build -j"$(nproc)" && sudo cmake --install build']);
  }

  /** Wire the feature checkboxes to the install tabs that follow them. */
  function setupFeatureMatrix(matrix, tabs) {
    var methods = Array.prototype.map.call(matrix.querySelectorAll("thead th[data-method]"),
                                           function (th) { return th.dataset.method; });
    var heads = matrix.querySelectorAll("thead th[data-method]");
    var rows = Array.prototype.slice.call(matrix.querySelectorAll("tr[data-feature]"));
    var panels = Array.prototype.slice.call(tabs.querySelectorAll(":scope > .tab-panel"));
    var buttons = Array.prototype.slice.call(tabs.querySelectorAll(":scope > .tab-list > button"));
    var status = matrix.querySelector(".fm-status");
    var refreshCounts = setupSections(matrix);
    var panelFor = function (name) {
      return panels.filter(function (p) { return p.dataset.tab === name; })[0];
    };
    var cellsOf = function (row) { return row.querySelectorAll("td"); };
    var has = function (row, k) { var c = cellsOf(row)[k]; return !!c && c.classList.contains("y"); };
    var labelOf = function (row) { return row.querySelector("th").textContent.trim(); };
    var update = function () {
      var picked = rows.filter(function (r) {
        var box = r.querySelector("input");
        return box && box.checked && !box.disabled;
      });
      var ok = methods.map(function (_, k) { return picked.every(function (r) { return has(r, k); }); });
      // Features that cannot be built together (data-conflicts, e.g. CUDA and SYCL).
      rows.forEach(function (r) {
        var box = r.querySelector("input");
        if (!box || box.checked) return;
        var clash = picked.filter(function (p) {
          return tokens(p.dataset.conflicts).indexOf(r.dataset.feature) !== -1 ||
            tokens(r.dataset.conflicts).indexOf(p.dataset.feature) !== -1;
        })[0];
        box.disabled = !!clash;
        if (clash) box.title = "Cannot be built together with " + labelOf(clash);
        else box.removeAttribute("title");
      });
      rows.forEach(function (r) {
        r.classList.toggle("picked", picked.indexOf(r) !== -1);
        Array.prototype.forEach.call(cellsOf(r), function (c, k) { c.classList.toggle("off", !ok[k]); });
      });
      Array.prototype.forEach.call(heads, function (th, k) { th.classList.toggle("off", !ok[k]); });
      panels.forEach(function (p, i) {
        var k = methods.indexOf(p.dataset.tab);
        var b = buttons[i];
        if (!b || k === -1) return;
        b.disabled = !ok[k];
        if (ok[k]) b.removeAttribute("title");
        else b.title = p.dataset.tab + " cannot install every ticked feature";
      });
      var current = buttons.filter(function (b) { return b.getAttribute("aria-selected") === "true"; })[0];
      if (current && current.disabled) {
        var first = buttons.filter(function (b) { return !b.disabled; })[0];
        if (first) first.click();
      }
      writeShell(panelFor("vcpkg"), vcpkgLines(picked));
      writeShell(panelFor("Spack"), spackLines(picked));
      writeShell(panelFor("Source"), sourceLines(picked));
      refreshCounts();
      if (status) {
        var names = methods.filter(function (_, k) { return ok[k]; });
        var isPipDefault = rows.every(function (r) {
          var box = r.querySelector("input");
          return !box || box.disabled || box.checked === (r.dataset.default === "pip");
        });
        status.textContent = (isPipDefault ? "Ticked: what pip installs by default. " : "") +
          (picked.length
            ? picked.length + (picked.length === 1 ? " feature" : " features") + " ticked. Available from " +
              (names.length ? names.join(", ") : "no single method") + "."
            : "Tick features to build one command for them. Methods that cannot provide a ticked feature are greyed out.");
      }
    };

    rows.forEach(function (r) {
      var box = r.querySelector("input");
      if (!box) return;
      box.hidden = false;
      box.addEventListener("change", update);
    });
    if (status) status.hidden = false;
    update();
  }

  // Open every tab panel that contains the URL's #target, outermost first.
  var openHash = function () {
    var target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (!target) return;
    openers.forEach(function (o) { if (o.panel.contains(target)) o.open(); });
    // For a whole panel, show its choices too, so the selection is visible.
    (target.classList.contains("tab-panel") ? target.parentNode : target).scrollIntoView();
  };
  openHash();
  window.addEventListener("hashchange", openHash);

  // "On this page": mark the section currently being read.
  var tocLinks = Array.prototype.slice.call(document.querySelectorAll(".page-toc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var targets = tocLinks.map(function (a) { return document.getElementById(a.hash.slice(1)); }).filter(Boolean);
    var visible = new Set();
    var mark = function () {
      var current = targets.filter(function (t) { return visible.has(t); })[0];
      if (!current) return;
      tocLinks.forEach(function (a) { a.classList.toggle("active", a.hash === "#" + current.id); });
    };
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) visible.add(e.target); else visible.delete(e.target); });
      mark();
    }, { rootMargin: "-70px 0px -65% 0px" });
    targets.forEach(function (t) { io.observe(t); });
  }

  // One-time reveal for below-the-fold landing sections.
  var reveals = document.querySelectorAll(".reveal");
  if (reveals.length) {
    if (reduceMotion.matches || !("IntersectionObserver" in window)) {
      reveals.forEach(function (el) { el.classList.add("in"); });
    } else {
      var ro = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add("in"); ro.unobserve(e.target); }
        });
      }, { rootMargin: "0px 0px -10% 0px" });
      reveals.forEach(function (el) {
        ro.observe(el);
        el.addEventListener("focusin", function () { el.classList.add("in"); });
      });
    }
  }
})();
