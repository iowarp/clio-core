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
        if (e.key === "ArrowRight") select((i + 1) % n, true);
        else if (e.key === "ArrowLeft") select((i - 1 + n) % n, true);
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
