/* Progressive enhancement only. Every word on this page is in the HTML and
   readable with JavaScript disabled; this file adds motion and the link between
   a project and the skills it uses. Nothing here gates content. */

(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ---------------------------------------------------- reveal on scroll */

  function initReveal() {
    var items = document.querySelectorAll(".reveal");
    if (!items.length) return;

    // No IntersectionObserver, or the reader asked for less motion: show everything.
    if (!("IntersectionObserver" in window) || reduced.matches) {
      items.forEach(function (el) { el.classList.add("is-in"); });
      return;
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0 });

    items.forEach(function (el) {
      el.classList.add("is-armed");   // only now is it safe to hide it
      io.observe(el);
    });

    // Failsafe: if the observer never reports (an odd viewport, a headless
    // renderer, a browser bug), show everything rather than leave the page blank.
    window.setTimeout(function () {
      items.forEach(function (el) { el.classList.add("is-in"); });
    }, 1600);
  }

  /* ------------------------------------------- scroll progress indicator */

  function initProgress() {
    var bar = document.querySelector(".scroll-progress i");
    if (!bar) return;
    var ticking = false;

    function update() {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      var pct = max > 0 ? (h.scrollTop || document.body.scrollTop) / max : 0;
      bar.style.transform = "scaleX(" + Math.min(1, Math.max(0, pct)) + ")";
      ticking = false;
    }

    window.addEventListener("scroll", function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    update();
  }

  /* --------------------------------------------------- active section nav */

  function initNav() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".site-nav__link"));
    if (!links.length || !("IntersectionObserver" in window)) return;

    var byId = {};
    links.forEach(function (a) {
      var id = a.getAttribute("href").replace("#", "");
      var target = document.getElementById(id);
      if (target) byId[id] = a;
    });

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var a = byId[entry.target.id];
        if (!a) return;
        if (entry.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("is-active"); });
          a.classList.add("is-active");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });

    Object.keys(byId).forEach(function (id) {
      io.observe(document.getElementById(id));
    });
  }

  /* ------------------------------- tie the skills panel to the project in view */

  function initSkillLink() {
    var panel = document.querySelector(".layout__aside");
    var projects = Array.prototype.slice.call(document.querySelectorAll(".project[data-skills]"));
    if (!panel || !projects.length || !("IntersectionObserver" in window)) return;

    var hint = panel.querySelector(".aside__hint");
    var chipsByKey = {};
    panel.querySelectorAll("[data-skill]").forEach(function (chip) {
      chipsByKey[chip.getAttribute("data-skill")] = chip;
    });

    var current = null;

    function clear() {
      Object.keys(chipsByKey).forEach(function (k) {
        chipsByKey[k].classList.remove("is-lit");
      });
      panel.classList.remove("is-linked");
      if (hint) hint.textContent = hint.getAttribute("data-default") || "";
    }

    function light(project) {
      if (project === current) return;
      current = project;
      if (!project) { clear(); return; }

      Object.keys(chipsByKey).forEach(function (k) {
        chipsByKey[k].classList.remove("is-lit");
      });
      var keys = (project.getAttribute("data-skills") || "").split(/\s+/);
      var hits = 0;
      keys.forEach(function (k) {
        if (chipsByKey[k]) { chipsByKey[k].classList.add("is-lit"); hits++; }
      });
      panel.classList.toggle("is-linked", hits > 0);
      if (hint && hits > 0) {
        var title = project.querySelector(".project__title");
        hint.textContent = "Used in " + (title ? title.textContent : "this project") + ".";
      }
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) light(entry.target);
      });
    }, { rootMargin: "-40% 0px -45% 0px" });

    projects.forEach(function (p) { io.observe(p); });

    // Leaving the work section entirely resets the panel to its resting state.
    var work = document.getElementById("work");
    if (work) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (!e.isIntersecting) { current = null; clear(); } });
      }, { threshold: 0 }).observe(work);
    }
  }

  /* ------------------------------------------ details: swap the toggle label */

  function initDetails() {
    document.querySelectorAll(".project__detail").forEach(function (d) {
      var label = d.querySelector(".project__toggle-label");
      if (!label) return;
      d.addEventListener("toggle", function () {
        var key = d.open ? "data-open" : "data-closed";
        var text = label.getAttribute(key);
        if (text) label.textContent = text;
      });
    });
  }

  /* ------------------------------------------------------------------ init */

  function init() {
    initReveal();
    initProgress();
    initNav();
    initSkillLink();
    initDetails();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
