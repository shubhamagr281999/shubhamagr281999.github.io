/* Progressive enhancement only. Every project is in the HTML and reachable with
   JavaScript disabled; this file adds tag filtering and scroll reveals. */

(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  /* ---------------------------------------------------- reveal on scroll */

  function initReveal() {
    var items = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
    if (!items.length) return;

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

    // Failsafe: if the observer never reports, show everything rather than
    // leave the page blank.
    window.setTimeout(function () {
      items.forEach(function (el) { el.classList.add("is-in"); });
    }, 1600);
  }

  /* ------------------------------------------------------- tag filtering */

  function initFilters() {
    var buttons = Array.prototype.slice.call(document.querySelectorAll(".filter"));
    var grid = document.getElementById("project-grid");
    if (!buttons.length || !grid) return;

    var cards = Array.prototype.slice.call(grid.querySelectorAll(".card"));
    var count = document.querySelector(".filter-count");

    function apply(tag, announce) {
      var shown = 0;
      cards.forEach(function (card) {
        var tags = (card.getAttribute("data-tags") || "").split(/\s+/);
        var on = tag === "all" || tags.indexOf(tag) !== -1;
        card.hidden = !on;
        if (on) shown++;
      });
      buttons.forEach(function (b) {
        var isOn = b.getAttribute("data-filter") === tag;
        b.classList.toggle("is-on", isOn);
        b.setAttribute("aria-pressed", isOn ? "true" : "false");
      });
      if (count) {
        count.textContent = shown === cards.length
          ? cards.length + " projects"
          : shown + " of " + cards.length + " projects";
      }
      if (announce) {
        try {
          var url = tag === "all" ? location.pathname : location.pathname + "#" + tag;
          history.replaceState(null, "", url);
        } catch (e) { /* file:// or a locked-down browser — filtering still works */ }
      }
    }

    buttons.forEach(function (b) {
      b.setAttribute("aria-pressed", b.classList.contains("is-on") ? "true" : "false");
      b.addEventListener("click", function () {
        apply(b.getAttribute("data-filter"), true);
      });
    });

    // deep link: /projects.html#perception opens pre-filtered
    var initial = (location.hash || "").replace("#", "");
    var known = buttons.some(function (b) { return b.getAttribute("data-filter") === initial; });
    apply(known ? initial : "all", false);
  }

  /* ------------------------------------------------------------------ init */

  function init() {
    initReveal();
    initFilters();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
