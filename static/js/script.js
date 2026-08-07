document.addEventListener("DOMContentLoaded", function () {
  setupThemeToggle();
  setupNavToggle();
  setActiveNavLink();
  setupNavScrollState();
  animateResultMeter();
  setupFormFeedback();
  setupCharCounter();
  setupLightbox();
  setupBackToTop();
  setupInsightsScrollSpy();
});

// Highlight the current section in the Project Insights rail nav as the
// reader scrolls, so the rail always reflects where they are on the page.
function setupInsightsScrollSpy() {
  var navLinks = document.querySelectorAll(".insights-nav a");
  if (!navLinks.length) return;

  var sections = Array.prototype.map.call(navLinks, function (link) {
    var id = link.getAttribute("href").slice(1);
    return document.getElementById(id);
  }).filter(Boolean);

  if (!sections.length) return;

  function updateActive() {
    var scrollPos = window.scrollY + window.innerHeight * 0.25;
    var current = sections[0];

    sections.forEach(function (section) {
      if (section.offsetTop <= scrollPos) {
        current = section;
      }
    });

    navLinks.forEach(function (link) {
      var isActive = link.getAttribute("href") === "#" + current.id;
      link.classList.toggle("active", isActive);
    });
  }

  window.addEventListener("scroll", updateActive, { passive: true });
  window.addEventListener("resize", updateActive);
  updateActive();
}

/* Light / dark theme toggle, persisted across visits */
function setupThemeToggle() {
  var root = document.documentElement;
  var button = document.querySelector("#theme-toggle");

  var stored = null;
  try {
    stored = localStorage.getItem("theme");
  } catch (e) {
    stored = null;
  }

  var prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  var initial = stored || (prefersLight ? "light" : "dark");

  applyTheme(initial);

  if (!button) return;

  button.addEventListener("click", function () {
    var current = root.getAttribute("data-theme") === "light" ? "light" : "dark";
    var next = current === "light" ? "dark" : "light";
    applyTheme(next);
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      /* storage unavailable, theme still applies for this session */
    }
  });

  function applyTheme(theme) {
    if (theme === "light") {
      root.setAttribute("data-theme", "light");
    } else {
      root.removeAttribute("data-theme");
    }
    if (button) {
      button.textContent = theme === "light" ? "\u2600" : "\u263D";
      button.setAttribute("aria-label", theme === "light" ? "Switch to dark theme" : "Switch to light theme");
    }
  }
}

// Give the floating navbar a glass background once the page scrolls
function setupNavScrollState() {
  var nav = document.querySelector(".navbar");
  if (!nav) return;

  function update() {
    if (window.scrollY > 24) {
      nav.classList.add("is-scrolled");
    } else {
      nav.classList.remove("is-scrolled");
    }
  }

  window.addEventListener("scroll", update, { passive: true });
  update();
}

/* Mobile navbar toggle. */
function setupNavToggle() {
  var toggle = document.querySelector(".navbar__toggle");
  var links = document.querySelector(".navbar__links");

  if (!toggle || !links) return;

  toggle.addEventListener("click", function () {
    links.classList.toggle("open");
  });
}

// Highlight the active navigation link
function setActiveNavLink() {
  var currentPath = window.location.pathname;
  var links = document.querySelectorAll(".navbar__links a");

  links.forEach(function (link) {
    var linkPath = link.getAttribute("href");
    if (linkPath === currentPath) {
      link.classList.add("active");
    }
  });
}

// Animate confidence meters using the prediction value from the page.
// Only targets meters with a data-confidence attribute, since some meters
// (e.g. similarity scores on placeholder cards) set their width directly.
function animateResultMeter() {
  var fills = document.querySelectorAll(".meter__fill[data-confidence]");
  if (!fills.length) return;

  fills.forEach(function (fill) {
    var confidence = parseFloat(fill.getAttribute("data-confidence")) || 0;

    // Delay the update so the progress animation is visible
    window.requestAnimationFrame(function () {
      setTimeout(function () {
        fill.style.width = confidence + "%";
      }, 100);
    });
  });
}

// Update description character count while typing
function setupCharCounter() {
  var textarea = document.querySelector("#description");
  var counter = document.querySelector("#description-counter");

  if (!textarea || !counter) return;

  var max = parseInt(textarea.getAttribute("maxlength"), 10) || 500;

  function updateCounter() {
    var length = textarea.value.length;
    counter.textContent = length + " / " + max;
    counter.classList.toggle("char-counter--warn", length > max * 0.9);
  }

  textarea.addEventListener("input", updateCounter);
  updateCounter();
}

// Basic form validation before submitting prediction
function setupFormFeedback() {
  var form = document.querySelector("#predict-form");
  if (!form) return;

  form.addEventListener("submit", function (event) {
    var title = form.querySelector("#title");
    var numericFields = [
      form.querySelector("#subscriber_count"),
      form.querySelector("#duration_seconds"),
      form.querySelector("#category_id")
    ];

    var hasError = false;

    if (title && title.value.trim() === "") {
      title.style.borderColor = "var(--signal-low)";
      hasError = true;
    } else if (title) {
      title.style.borderColor = "";
    }

    numericFields.forEach(function (field) {
      if (!field) return;

      if (field.value === "" || isNaN(field.value) || Number(field.value) < 0) {
        field.style.borderColor = "var(--signal-low)";
        hasError = true;
      } else {
        field.style.borderColor = "";
      }
    });

    if (hasError) {
      event.preventDefault();
    }
  });
}

// Open chart images in an enlarged view
function setupLightbox() {
  var triggers = document.querySelectorAll(".lightbox-trigger");
  var lightbox = document.querySelector("#lightbox");

  if (!triggers.length || !lightbox) return;

  var lightboxImage = document.querySelector("#lightbox-image");
  var lightboxCaption = document.querySelector("#lightbox-caption");
  var closeButton = document.querySelector("#lightbox-close");

  function openLightbox(imgEl) {
    lightboxImage.src = imgEl.getAttribute("src");
    lightboxImage.alt = imgEl.getAttribute("alt") || "";

    var figcaption = imgEl.closest("figure");
    lightboxCaption.textContent = figcaption
      ? figcaption.querySelector("figcaption").textContent.trim()
      : "";

    lightbox.classList.add("open");
    lightbox.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function closeLightbox() {
    lightbox.classList.remove("open");
    lightbox.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }

  triggers.forEach(function (img) {
    img.addEventListener("click", function () {
      openLightbox(img);
    });
  });

  closeButton.addEventListener("click", closeLightbox);

  // Close when clicking outside the image
  lightbox.addEventListener("click", function (event) {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  // Close with ESC key
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && lightbox.classList.contains("open")) {
      closeLightbox();
    }
  });
}

// Show button after scrolling and return to top when clicked.
// Threshold is set relative to viewport height so it appears reliably
// on both short and long pages, rather than a fixed pixel value that
// never triggers on shorter pages.
function setupBackToTop() {
  var button = document.querySelector("#back-to-top");
  if (!button) return;

  var threshold = Math.max(280, window.innerHeight * 0.6);

  function toggleVisibility() {
    if (window.scrollY > threshold) {
      button.classList.add("visible");
    } else {
      button.classList.remove("visible");
    }
  }

  window.addEventListener("scroll", toggleVisibility, { passive: true });
  window.addEventListener("resize", function () {
    threshold = Math.max(280, window.innerHeight * 0.6);
    toggleVisibility();
  });

  toggleVisibility();

  button.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}
