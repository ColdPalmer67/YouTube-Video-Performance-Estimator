document.addEventListener("DOMContentLoaded", function () {
  setupNavToggle();
  setActiveNavLink();
  animateResultMeter();
  setupFormFeedback();
  setupCharCounter();
  setupLightbox();
  setupBackToTop();
});

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

// Animate the confidence meter using the prediction value from the page
function animateResultMeter() {
  var fill = document.querySelector(".meter__fill");
  if (!fill) return;

  var confidence = parseFloat(fill.getAttribute("data-confidence")) || 0;

  // Delay the update so the progress animation is visible
  window.requestAnimationFrame(function () {
    setTimeout(function () {
      fill.style.width = confidence + "%";
    }, 100);
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

// Show button after scrolling and return to top when clicked
function setupBackToTop() {
  var button = document.querySelector("#back-to-top");
  if (!button) return;

  window.addEventListener("scroll", function () {
    if (window.scrollY > 400) {
      button.classList.add("visible");
    } else {
      button.classList.remove("visible");
    }
  });

  button.addEventListener("click", function () {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}
