function applySettings() {
  const theme = document.getElementById("theme").value;
  const accent = document.getElementById("accent").value;

  // Save settings
  localStorage.setItem("theme", theme);
  localStorage.setItem("accent", accent);

  // Apply and reload for sync across pages
  applyThemeFromStorage();
  location.reload();
}

// Optional: pre-fill current settings in the form
document.addEventListener("DOMContentLoaded", () => {
  const savedTheme = localStorage.getItem("theme") || "light";
  const savedAccent = localStorage.getItem("accent") || "#3498db";

  document.getElementById("theme").value = savedTheme;
  document.getElementById("accent").value = savedAccent;
});