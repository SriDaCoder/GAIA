function applyThemeFromStorage() {
  const theme = localStorage.getItem("theme") || "light";
  const accent = localStorage.getItem("accent") || "#0083ff";

  if (theme === "dark") {
    document.documentElement.style.setProperty("--bg", "#121212");
    document.documentElement.style.setProperty("--text", "#ffffff");
    document.documentElement.style.setProperty("--panel-bg", "#1e1e1e");
    document.documentElement.style.setProperty("--panel-text", "#fff");
    document.documentElement.style.setProperty("--menu-bar", "#ccc");
    document.documentElement.style.setProperty("--menu-bar-active", "#fff");
    document.documentElement.style.setProperty("--hr", "#ffffff");
  } else {
    document.documentElement.style.setProperty("--bg", "#f4f4f4");
    document.documentElement.style.setProperty("--text", "#000");
    document.documentElement.style.setProperty("--panel-bg", "#333");
    document.documentElement.style.setProperty("--panel-text", "#fff");
    document.documentElement.style.setProperty("--menu-bar", "#333");
    document.documentElement.style.setProperty("--menu-bar-active", "#fff");
    document.documentElement.style.setProperty("--hr", "#000");
  }

  document.documentElement.style.setProperty("--button-bg", accent);
  document.documentElement.style.setProperty("--button-text", "#fff");
}

applyThemeFromStorage();