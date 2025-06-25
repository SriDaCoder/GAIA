function toggleMenu(x) {
  x.classList.toggle("change");
  const panel = document.getElementById("panel");
  if (panel.style.width === "200px") {
    panel.style.width = "0";
  } else {
    panel.style.width = "200px";
  }
}
