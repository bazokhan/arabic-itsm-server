const NAV_ITEMS = [
  { href: "/models", key: "models", label: "Models", icon: "layout-grid" },
  { href: "/research", key: "research", label: "Research", icon: "flask-conical" },
  { href: "/dashboard", key: "dashboard", label: "Dashboard", icon: "bar-chart-3" },
  { href: "/monitoring", key: "monitoring", label: "Monitoring", icon: "activity" },
  { href: "/admin/export", key: "export", label: "Export", icon: "download" },
];

const PALETTES = [
  { value: "default", label: "Default" },
  { value: "teal", label: "Teal" },
  { value: "violet", label: "Violet" },
  { value: "amber", label: "Amber" },
];

const CHART_STYLES = [
  { value: "rounded", label: "Rounded" },
  { value: "sharp", label: "Sharp" },
  { value: "soft", label: "Soft" },
];

function currentPageKey(pathname) {
  if (pathname === "/" || pathname.startsWith("/models")) return "models";
  if (pathname.startsWith("/research")) return "research";
  if (pathname.startsWith("/dashboard")) return "dashboard";
  if (pathname.startsWith("/monitoring")) return "monitoring";
  if (pathname.startsWith("/admin/export")) return "export";
  return "";
}

function buildSelect(id, options, selectedValue) {
  const el = document.createElement("select");
  el.id = id;
  el.className = "shell-select";
  options.forEach((opt) => {
    const optionEl = document.createElement("option");
    optionEl.value = opt.value;
    optionEl.textContent = opt.label;
    if (opt.value === selectedValue) optionEl.selected = true;
    el.appendChild(optionEl);
  });
  return el;
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("theme", theme);
}

function applyPalette(palette) {
  document.documentElement.setAttribute("data-palette", palette);
  localStorage.setItem("palette", palette);
}

function applyChartStyle(style) {
  document.documentElement.setAttribute("data-chart-style", style);
  localStorage.setItem("chartStyle", style);
}

export function initAppShell() {
  const pageRoot = document.querySelector(".page-wide") || document.querySelector(".page");
  if (!pageRoot || document.querySelector(".app-shell")) return;

  const savedTheme = localStorage.getItem("theme") || "light";
  const savedPalette = localStorage.getItem("palette") || "default";
  const savedChartStyle = localStorage.getItem("chartStyle") || "rounded";
  const collapsed = localStorage.getItem("sidebarCollapsed") === "1";

  applyTheme(savedTheme);
  applyPalette(savedPalette);
  applyChartStyle(savedChartStyle);

  const shell = document.createElement("div");
  shell.className = "app-shell";
  if (collapsed) shell.classList.add("collapsed");

  const sidebar = document.createElement("aside");
  sidebar.className = "app-sidebar";
  const pageKey = currentPageKey(window.location.pathname);

  const nav = document.createElement("nav");
  nav.className = "app-nav";
  NAV_ITEMS.forEach((item) => {
    const a = document.createElement("a");
    a.href = item.href;
    a.className = "app-nav-item";
    if (item.key === pageKey) a.classList.add("active");
    a.innerHTML = `<i data-lucide="${item.icon}" style="width:16px;height:16px;"></i><span>${item.label}</span>`;
    a.addEventListener("click", () => {
      if (window.innerWidth <= 900) shell.classList.remove("mobile-open");
    });
    nav.appendChild(a);
  });

  const controls = document.createElement("div");
  controls.className = "shell-controls";

  const collapseBtn = document.createElement("button");
  collapseBtn.className = "btn shell-collapse-btn";
  collapseBtn.type = "button";
  collapseBtn.innerHTML = `<i data-lucide="panel-left-close" style="width:14px;height:14px;"></i><span>Collapse</span>`;
  collapseBtn.addEventListener("click", () => {
    shell.classList.toggle("collapsed");
    localStorage.setItem("sidebarCollapsed", shell.classList.contains("collapsed") ? "1" : "0");
  });

  const themeBtn = document.createElement("button");
  themeBtn.className = "btn shell-theme-btn";
  themeBtn.type = "button";
  themeBtn.innerHTML = `<i data-lucide="moon-star" style="width:14px;height:14px;"></i><span>Theme</span>`;
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    document.querySelector(".icon-dark")?.classList.toggle("hidden", next === "dark");
    document.querySelector(".icon-light")?.classList.toggle("hidden", next === "light");
  });

  const paletteLabel = document.createElement("label");
  paletteLabel.className = "shell-label";
  paletteLabel.textContent = "Palette";
  const paletteSelect = buildSelect("shellPalette", PALETTES, savedPalette);
  paletteSelect.addEventListener("change", () => applyPalette(paletteSelect.value));

  const chartLabel = document.createElement("label");
  chartLabel.className = "shell-label";
  chartLabel.textContent = "Chart Style";
  const chartSelect = buildSelect("shellChartStyle", CHART_STYLES, savedChartStyle);
  chartSelect.addEventListener("change", () => applyChartStyle(chartSelect.value));

  controls.appendChild(collapseBtn);
  controls.appendChild(themeBtn);
  controls.appendChild(paletteLabel);
  controls.appendChild(paletteSelect);
  controls.appendChild(chartLabel);
  controls.appendChild(chartSelect);

  sidebar.innerHTML = `
    <div class="app-sidebar-head">
      <div class="app-sidebar-title">Arabic ITSM</div>
      <div class="app-sidebar-sub">Data Console</div>
    </div>
  `;
  sidebar.appendChild(nav);
  sidebar.appendChild(controls);

  const main = document.createElement("main");
  main.className = "app-main";
  main.appendChild(pageRoot);

  const mobileToggle = document.createElement("button");
  mobileToggle.className = "btn shell-mobile-toggle";
  mobileToggle.type = "button";
  mobileToggle.setAttribute("aria-label", "Toggle navigation");
  mobileToggle.innerHTML = `<i data-lucide="menu" style="width:15px;height:15px;"></i>`;
  mobileToggle.addEventListener("click", () => {
    shell.classList.toggle("mobile-open");
  });

  shell.appendChild(sidebar);
  shell.appendChild(main);
  document.body.prepend(shell);
  document.body.appendChild(mobileToggle);

  document.addEventListener("click", (event) => {
    if (window.innerWidth > 900) return;
    if (!shell.classList.contains("mobile-open")) return;
    const target = event.target;
    if (!(target instanceof Element)) return;
    const inSidebar = target.closest(".app-sidebar");
    const inToggle = target.closest(".shell-mobile-toggle");
    if (!inSidebar && !inToggle) shell.classList.remove("mobile-open");
  });

  if (window.lucide && typeof window.lucide.createIcons === "function") {
    window.lucide.createIcons();
  }
}
