---
hide:
  - toc
  - path
  - title
---

<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="preconnect" href="https://cdnjs.cloudflare.com" crossorigin>
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<script src="/assets/js/vue.global.prod.js"></script>
<script>
  define = undefined;
</script>
<script src="/assets/js/primevue.min.js"></script>
<script src="/assets/js/aura.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/primeicons@latest/primeicons.css"/>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=Fira+Code:wght@400&family=Electrolize&display=swap" rel="stylesheet">

<style>
  /* ===== MkDocs full-width overrides ===== */
  .md-main__inner { max-width: 100% !important; margin: 0 !important; }
  .md-content { max-width: 100% !important; margin: 0 !important; }
  .md-content__inner { max-width: 100% !important; padding: 0 !important; margin: 0 !important; }
  @media (min-width: 76.25em) { .md-sidebar { display: none !important; } }
  .md-grid { max-width: 100% !important; margin: 0 !important; }
  .md-content h1, .md-content__inner > h1:first-child, .md-typeset h1 { display: none !important; }

  /* ===== Design tokens ===== */
  #status-app {
    --nn-font-display: 'Syne', sans-serif;
    --nn-font-body: 'DM Sans', sans-serif;
    --nn-font-mono: 'Fira Code', monospace;
    --nn-font-card: 'Electrolize', sans-serif;
    --nn-hot: #10b981;
    --nn-warm: #f59e0b;
    --nn-cold: #6366f1;
    --nn-danger: #ef4444;
    --nn-info: #3b82f6;
    --nn-muted: #9ca3af;
    --nn-radius: 8px;
    --nn-radius-sm: 4px;
    --nn-ease: 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    font-family: var(--nn-font-body);

    --p-inputtext-background: var(--md-default-bg-color) !important;
    --p-select-background: var(--md-default-bg-color) !important;
    --p-floatlabel-on-active-background: var(--md-default-bg-color) !important;
    --p-select-overlay-background: var(--md-default-bg-color) !important;
    --p-select-color: var(--md-default-fg-color) !important;
    --p-inputtext-color: var(--md-default-fg-color) !important;
    --p-select-overlay-color: var(--md-default-fg-color) !important;
  }

  /* Light mode */
  [data-md-color-scheme="default"] #status-app {
    --nn-card-bg: #ffffff;
    --nn-card-border: rgba(0, 0, 0, 0.08);
    --nn-card-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    --nn-card-shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.1);
    --nn-chip-bg: rgba(0, 0, 0, 0.04);
    --nn-chip-bg-hover: rgba(0, 0, 0, 0.07);
    --nn-sidebar-bg: #ffffff;
    --nn-badge-text: #ffffff;
  }

  /* Dark mode */
  [data-md-color-scheme="slate"] #status-app {
    --nn-card-bg: #181818;
    --nn-card-border: rgba(255, 255, 255, 0.06);
    --nn-card-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    --nn-card-shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.4);
    --nn-chip-bg: rgba(255, 255, 255, 0.06);
    --nn-chip-bg-hover: rgba(255, 255, 255, 0.1);
    --nn-sidebar-bg: #181818;
    --nn-badge-text: #000000;
  }

  /* ===== Layout ===== */
  #status-app {
    display: flex;
    flex-direction: column;
    width: 100%;
    min-height: 80vh;
  }

  #status-main {
    display: flex;
    flex-direction: row;
    width: 100%;
    padding: 1.5rem 3rem;
    gap: 2rem;
  }

  /* ===== Loading Banner ===== */
  #status-banner {
    padding: 0.75rem 2rem;
    text-align: center;
    font-family: var(--nn-font-display);
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1;
    letter-spacing: 0.02em;
    color: white;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    max-height: 60px;
    overflow: hidden;
    transition: max-height 0.5s ease, opacity 0.4s ease, padding 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  #status-banner.nn-error {
    background: linear-gradient(135deg, #ef4444, #dc2626);
  }

  #status-banner.nn-collapsed {
    max-height: 0;
    opacity: 0;
    padding: 0;
  }

  .nn-pulse {
    display: inline-block;
    animation: nn-pulse 1.5s ease-in-out infinite;
  }

  @keyframes nn-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* ===== Sidebar ===== */
  #status-sidebar {
    width: 300px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    position: sticky;
    top: 80px;
    align-self: flex-start;
  }

  .nn-panel {
    background: var(--nn-sidebar-bg);
    border: 1px solid var(--nn-card-border);
    border-radius: var(--nn-radius);
    box-shadow: var(--nn-card-shadow);
    overflow: hidden;
  }

  .nn-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.25rem;
    cursor: pointer;
    user-select: none;
    transition: background var(--nn-ease);
  }

  .nn-panel-header:hover {
    background: var(--nn-chip-bg);
  }

  .nn-panel-title {
    font-family: var(--nn-font-display);
    font-weight: 700;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--nn-muted);
    margin: 0;
  }

  .nn-panel-chevron {
    font-size: 0.75rem;
    color: var(--nn-muted);
    transition: transform 0.25s ease;
  }

  .nn-panel-chevron.collapsed {
    transform: rotate(-90deg);
  }

  .nn-panel-body {
    padding: 0 1.25rem 1.25rem;
    transition: max-height 0.3s ease, padding 0.3s ease, opacity 0.2s ease;
    max-height: 800px;
    opacity: 1;
    overflow: hidden;
  }

  .nn-panel-body.collapsed {
    max-height: 0;
    padding-top: 0;
    padding-bottom: 0;
    opacity: 0;
  }

  .nn-sidebar-links {
    display: flex;
    flex-direction: row;
    gap: 0.5rem;
    justify-content: center;
  }

  .nn-sidebar-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: var(--nn-radius);
    text-decoration: none;
    color: var(--md-default-fg-color);
    transition: background var(--nn-ease), color var(--nn-ease);
    position: relative;
  }

  .nn-sidebar-link:hover {
    background: var(--nn-chip-bg-hover);
    color: var(--nn-cold);
  }

  .nn-sidebar-link i {
    font-size: 1.4rem;
  }

  /* ===== Content ===== */
  #status-content {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  /* ===== Filter bar ===== */
  #filter-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .nn-filter-left {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    flex-shrink: 0;
  }

  .nn-filter-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
  }

  .nn-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid transparent;
    background: var(--nn-chip-bg);
    color: var(--md-default-fg-color);
    transition: all var(--nn-ease);
    user-select: none;
  }

  .nn-chip:hover { background: var(--nn-chip-bg-hover); }

  .nn-chip.active { border-color: var(--md-default-fg-color); }
  .nn-chip.active.hot { color: var(--nn-hot); background: rgba(16, 185, 129, 0.1); border-color: var(--nn-hot); }
  .nn-chip.active.warm { color: var(--nn-warm); background: rgba(245, 158, 11, 0.1); border-color: var(--nn-warm); }
  .nn-chip.active.cold { color: var(--nn-cold); background: rgba(99, 102, 241, 0.1); border-color: var(--nn-cold); }

  .nn-chip-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .nn-chip-dot.hot { background: var(--nn-hot); }
  .nn-chip-dot.warm { background: var(--nn-warm); }
  .nn-chip-dot.cold { background: var(--nn-cold); }

  .nn-chip-count {
    font-family: var(--nn-font-mono);
    font-size: 0.7rem;
    opacity: 0.7;
  }

  /* PrimeVue overrides */
  .p-paginator {
    background: transparent !important;
    gap: 0.25rem;
    padding: 0.4rem 0 !important;
    font-family: var(--nn-font-body);
  }

  .p-paginator .p-paginator-prev,
  .p-paginator .p-paginator-next,
  .p-paginator .p-paginator-first,
  .p-paginator .p-paginator-last {
    min-width: 2rem !important;
    height: 2rem !important;
    border-radius: var(--nn-radius-sm) !important;
    background: var(--nn-chip-bg) !important;
    color: var(--md-default-fg-color) !important;
    border: 1px solid var(--nn-card-border) !important;
    transition: all var(--nn-ease) !important;
  }

  .p-paginator .p-paginator-prev:hover,
  .p-paginator .p-paginator-next:hover,
  .p-paginator .p-paginator-first:hover,
  .p-paginator .p-paginator-last:hover {
    background: var(--nn-chip-bg-hover) !important;
  }

  .p-paginator .p-paginator-page {
    min-width: 2rem !important;
    height: 2rem !important;
    border-radius: var(--nn-radius-sm) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--md-default-fg-color) !important;
    border: 1px solid transparent !important;
    transition: all var(--nn-ease) !important;
  }

  .p-paginator .p-paginator-page:hover {
    background: var(--nn-chip-bg) !important;
  }

  .p-paginator .p-paginator-page.p-highlight {
    background: var(--nn-cold) !important;
    color: #fff !important;
    border-color: var(--nn-cold) !important;
  }

  .p-paginator .p-paginator-current {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--nn-muted) !important;
    font-family: var(--nn-font-mono) !important;
    padding: 0 0.5rem !important;
  }

  .p-select {
    border-radius: var(--nn-radius-sm) !important;
    border: 1px solid var(--nn-card-border) !important;
    background: var(--nn-card-bg) !important;
    transition: border-color var(--nn-ease) !important;
  }

  .p-select:hover {
    border-color: var(--nn-muted) !important;
  }

  .p-select.p-focus {
    border-color: var(--nn-cold) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
  }

  .p-select-label {
    padding: 0.35rem 0.6rem !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
  }

  /* Select overlay is teleported to body - must use root-level color scheme selectors */
  [data-md-color-scheme="default"] .p-select-overlay {
    background: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
  }

  [data-md-color-scheme="slate"] .p-select-overlay {
    background: #181818 !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
  }

  .p-select-overlay {
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15) !important;
    overflow: hidden !important;
  }

  [data-md-color-scheme="default"] .p-select-option {
    color: #1a1a1a !important;
  }

  [data-md-color-scheme="default"] .p-select-option:hover,
  [data-md-color-scheme="default"] .p-select-option.p-highlight {
    background: rgba(0, 0, 0, 0.05) !important;
  }

  [data-md-color-scheme="slate"] .p-select-option {
    color: #e0e0e0 !important;
  }

  [data-md-color-scheme="slate"] .p-select-option:hover,
  [data-md-color-scheme="slate"] .p-select-option.p-highlight {
    background: rgba(255, 255, 255, 0.08) !important;
  }

  .p-select-option {
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.5rem 0.75rem !important;
  }

  .p-inputtext {
    padding: 0.35rem 0.6rem !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: var(--nn-radius-sm) !important;
    border: 1px solid var(--nn-card-border) !important;
    background: var(--nn-card-bg) !important;
    transition: border-color var(--nn-ease) !important;
  }

  .p-inputtext:hover { border-color: var(--nn-muted) !important; }
  .p-inputtext:focus {
    border-color: var(--nn-cold) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
  }

  .p-iconfield {
    display: flex !important;
    align-items: center !important;
    position: relative !important;
  }

  .p-iconfield .p-inputtext {
    padding-left: 2rem !important;
  }

  .p-iconfield .p-inputicon {
    position: absolute;
    left: 0.6rem;
    z-index: 1;
    font-size: 0.82rem;
    color: var(--nn-muted);
  }

  .p-floatlabel label {
    font-size: 0.82rem !important;
    font-family: var(--nn-font-body) !important;
  }

  /* ===== Cards grid ===== */
  #deployments {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 1rem;
  }

  /* ===== Card ===== */
  .nn-deployment {
    background: var(--nn-card-bg);
    border: 1px solid var(--nn-card-border);
    border-left: 3px solid var(--nn-muted);
    border-radius: var(--nn-radius);
    box-shadow: var(--nn-card-shadow);
    transition: transform var(--nn-ease), box-shadow var(--nn-ease);
    position: relative;
    overflow: visible;
    display: flex;
    flex-direction: column;
    animation: nn-fadeIn 0.3s ease both;
  }

  .nn-deployment.level-hot { border-left-color: var(--nn-hot); }
  .nn-deployment.level-warm { border-left-color: var(--nn-warm); }
  .nn-deployment.level-cold { border-left-color: var(--nn-cold); }

  .nn-deployment:hover {
    transform: translateY(-2px);
    box-shadow: var(--nn-card-shadow-hover);
  }

  [data-md-color-scheme="slate"] .nn-deployment.level-hot:hover {
    box-shadow: var(--nn-card-shadow-hover), 0 0 20px rgba(16, 185, 129, 0.06);
  }
  [data-md-color-scheme="slate"] .nn-deployment.level-warm:hover {
    box-shadow: var(--nn-card-shadow-hover), 0 0 20px rgba(245, 158, 11, 0.06);
  }
  [data-md-color-scheme="slate"] .nn-deployment.level-cold:hover {
    box-shadow: var(--nn-card-shadow-hover), 0 0 20px rgba(99, 102, 241, 0.06);
  }

  @keyframes nn-fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .nn-card-body {
    padding: 1rem 1.15rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    flex: 1;
  }

  .nn-repo-id {
    font-family: var(--nn-font-card);
    font-size: 0.88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 2rem;
  }

  .nn-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }

  .nn-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 7px;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.6;
  }

  .nn-badge-success { background: var(--nn-hot); color: var(--nn-badge-text); }
  .nn-badge-warning { background: var(--nn-warm); color: var(--nn-badge-text); }
  .nn-badge-danger { background: var(--nn-danger); color: var(--nn-badge-text); }
  .nn-badge-info { background: var(--nn-info); color: var(--nn-badge-text); }
  .nn-badge-primary { background: var(--nn-cold); color: var(--nn-badge-text); }
  .nn-badge-secondary { background: #6b7280; color: var(--nn-badge-text); }
  .nn-badge-muted { background: var(--nn-muted); color: var(--nn-badge-text); }

  .nn-badge-outline {
    background: transparent;
    border: 1px solid var(--md-default-fg-color--light);
    color: var(--md-default-fg-color--light);
  }

  .nn-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.7rem;
    color: var(--md-default-fg-color--light);
    padding-top: 2px;
  }

  .nn-meta-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  /* ===== Copy button ===== */
  .nn-copy-btn {
    position: absolute;
    top: 0.6rem;
    right: 0.6rem;
    z-index: 2;
  }

  .nn-copy-btn button {
    background: var(--nn-chip-bg);
    border: 1px solid var(--nn-card-border);
    border-radius: var(--nn-radius-sm);
    cursor: pointer;
    padding: 4px 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background var(--nn-ease);
    z-index: 3;
    position: relative;
  }

  .nn-copy-btn button:hover { background: var(--nn-chip-bg-hover); }

  .nn-copy-btn button svg {
    width: 16px;
    height: 16px;
    stroke: var(--md-default-fg-color);
    fill: none;
  }

  /* ===== Code snippet popover (teleported to body) ===== */
  .p-popover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }

  .p-popover .p-popover-content {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
  }

  .p-popover::before,
  .p-popover::after {
    display: none !important;
  }

  .nn-snippet {
    border-radius: 8px;
    padding: 1rem;
    overflow-x: auto;
    width: max-content;
  }

  .nn-snippet pre {
    margin: 0;
    font-family: 'Fira Code', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
  }

  /* Snippet theming - light mode */
  [data-md-color-scheme="default"] .nn-snippet {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.08);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  }

  [data-md-color-scheme="default"] .nn-snippet pre { color: #4c4f69; }
  [data-md-color-scheme="default"] .nn-snippet .kn { color: #8839ef; }
  [data-md-color-scheme="default"] .nn-snippet .nn { color: #1e66f5; }
  [data-md-color-scheme="default"] .nn-snippet .n { color: #4c4f69; }
  [data-md-color-scheme="default"] .nn-snippet .o { color: #04a5e5; }
  [data-md-color-scheme="default"] .nn-snippet .p { color: #4c4f69; }
  [data-md-color-scheme="default"] .nn-snippet .s1,
  [data-md-color-scheme="default"] .nn-snippet .s2 { color: #40a02b; }
  [data-md-color-scheme="default"] .nn-snippet .k { color: #8839ef; }

  /* Snippet theming - dark mode */
  [data-md-color-scheme="slate"] .nn-snippet {
    background: #181818;
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }

  [data-md-color-scheme="slate"] .nn-snippet pre { color: #cdd6f4; }
  [data-md-color-scheme="slate"] .nn-snippet .kn { color: #c792ea; }
  [data-md-color-scheme="slate"] .nn-snippet .nn { color: #82aaff; }
  [data-md-color-scheme="slate"] .nn-snippet .n { color: #cdd6f4; }
  [data-md-color-scheme="slate"] .nn-snippet .o { color: #89ddff; }
  [data-md-color-scheme="slate"] .nn-snippet .p { color: #cdd6f4; }
  [data-md-color-scheme="slate"] .nn-snippet .s1,
  [data-md-color-scheme="slate"] .nn-snippet .s2 { color: #c3e88d; }
  [data-md-color-scheme="slate"] .nn-snippet .k { color: #c792ea; }

  /* ===== Stretched link ===== */
  .nn-stretched-link::after {
    position: absolute;
    top: 0; right: 0; bottom: 0; left: 0;
    z-index: 1;
    content: "";
  }

  .nn-stretched-link { text-decoration: none; }

  .nn-stretched-link span {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  /* ===== Tooltips ===== */
  [data-nn-tooltip] {
    position: relative;
    cursor: help;
    z-index: 3;
  }

  [data-nn-tooltip].cursor-pointer { cursor: pointer; }

  [data-nn-tooltip]:hover::after {
    content: attr(data-nn-tooltip);
    z-index: 6;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    padding: 6px 10px;
    background-color: var(--nn-card-bg);
    color: var(--md-default-fg-color);
    border: 1px solid var(--nn-card-border);
    border-radius: var(--nn-radius-sm);
    font-size: 0.72rem;
    white-space: normal;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-family: var(--nn-font-body);
    font-weight: normal;
    width: max-content;
    max-width: 260px;
    text-align: center;
    pointer-events: none;
  }

  /* ===== Skeleton cards ===== */
  .nn-skeleton-card {
    height: 90px;
    border-radius: var(--nn-radius);
    overflow: hidden;
  }

  /* ===== No results ===== */
  .nn-empty {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--nn-muted);
    width: 100%;
  }

  .nn-empty i {
    font-size: 2rem;
    margin-bottom: 0.75rem;
    display: block;
  }

  /* ===== Cluster panel ===== */
  .nn-cluster-summary {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  .nn-cluster-metric {
    flex: 1;
    text-align: center;
    padding: 0.6rem 0.4rem;
    background: var(--nn-chip-bg);
    border-radius: var(--nn-radius-sm);
  }

  .nn-cluster-metric-value {
    font-family: var(--nn-font-mono);
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.25rem;
  }

  .nn-cluster-metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--nn-muted);
  }

  .nn-cluster-bar {
    width: 100%;
    height: 8px;
    background: var(--nn-chip-bg);
    border-radius: 100px;
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .nn-cluster-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: var(--nn-hot);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .nn-cluster-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--nn-muted);
    margin-bottom: 0.3rem;
    font-weight: 500;
  }

  .nn-node-list {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }

  .nn-node {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }

  .nn-node-id {
    font-family: var(--nn-font-mono);
    font-size: 0.65rem;
    color: var(--nn-muted);
    width: 3.5rem;
    flex-shrink: 0;
    text-align: right;
  }

  .nn-node-gpus {
    display: flex;
    gap: 3px;
    flex: 1;
  }

  .nn-gpu-block {
    flex: 1;
    height: 18px;
    border-radius: 3px;
    background: var(--nn-chip-bg);
    transition: background var(--nn-ease);
  }

  .nn-gpu-block.used {
    background: var(--nn-cold);
  }

  .nn-gpu-block.used:hover {
    opacity: 0.8;
  }

  .nn-node-vram {
    font-family: var(--nn-font-mono);
    font-size: 0.65rem;
    color: var(--nn-muted);
    width: 3rem;
    flex-shrink: 0;
    text-align: right;
  }

  /* ===== Responsive ===== */
  @media (max-width: 768px) {
    #status-main {
      flex-direction: column;
      padding: 0.75rem;
      gap: 1rem;
    }

    #status-sidebar {
      width: 100%;
      position: static;
      flex-direction: row;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .nn-panel { flex: 1; min-width: 200px; }
    .nn-cluster-section { display: none !important; }

    #filter-bar {
      flex-direction: column;
      align-items: stretch;
      gap: 0.75rem;
    }

    .nn-filter-left, .nn-filter-right {
      justify-content: center;
      flex-wrap: wrap;
    }

    #deployments {
      grid-template-columns: 1fr;
    }
  }

  .desktop-only { display: block; }
  @media (max-width: 768px) { .desktop-only { display: none !important; } }
</style>

<div id="status-app">
  <div id="status-banner"
       :class="{ 'nn-collapsed': status === 'success', 'nn-error': status === 'error' }">
    <span v-if="status === 'loading'" class="nn-pulse">
      <i class="pi pi-spin pi-spinner"></i>&nbsp; Connecting to NDIF...
    </span>
    <span v-else-if="status === 'error'">
      <i class="pi pi-exclamation-triangle"></i>&nbsp; Unable to reach NDIF. Please try again later.
    </span>
  </div>

  <div id="status-main">
    <div id="status-sidebar">
      <div class="nn-panel">
        <div class="nn-panel-header" @click="panels.resources = !panels.resources">
          <div class="nn-panel-title">Resources</div>
          <i class="pi pi-chevron-down nn-panel-chevron" :class="{ collapsed: !panels.resources }"></i>
        </div>
        <div class="nn-panel-body" :class="{ collapsed: !panels.resources }">
          <div class="nn-sidebar-links">
            <a v-if="calendar_id"
               :href="'https://calendar.google.com/calendar/embed?src=' + encodeURIComponent(calendar_id)"
               target="_blank"
               class="nn-sidebar-link"
               title="Deployment Calendar">
              <i class="pi pi-calendar"></i>
            </a>
            <a href="/features/13_remote_execution/" target="_blank"
               class="nn-sidebar-link"
               title="How do I use NDIF?">
              <i class="pi pi-question-circle"></i>
            </a>
            <a href="https://login.ndif.us/" target="_blank"
               class="nn-sidebar-link"
               title="Login to NDIF">
              <i class="pi pi-sign-in"></i>
            </a>
          </div>
        </div>
      </div>

      <div v-if="clusterInfo" class="nn-panel nn-cluster-section">
        <div class="nn-panel-header" @click="panels.cluster = !panels.cluster">
          <div class="nn-panel-title">Cluster</div>
          <i class="pi pi-chevron-down nn-panel-chevron" :class="{ collapsed: !panels.cluster }"></i>
        </div>
        <div class="nn-panel-body" :class="{ collapsed: !panels.cluster }">
          <div class="nn-cluster-summary">
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value">{{ clusterInfo.totalGpus }}</div>
              <div class="nn-cluster-metric-label">GPUs</div>
            </div>
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value">{{ clusterInfo.usedGpus }}</div>
              <div class="nn-cluster-metric-label">In Use</div>
            </div>
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value">{{ clusterInfo.totalVram }}</div>
              <div class="nn-cluster-metric-label">VRAM</div>
            </div>
          </div>
          <div class="nn-cluster-bar-label">
            <span>GPU Utilization</span>
            <span>{{ clusterInfo.utilizationPct }}%</span>
          </div>
          <div class="nn-cluster-bar">
            <div class="nn-cluster-bar-fill" :style="{ width: clusterInfo.utilizationPct + '%' }"></div>
          </div>
          <div class="nn-panel-title" style="margin-bottom: 0.6rem;">Nodes</div>
          <div class="nn-node-list">
            <div v-for="node in clusterInfo.nodes" :key="node.id" class="nn-node">
              <span class="nn-node-id">{{ node.id }}</span>
              <div class="nn-node-gpus">
                <template v-if="node.gpuDetails">
                  <div v-for="g in node.gpuDetails" :key="g.index"
                       class="nn-gpu-block"
                       :class="{ used: g.isUsed }"
                       :data-nn-tooltip="g.isUsed ? 'GPU ' + g.index + ': ' + formatBytes(g.usedBytes) + ' / ' + formatBytes(g.memoryBytes) + (g.model ? ' (' + g.model + ')' : '') : 'GPU ' + g.index + ': Available'">
                  </div>
                </template>
                <template v-else>
                  <div v-for="g in node.totalGpus" :key="g"
                       class="nn-gpu-block"
                       :class="{ used: g <= node.usedGpus }"
                       :data-nn-tooltip="g <= node.usedGpus ? 'In use' : 'Available'">
                  </div>
                </template>
              </div>
              <span class="nn-node-vram">{{ node.vram }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div id="status-content">
      <div id="filter-bar">
        <div class="nn-filter-left">
          <span class="nn-chip" :class="{ active: !selectedLevel }" @click="selectedLevel = null; currentPage = 1">
            All <span class="nn-chip-count">{{ deployments.length }}</span>
          </span>
          <span class="nn-chip hot" :class="{ active: selectedLevel === 'HOT' }" @click="toggleLevel('HOT')">
            <span class="nn-chip-dot hot"></span> Hot <span class="nn-chip-count">{{ stats.hot }}</span>
          </span>
          <span class="nn-chip warm" :class="{ active: selectedLevel === 'WARM' }" @click="toggleLevel('WARM')">
            <span class="nn-chip-dot warm"></span> Warm <span class="nn-chip-count">{{ stats.warm }}</span>
          </span>
          <span class="nn-chip cold" :class="{ active: selectedLevel === 'COLD' }" @click="toggleLevel('COLD')">
            <span class="nn-chip-dot cold"></span> Cold <span class="nn-chip-count">{{ stats.cold }}</span>
          </span>
        </div>
        <div class="nn-filter-right">
          <p-iconfield>
            <p-inputicon class="pi pi-search"></p-inputicon>
            <p-inputtext
              v-model="searchQuery"
              placeholder="Search models..."
              @input="currentPage = 1"
              style="min-width: 180px"
            ></p-inputtext>
          </p-iconfield>
          <p-floatlabel variant="on">
            <p-select
              v-model="selectedSortBy"
              :options="sortByOptions"
              inputId="sort_by"
              option-label="label"
              option-value="value"
              @change="currentPage = 1"
              style="min-width: 110px"
            ></p-select>
            <label for="sort_by">Sort</label>
          </p-floatlabel>
          <p-paginator
            v-if="filteredDeployments.length > 0"
            :first="(currentPage - 1) * pageSize"
            :rows="pageSize"
            :total-records="filteredDeployments.length"
            @page="onPageChange"
            template="PrevPageLink CurrentPageReport NextPageLink"
          ></p-paginator>
        </div>
      </div>

      <div id="deployments">
        <template v-if="status === 'loading'">
          <div v-for="n in 18" :key="'sk-' + n" class="nn-skeleton-card">
            <p-skeleton width="100%" height="100%"></p-skeleton>
          </div>
        </template>

        <deployment-component
          v-for="(deployment, index) in paginatedDeployments"
          :key="deployment.model_key"
          :index="index"
          :model_key="deployment.model_key"
          :dedicated="deployment.dedicated"
          :n_params="deployment.n_params"
          :deployment_level="deployment.deployment_level"
          :schedule="deployment.schedule"
          :application_state="deployment.application_state"
          :repo_id="deployment.repo_id"
          :revision="deployment.revision"
          :style="{ 'animation-delay': (index * 30) + 'ms' }"
        ></deployment-component>

        <div v-if="status === 'success' && paginatedDeployments.length === 0" class="nn-empty">
          <i class="pi pi-search"></i>
          No models match your filters.
        </div>
      </div>

      <div v-if="totalPages > 1" style="display: flex; justify-content: center;">
        <p-paginator
          :first="(currentPage - 1) * pageSize"
          :rows="pageSize"
          :total-records="filteredDeployments.length"
          @page="onPageChange"
          template="PrevPageLink PageLinks NextPageLink"
        ></p-paginator>
      </div>
    </div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {

  const BadgeComponent = {
    props: {
      content: String,
      bg: { type: Boolean, default: true },
      sdcls: String,
      cls: { type: String, default: undefined },
      tooltip: { type: String, default: undefined },
    },
    template: `<span v-if="content" class="nn-badge" :class="[bg ? 'nn-badge-' + sdcls : 'nn-badge-outline', cls]" :data-nn-tooltip="tooltip" v-html="content"></span>`,
  };

  const DeploymentComponent = {
    components: { BadgeComponent },
    data() {
      return { copied: false };
    },
    props: {
      model_key: String, deployment_level: String, dedicated: Boolean,
      schedule: Object, application_state: String, repo_id: String,
      n_params: Number, index: Number, revision: String,
    },
    computed: {
      levelClass() {
        return 'level-' + (this.deployment_level || '').toLowerCase();
      },
    },
    template: `
      <div class="nn-deployment" :class="levelClass">
        <div class="nn-card-body">
          <div v-if="model_key" class="nn-copy-btn">
            <button @mouseenter="showPopover" @mouseleave="hidePopover" @click.stop.prevent="copySnippet">
              <svg v-if="!copied" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none">
                <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                <rect x="8" y="8" width="12" height="12" rx="2"></rect>
                <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-width="2" stroke="#10b981" fill="none">
                <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                <path d="M5 12l5 5l10 -10"></path>
              </svg>
            </button>
            <p-popover ref="popover">
              <div class="nn-snippet"><pre><span v-html="getCodeSnippet()"></span></pre></div>
            </p-popover>
          </div>
          <div class="nn-repo-id">{{ repo_id }}</div>
          <div class="nn-badges">
            <badge-component v-if="schedule" v-bind="getScheduleInfoBadge()"></badge-component>
            <badge-component v-if="schedule" v-bind="getScheduleBadge()"></badge-component>
            <badge-component v-if="isPilotOnly()" v-bind="getPilotOnlyBadge()" @click="onPilotBadgeClick"></badge-component>
            <badge-component v-if="model_key" v-bind="getModelClassBadge()"></badge-component>
            <badge-component v-bind="getDeploymentLevelBadge()"></badge-component>
            <badge-component v-if="application_state" v-bind="getApplicationStateBadge()"></badge-component>
          </div>
          <div class="nn-meta" v-if="hasInfo()">
            <div v-if="revision" class="nn-meta-item" data-nn-tooltip="Revision">
              <i class="fa-regular fa-bookmark"></i><span>{{ revision }}</span>
            </div>
            <div v-if="n_params" class="nn-meta-item" data-nn-tooltip="# Params">
              <i class="fa-solid fa-border-none"></i>
              <span>{{ formatParams(n_params) }}</span>
            </div>
          </div>
        </div>
        <a :href="'http://huggingface.co/' + repo_id" target="_blank" class="nn-stretched-link"><span>HuggingFace</span></a>
      </div>`,
    methods: {
      hasInfo() { return this.n_params || this.revision; },
      className() { return this.model_key ? this.model_key.split(":")[0] : undefined; },
      formatParams(n) { return n / 1e9 < 1 ? (n / 1e9).toFixed(1) + 'B' : Math.round(n / 1e9) + 'B'; },
      getCodeSnippet() {
        let cn = this.className(), parts = cn.split("."), imp = parts.slice(0,-1).join("."), obj = parts[parts.length-1];
        const rev = this.revision ? `<span class="p">,</span> <span class="n">revision</span><span class="o">=</span><span class="s1">"${this.revision}"</span>` : "";
        return `<span class="kn">from</span> <span class="nn">${imp}</span> <span class="kn">import</span> <span class="n">${obj}</span>\n\n<span class="n">model</span> <span class="o">=</span> <span class="n">${obj}</span><span class="p">(</span><span class="s1">"${this.repo_id}"</span>${rev}<span class="p">)</span>\n\n<span class="k">with</span> <span class="n">model</span><span class="o">.</span><span class="n">trace</span><span class="p">(</span><span class="s2">"The Eiffel Tower is in the city of"</span><span class="p">,</span> <span class="n">remote</span><span class="o">=</span><span class="s1">True</span><span class="p">)</span><span class="p">:</span>\n    <span class="n">output</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">output</span><span class="o">.</span><span class="n">save</span><span class="p">()</span>`;
      },
      isPilotOnly() { return !(this.schedule && this.schedule.start_time); },
      getDeploymentLevelBadge() {
        const m = {
          HOT: ["This model is on GPU and ready.", "success", '<i class="fa-solid fa-microchip"></i> Hot'],
          WARM: ["Cached on CPU, quick to load.", "warning", '<i class="fa-solid fa-fire"></i> Warm'],
          COLD: ["Downloaded, slower to load.", "primary", '<i class="fa-regular fa-snowflake"></i> Cold'],
        };
        const [t, c, x] = m[this.deployment_level] || ["", "secondary", this.deployment_level];
        return { content: x, bg: true, sdcls: c, tooltip: t };
      },
      getApplicationStateBadge() {
        const m = {
          NOT_STARTED: ["warning", '<i class="fa-solid fa-gear fa-spin"></i> Not Started'],
          DEPLOYING: ["warning", '<i class="fa-solid fa-gear fa-spin"></i> Deploying'],
          DEPLOY_FAILED: ["danger", '<i class="fa-solid fa-xmark"></i> Deploy Failed'],
          RUNNING: ["success", '<i class="fa-solid fa-check"></i> Running'],
          UNHEALTHY: ["danger", '<i class="fa-solid fa-xmark"></i> Unhealthy'],
        };
        const [c, x] = m[this.application_state] || ["secondary", this.application_state];
        return { content: x, bg: true, sdcls: c };
      },
      getModelClassBadge() {
        return this.model_key.includes("LanguageModel")
          ? { content: '<i class="fa-solid fa-language"></i> Language', bg: true, sdcls: "secondary" }
          : { content: "", bg: true, sdcls: "secondary" };
      },
      formatTimeRemaining(endTime) {
        const diff = new Date(endTime) - new Date();
        if (diff < 0) return "Ended";
        let s = Math.floor(diff/1000), d = Math.floor(s/86400); s %= 86400;
        let h = Math.floor(s/3600); s %= 3600; let m = Math.floor(s/60);
        return d > 10 ? ">10d remaining" : `${d}d ${h}h ${m}m remaining`;
      },
      getScheduleInfoBadge() {
        if (!this.schedule.end_time || new Date() > new Date(this.schedule.end_time)) return { content: "" };
        if (this.schedule.start_time && new Date() < new Date(this.schedule.start_time)) {
          const s = new Date(this.schedule.start_time).toLocaleString(undefined, { month:"short", day:"numeric", hour:"numeric", minute:"numeric", hour12:true });
          return { content: `Starts ${s}`, bg: false, sdcls: "muted", tooltip: "Scheduled for later." };
        }
        return { content: this.formatTimeRemaining(this.schedule.end_time), bg: false, sdcls: "muted" };
      },
      getScheduleBadge() {
        if (!this.schedule.end_time || new Date() > new Date(this.schedule.end_time)) return { content: "" };
        return this.schedule.start_time && new Date() < new Date(this.schedule.start_time)
          ? { content: '<i class="fa-solid fa-clock"></i> Scheduled', bg: true, sdcls: "info", tooltip: "Scheduled for later." }
          : { content: '<i class="fa-solid fa-thumbtack"></i> Pinned', bg: true, sdcls: "info", tooltip: "Pinned deployment." };
      },
      getPilotOnlyBadge() {
        return { content: '<i class="fa-solid fa-lock"></i> Pilot Only', bg: true, sdcls: "muted", cls: "cursor-pointer", tooltip: "Restricted to pilot program. Click to sign up!" };
      },
      getPlainTextSnippet() {
        let cn = this.className(), parts = cn.split("."), imp = parts.slice(0,-1).join("."), obj = parts[parts.length-1];
        const rev = this.revision ? `, revision="${this.revision}"` : "";
        return `from ${imp} import ${obj}\n\nmodel = ${obj}("${this.repo_id}"${rev})\n\nwith model.trace("The Eiffel Tower is in the city of", remote=True):\n    output = model.output.save()`;
      },
      copySnippet() {
        navigator.clipboard.writeText(this.getPlainTextSnippet()).then(() => {
          this.copied = true;
          setTimeout(() => { this.copied = false; }, 1500);
        });
      },
      onPilotBadgeClick() { window.open("https://forms.gle/ZBtYvvnSdpiEdQEk6", "_blank"); },
      showPopover(e) { this.$refs.popover.show(e); },
      hidePopover() { this.$refs.popover.hide(); },
    },
  };

  const app = Vue.createApp({
    components: { DeploymentComponent },
    data() {
      return {
        ndif_url: "https://api.ndif.us",
        status: 'loading',
        deployments: [],
        cluster: null,
        calendar_id: undefined,
        currentPage: 1,
        pageSize: 18,
        selectedLevel: null,
        selectedSortBy: "deployment_level",
        searchQuery: "",
        sortByOptions: [
          { label: "Status", value: "deployment_level" },
          { label: "Name", value: "repo_id" },
          { label: "Params", value: "n_params" },
        ],
        panels: { resources: true, cluster: true },
      };
    },
    computed: {
      stats() {
        const d = this.deployments;
        return {
          hot: d.filter(x => x.deployment_level === 'HOT').length,
          warm: d.filter(x => x.deployment_level === 'WARM').length,
          cold: d.filter(x => x.deployment_level === 'COLD').length,
        };
      },
      clusterInfo() {
        if (!this.cluster || !this.cluster.nodes) return null;
        const entries = Object.entries(this.cluster.nodes);
        let totalGpus = 0, usedGpusCount = 0, totalVramBytes = 0;
        const nodes = entries.map(([id, node]) => {
          const res = node.resources;
          if (res.gpu_details) {
            const gpus = res.gpu_details;
            const t = gpus.length;
            const deployedGpuIndices = new Set();
            if (node.deployments) {
              Object.values(node.deployments).forEach(dep => {
                if (dep.gpus) Object.keys(dep.gpus).forEach(idx => deployedGpuIndices.add(parseInt(idx)));
              });
            }
            const used = deployedGpuIndices.size;
            const vramBytes = gpus.reduce((sum, g) => sum + g.memory_bytes, 0);
            totalGpus += t;
            usedGpusCount += used;
            totalVramBytes += vramBytes;
            const vramGb = vramBytes / 1e9;
            const gpuDetails = gpus.map(g => {
              const usedBytes = g.memory_bytes - g.available_memory_bytes;
              let modelOnGpu = null;
              if (node.deployments) {
                for (const [modelKey, dep] of Object.entries(node.deployments)) {
                  if (dep.gpus && dep.gpus[String(g.index)] !== undefined) {
                    const repoMatch = modelKey.match(/"repo_id":\s*"([^"]+)"/);
                    modelOnGpu = repoMatch ? repoMatch[1].split('/').pop() : modelKey.split(':').pop();
                    break;
                  }
                }
              }
              return {
                index: g.index,
                memoryBytes: g.memory_bytes,
                usedBytes,
                usedPct: Math.round((usedBytes / g.memory_bytes) * 100),
                isUsed: usedBytes > 0,
                model: modelOnGpu,
              };
            });
            return {
              id: id.slice(0, 6),
              totalGpus: t,
              usedGpus: used,
              vram: vramGb >= 100 ? Math.round(vramGb) + 'G' : vramGb.toFixed(0) + 'G',
              gpuDetails,
            };
          } else {
            const t = Math.round(res.total_gpus);
            const a = res.available_gpus.length;
            totalGpus += t;
            usedGpusCount += (t - a);
            totalVramBytes += res.gpu_memory_bytes;
            const vramGb = res.gpu_memory_bytes / 1e9;
            return {
              id: id.slice(0, 6),
              totalGpus: t,
              usedGpus: t - a,
              vram: vramGb >= 100 ? Math.round(vramGb) + 'G' : vramGb.toFixed(0) + 'G',
              gpuDetails: null,
            };
          }
        }).sort((a, b) => b.totalGpus - a.totalGpus);
        const totalVramTb = totalVramBytes / 1e12;
        return {
          totalGpus,
          usedGpus: usedGpusCount,
          totalVram: totalVramTb >= 1 ? totalVramTb.toFixed(1) + ' TB' : Math.round(totalVramBytes / 1e9) + ' GB',
          utilizationPct: totalGpus > 0 ? Math.round((usedGpusCount / totalGpus) * 100) : 0,
          nodes,
        };
      },
      filteredDeployments() {
        let f = [...this.deployments];
        if (this.selectedLevel) f = f.filter(d => d.deployment_level === this.selectedLevel);
        if (this.searchQuery.trim()) f = f.filter(d => d.repo_id.toLowerCase().includes(this.searchQuery.toLowerCase().trim()));
        if (this.selectedSortBy === "deployment_level") {
          const o = ["HOT", "WARM", "COLD"];
          f.sort((a, b) => {
            const al = o.indexOf(a.deployment_level), bl = o.indexOf(b.deployment_level);
            if (al === bl) return (b.schedule ? 1 : 0) - (a.schedule ? 1 : 0) || a.repo_id.localeCompare(b.repo_id);
            return (al === -1 ? o.length : al) - (bl === -1 ? o.length : bl);
          });
        } else if (this.selectedSortBy === "repo_id") {
          f.sort((a, b) => a.repo_id.localeCompare(b.repo_id));
        } else if (this.selectedSortBy === "n_params") {
          f.sort((a, b) => (b.n_params || 0) - (a.n_params || 0));
        }
        return f;
      },
      paginatedDeployments() {
        return this.filteredDeployments.slice((this.currentPage - 1) * this.pageSize, this.currentPage * this.pageSize);
      },
      totalPages() {
        return Math.ceil(this.filteredDeployments.length / this.pageSize);
      },
    },
    methods: {
      formatBytes(bytes) {
        if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
        if (bytes >= 1e6) return Math.round(bytes / 1e6) + ' MB';
        return Math.round(bytes / 1e3) + ' KB';
      },
      toggleLevel(level) {
        this.selectedLevel = this.selectedLevel === level ? null : level;
        this.currentPage = 1;
      },
      onPageChange(e) {
        this.currentPage = Math.floor(e.first / this.pageSize) + 1;
      },
      getStatus() {
        this.status = 'loading';
        fetch(this.ndif_url + "/ping")
          .then(r => {
            if (r.status === 200) {
              fetch(this.ndif_url + "/status")
                .then(r => r.status === 200
                  ? r.json().then(d => {
                      this.deployments = Object.values(d.deployments).filter(dep => dep.repo_id);
                      this.calendar_id = d.calendar_id;
                      this.cluster = d.cluster;
                      this.status = 'success';
                    })
                  : (this.status = 'error'))
                .catch(() => (this.status = 'error'));
            } else {
              this.status = 'error';
            }
          })
          .catch(() => (this.status = 'error'));
      },
    },
    mounted() { this.getStatus(); },
  });

  app.use(PrimeVue.Config, { theme: { preset: PrimeUIX.Themes.Aura } });
  app.component("p-paginator", PrimeVue.Paginator);
  app.component("p-select", PrimeVue.Select);
  app.component("p-inputtext", PrimeVue.InputText);
  app.component("p-iconfield", PrimeVue.IconField);
  app.component("p-inputicon", PrimeVue.InputIcon);
  app.component("p-skeleton", PrimeVue.Skeleton);
  app.component("p-floatlabel", PrimeVue.FloatLabel);
  app.component("p-popover", PrimeVue.Popover);

  app.mount("#status-app");
});
</script>
