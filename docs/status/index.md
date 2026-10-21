---
hide:
  - toc
  - path
card_type: ""
social:
  cards_layout_options:
    title: "What's hot on NDIF?"
---

<link rel="preconnect" href="https://cdnjs.cloudflare.com" crossorigin>
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<script src="/assets/js/vue.global.prod.js"></script>
<script>
  define = undefined;
</script>
<script src="/assets/js/primevue.min.js"></script>
<script src="/assets/js/aura.js"></script>
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
    width: 100%;
    font: inherit;
    color: inherit;
    background: none;
    border: 0;
    text-align: left;
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
  /* Wraps when the chips and the controls do not fit on one line; the
     controls then sit right-aligned on their own line. */
  #filter-bar {
    display: flex;
    flex-wrap: wrap;
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
    flex-wrap: wrap;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
    margin-left: auto;
  }

  .nn-chip {
    font-family: inherit;
    line-height: inherit;
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
  .p-paginator .p-paginator-next {
    min-width: 2rem !important;
    height: 2rem !important;
    border-radius: var(--nn-radius-sm) !important;
    background: var(--nn-chip-bg) !important;
    color: var(--md-default-fg-color) !important;
    border: 1px solid var(--nn-card-border) !important;
    transition: all var(--nn-ease) !important;
  }

  .p-paginator .p-paginator-prev:hover,
  .p-paginator .p-paginator-next:hover {
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
    grid-template-columns: repeat(auto-fill, minmax(min(360px, 100%), 1fr));
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
    /* A grid item's default minimum width is its content's, and the nowrap
       title would widen the card past its column; let it shrink so the title
       truncates instead. */
    min-width: 0;
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
    display: block;
    font-family: var(--nn-font-card);
    font-size: 0.88rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 2rem;
    color: inherit;
    text-decoration: none;
  }

  /* Outranks Material's .md-typeset a colour rules. */
  .nn-deployment .nn-repo-id,
  .nn-deployment .nn-repo-id:hover,
  .nn-deployment .nn-repo-id:focus { color: inherit; }

  /* A clipped name carries its full text as a tooltip on this wrapper (the
     name itself clips its overflow, which would clip the tooltip too). The
     wrapper's raised z-index also lets the name link win over the card's
     stretched link. */
  .nn-repo-id-wrap[data-nn-tooltip] { cursor: pointer; }

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

  .nn-badge-outline {
    background: transparent;
    border: 1px solid var(--md-default-fg-color--light);
    color: var(--md-default-fg-color--light);
  }

  /* Items never break inside their own text: when the row is too long the
     whole item moves to a second row. */
  .nn-meta {
    display: flex;
    flex-wrap: wrap;
    column-gap: 1rem;
    row-gap: 0.25rem;
    font-size: 0.7rem;
    color: var(--md-default-fg-color--light);
    padding-top: 2px;
  }

  .nn-meta-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    white-space: nowrap;
    flex-shrink: 0;
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
  [data-md-color-scheme="default"] .nn-snippet .o { color: #04a5e5; }
  [data-md-color-scheme="default"] .nn-snippet .s1 { color: #40a02b; }

  /* Snippet theming - dark mode */
  [data-md-color-scheme="slate"] .nn-snippet {
    background: #181818;
    border: 1px solid rgba(255, 255, 255, 0.06);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }

  [data-md-color-scheme="slate"] .nn-snippet pre { color: #cdd6f4; }
  [data-md-color-scheme="slate"] .nn-snippet .kn { color: #c792ea; }
  [data-md-color-scheme="slate"] .nn-snippet .nn { color: #82aaff; }
  [data-md-color-scheme="slate"] .nn-snippet .o { color: #89ddff; }
  [data-md-color-scheme="slate"] .nn-snippet .s1 { color: #c3e88d; }

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
    white-space: pre-line;
    line-height: 1.5;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-family: var(--nn-font-body);
    font-weight: normal;
    width: max-content;
    max-width: 260px;
    overflow-wrap: anywhere;
    text-align: center;
    pointer-events: none;
  }

  /* ===== Skeleton cards =====
     A skeleton is a real .nn-deployment whose text is transparent and painted
     over with a shimmer, so every line box has exactly the height the loaded
     card will have and the cards appear in place when the data arrives. */
  .nn-deployment.nn-skeleton {
    border-left-color: var(--nn-card-border);
    animation: none;
    pointer-events: none;
  }

  /* The ellipsis a clipped title would draw belongs to the block, not the span. */
  .nn-deployment.nn-skeleton .nn-repo-id { color: transparent; }

  .nn-deployment.nn-skeleton:hover {
    transform: none;
    box-shadow: var(--nn-card-shadow);
  }

  .nn-sk {
    color: transparent !important;
    border-radius: 3px;
    user-select: none;
    background: linear-gradient(90deg, var(--nn-chip-bg) 25%, var(--nn-chip-bg-hover) 50%, var(--nn-chip-bg) 75%);
    background-size: 200% 100%;
    animation: nn-shimmer 1.4s ease-in-out infinite;
  }

  i.nn-sk {
    display: inline-block;
    width: 1em;
    height: 1em;
    border-radius: 2px;
  }

  @keyframes nn-shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
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
    position: relative;
    flex: 1;
    height: 18px;
    border-radius: 3px;
    background: var(--nn-chip-bg);
  }

  /* Width tracks allocated memory, so a half-full GPU reads as half full
     rather than looking identical to a saturated one. */
  .nn-gpu-fill {
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    min-width: 2px;
    border-radius: 3px;
    background: var(--nn-cold);
    transition: width var(--nn-ease);
  }

  .nn-gpu-block:not(.used) .nn-gpu-fill { display: none; }

  .nn-gpu-block.used:hover .nn-gpu-fill {
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

  /* Shown only under 768px, where the full cluster panel is hidden. */
  .nn-cluster-compact {
    display: none;
    width: 100%;
    padding: 0.5rem 0.7rem;
    border: 1px solid var(--nn-card-border);
    border-radius: var(--nn-radius-sm);
    background: var(--nn-card-bg);
    font-family: var(--nn-font-mono);
    font-size: 0.7rem;
    color: var(--md-default-fg-color);
    text-align: center;
  }

  /* ===== Responsive ===== */
  /* Below Material's own sidebar breakpoint (76.25em) the page sidebar moves
     above the grid as a row of panels, and the card minimum drops so more
     columns fit. */
  @media (max-width: 76.1875em) {
    #status-main {
      flex-direction: column;
      padding: 1rem 1.5rem;
      gap: 1rem;
    }

    #status-sidebar {
      width: 100%;
      position: static;
      flex-direction: row;
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 0.75rem;
    }

    .nn-panel { flex: 1; min-width: 200px; }
    .nn-panel.nn-resources { flex: 0 1 220px; }

    #deployments {
      grid-template-columns: repeat(auto-fill, minmax(min(320px, 100%), 1fr));
    }
  }

  @media (max-width: 768px) {
    #status-main { padding: 0.75rem; }
    /* Phones: every sidebar item spans the full width. */
    .nn-panel.nn-resources { flex: 1 1 100%; }
    .nn-cluster-section { display: none !important; }
    .nn-cluster-compact { display: block; }

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
      grid-template-columns: minmax(0, 1fr);
    }
  }

</style>

<div id="status-app">
  <div id="status-banner" role="status" aria-live="polite"
       :class="{ 'nn-collapsed': status === 'success', 'nn-error': status === 'error' }">
    <span v-if="status === 'loading'" class="nn-pulse">
      <i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i>&nbsp; Connecting to NDIF...
    </span>
    <span v-else-if="status === 'error'">
      <i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i>&nbsp; Unable to reach NDIF. Please try again later.
    </span>
  </div>

  <div id="status-main">
    <div id="status-sidebar">
      <div class="nn-panel nn-resources">
        <button type="button" class="nn-panel-header" :aria-expanded="panels.resources" @click="panels.resources = !panels.resources">
          <span class="nn-panel-title">Resources</span>
          <i class="fa-solid fa-chevron-down nn-panel-chevron" :class="{ collapsed: !panels.resources }" aria-hidden="true"></i>
        </button>
        <div class="nn-panel-body" :class="{ collapsed: !panels.resources }">
          <div class="nn-sidebar-links">
            <a href="/features/15_remote_execution/" target="_blank"
               class="nn-sidebar-link"
               title="How do I use NDIF?" aria-label="How do I use NDIF?">
              <i class="fa-regular fa-circle-question" aria-hidden="true"></i>
            </a>
            <a href="https://login.ndif.us/" target="_blank"
               class="nn-sidebar-link"
               title="Login to NDIF" aria-label="Login to NDIF">
              <i class="fa-solid fa-right-to-bracket" aria-hidden="true"></i>
            </a>
          </div>
        </div>
      </div>

      <!-- While loading, clusterView is a placeholder with the real panel's
           shape, so the sidebar does not jump when the data arrives. -->
      <div v-if="clusterView" class="nn-panel nn-cluster-section" :aria-busy="clusterView.skeleton">
        <button type="button" class="nn-panel-header" :aria-expanded="panels.cluster" @click="panels.cluster = !panels.cluster">
          <span class="nn-panel-title">Cluster</span>
          <i class="fa-solid fa-chevron-down nn-panel-chevron" :class="{ collapsed: !panels.cluster }" aria-hidden="true"></i>
        </button>
        <div class="nn-panel-body" :class="{ collapsed: !panels.cluster }">
          <div class="nn-cluster-summary">
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value"><span :class="{ 'nn-sk': clusterView.skeleton }">{{ clusterView.totalGpus }}</span></div>
              <div class="nn-cluster-metric-label">GPUs</div>
            </div>
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value"><span :class="{ 'nn-sk': clusterView.skeleton }">{{ clusterView.busyGpus }}</span></div>
              <div class="nn-cluster-metric-label">Holding a model</div>
            </div>
            <div class="nn-cluster-metric">
              <div class="nn-cluster-metric-value"><span :class="{ 'nn-sk': clusterView.skeleton }">{{ clusterView.totalVram }}</span></div>
              <div class="nn-cluster-metric-label">Total VRAM</div>
            </div>
          </div>
          <div class="nn-cluster-bar-label">
            <span data-nn-tooltip="Share of the cluster's total GPU memory currently holding model weights.">GPU memory allocated</span>
            <span :class="{ 'nn-sk': clusterView.skeleton }">{{ clusterView.memoryPct }}%</span>
          </div>
          <div class="nn-cluster-bar">
            <div class="nn-cluster-bar-fill" :style="{ width: clusterView.memoryPct + '%' }"></div>
          </div>
          <div class="nn-panel-title" style="margin-bottom: 0.6rem;">Nodes</div>
          <div class="nn-node-list">
            <div v-for="node in clusterView.nodes" :key="node.rawId" class="nn-node">
              <span class="nn-node-id" :class="{ 'nn-sk': clusterView.skeleton }" :data-nn-tooltip="node.tooltip">{{ node.label }}</span>
              <div class="nn-node-gpus">
                <div v-for="g in node.gpuDetails" :key="g.index"
                     class="nn-gpu-block"
                     :class="{ used: g.isUsed }"
                     :data-nn-tooltip="g.tooltip">
                  <div class="nn-gpu-fill" :style="{ width: g.usedPct + '%' }"></div>
                </div>
              </div>
              <span class="nn-node-vram" :class="{ 'nn-sk': clusterView.skeleton }">{{ node.vram }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="clusterView" class="nn-cluster-compact">
        <span :class="{ 'nn-sk': clusterView.skeleton }">{{ clusterView.totalGpus }} GPUs &middot; {{ clusterView.nodes.length }} nodes &middot;
        {{ clusterView.totalVram }} &middot; {{ clusterView.memoryPct }}% allocated</span>
      </div>
    </div>

    <div id="status-content">
      <div id="filter-bar">
        <div class="nn-filter-left" role="group" aria-label="Filter by status">
          <button type="button" class="nn-chip" :class="{ active: !selectedLevel }" :aria-pressed="!selectedLevel" @click="selectedLevel = null; currentPage = 1">
            All <span v-if="status === 'success'" class="nn-chip-count">{{ deployments.length }}</span>
          </button>
          <button type="button" class="nn-chip hot" :class="{ active: selectedLevel === 'HOT' }" :aria-pressed="selectedLevel === 'HOT'" @click="toggleLevel('HOT')">
            <span class="nn-chip-dot hot"></span> Hot <span v-if="status === 'success'" class="nn-chip-count">{{ stats.hot }}</span>
          </button>
          <button type="button" class="nn-chip warm" :class="{ active: selectedLevel === 'WARM' }" :aria-pressed="selectedLevel === 'WARM'" @click="toggleLevel('WARM')">
            <span class="nn-chip-dot warm"></span> Warm <span v-if="status === 'success'" class="nn-chip-count">{{ stats.warm }}</span>
          </button>
          <button type="button" class="nn-chip cold" :class="{ active: selectedLevel === 'COLD' }" :aria-pressed="selectedLevel === 'COLD'" @click="toggleLevel('COLD')">
            <span class="nn-chip-dot cold"></span> Cold <span v-if="status === 'success'" class="nn-chip-count">{{ stats.cold }}</span>
          </button>
        </div>
        <div class="nn-filter-right">
          <p-iconfield>
            <p-inputicon class="fa-solid fa-magnifying-glass"></p-inputicon>
            <p-inputtext
              v-model="searchQuery"
              placeholder="Search models..."
              aria-label="Search models"
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
          <div v-for="n in pageSize" :key="'sk-' + n" class="nn-deployment nn-skeleton" aria-hidden="true">
            <div class="nn-card-body">
              <div class="nn-repo-id-wrap"><div class="nn-repo-id"><span class="nn-sk">{{ skeletonText(n, 0) }}</span></div></div>
              <div class="nn-badges">
                <span class="nn-badge nn-sk">Pinned</span>
                <span class="nn-badge nn-sk">Hot</span>
                <span class="nn-badge nn-sk">Running</span>
              </div>
              <div class="nn-meta">
                <div class="nn-meta-item"><i class="nn-sk"></i><span class="nn-sk">{{ skeletonText(n, 1) }}</span></div>
                <div class="nn-meta-item"><i class="nn-sk"></i><span class="nn-sk">{{ skeletonText(n, 2) }}</span></div>
                <div class="nn-meta-item"><i class="nn-sk"></i><span class="nn-sk">bfloat16</span></div>
              </div>
            </div>
          </div>
        </template>

        <deployment-component
          v-for="(deployment, index) in paginatedDeployments"
          :key="deployment.model_key || deployment.repo_id"
          :model_key="deployment.model_key"
          :pinned="deployment.pinned"
          :n_params="deployment.n_params"
          :deployment_level="deployment.deployment_level"
          :schedule="deployment.schedule"
          :application_state="deployment.application_state"
          :repo_id="deployment.repo_id"
          :revision="deployment.revision"
          :task="deployment.task"
          :dtype="deployment.dtype"
          :trust_remote_code="deployment.trust_remote_code"
          :replicas_count="deployment.replicas_count"
          :style="{ 'animation-delay': (index * 30) + 'ms' }"
        ></deployment-component>

        <div v-if="status === 'success' && paginatedDeployments.length === 0" class="nn-empty">
          <i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i>
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
      tooltip: { type: String, default: undefined },
    },
    template: `<span v-if="content" class="nn-badge" :class="bg ? 'nn-badge-' + sdcls : 'nn-badge-outline'" :data-nn-tooltip="tooltip" v-html="content"></span>`,
  };

  const DeploymentComponent = {
    components: { BadgeComponent },
    data() {
      // clipped: the name is wider than the card and shows an ellipsis.
      return { copied: false, clipped: false };
    },
    props: {
      model_key: String, deployment_level: String, pinned: Boolean,
      schedule: Object, application_state: String, repo_id: String,
      n_params: Number, revision: String, task: String, dtype: String,
      trust_remote_code: Boolean,
      // Total HOT+WARM replicas behind this card (post-aggregation in
      // the parent app). 0 for COLD HF-cache shadows.
      replicas_count: { type: Number, default: 0 },
    },
    computed: {
      levelClass() {
        return 'level-' + (this.deployment_level || '').toLowerCase();
      },
      hfUrl() { return 'https://huggingface.co/' + this.repo_id; },
    },
    mounted() {
      const name = this.$refs.name;
      const check = () => { this.clipped = name.scrollWidth > name.clientWidth; };
      check();
      if (typeof ResizeObserver !== 'undefined') new ResizeObserver(check).observe(name);
    },
    template: `
      <div class="nn-deployment" :class="levelClass">
        <div class="nn-card-body">
          <div v-if="model_key" class="nn-copy-btn">
            <button type="button" aria-label="Copy nnsight snippet" @mouseenter="showPopover" @mouseleave="hidePopover" @click.stop.prevent="copySnippet">
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
          <div class="nn-repo-id-wrap" :data-nn-tooltip="clipped ? repo_id : null">
            <a ref="name" class="nn-repo-id" :href="hfUrl" target="_blank">{{ repo_id }}</a>
          </div>
          <div class="nn-badges">
            <badge-component v-if="pinned" v-bind="getPinnedBadge()"></badge-component>
            <badge-component v-if="hasAutoEvictWindow()" v-bind="getAutoEvictBadge()"></badge-component>
            <badge-component v-if="model_key" v-bind="getModelClassBadge()"></badge-component>
            <badge-component v-bind="getDeploymentLevelBadge()"></badge-component>
            <badge-component v-if="application_state" v-bind="getApplicationStateBadge()"></badge-component>
          </div>
          <div class="nn-meta" v-if="hasInfo()">
            <div v-if="task" class="nn-meta-item" data-nn-tooltip="Task">
              <i class="fa-solid fa-list-check"></i><span>{{ task }}</span>
            </div>
            <div v-if="revision" class="nn-meta-item" data-nn-tooltip="Revision">
              <i class="fa-regular fa-bookmark"></i><span>{{ revision }}</span>
            </div>
            <div v-if="trust_remote_code" class="nn-meta-item" data-nn-tooltip="Runs the checkpoint's own modelling code; load with trust_remote_code=True">
              <i class="fa-solid fa-code"></i><span>trust_remote_code</span>
            </div>
            <div v-if="n_params" class="nn-meta-item" data-nn-tooltip="# Params">
              <i class="fa-solid fa-border-none"></i>
              <span>{{ formatParams(n_params) }}</span>
            </div>
            <div v-if="dtype" class="nn-meta-item" data-nn-tooltip="Weight dtype">
              <i class="fa-solid fa-memory"></i><span>{{ dtype }}</span>
            </div>
            <div v-if="replicas_count > 1" class="nn-meta-item" data-nn-tooltip="HOT/WARM replicas serving this model">
              <i class="fa-solid fa-layer-group"></i>
              <span>{{ replicas_count }} replicas</span>
            </div>
          </div>
        </div>
        <a :href="hfUrl" target="_blank" class="nn-stretched-link" tabindex="-1" aria-hidden="true"><span>HuggingFace</span></a>
      </div>`,
    methods: {
      hasInfo() { return this.task || this.dtype || this.trust_remote_code || this.n_params || this.revision || this.replicas_count > 1; },
      className() { return this.model_key ? this.model_key.split(":")[0] : undefined; },
      formatParams(n) { return n / 1e9 < 1 ? (n / 1e9).toFixed(1) + 'B' : Math.round(n / 1e9) + 'B'; },
      // The highlighted popover and the clipboard text come from the same
      // string, so the two can never drift apart.
      getCodeSnippet() {
        const esc = this.getPlainTextSnippet().replace(/&/g, "&amp;").replace(/</g, "&lt;");
        return esc.replace(/"[^"]*"|\bTrue\b|\b(?:from|import|with)\b|\b[\w.]+(?= import)|[=.]/g, (m) => {
          const cls = m[0] === '"' || m === "True" ? "s1"
            : m === "from" || m === "import" || m === "with" ? "kn"
            : m === "=" || m === "." ? "o"
            : "nn";
          return `<span class="${cls}">${m}</span>`;
        });
      },
      hasAutoEvictWindow() {
        // The non-pinned HOT path may carry a schedule.end_time meaning
        // "eligible for eviction after this timestamp" (NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS window).
        return !this.pinned
            && this.schedule
            && this.schedule.end_time
            && new Date() < new Date(this.schedule.end_time);
      },
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
        // Match on the bare class name parsed from the model_key prefix
        // (e.g. "nnsight.modeling.vlm.VisionLanguageModel:..." → "VisionLanguageModel").
        // ``includes("LanguageModel")`` was substring-matching so VLMs were
        // labeled "Language" alongside plain LMs — this map keeps them
        // distinct and is easy to extend for new envoy types.
        const cn = this.className();
        const base = cn ? cn.split(".").pop() : null;
        const map = {
          LanguageModel: { content: '<i class="fa-solid fa-language"></i> Language', bg: true, sdcls: "secondary" },
          VisionLanguageModel: { content: '<i class="fa-solid fa-eye"></i> Vision-Language', bg: true, sdcls: "secondary" },
          DiffusionModel: { content: '<i class="fa-solid fa-wand-magic-sparkles"></i> Diffusion', bg: true, sdcls: "secondary" },
        };
        return map[base] || { content: "", bg: true, sdcls: "secondary" };
      },
      formatTimeRemaining(endTime) {
        const diff = new Date(endTime) - new Date();
        if (diff < 0) return "Ended";
        let s = Math.floor(diff/1000), d = Math.floor(s/86400); s %= 86400;
        let h = Math.floor(s/3600); s %= 3600; let m = Math.floor(s/60);
        return d > 10 ? ">10d remaining" : `${d}d ${h}h ${m}m remaining`;
      },
      getPinnedBadge() {
        return {
          content: '<i class="fa-solid fa-thumbtack"></i> Pinned',
          bg: true, sdcls: "info",
          tooltip: "Pinned deployment — always kept HOT.",
        };
      },
      getAutoEvictBadge() {
        return {
          content: this.formatTimeRemaining(this.schedule.end_time),
          bg: false, sdcls: "muted",
          tooltip: "Hot-swap deployment — eligible for eviction after this window ends.",
        };
      },
      getPlainTextSnippet() {
        const parts = this.className().split("."), imp = parts.slice(0, -1).join("."), obj = parts[parts.length - 1];
        // task and trust_remote_code are part of the deployment's identity:
        // a client that leaves either off mints a different model key and may
        // target another deployment.
        const task = this.task ? `, task="${this.task}"` : "";
        const trc = this.trust_remote_code ? `, trust_remote_code=True` : "";
        const rev = this.revision ? `, revision="${this.revision}"` : "";
        return `from ${imp} import ${obj}\n\nmodel = ${obj}("${this.repo_id}"${task}${trc}${rev})\n\nwith model.trace("The Eiffel Tower is in the city of", remote=True):\n    output = model.output.save()`;
      },
      copySnippet() {
        navigator.clipboard.writeText(this.getPlainTextSnippet()).then(() => {
          this.copied = true;
          setTimeout(() => { this.copied = false; }, 1500);
        });
      },
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
        currentPage: 1,
        // Columns the card grid currently renders; measured from the grid so
        // every full page ends on a complete row.
        columns: 1,
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
      // Closest multiple of the column count to the nominal 18 cards, so a
      // page never ends with a partial row on any viewport width.
      pageSize() {
        return this.columns * Math.max(1, Math.round(18 / this.columns));
      },
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
        let totalGpus = 0, busyGpus = 0, totalVramBytes = 0, allocatedBytes = 0;

        const repoOf = (modelKey) => {
          const m = modelKey.match(/"repo_id":\s*"([^"]+)"/);
          return m ? m[1] : modelKey.split(":").pop();
        };
        // One checkpoint can be deployed under several tasks, each its own
        // model_key; name the task from the /status entry so they stay apart.
        const fieldsByKey = {};
        this.deployments.forEach((d) => {
          if (!d.model_key) return;
          const fields = [d.task, d.trust_remote_code ? "trust_remote_code" : null].filter(Boolean);
          if (fields.length) fieldsByKey[d.model_key] = fields.join(", ");
        });
        const labelOf = (modelKey) => repoOf(modelKey) + (fieldsByKey[modelKey] ? " (" + fieldsByKey[modelKey] + ")" : "");

        // node.deployments is model_key -> replica_id -> { gpus: {index: bytes} }.
        // Returns gpu index -> [{repo, bytes}], because several models can share
        // one GPU and the panel should name all of them.
        const occupantsByGpu = (node) => {
          const byIndex = {};
          Object.entries(node.deployments || {}).forEach(([modelKey, replicas]) => {
            Object.values(replicas || {}).forEach((replica) => {
              const gpus = replica && replica.gpus;
              if (!gpus) return;
              Object.entries(gpus).forEach(([idx, bytes]) => {
                (byIndex[idx] = byIndex[idx] || []).push({
                  repo: labelOf(modelKey),
                  bytes: Number(bytes) || 0,
                });
              });
            });
          });
          Object.values(byIndex).forEach((list) => list.sort((a, b) => b.bytes - a.bytes));
          return byIndex;
        };

        const nodes = entries.map(([id, node]) => {
          const gpus = node.resources.gpu_details || [];
          const occupants = occupantsByGpu(node);
          const vramBytes = gpus.reduce((sum, g) => sum + g.memory_bytes, 0);
          totalGpus += gpus.length;
          totalVramBytes += vramBytes;

          const gpuDetails = gpus.map((g) => {
            const usedBytes = Math.max(0, g.memory_bytes - g.available_memory_bytes);
            const isUsed = usedBytes > 0;
            if (isUsed) busyGpus += 1;
            allocatedBytes += usedBytes;
            const here = occupants[String(g.index)] || [];
            const head = "GPU " + g.index + " \u00b7 " + (isUsed
              ? this.formatBytes(usedBytes) + " of " + this.formatBytes(g.memory_bytes) + " in use"
              : "free \u00b7 " + this.formatBytes(g.memory_bytes));
            const lines = here.map((o) => o.repo + "  " + this.formatBytes(o.bytes));
            return {
              index: g.index,
              memoryBytes: g.memory_bytes,
              usedBytes,
              usedPct: g.memory_bytes ? Math.round((usedBytes / g.memory_bytes) * 100) : 0,
              isUsed,
              models: here,
              tooltip: [head].concat(lines).join("\n"),
            };
          });

          return {
            rawId: id,
            totalGpus: gpus.length,
            usedGpus: gpuDetails.filter((g) => g.isUsed).length,
            vram: this.formatVram(vramBytes),
            gpuDetails,
          };
        }).sort((a, b) => b.totalGpus - a.totalGpus);

        // Ray node ids are 56 hex characters and change on restart, so they are
        // labelled by position and the real id lives in the tooltip.
        nodes.forEach((node, i) => {
          node.label = "Node " + (i + 1);
          const size = node.gpuDetails.length
            ? " \u00b7 " + this.formatBytes(node.gpuDetails[0].memoryBytes) + " each"
            : "";
          node.tooltip = node.totalGpus + " GPUs" + size + "\n" + node.rawId;
        });

        const totalVramTb = totalVramBytes / 1e12;
        return {
          totalGpus,
          busyGpus,
          totalVram: totalVramTb >= 1 ? totalVramTb.toFixed(1) + " TB" : Math.round(totalVramBytes / 1e9) + " GB",
          memoryPct: totalVramBytes > 0 ? Math.round((allocatedBytes / totalVramBytes) * 100) : 0,
          nodes,
        };
      },
      // What the cluster panel renders: the live data, or while loading a
      // placeholder with the same shape (typical node count and width) whose
      // values the template paints as skeleton bars.
      clusterView() {
        if (this.clusterInfo) return this.clusterInfo;
        if (this.status !== 'loading') return null;
        const gpu = (index) => ({ index, isUsed: false, usedPct: 0 });
        const nodes = Array.from({ length: 8 }, (_, i) => ({
          rawId: 'sk-' + i, label: 'Node ' + (i + 1), vram: '191G',
          gpuDetails: [gpu(0), gpu(1), gpu(2), gpu(3)],
        }));
        return { skeleton: true, totalGpus: 36, busyGpus: 35, totalVram: '2.5 TB', memoryPct: 0, nodes };
      },
      filteredDeployments() {
        let f = [...this.deployments];
        if (this.selectedLevel) f = f.filter(d => d.deployment_level === this.selectedLevel);
        if (this.searchQuery.trim()) {
          const q = this.searchQuery.toLowerCase().trim();
          f = f.filter(d => d.repo_id.toLowerCase().includes(q) || (d.task || "").toLowerCase().includes(q));
        }
        if (this.selectedSortBy === "deployment_level") {
          const o = ["HOT", "WARM", "COLD"];
          f.sort((a, b) => {
            const al = o.indexOf(a.deployment_level), bl = o.indexOf(b.deployment_level);
            if (al === bl) return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0) || a.repo_id.localeCompare(b.repo_id) || (a.task || "").localeCompare(b.task || "");
            return (al === -1 ? o.length : al) - (bl === -1 ? o.length : bl);
          });
        } else if (this.selectedSortBy === "repo_id") {
          f.sort((a, b) => a.repo_id.localeCompare(b.repo_id) || (a.task || "").localeCompare(b.task || ""));
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
    watch: {
      pageSize() {
        if (this.currentPage > this.totalPages) this.currentPage = Math.max(1, this.totalPages);
      },
    },
    methods: {
      measureColumns() {
        const grid = document.getElementById('deployments');
        if (!grid) return;
        // Resolved track list, e.g. "409px 409px 409px 409px".
        const tracks = getComputedStyle(grid).gridTemplateColumns.split(' ').filter(Boolean).length;
        this.columns = Math.max(1, tracks);
      },
      // Placeholder strings of varied, realistic lengths. Only their widths
      // matter: the text itself is transparent.
      skeletonText(n, slot) {
        const names = ["meta-llama/Llama-3.1-8B", "google/gemma-2-9b-it", "EleutherAI/gpt-j-6b",
                       "Qwen/Qwen2.5-14B", "mistralai/Mistral-7B-v0.3", "openai-community/gpt2"];
        const tasks = ["text-generation", "image-text-to-text"];
        const sizes = ["8B", "70B", "12B"];
        const pick = (list, k) => list[(n * 7 + k) % list.length];
        return slot === 0 ? pick(names, 0) : slot === 1 ? pick(tasks, 1) : pick(sizes, 2);
      },
      formatBytes(bytes) {
        if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
        if (bytes >= 1e6) return Math.round(bytes / 1e6) + ' MB';
        return Math.round(bytes / 1e3) + ' KB';
      },
      // Compact, for the node row: 1.2T / 192G.
      formatVram(bytes) {
        const gb = bytes / 1e9;
        return gb >= 1000 ? (gb / 1000).toFixed(1) + 'T' : Math.round(gb) + 'G';
      },
      toggleLevel(level) {
        this.selectedLevel = this.selectedLevel === level ? null : level;
        this.currentPage = 1;
      },
      onPageChange(e) {
        this.currentPage = Math.floor(e.first / this.pageSize) + 1;
      },
      // Collapse one-entry-per-replica into one-entry-per-model_key. The key
      // carries the pipeline task, so one repo deployed under several tasks
      // stays one card per task.
      // Mirrors the dashboard backend's _aggregate_by_model_key (see
      // src/ndif/services/dashboard/backend/routers/deploy.py) so the
      // public status page stays consistent with the admin view: HOT
      // outranks WARM outranks COLD, RUNNING outranks DEPLOYING etc.
      // COLD HF-cache entries have no model_key and pass through as-is.
      aggregateByModelKey(entries) {
        const LEVEL_RANK = { HOT: 0, WARM: 1, COLD: 2 };
        const STATE_RANK = { RUNNING: 0, DEPLOYING: 1, NOT_STARTED: 2, UNHEALTHY: 3 };
        const out = new Map();
        for (const e of entries) {
          const mk = e.model_key;
          if (!mk) {
            out.set(e.repo_id, { ...e, replicas_count: 0 });
            continue;
          }
          let card = out.get(mk);
          if (!card) {
            card = { ...e, replicas_count: 0 };
            out.set(mk, card);
          }
          const newLevel = e.deployment_level;
          if (newLevel && (LEVEL_RANK[newLevel] ?? 99) < (LEVEL_RANK[card.deployment_level] ?? 99)) {
            card.deployment_level = newLevel;
          }
          const newState = e.application_state;
          if (newState && (
            !card.application_state ||
            (STATE_RANK[newState] ?? 99) < (STATE_RANK[card.application_state] ?? 99)
          )) {
            card.application_state = newState;
          }
          if (e.pinned) {
            card.pinned = true;
            if (e.schedule && !card.schedule) card.schedule = e.schedule;
          }
          card.replicas_count += 1;
        }
        return Array.from(out.values());
      },
      getStatus() {
        this.status = 'loading';
        fetch(this.ndif_url + "/status")
          .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
          .then(d => {
            const raw = Object.values(d.deployments).filter(dep => dep.repo_id);
            this.deployments = this.aggregateByModelKey(raw);
            this.cluster = d.cluster;
            this.status = 'success';
          })
          .catch(() => { this.status = 'error'; });
      },
    },
    mounted() {
      this.measureColumns();
      const grid = document.getElementById('deployments');
      if (grid && typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => this.measureColumns()).observe(grid);
      } else {
        window.addEventListener('resize', () => this.measureColumns());
      }
      this.getStatus();
    },
  });

  app.use(PrimeVue.Config, { theme: { preset: PrimeUIX.Themes.Aura } });
  app.component("p-paginator", PrimeVue.Paginator);
  app.component("p-select", PrimeVue.Select);
  app.component("p-inputtext", PrimeVue.InputText);
  app.component("p-iconfield", PrimeVue.IconField);
  app.component("p-inputicon", PrimeVue.InputIcon);
  app.component("p-floatlabel", PrimeVue.FloatLabel);
  app.component("p-popover", PrimeVue.Popover);

  app.mount("#status-app");
});
</script>
