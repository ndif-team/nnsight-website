---
hide:
  - navigation
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

<style>
  @import url("https://fonts.googleapis.com/css2?family=Electrolize&family=Zen+Dots&display=swap");

  /* Make content area full width for status page */
  .md-main__inner {
    max-width: 100% !important;
    margin: 0 !important;
  }
  
  .md-content {
    max-width: 100% !important;
    margin: 0 !important;
  }
  
  .md-content__inner {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
  }
  
  /* Hide sidebar completely */
  .md-sidebar {
    display: none !important;
  }
  
  /* Make grid full width */
  .md-grid {
    max-width: 100% !important;
    margin: 0 !important;
  }
  
  /* Hide page title */
  .md-content h1,
  .md-content__inner > h1:first-child,
  .md-typeset h1 {
    display: none !important;
  }

  /* PrimeVue styles */
  .p-paginator {
    background: transparent !important;
  }

  .p-select-label {
    padding: 4px 7px !important;
    font-size: 0.9rem !important;
  }

  .p-inputtext {
    padding-block: 4px !important;
    font-size: 0.9rem !important;
  }

  /* Core elements styles */
  #status-app {
    display: flex;
    flex-direction: row;
    width: 100%;
    justify-content: center;
    padding: 1rem;

    --p-inputtext-background: var(--md-default-bg-color) !important;
    --p-select-background: var(--md-default-bg-color) !important;
    --p-floatlabel-on-active-background: var(--md-default-bg-color) !important;
    --p-select-overlay-background: var(--md-default-bg-color) !important;
    --p-card-background: var(--md-default-bg-color) !important;
    --p-panel-background: var(--md-default-bg-color) !important;

    --p-select-color: var(--md-default-fg-color) !important;
    --p-inputtext-color: var(--md-default-fg-color) !important;
    --p-select-overlay-color: var(--md-default-fg-color) !important;
    --p-panel-color: var(--md-default-fg-color) !important;
    --p-panel-header-color: var(--md-default-fg-color) !important;
  }

  #status-display {
    /* max-width: 1100px; */
    width: 80%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  #status-bar {
    border-radius: 15px 15px 5px 5px;
    overflow: hidden;
  }

  #status-sidebar {
    width: 300px;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  #deployments {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  #filter-bar {
    display: flex;
    flex-direction: row;
    gap: 0.1rem;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
  }

  .filter-item {
    display: flex;
    flex-direction: row;
  }

  .filter-group {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 2rem;
    justify-content: center;
    flex-wrap: wrap;
  }

  /* Status bar styles */
  .status-card {
    padding: 1rem 2rem;
    text-align: center;
    font-weight: bold;
    font-size: 1.1rem;
  }

  .status-success { background: linear-gradient(135deg, #10b981, #059669); color: white; }
  .status-info { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
  .status-warning { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
  .status-danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
  .status-primary { background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; }

  /* Deployment card styles */
  .nn-deployment {
    flex: 1 1 400px;
    border-radius: 0.5rem;
    max-width: 475px;
    background: var(--md-default-bg-color);
    border: 1px solid var(--md-default-fg-color--lightest);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .nn-deployment:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 3;
  }

  .nn-card-body {
    display: flex;
    flex-direction: column;
    border-radius: 0.5rem;
    justify-content: space-between;
    padding: 1rem;
    background-image: linear-gradient(to right, var(--md-default-fg-color--lightest), var(--md-default-bg-color)) !important;
    flex: 1;
  }

  .nn-deployment-badges {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 6px;
  }

  .nn-deployment-other {
    font-size: 0.7rem;
    display: flex;
    flex-direction: row;
    gap: 1rem;
    color: var(--md-default-fg-color--light);
    padding-top: 5px;
  }

  .nn-deployment-other-item {
    display: flex;
    flex-direction: row;
    gap: 0.3rem;
    align-items: center;
  }

  .nn-deployment-repo-id {
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
    margin-right: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .nn-copy-button {
    position: absolute !important;
    right: 2.3%;
    top: 4%;
    z-index: 2;
  }

  .nn-copy-button button {
    z-index: 3;
    opacity: 1;
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px;
  }

  .nn-copy-button button svg {
    stroke: var(--md-default-fg-color);
    width: 20px;
    height: 20px;
  }

  .nn-snippet {
    width: max-content;
  }

  /* Badge styles */
  .nn-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: bolder;
  }

  /* Light mode badge text - white */
  [data-md-color-scheme="default"] .nn-badge-success,
  [data-md-color-scheme="default"] .nn-badge-warning,
  [data-md-color-scheme="default"] .nn-badge-danger,
  [data-md-color-scheme="default"] .nn-badge-info,
  [data-md-color-scheme="default"] .nn-badge-primary,
  [data-md-color-scheme="default"] .nn-badge-secondary,
  [data-md-color-scheme="default"] .nn-badge-muted {
    color: #ffffff;
  }

  /* Dark mode badge text - black */
  [data-md-color-scheme="slate"] .nn-badge-success,
  [data-md-color-scheme="slate"] .nn-badge-warning,
  [data-md-color-scheme="slate"] .nn-badge-danger,
  [data-md-color-scheme="slate"] .nn-badge-info,
  [data-md-color-scheme="slate"] .nn-badge-primary,
  [data-md-color-scheme="slate"] .nn-badge-secondary,
  [data-md-color-scheme="slate"] .nn-badge-muted {
    color: #000000;
  }

  .nn-badge-success { background: #10b981; }
  .nn-badge-warning { background: #f59e0b; }
  .nn-badge-danger { background: #ef4444; }
  .nn-badge-info { background: #3b82f6; }
  .nn-badge-primary { background: #6366f1; }
  .nn-badge-secondary { background: #6b7280; }
  .nn-badge-muted { background: #9ca3af; }

  .nn-badge-outline-muted {
    background: transparent;
    border: 1px solid var(--md-default-fg-color--light);
    color: var(--md-default-fg-color--light);
  }

  .electrolize {
    font-family: "Electrolize", sans-serif;
    font-weight: 1000;
    font-style: normal;
  }

  .nn-stretched-link::after {
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
    z-index: 1;
    content: "";
  }

  .nn-stretched-link {
    text-decoration: none;
  }

  .nn-stretched-link span {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  [data-nn-tooltip] {
    position: relative;
    cursor: help;
    z-index: 3;
  }

  [data-nn-tooltip].cursor-pointer {
    cursor: pointer;
  }

  [data-nn-tooltip]:hover::after {
    content: attr(data-nn-tooltip);
    z-index: 6;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 12px;
    background-color: var(--md-default-bg-color);
    color: var(--md-default-fg-color);
    border: 1px solid var(--md-default-fg-color--lightest);
    border-radius: 4px;
    font-size: 12px;
    white-space: normal;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    font-family: inherit;
    font-weight: normal;
    width: max-content;
    max-width: 300px;
    text-align: center;
    pointer-events: none;
  }

  .p-popover {
    background: var(--md-default-bg-color) !important;
    color: var(--md-default-fg-color) !important;
  }

  .p-popover .p-popover-content {
    background: var(--md-default-bg-color) !important;
    border: 1px solid var(--md-default-fg-color--lightest) !important;
    border-radius: 4px !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
  }

  .p-popover .nn-snippet pre {
    background: var(--md-default-bg-color) !important;
    color: var(--md-default-fg-color);
    padding: 1rem;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 0.85rem;
  }

  .p-popover::before,
  .p-popover::after {
    display: none !important;
  }

  .desktop-only {
    display: block;
  }

  /* Responsive styles */
  @media (max-width: 768px) {
    #status-app {
      flex-direction: column;
      padding: 0.5rem;
    }

    .desktop-only {
      display: none !important;
    }

    #status-sidebar {
      display: none !important;
    }

    #status-display {
      max-width: 100%;
    }

    #filter-bar {
      flex-direction: column;
      gap: 1rem;
      align-items: stretch;
    }

    .filter-group {
      justify-content: center;
      gap: 1rem;
    }

    #deployments {
      justify-content: center;
    }

    .nn-deployment {
      max-width: 100%;
      flex: 1 1 auto;
    }
  }

  /* Syntax highlighting for code snippets - dark mode */
  [data-md-color-scheme="slate"] .highlight .kn { color: #c792ea; }
  [data-md-color-scheme="slate"] .highlight .nn { color: #82aaff; }
  [data-md-color-scheme="slate"] .highlight .n { color: var(--md-default-fg-color); }
  [data-md-color-scheme="slate"] .highlight .o { color: #89ddff; }
  [data-md-color-scheme="slate"] .highlight .p { color: var(--md-default-fg-color); }
  [data-md-color-scheme="slate"] .highlight .s1,
  [data-md-color-scheme="slate"] .highlight .s2 { color: #c3e88d; }
  [data-md-color-scheme="slate"] .highlight .k { color: #c792ea; }

  /* Syntax highlighting for code snippets - light mode */
  [data-md-color-scheme="default"] .highlight .kn { color: #8839ef; }
  [data-md-color-scheme="default"] .highlight .nn { color: #1e66f5; }
  [data-md-color-scheme="default"] .highlight .n { color: #4c4f69; }
  [data-md-color-scheme="default"] .highlight .o { color: #04a5e5; }
  [data-md-color-scheme="default"] .highlight .p { color: #4c4f69; }
  [data-md-color-scheme="default"] .highlight .s1,
  [data-md-color-scheme="default"] .highlight .s2 { color: #40a02b; }
  [data-md-color-scheme="default"] .highlight .k { color: #8839ef; }
</style>

<div id="status-app">
  <div id="status-sidebar">
    <div class="filter-group">
      <a
        v-if="calendar_id"
        :href="'https://calendar.google.com/calendar/embed?src=' + encodeURIComponent(calendar_id)"
        target="_blank"
        title="View Dedicated Deployment Calendar"
        style="display: inline-block"
      >
        <i class="pi pi-calendar-clock" style="font-size: 3rem"></i>
      </a>
      <a
        href="/features/remote_execution/"
        target="_blank"
        title="How do I use NDIF?"
        style="display: inline-block"
      >
        <i class="pi pi-question-circle" style="font-size: 3rem"></i>
      </a>
    </div>
    <cluster-component :cluster="cluster"></cluster-component>
  </div>
  <p-divider layout="vertical" class="desktop-only"></p-divider>
  <div id="status-display">
    <div id="status-bar">
      <status-bar-component ref="statusbar"></status-bar-component>
    </div>

    <div id="filter-bar">
      <div class="filter-group">
        <div class="filter-item">
          <p-floatlabel variant="on">
            <p-select
              v-model="selectedSortBy"
              :options="sortByOptions"
              inputId="sort_by_label"
              option-label="label"
              option-value="value"
              placeholder="Sort by"
              @change="onSortChange"
              style="min-width: 120px"
            ></p-select>
            <label for="sort_by_label">Sort</label>
          </p-floatlabel>
        </div>
        <div class="filter-item">
          <p-iconfield>
            <p-inputicon class="pi pi-search"></p-inputicon>
            <p-inputtext
              v-model="searchQuery"
              placeholder="Search..."
              @input="onSearchChange"
              style="min-width: 200px"
            ></p-inputtext>
          </p-iconfield>
        </div>
      </div>
      <div class="filter-group">
        <p-paginator
          v-model:first="firstRecord"
          :rows="pageSize"
          :total-records="filteredDeployments.length"
          @page="onPageChange"
          template="PrevPageLink CurrentPageReport NextPageLink"
        >
        </p-paginator>
      </div>
    </div>

    <div id="deployments">
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
      ></deployment-component>
      <p-skeleton
        v-if="deployments.length === 0"
        v-for="n in 18"
        :key="n"
        width="350px"
        height="83px"
      ></p-skeleton>
    </div>

    <div v-if="totalPages > 1" class="pagination-container">
      <p-paginator
        v-model:first="firstRecord"
        :rows="pageSize"
        :total-records="filteredDeployments.length"
        @page="onPageChange"
        :template="{ '640px': 'PrevPageLink PageLinks NextPageLink', '960px': 'FirstPageLink PrevPageLink PageLinks NextPageLink LastPageLink', '1300px': 'FirstPageLink PrevPageLink PageLinks NextPageLink LastPageLink' }"
      ></p-paginator>
    </div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const StatusBarComponent = {
    data() {
      return { text: "Getting Status...", cls: "primary" };
    },
    methods: {
      update(text, cls) { this.text = text; this.cls = cls; },
    },
    template: `<div :class="['status-card', 'status-' + cls]">{{ text }}</div>`,
  };

  const BadgeComponent = {
    props: {
      content: String,
      bg: { type: Boolean, default: true },
      sdcls: String,
      cls: { type: String, default: undefined },
      tooltip: { type: String, default: undefined },
    },
    template: `<span v-if="content" class="nn-badge" :class="[bg ? 'nn-badge-' + sdcls : 'nn-badge-outline-' + sdcls, cls]" :data-nn-tooltip="tooltip" v-html="content"></span>`,
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
    template: `
      <div class="nn-deployment electrolize">
        <div class="nn-card-body">
          <div v-if="model_key" class="nn-copy-button">
            <button @mouseenter="showPopover" @mouseleave="hidePopover" @click="copySnippet">
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
              <div class="nn-snippet highlight"><pre :id="'codecell' + index"><span v-html="getCodeSnippet()"></span></pre></div>
            </p-popover>
          </div>
          <div class="nn-deployment-repo-id">{{repo_id}}</div>
          <div class="nn-deployment-badges">
            <badge-component v-if="schedule" v-bind="getScheduleInfoBadge()"></badge-component>
            <badge-component v-if="schedule" v-bind="getScheduleBadge()"></badge-component>
            <badge-component v-if="isPilotOnly()" v-bind="getPilotOnlyBadge()" @click="onPilotBadgeClick"></badge-component>
            <badge-component v-if="model_key" v-bind="getModelClassBadge()"></badge-component>
            <badge-component v-bind="getDeploymentLevelBadge()"></badge-component>
            <badge-component v-if="application_state" v-bind="getApplicationStateBadge()"></badge-component>
          </div>
          <div class="nn-deployment-other" v-if="hasInfo()">
            <div v-if="revision" class="nn-deployment-other-item" data-nn-tooltip="Revision">
              <i class="fa-regular fa-bookmark"></i><span>{{ revision }}</span>
            </div>
            <div v-if="n_params" class="nn-deployment-other-item" data-nn-tooltip="# Params">
              <i class="fa-solid fa-border-none"></i>
              <span>{{ n_params / 1e9 < 1 ? (n_params / 1e9).toFixed(1) + 'B' : Math.round(n_params / 1e9) + 'B' }}</span>
            </div>
          </div>
        </div>
        <a :href="'http://huggingface.co/' + repo_id" target="_blank" class="nn-stretched-link"><span>HuggingFace</span></a>
      </div>`,
    methods: {
      hasInfo() { return this.n_params || this.revision; },
      className() { return this.model_key ? this.model_key.split(":")[0] : undefined; },
      getCodeSnippet() {
        let cn = this.className(), parts = cn.split("."), imp = parts.slice(0,-1).join("."), obj = parts[parts.length-1];
        const rev = this.revision ? `<span class="p">,</span> <span class="n">revision</span><span class="o">=</span><span class="s1">"${this.revision}"</span>` : "";
        return `<span class="kn">from</span> <span class="nn">${imp}</span> <span class="kn">import</span> <span class="n">${obj}</span>\n\n<span class="n">model</span> <span class="o">=</span> <span class="n">${obj}</span><span class="p">(</span><span class="s1">"${this.repo_id}"</span>${rev}<span class="p">)</span>\n\n<span class="k">with</span> <span class="n">model</span><span class="o">.</span><span class="n">trace</span><span class="p">(</span><span class="s2">"The Eiffel Tower is in the city of"</span><span class="p">,</span> <span class="n">remote</span><span class="o">=</span><span class="s1">True</span><span class="p">)</span><span class="p">:</span>\n    <span class="n">output</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">output</span><span class="o">.</span><span class="n">save</span><span class="p">()</span>`;
      },
      isPilotOnly() { return !(this.schedule && this.schedule.start_time); },
      getDeploymentLevelBadge() {
        const m = { HOT: ["This model is on GPU and ready.", "success", '<i class="fa-solid fa-microchip"></i> Hot'],
                    WARM: ["Cached on CPU, quick to load.", "warning", '<i class="fa-solid fa-fire"></i> Warm'],
                    COLD: ["Downloaded, slower to load.", "primary", '<i class="fa-regular fa-snowflake"></i> Cold'] };
        const [t, c, x] = m[this.deployment_level] || ["", "secondary", this.deployment_level];
        return { content: x, bg: true, sdcls: c, tooltip: t };
      },
      getApplicationStateBadge() {
        const m = { NOT_STARTED: ["warning", '<i class="fa-solid fa-gear fa-spin"></i> Not Started'],
                    DEPLOYING: ["warning", '<i class="fa-solid fa-gear fa-spin"></i> Deploying'],
                    DEPLOY_FAILED: ["danger", '<i class="fa-solid fa-xmark"></i> Deploy Failed'],
                    RUNNING: ["success", '<i class="fa-solid fa-check"></i> Running'],
                    UNHEALTHY: ["danger", '<i class="fa-solid fa-xmark"></i> Unhealthy'] };
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

  const ClusterComponent = {
    components: { BadgeComponent },
    props: { cluster: Object },
    template: `<div v-if="false"></div>`,
  };

  const app = Vue.createApp({
    components: { StatusBarComponent, DeploymentComponent, ClusterComponent },
    data() {
      return {
        ndif_url: "https://api.ndif.us",
        deployments: [], cluster: null, currentPage: 1, pageSize: 18,
        selectedDeploymentLevel: null, selectedSortBy: "deployment_level",
        searchQuery: "", calendar_id: undefined,
        sortByOptions: [
          { label: "Status", value: "deployment_level" },
          { label: "Name", value: "repo_id" },
          { label: "Params", value: "n_params" },
        ],
      };
    },
    computed: {
      sortedAndFilteredDeployments() {
        let f = [...this.deployments];
        if (this.selectedDeploymentLevel) f = f.filter(d => d.deployment_level === this.selectedDeploymentLevel);
        if (this.searchQuery.trim()) f = f.filter(d => d.repo_id.toLowerCase().includes(this.searchQuery.toLowerCase().trim()));
        if (this.selectedSortBy === "deployment_level") {
          const o = ["HOT", "WARM", "COLD"];
          f.sort((a,b) => {
            const al = o.indexOf(a.deployment_level), bl = o.indexOf(b.deployment_level);
            if (al === bl) return (b.schedule?1:0)-(a.schedule?1:0) || a.repo_id.localeCompare(b.repo_id);
            return (al===-1?o.length:al)-(bl===-1?o.length:bl);
          });
        } else if (this.selectedSortBy === "repo_id") f.sort((a,b) => a.repo_id.localeCompare(b.repo_id));
        else if (this.selectedSortBy === "n_params") f.sort((a,b) => (b.n_params||0)-(a.n_params||0));
        return f;
      },
      filteredDeployments() { return this.sortedAndFilteredDeployments; },
      paginatedDeployments() { return this.filteredDeployments.slice((this.currentPage-1)*this.pageSize, this.currentPage*this.pageSize); },
      totalPages() { return Math.ceil(this.filteredDeployments.length / this.pageSize); },
      firstRecord() { return (this.currentPage - 1) * this.pageSize; },
    },
    methods: {
      onSortChange() { this.currentPage = 1; },
      onSearchChange() { this.currentPage = 1; },
      onPageChange(e) { this.currentPage = Math.floor(e.first / this.pageSize) + 1; },
      updateDeployments(data) { this.deployments = Object.values(data.deployments); },
      getStatus() {
        const sb = this.$refs.statusbar;
        sb.update("Fetching NDIF status...", "info");
        fetch(this.ndif_url + "/ping")
          .then(r => {
            if (r.status === 200) {
              sb.update("NDIF is up. Fetching model status...", "info");
              fetch(this.ndif_url + "/status")
                .then(r => r.status === 200 ? r.json().then(d => { sb.update("NDIF is operational", "success"); this.updateDeployments(d); this.calendar_id = d.calendar_id; this.cluster = d.cluster; }) : sb.update("Unable to get status", "danger"))
                .catch(() => sb.update("Unable to get status", "danger"));
            } else sb.update("NDIF is unavailable", "danger");
          })
          .catch(() => sb.update("NDIF is unavailable", "danger"));
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
  app.component("p-divider", PrimeVue.Divider);
  app.component("p-floatlabel", PrimeVue.FloatLabel);
  app.component("p-panel", PrimeVue.Panel);
  app.component("p-card", PrimeVue.Card);
  app.component("p-popover", PrimeVue.Popover);

  app.mount("#status-app");
});
</script>
