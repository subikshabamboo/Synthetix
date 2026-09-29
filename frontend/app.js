/**
 * Synthetix — Autonomous Research Application Logic
 * Direct API Integration with FastAPI & Redis Backend
 */

let currentRunId = null;
let pollTimer = null;
let eventSource = null;
let activeRunData = null;

// Initial bootstrap
document.addEventListener("DOMContentLoaded", () => {
  loadRecentRuns();
  
  // Enter key support for query input
  const queryInput = document.getElementById("queryInput");
  if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        startResearch();
      }
    });
  }
});


/**
 * Focus and scroll to search
 */
function focusSearch() {
  scrollToSection("hero");
  const input = document.getElementById("queryInput");
  if (input) {
    input.focus();
    input.select();
  }
}

/**
 * Set a preset search query
 */
function setPreset(queryText) {
  const input = document.getElementById("queryInput");
  if (input) {
    input.value = queryText;
    input.focus();
  }
}

/**
 * Smooth scrolling
 */
function scrollToSection(sectionId) {
  const el = document.getElementById(sectionId);
  if (el) {
    el.scrollIntoView({ behavior: "smooth" });
  }
  
  // Update active nav button
  document.querySelectorAll(".nav-item-btn").forEach(btn => {
    btn.classList.remove("active");
  });
}

/**
 * Start a new research run
 */
async function startResearch() {
  const input = document.getElementById("queryInput");
  const question = input.value.trim();
  if (!question) {
    alert("Please enter a research topic.");
    return;
  }

  const startBtn = document.getElementById("startResearchBtn");
  startBtn.disabled = true;
  startBtn.innerHTML = `<span class="spin">⚙️</span> Starting...`;

  try {
    const res = await fetch("/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }

    const data = await res.json();
    currentRunId = data.run_id;

    // Update UI for active run
    document.getElementById("currentRunIdBadge").innerText = `Run: ${currentRunId.substring(0, 8)}...`;
    
    // Clear previous view
    resetStudioView();
    scrollToSection("studio");

    // Follow the run: SSE push with polling fallback
    followRun(currentRunId);
    
    // Refresh recent runs list after a brief moment
    setTimeout(loadRecentRuns, 1000);

  } catch (err) {
    alert(`Failed to start research: ${err.message}`);
  } finally {
    startBtn.disabled = false;
    startBtn.innerHTML = `<span>Initiate Run</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>`;
  }
}

function stopLiveUpdates() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

/**
 * Follow a run via SSE (push) with automatic polling fallback.
 *
 * THEORY: EventSource is the primary channel — the server pushes state
 * transitions the moment Redis sees them, so the stepper moves in step
 * with the actual pipeline instead of on a blind 1.5s cadence. EventSource
 * auto-reconnects on network blips; if SSE is fundamentally unavailable
 * (old proxy, non-200 first response) we degrade to the polling loop,
 * which remains fully functional. Only one channel is ever active.
 */
function followRun(runId) {
  stopLiveUpdates();

  if (typeof EventSource !== "undefined") {
    const es = new EventSource(`/research/${runId}/stream`);
    eventSource = es;

    es.addEventListener("state", (ev) => {
      try {
        const state = JSON.parse(ev.data);
        activeRunData = state;
        renderRunState(state);
        const terminal = state.status === "done" || state.status === "failed" ||
          state.status === "failed_citation_validation";
        if (terminal) {
          stopLiveUpdates();
          loadRecentRuns();
        }
      } catch (e) {
        console.error("Bad SSE payload:", e);
      }
    });

    es.addEventListener("gone", () => {
      stopLiveUpdates();
    });

    es.onerror = () => {
      // EventSource retries on its own for transient errors; a hard
      // failure (e.g. proxy that can't stream) surfaces here. Give up
      // after the first error and fall back to polling.
      if (es.readyState === EventSource.CLOSED) {
        stopLiveUpdates();
        startPolling(runId);
      }
    };
    return;
  }

  // No EventSource support at all: plain polling.
  startPolling(runId);
}

/**
 * Start polling the research run status
 */
function startPolling(runId) {
  if (pollTimer) clearInterval(pollTimer);
  
  // Poll immediately, then every 1500ms
  pollRunStatus(runId);
  pollTimer = setInterval(() => {
    pollRunStatus(runId);
  }, 1500);
}

/**
 * Poll a specific run status
 */
async function pollRunStatus(runId) {
  try {
    const res = await fetch(`/research/${runId}`);
    if (!res.ok) {
      if (res.status === 404) {
        clearInterval(pollTimer);
        return;
      }
      throw new Error(`Status ${res.status}`);
    }

    const state = await res.json();
    activeRunData = state;
    renderRunState(state);

    if (state.status === "done" || state.status === "failed" || state.status === "failed_citation_validation") {
      clearInterval(pollTimer);
      pollTimer = null;
      loadRecentRuns(); // update recent runs list with final stats
    }
  } catch (err) {
    console.error("Error polling run status:", err);
  }
}

/**
 * Format a duration (seconds) as "42s" or "1m 23s"
 */
function formatDuration(seconds) {
  if (seconds == null) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

/**
 * Render Run State to UI
 */
function renderRunState(state) {
  document.getElementById("currentRunIdBadge").innerText = `Run: ${state.run_id.substring(0, 10)}`;

  // 1. Update Pipeline Stepper Nodes
  updateStepper(state.status);

  // 2. Update Telemetry
  if (state.budget) {
    const tokens = state.budget.tokens_used || 0;
    const maxTokens = state.budget.max_total_tokens || 150000;
    document.getElementById("teleTokensVal").innerText = tokens.toLocaleString();
    document.getElementById("teleTokensText").innerText = `${tokens.toLocaleString()} / ${maxTokens.toLocaleString()}`;
    const tokenPct = Math.min(100, Math.round((tokens / maxTokens) * 100));
    document.getElementById("teleTokensBar").style.width = `${tokenPct}%`;

    const searches = state.budget.searches_used || 0;
    const maxSearches = (state.budget.max_searches_per_sub_question || 3) * (state.budget.max_sub_questions || 5);
    document.getElementById("teleSearchesVal").innerText = searches;
    document.getElementById("teleSearchesText").innerText = `${searches} / ${maxSearches}`;
    const searchPct = Math.min(100, Math.round((searches / maxSearches) * 100));
    document.getElementById("teleSearchesBar").style.width = `${searchPct}%`;
  }

  // Revision count
  const revCount = state.revision_count || 0;
  document.getElementById("teleRevisionsVal").innerText = `${revCount} / 1`;
  document.getElementById("teleRevisionsBar").style.width = revCount > 0 ? "100%" : "0%";

  // 3. Render Sub-Questions Plan
  renderSubQuestions(state.plan);

  // 4. Render Extracted Findings
  renderFindings(state.results);

  // 5. Render Report
  renderReport(state);

  // 5b. Render Cross-Source Contradictions (auditor output)
  renderConflicts(state.conflicts);

  // 6. Update Trace
  if (state.trace && state.trace.length > 0) {
    renderTrace(state.trace);
  }

  // 7. Show/Hide Resume button if failed
  const resumeBtn = document.getElementById("resumeBtn");
  if (state.status === "failed") {
    resumeBtn.style.display = "inline-flex";
  } else {
    resumeBtn.style.display = "none";
  }
}

/**
 * Update Stepper Node States
 */
function updateStepper(status) {
  // "auditing" is the contradiction-auditor pass between research and
  // writing; visually it maps onto the Reviewing step.
  const steps = ["planning", "researching", "reviewing", "writing", "done"];
  const stepIdx = steps.indexOf(status);
  const displayIdx = status === "auditing" ? steps.indexOf("reviewing") : stepIdx;

  steps.forEach((step, idx) => {
    const el = document.getElementById(`step-${step}`);
    const statusText = document.getElementById(`step-${step}-status`);
    if (!el) return;

    el.className = "step-node";

    if (status === "failed") {
      if (idx === stepIdx || (stepIdx === -1 && idx === 0)) {
        el.classList.add("failed");
        statusText.innerText = "Failed";
      } else {
        statusText.innerText = "Halted";
      }
    } else if (status === "failed_citation_validation") {
      if (step === "writing" || step === "done") {
        el.classList.add("failed");
        statusText.innerText = "Validation Alert";
      }
    } else if (idx < displayIdx || status === "done") {
      el.classList.add("completed");
      statusText.innerText = "Completed";
    } else if (idx === displayIdx) {
      el.classList.add("active");
      statusText.innerText = status === "auditing" ? "Auditing..." : "Active...";
    } else {
      statusText.innerText = "Pending";
    }
  });
}

/**
 * Render Sub-Questions
 */
function renderSubQuestions(plan) {
  const listEl = document.getElementById("subQuestionsList");
  const countEl = document.getElementById("subQuestionsCount");

  if (!plan || !plan.sub_questions || plan.sub_questions.length === 0) {
    countEl.innerText = "0 items";
    return;
  }

  countEl.innerText = `${plan.sub_questions.length} items`;
  listEl.innerHTML = plan.sub_questions.map((sq, i) => `
    <div class="sub-q-card">
      <div class="sub-q-top">
        <span class="sub-q-id">SUB-Q #${i + 1} [${sq.id || 'id'}]</span>
      </div>
      <div class="sub-q-text">${escapeHtml(sq.question)}</div>
      <div class="sub-q-rationale">Rationale: ${escapeHtml(sq.rationale)}</div>
    </div>
  `).join("");
}

/**
 * Render Findings
 */
function renderFindings(results) {
  const listEl = document.getElementById("findingsList");
  const countEl = document.getElementById("findingsCountBadge");
  const teleFindingsVal = document.getElementById("teleFindingsVal");
  const teleFindingsText = document.getElementById("teleFindingsText");
  const teleFindingsBar = document.getElementById("teleFindingsBar");

  if (!results || results.length === 0) {
    if (countEl) countEl.innerText = "0 extracted";
    if (teleFindingsVal) teleFindingsVal.innerText = "0";
    return;
  }

  let totalFindings = 0;
  let allHtml = "";

  results.forEach(res => {
    if (res.findings && res.findings.length > 0) {
      totalFindings += res.findings.length;
      res.findings.forEach(f => {
        const confClass = f.confidence === "high" ? "conf-high" : f.confidence === "medium" ? "conf-medium" : "conf-low";
        const claimText = f.claim || f.finding || "Claim extracted from source.";
        allHtml += `
          <div class="finding-card">
            <div class="finding-header">
              <span class="finding-id-tag">ID: ${escapeHtml(f.id)}</span>
              <span class="finding-confidence ${confClass}">Conf: ${escapeHtml(f.confidence || 'med')}</span>
            </div>
            <div class="finding-summary">${escapeHtml(claimText)}</div>
            ${f.source && f.source.snippet ? `<div class="finding-quote">"${escapeHtml(f.source.snippet)}"</div>` : ''}
            <div>
              <a href="${escapeHtml(f.source ? f.source.url : '#')}" target="_blank" rel="noopener noreferrer" class="finding-source-url">
                🔗 ${escapeHtml((f.source && f.source.title) || (f.source && f.source.url) || 'Source Link')}
              </a>
            </div>
          </div>
        `;
      });
    }
  });

  if (countEl) countEl.innerText = `${totalFindings} extracted`;
  if (teleFindingsVal) teleFindingsVal.innerText = totalFindings;
  if (teleFindingsText) teleFindingsText.innerText = `${totalFindings} atomic records`;
  if (teleFindingsBar) teleFindingsBar.style.width = `${Math.min(100, totalFindings * 10)}%`;

  if (allHtml && listEl) {
    listEl.innerHTML = allHtml;
  }
}

/**
 * Render Synthesized Report
 */
function renderReport(state) {
  const container = document.getElementById("reportContainer");
  if (!container) return;

  if (state.status === "failed_citation_validation") {
    const problems = state.validation_problems || [];
    container.innerHTML = `
      <div style="background: #fef2f2; border: 1px solid #fca5a5; border-radius: var(--radius-md); padding: 1.5rem;">
        <h4 style="color: #991b1b; font-weight: 800; font-size: 1.1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
          <span>⚠️ CITATION VALIDATION FAILED</span>
        </h4>
        <p style="color: #7f1d1d; font-size: 0.9rem; margin-bottom: 1rem;">
          The report was generated, but the defensive validation engine intercepted it because one or more citations do not resolve to verified finding records:
        </p>
        <ul style="color: #991b1b; font-family: var(--font-mono); font-size: 0.8rem; padding-left: 1.5rem; display: flex; flex-direction: column; gap: 0.4rem;">
          ${problems.map(p => `<li>${escapeHtml(p)}</li>`).join("")}
        </ul>
      </div>
    `;
    return;
  }

  if (!state.report) {
    return;
  }

  const report = state.report;
  let sectionsHtml = "";

  if (report.sections && report.sections.length > 0) {
    sectionsHtml = report.sections.map(sec => {
      // Replace citations in body e.g. [finding_id] with clickable chips
      let bodyFormatted = escapeHtml(sec.body);
      return `
        <div class="report-section">
          <h5 class="report-section-title">${escapeHtml(sec.heading)}</h5>
          <div class="report-section-body">${bodyFormatted}</div>
        </div>
      `;
    }).join("");
  }

  // Render Citations Table
  let citationsTableHtml = "";
  if (report.citations && report.citations.length > 0) {
    citationsTableHtml = `
      <div class="citations-table-wrap">
        <table class="citations-table">
          <thead>
            <tr>
              <th>Citation</th>
              <th>Finding ID</th>
              <th>Verified Source URL</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${report.citations.map((c, idx) => `
              <tr>
                <td><span class="citation-chip" onclick="openCitationProvenance('${escapeHtml(c.finding_id)}')">[C${idx + 1}]</span></td>
                <td><code>${escapeHtml(c.finding_id)}</code></td>
                <td><a href="${escapeHtml(c.source_url)}" target="_blank" rel="noopener noreferrer" style="color: #0284c7; text-decoration: none;">${escapeHtml(c.source_url)}</a></td>
                <td><button class="btn-secondary" style="padding: 0.2rem 0.6rem; font-size: 0.72rem;" onclick="openCitationProvenance('${escapeHtml(c.finding_id)}')">Inspect</button></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="report-summary-box">
      <div class="report-summary-title">Executive Summary</div>
      <div class="report-summary-content">${escapeHtml(report.summary)}</div>
    </div>

    ${sectionsHtml}

    ${report.conclusion ? `
      <div class="report-section">
        <h5 class="report-section-title">Conclusion & Strategic Recommendations</h5>
        <div class="report-section-body">${escapeHtml(report.conclusion)}</div>
      </div>
    ` : ''}

    <div style="margin-top: 1rem;">
      <h5 style="font-family: var(--font-display); font-size: 1rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem;">
        Citation Provenance Index (${report.citations ? report.citations.length : 0} citations)
      </h5>
      ${citationsTableHtml}
    </div>
  `;
}

/**
 * Render Cross-Source Contradictions detected by the auditor node.
 * Shown above the report: the auditor flags disagreements BETWEEN sources
 * (negation pairs, incompatible numbers, opposing comparatives) so the
 * reader can weigh them; findings still flow into the report unchanged.
 */
function renderConflicts(conflicts) {
  const container = document.getElementById("reportContainer");
  if (!container || !conflicts || conflicts.length === 0) return;

  // renderReport rebuilds the container on terminal states; in between
  // (status ticks with no report yet) drop any banner we rendered before.
  const existing = document.getElementById("conflictBanner");
  if (existing) existing.remove();

  const REASON_LABELS = {
    negation: "Directly opposing statements",
    numeric_disagreement: "Incompatible numbers for the same metric",
    opposing_comparative: "Opposing comparisons between the same entities",
  };

  const cards = conflicts.map(c => {
    const label = REASON_LABELS[c.reason] || (c.reason || "Conflict").replace(/_/g, " ");
    const row = (side) => `
      <div class="conflict-claim">
        <div class="conflict-claim-meta">
          <code>${escapeHtml(c[side].finding_id)}</code>
          <a href="${escapeHtml(c[side].source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(c[side].source_url)}</a>
        </div>
        <div class="conflict-claim-text">"${escapeHtml(c[side].claim)}"</div>
      </div>
    `;
    return `
      <div class="conflict-card">
        <div class="conflict-reason">⚠ ${escapeHtml(label)}</div>
        ${row("claim_a")}
        <div class="conflict-vs">vs</div>
        ${row("claim_b")}
      </div>
    `;
  }).join("");

  const banner = document.createElement("div");
  banner.id = "conflictBanner";
  banner.innerHTML = `
    <div class="conflict-banner">
      <div class="conflict-banner-title">
        ⚡ ${conflicts.length} cross-source contradiction${conflicts.length > 1 ? "s" : ""} detected
      </div>
      <p class="conflict-banner-sub">
        The auditor flagged the disagreements below between sources. They remain in the report with their citations — weigh them yourself.
      </p>
      ${cards}
    </div>
  `;
  container.prepend(banner);
}

/**
 * Render Step-Level Trace Timeline
 */
function renderTrace(trace) {
  const timeline = document.getElementById("traceTimeline");
  if (!trace || trace.length === 0) return;

  timeline.innerHTML = trace.map(item => {
    const node = item.node || "system";
    let badgeClass = "node-api";
    if (node === "planner") badgeClass = "node-planner";
    else if (node === "researcher") badgeClass = "node-researcher";
    else if (node === "supervisor") badgeClass = "node-supervisor";
    else if (node === "auditor") badgeClass = "node-auditor";
    else if (node === "writer") badgeClass = "node-writer";

    // Trace entries carry their time in `at` (ISO 8601 from graph._log),
    // not `timestamp` — before this fix the timeline rendered blank clocks.
    const timestamp = item.at ? new Date(item.at).toLocaleTimeString() : "";
    const tokens = item.tokens_used_so_far ? `${item.tokens_used_so_far.toLocaleString()} tokens` : "";

    return `
      <div class="trace-step">
        <div class="trace-dot"></div>
        <div class="trace-step-top">
          <span class="trace-node-badge ${badgeClass}">${escapeHtml(node)}</span>
          <span class="trace-timestamp">${timestamp}</span>
        </div>
        <div class="trace-note">${escapeHtml(item.note || "")}</div>
        ${tokens ? `<div class="trace-tokens">Cumulative Spend: ${tokens}</div>` : ''}
      </div>
    `;
  }).join("");
}

/**
 * Refresh Trace manually
 */
async function refreshTrace() {
  if (!currentRunId) return;
  try {
    const res = await fetch(`/research/${currentRunId}/trace`);
    if (res.ok) {
      const data = await res.json();
      renderTrace(data.trace);
    }
  } catch (e) {
    console.error("Failed to refresh trace:", e);
  }
}

/**
 * Load Recent Runs from Redis
 */
async function loadRecentRuns() {
  try {
    const res = await fetch("/research/recent");
    if (!res.ok) return;

    const data = await res.json();
    const listEl = document.getElementById("recentRunsList");
    if (!data.runs || data.runs.length === 0) {
      listEl.innerHTML = `<div style="color: var(--text-light); font-size: 0.85rem; padding: 1rem 0;">No persisted runs found in Redis.</div>`;
      return;
    }

    listEl.innerHTML = data.runs.map(r => {
      const statusColor =
        r.status === "done" ? "#10b981" :
        (r.status === "failed" || r.status === "failed_citation_validation") ? "#ef4444" :
        "#6366f1";
      return `
        <div class="run-history-item" onclick="loadRunById('${r.run_id}')">
          <div class="run-history-top">
            <span style="font-family: var(--font-mono); font-size: 0.72rem; color: #64748b;">${r.run_id.substring(0, 8)}...</span>
            <span style="font-size: 0.7rem; font-weight: 700; color: ${statusColor}; text-transform: uppercase;">${r.status}</span>
          </div>
          <div class="run-history-title">${escapeHtml(r.question)}</div>
          <div class="run-history-stats">
            <span>⚡ ${r.tokens_used.toLocaleString()} tok</span>
            <span>🔍 ${r.searches_used} searches</span>
            <span>📑 ${r.findings_count} findings</span>
            ${r.duration_seconds != null ? `<span>⏱ ${formatDuration(r.duration_seconds)}${r.status === "done" || r.status === "failed" ? "" : "…"}</span>` : ""}
          </div>
        </div>
      `;
    }).join("");

  } catch (err) {
    console.error("Failed to load recent runs:", err);
  }
}

/**
 * Load a run by ID when clicked from history
 */
async function loadRunById(runId) {
  currentRunId = runId;
  scrollToSection("studio");
  resetStudioView();
  followRun(runId);
}

/**
 * Resume a crashed / halted run
 */
async function resumeCurrentRun() {
  if (!currentRunId) return;
  try {
    const res = await fetch(`/research/${currentRunId}/resume`, { method: "POST" });
    if (res.ok) {
      followRun(currentRunId);
    }
  } catch (err) {
    alert(`Resume failed: ${err.message}`);
  }
}

/**
 * Citation Sandbox: Corrupt Citation in Redis
 */
async function corruptCitationTest() {
  if (!currentRunId) {
    alert("Please run or load a completed research report first.");
    return;
  }
  try {
    const res = await fetch(`/research/${currentRunId}/corrupt-citation`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to corrupt citation");
    }
    // Re-poll status immediately to see failure
    pollRunStatus(currentRunId);
  } catch (err) {
    alert(`Corrupt test failed: ${err.message}`);
  }
}

/**
 * Citation Sandbox: Restore / Re-verify
 */
function restoreCitationTest() {
  if (currentRunId) {
    pollRunStatus(currentRunId);
  }
}

/**
 * Open Citation Provenance Modal
 */
function openCitationProvenance(findingId) {
  if (!activeRunData) return;

  // Find finding across all results
  let matchedFinding = null;
  if (activeRunData.results) {
    for (const r of activeRunData.results) {
      if (r.findings) {
        for (const f of r.findings) {
          if (f.id === findingId) {
            matchedFinding = f;
            break;
          }
        }
      }
      if (matchedFinding) break;
    }
  }

  const modal = document.getElementById("provenanceModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");

  modalTitle.innerText = `Citation Provenance: ${findingId}`;

  if (!matchedFinding) {
    modalBody.innerHTML = `
      <div style="color: #991b1b; background: #fee2e2; padding: 1rem; border-radius: var(--radius-md);">
        <strong>Validation Anomaly:</strong> Finding ID <code>${escapeHtml(findingId)}</code> was not found in the run's extracted evidence results.
      </div>
    `;
  } else {
    const claimVal = matchedFinding.claim || matchedFinding.finding || "Extracted claim";
    const snippetVal = (matchedFinding.source && matchedFinding.source.snippet) || matchedFinding.quote || "";
    modalBody.innerHTML = `
      <div>
        <label style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase;">Extracted Finding Claim</label>
        <div style="font-size: 0.95rem; font-weight: 600; color: var(--text-main); margin-top: 0.25rem;">
          ${escapeHtml(claimVal)}
        </div>
      </div>

      ${snippetVal ? `
        <div>
          <label style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase;">Original Verbatim Source Quote</label>
          <div style="background: #f8fafc; border-left: 3px solid #0284c7; padding: 0.75rem 1rem; font-family: var(--font-mono); font-size: 0.82rem; color: #334155; margin-top: 0.25rem;">
            "${escapeHtml(snippetVal)}"
          </div>
        </div>
      ` : ''}

      <div>
        <label style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--text-muted); text-transform: uppercase;">Verified Source</label>
        <div style="margin-top: 0.25rem;">
          <a href="${escapeHtml((matchedFinding.source && matchedFinding.source.url) || '#')}" target="_blank" rel="noopener noreferrer" style="color: #0284c7; font-size: 0.88rem; font-weight: 600; text-decoration: none;">
            🔗 ${escapeHtml((matchedFinding.source && (matchedFinding.source.title || matchedFinding.source.url)) || 'Source Link')}
          </a>
        </div>
      </div>

      <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-family: var(--font-mono); color: var(--text-muted); padding-top: 0.75rem; border-top: 1px solid var(--border-color);">
        <span>Confidence: <strong>${escapeHtml(matchedFinding.confidence || 'high')}</strong></span>
        <span>ID: <code>${escapeHtml(matchedFinding.id)}</code></span>
      </div>
    `;
  }

  modal.classList.add("active");
}

function closeProvenanceModal(e) {
  const modal = document.getElementById("provenanceModal");
  modal.classList.remove("active");
}

/**
 * Copy / Export Helpers
 */
function copyCurrentRunId() {
  if (!currentRunId) {
    alert("No active run ID.");
    return;
  }
  navigator.clipboard.writeText(currentRunId);
  alert(`Copied run ID: ${currentRunId}`);
}

function copyReportMarkdown() {
  if (!activeRunData || !activeRunData.report) {
    alert("No report to copy.");
    return;
  }
  const r = activeRunData.report;
  let md = `# Research Report: ${activeRunData.question}\n\n`;
  md += `## Executive Summary\n${r.summary}\n\n`;
  if (r.sections) {
    r.sections.forEach(s => {
      md += `## ${s.heading}\n${s.body}\n\n`;
    });
  }
  if (r.conclusion) {
    md += `## Conclusion\n${r.conclusion}\n\n`;
  }
  if (r.citations) {
    md += `## Citations\n`;
    r.citations.forEach((c, idx) => {
      md += `- [C${idx + 1}] (${c.finding_id}) ${c.source_url}\n`;
    });
  }
  navigator.clipboard.writeText(md);
  alert("Report copied to clipboard as Markdown!");
}

function exportReportJSON() {
  if (!activeRunData) {
    alert("No active run data.");
    return;
  }
  const blob = new Blob([JSON.stringify(activeRunData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `research_run_${currentRunId || 'export'}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function resetStudioView() {
  document.getElementById("subQuestionsList").innerHTML = `<div style="color: var(--text-light); font-size: 0.88rem; font-style: italic; padding: 1rem 0;">Decomposing research query...</div>`;
  document.getElementById("findingsList").innerHTML = `<div style="color: var(--text-light); font-size: 0.88rem; font-style: italic; padding: 1rem 0;">Executing web queries & extracting findings...</div>`;
  document.getElementById("reportContainer").innerHTML = `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 350px; color: var(--text-light); text-align: center;">
      <div class="spin" style="font-size: 2rem; margin-bottom: 1rem;">⚙️</div>
      <p style="font-weight: 600; font-size: 1rem; color: var(--text-muted); margin-bottom: 0.25rem;">Research in Progress</p>
      <p style="font-size: 0.85rem; max-width: 320px;">The multi-agent pipeline is executing across planning, search, supervisor review, and synthesis.</p>
    </div>
  `;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
