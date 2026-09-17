(() => {
  "use strict";

  const state = {
    caseId: null,
    lastResponse: null,
    selectedFile: null,
    objectUrl: null,
    cameraStream: null,
    cameraSessionId: null,
    lastDocumentFile: null,
    docZoom: 1,
    identityGateRequired: false,
    identityComplete: false,
    autoCaptureArmed: false,
    qualityTimer: null,
    readyStreak: 0,
  };

  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const previewWrap = $("preview-wrap");
  const preview = $("preview");
  const fileName = $("file-name");
  const fileMeta = $("file-meta");
  const btnAnalyze = $("btn-analyze");
  const btnRemove = $("btn-remove");
  const btnClear = $("btn-clear");
  const statusEl = $("status");
  const analyzeProgress = $("analyze-progress");
  const emptyState = $("empty-state");
  const resultCard = $("result-card");
  const liveBanner = $("live-verify-banner");

  const ALLOWED = new Set([
    "image/png", "image/jpeg", "image/webp", "image/jpg", "application/pdf",
  ]);

  function pct(v) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return `${(Number(v) * 100).toFixed(1)}%`;
  }

  function setStatus(msg, kind) {
    if (!statusEl) return;
    statusEl.textContent = msg || "";
    statusEl.classList.remove("is-error", "is-ok");
    if (kind) statusEl.classList.add(kind);
  }

  function setCameraState(s) {
    const el = $("camera-state");
    if (el) el.textContent = `CAMERA: ${s}`;
  }

  function showProgress(steps) {
    if (!analyzeProgress) return;
    analyzeProgress.classList.remove("is-hidden");
    analyzeProgress.innerHTML = "";
    steps.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      analyzeProgress.appendChild(li);
    });
  }

  /* —— Navigation —— */
  function showView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    document.querySelectorAll(".side-link").forEach((b) => {
      b.classList.toggle("is-active", b.dataset.view === name);
    });
    const view = $(`view-${name}`);
    if (view) view.classList.add("is-active");
    requestAnimationFrame(() => {
      window.DocshieldThree?.resizeAll?.();
      window.dispatchEvent(new Event("resize"));
      if (name === "result" && state.lastResponse) {
        renderCompareAndRegions(state.lastResponse);
      }
    });
  }

  document.querySelectorAll(".side-link").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });
  $("btn-start-screening")?.addEventListener("click", () => showView("screening"));
  $("btn-sidebar")?.addEventListener("click", () => {
    $("sidebar")?.classList.toggle("is-collapsed");
  });
  $("btn-low-gfx")?.addEventListener("click", (e) => {
    const on = localStorage.getItem("docshield_low_gfx") === "1";
    localStorage.setItem("docshield_low_gfx", on ? "0" : "1");
    e.currentTarget.setAttribute("aria-pressed", on ? "false" : "true");
    location.reload();
  });
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.body.classList.add("reduced-motion");
  }

  /* —— System status from /api/health —— */
  async function refreshHealth() {
    const box = $("sys-status");
    if (!box) return;
    box.innerHTML = "";
    try {
      const h = await fetch("/api/health").then((r) => r.json());
      if (h.demo_mode) $("demo-chip")?.classList.remove("is-hidden");
      const map = [
        ["AI", h.components?.efficientnet || h.components?.ml_mode],
        ["OCR", h.components?.ocr],
        ["FORENSICS", h.components?.tampering],
        ["FACE", h.components?.face_detection],
        ["CAMERA", h.components?.live_camera],
        ["API", h.components?.api],
      ];
      map.forEach(([label, val]) => {
        const span = document.createElement("span");
        span.className = "sys-dot";
        const v = String(val || "").toLowerCase();
        let state = "LOADING";
        if (!v) {
          state = "OFFLINE";
          span.classList.add("is-offline");
        } else if (v.includes("not_connected") || v.includes("unavailable") || v.includes("not_implemented") || v.includes("disabled") || v.includes("not_configured")) {
          state = v.includes("not_connected") || v.includes("not_configured") ? "NOT_CONFIGURED" : "DEGRADED";
          span.classList.add("is-degraded");
        } else if (v.includes("working") || v === "ok" || v.includes("mock")) {
          state = "ONLINE";
          span.classList.add("is-online");
        } else {
          state = "DEGRADED";
          span.classList.add("is-degraded");
        }
        span.textContent = `${label} ${state}`;
        span.title = `${label}: ${val || "UNKNOWN"} (${state})`;
        span.setAttribute("aria-label", `${label} ${state}`);
        box.appendChild(span);
      });
    } catch {
      const span = document.createElement("span");
      span.className = "sys-dot is-offline";
      span.textContent = "API OFFLINE";
      box.appendChild(span);
    }
  }

  /* —— Fill helpers —— */
  function fillKv(listEl, rows) {
    if (!listEl) return;
    listEl.innerHTML = "";
    (rows || []).forEach(([k, v]) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${k}</span><strong>${v ?? "—"}</strong>`;
      listEl.appendChild(li);
    });
  }

  function fillQuality(q) {
    const s = (q && q.scores) || {};
    fillKv($("quality-list"), [
      ["Overall", s.overall_quality != null ? `${s.overall_quality}%` : pct(q?.overall_quality)],
      ["Resolution", q?.resolution],
      ["Blur", q?.blur],
      ["Brightness", q?.brightness],
      ["Perspective", q?.perspective],
      ["Noise", q?.noise],
      ["Crop", q?.crop],
      ["Visibility", q?.document_visibility],
    ]);
  }

  function fillFields(fieldsPayload, mrz, ocr) {
    const fieldsList = $("fields-list");
    if (!fieldsList) return;
    fieldsList.innerHTML = "";
    const rows = (fieldsPayload && (fieldsPayload.fields || fieldsPayload.field_inventory)) || [];
    if (ocr) {
      const eng = document.createElement("li");
      eng.innerHTML = `<span>OCR engine</span><strong>${escapeHtml(ocr.engine || "—")} · ${escapeHtml(ocr.status || "—")} · boxes ${ocr.text_box_count ?? "—"}</strong>`;
      fieldsList.appendChild(eng);
      if (ocr.mean_ocr_confidence != null) {
        const conf = document.createElement("li");
        conf.innerHTML = `<span>Mean OCR confidence</span><strong>${(Number(ocr.mean_ocr_confidence) * 100).toFixed(1)}%</strong>`;
        fieldsList.appendChild(conf);
      }
    }
    if (fieldsPayload?.field_completeness != null) {
      const li = document.createElement("li");
      li.innerHTML = `<span>Completeness</span><strong>${(fieldsPayload.field_completeness * 100).toFixed(1)}%</strong>`;
      fieldsList.appendChild(li);
    }
    if (!rows.length) {
      const li = document.createElement("li");
      const why = (ocr && (ocr.engine === "mock" || !ocr.text_box_count))
        ? `${fieldsPayload?.status || "NO_FIELDS"} — OCR returned no text (install/enable Tesseract or use a clearer scan)`
        : (fieldsPayload?.status || "NONE");
      li.innerHTML = `<span>Fields</span><strong>${escapeHtml(why)}</strong>`;
      fieldsList.appendChild(li);
    } else {
      rows.forEach((f) => {
        const li = document.createElement("li");
        const conf =
          f.ocr_confidence != null && f.status !== "MISSING"
            ? ` · OCR ${(Number(f.ocr_confidence) * 100).toFixed(0)}%`
            : "";
        let status = f.status || f.validation || "—";
        if (f.ocr_confidence != null && Number(f.ocr_confidence) < 0.15 && status === "PASS") {
          status = "UNCERTAIN";
        }
        const val = f.status === "MISSING" || f.status === "NOT_APPLICABLE" ? "—" : (f.masked_value || f.value || "—");
        li.innerHTML = `<span>${escapeHtml(f.field || f.name || "field")}${conf}</span><strong>${escapeHtml(String(val))} · ${escapeHtml(status)}</strong>`;
        fieldsList.appendChild(li);
      });
    }
    const mrzStatus = $("mrz-status");
    const mrzChecklist = $("mrz-checklist");
    if (mrz) {
      if (mrzStatus) mrzStatus.textContent = `MRZ: ${mrz.status || "—"}`;
      if (mrzChecklist) {
        mrzChecklist.innerHTML = "";
        if (mrz.status === "NOT_APPLICABLE") {
          const li = document.createElement("li");
          li.innerHTML = `<span>MRZ checks</span><strong>NOT_APPLICABLE for this document type</strong>`;
          mrzChecklist.appendChild(li);
        } else {
          (mrz.checklist || []).forEach((row) => {
            const li = document.createElement("li");
            li.innerHTML = `<span>${escapeHtml(row.label || row.item)}</span><strong>${escapeHtml(row.status || "—")}</strong>`;
            mrzChecklist.appendChild(li);
          });
        }
      }
    } else if (mrzStatus) mrzStatus.textContent = "";
  }

  function fillVision(ai, fusion) {
    const list = $("vision-list");
    if (!list) return;
    const en = (ai && ai.efficientnet) || {};
    const vit = (ai && ai.vit) || {};
    const clip = (ai && ai.clip) || {};
    const eh = en.authenticity_head || {};
    const vh = vit.authenticity_head || {};
    const cs = clip.supporting_evidence || {};
    fillKv(list, [
      ["Vision status", ai?.status || "—"],
      ["EfficientNet", en.status || "—"],
      ["EffNet head", eh.status || "NOT_SCORED"],
      ["EffNet auth (uncal)", eh.authentic_probability != null ? pct(eh.authentic_probability) : "—"],
      ["ViT", vit.status || "—"],
      ["ViT head", vh.status || "NOT_SCORED"],
      ["ViT auth (uncal)", vh.authentic_probability != null ? pct(vh.authentic_probability) : "—"],
      ["CLIP", clip.status || "—"],
      ["CLIP align", (cs.document_type_alignment || {}).agreement || "—"],
      ["Fusion", fusion?.status || "—"],
      ["Fusion auth (raw)", fusion?.authentic_probability != null ? pct(fusion.authentic_probability) : "—"],
      ["Concern", fusion?.concern || eh.concern || "—"],
    ]);
    const mc = $("model-compare");
    if (mc) {
      mc.innerHTML = "";
      [
        ["EfficientNet", eh.authentic_probability, eh.status || en.status],
        ["ViT", vh.authentic_probability, vh.status || vit.status],
        ["CLIP top", (cs.document_type_alignment || {}).top_probability, cs.status || clip.status],
        ["Fusion", fusion?.authentic_probability, fusion?.status],
      ].forEach(([name, val, st]) => {
        const div = document.createElement("div");
        div.className = "score";
        const barW = val != null ? Math.round(Number(val) * 100) : 0;
        div.innerHTML = `<span>${name}</span><strong>${st || "NOT_AVAILABLE"}</strong>
          <div class="bar" aria-hidden="true"><i style="width:${val != null ? barW : 0}%"></i></div>
          <span class="tiny muted">${val != null ? pct(val) : "NOT_AVAILABLE"}</span>`;
        mc.appendChild(div);
      });
    }
  }

  function fillTampering(tamp) {
    const tamperList = $("tamper-list");
    const heatmapWrap = $("heatmap-wrap");
    const heatmapPreview = $("heatmap-preview");
    if (!tamperList) return;
    if (heatmapWrap) heatmapWrap.classList.add("is-hidden");
    const sig = (tamp && tamp.signals) || {};
    const loc = (tamp && tamp.localization) || {};
    fillKv(tamperList, [
      ["Status", tamp?.status || "NOT_ASSESSED"],
      ["Severity", tamp?.severity || "—"],
      ["Review score", tamp?.review_score != null ? pct(tamp.review_score) : "—"],
      ["ELA", sig.ela_inconsistency != null ? pct(sig.ela_inconsistency) : "—"],
      ["Copy-move", sig.copy_move_indicators != null ? pct(sig.copy_move_indicators) : "—"],
      ["Noise", sig.noise_inconsistency != null ? pct(sig.noise_inconsistency) : "—"],
      ["Flags", (tamp?.flags || []).join(", ") || "none"],
      ["Localization", loc.status || "NOT_ASSESSED"],
    ]);
    if (loc.status === "LOCALIZED" && loc.overlay_png_base64 && heatmapWrap && heatmapPreview) {
      heatmapPreview.src = `data:image/png;base64,${loc.overlay_png_base64}`;
      heatmapWrap.classList.remove("is-hidden");
    }
  }

  function fillFace(face) {
    const primary = (face?.faces && face.faces[0]) || null;
    const q = (primary && primary.quality) || {};
    const ver = face?.verification || {};
    const live = face?.liveness || {};
    const docF = face?.document_face || {};
    const liveF = face?.live_face || {};
    const dbg = face?.verification_debug || {};
    fillKv($("face-list"), [
      ["Document face", docF.status || face?.document_face_status || face?.status || "NOT_ASSESSED"],
      ["Live face", liveF.status || (face?.camera?.status === "CAPTURED" ? face?.status : "—") || "—"],
      ["Face count (live)", liveF.face_count != null ? String(liveF.face_count) : (face?.face_count != null ? String(face.face_count) : "—")],
      ["Quality", q.label || (liveF.primary_quality && liveF.primary_quality.label) || "—"],
      ["Liveness", live.status || "NOT_ASSESSED"],
      ["Liveness decision", live.decision || "—"],
      ["Verification result", ver.result || "NOT_AVAILABLE"],
      ["Calibration", ver.calibration || (ver.calibrated === false ? "UNCALIBRATED" : "NOT_ASSESSED")],
      ["Similarity (not identity %)", ver.similarity != null ? pct(ver.similarity) : "—"],
      ["Metric", ver.metric || dbg.metric || "—"],
      ["Match threshold", ver.threshold_match_min != null ? String(ver.threshold_match_min) : "—"],
      ["Embedding sources", ver.embedding_sources || dbg.embedding_sources || "—"],
      ["Age-robustness", ver.age_robustness || "NOT_VALIDATED"],
      ["Concern", face?.concern || "—"],
      ["Note", "DETECTED ≠ MATCH · Liveness ≠ identity"],
    ]);
  }

  function fillCrossValidation(cv) {
    if (!cv) return fillKv($("crossval-list"), [["Status", "NOT_ASSESSED"]]);
    fillKv($("crossval-list"), [
      ["Status", cv.status],
      ["Overall", cv.overall],
      ["Severity", cv.severity],
      ["Pass rate", cv.pass_rate != null ? pct(cv.pass_rate) : "—"],
      ["Concern", cv.concern],
    ]);
    (cv.checks || []).slice(0, 12).forEach((c) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${c.item}</span><strong>${c.status}</strong>`;
      $("crossval-list")?.appendChild(li);
    });
  }

  function fillCalibration(cal) {
    fillKv($("calib-list"), [
      ["Status", cal?.status || "NOT_ASSESSED"],
      ["Raw fusion", cal?.raw_authentic_probability != null ? pct(cal.raw_authentic_probability) : "—"],
      ["Calibrated", cal?.calibrated_authentic_probability != null ? pct(cal.calibrated_authentic_probability) : "—"],
      ["Proxy", cal?.proxy_calibrated === true ? "yes" : "no"],
      ["Production", cal?.production_calibrated === true ? "yes" : "no"],
      ["Concern", cal?.concern || "—"],
    ]);
  }

  function fillConfidence(ce) {
    const list = $("conf-list");
    if (!list) return;
    list.innerHTML = "";
    if (!ce) return fillKv(list, [["Status", "NOT_ASSESSED"]]);
    fillKv(list, [["Overall", ce.overall_confidence != null ? pct(ce.overall_confidence) : "—"]]);
    (ce.factors || []).forEach((f) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${f.name}</span><strong>${pct(f.value)}</strong>`;
      list.appendChild(li);
    });
  }

  function fillExplain(ex) {
    const mc = (ex && ex.model_comparison) || {};
    fillKv($("explain-list"), [
      ["Status", ex?.status || "—"],
      ["EffNet", mc.efficientnet_auth != null ? pct(mc.efficientnet_auth) : "—"],
      ["ViT", mc.vit_auth != null ? pct(mc.vit_auth) : "—"],
      ["Fusion raw", mc.fusion_auth_raw != null ? pct(mc.fusion_auth_raw) : "—"],
      ["Proxy estimate", mc.authenticity_estimate_proxy != null ? pct(mc.authenticity_estimate_proxy) : "—"],
      ["Heatmap", ex?.heatmap_available ? "available" : "n/a"],
    ]);
  }

  function fillForensics(fmap, textVisual, stamp, meta, photo) {
    const tv = textVisual || {};
    const typo = tv.typography || {};
    fillKv($("forensics-list"), [
      ["Map", fmap?.status || "—"],
      ["Typography", typo.status || "—"],
      ["Text anomalies", typo.anomalies ? String(typo.anomalies.length) : "0"],
      ["Text splicing", (tv.text_splicing || {}).status || "—"],
      ["Photo", photo?.detection || photo?.status || "—"],
      ["Stamp", stamp?.detection || "—"],
      ["Metadata", meta?.status || "—"],
    ]);
    const bars = $("typo-bars");
    if (bars) {
      bars.innerHTML = "";
      (typo.anomalies || []).slice(0, 6).forEach((a) => {
        const d = document.createElement("div");
        d.style.marginBottom = "0.5rem";
        d.innerHTML = `<div class="tiny muted">${a.text || "region"} · ${a.status}</div>
          <div class="bar"><i style="width:${Math.min(100, (a.relative_height || 0) * 2000)}%;background:var(--warn)"></i></div>`;
        bars.appendChild(d);
      });
      if (!(typo.anomalies || []).length) {
        bars.innerHTML = `<p class="muted tiny">${typo.status === "ANALYZED" ? "CONSISTENT — no relative-size anomalies flagged" : typo.status || "NOT_ASSESSED"}</p>`;
      }
    }
  }

  function openDrawer(title, html) {
    const root = $("drawer-root");
    if (!root) return;
    $("drawer-title").textContent = title;
    $("drawer-body").innerHTML = html;
    root.classList.remove("is-hidden");
  }
  function closeDrawer() {
    $("drawer-root")?.classList.add("is-hidden");
  }
  $("drawer-close")?.addEventListener("click", closeDrawer);
  $("drawer-backdrop")?.addEventListener("click", closeDrawer);

  function buildResultCards(data, targetId) {
    const wrap = $(targetId || "result-cards");
    if (!wrap) return;
    wrap.innerHTML = "";
    const ev = data.evidence || {};
    const cards = [
      ["DOCUMENT", data.document_type?.label || "—", data.document_type?.label],
      ["OCR", ev.ocr?.status || "—", `boxes ${ev.ocr?.text_box_count ?? "—"}`],
      ["FORENSICS", ev.tampering?.severity || ev.tampering?.status || "—", `${(ev.tampering?.flags || []).length} flags`],
      ["IDENTITY", ev.face?.status || "—", `faces ${ev.face?.face_count ?? 0}`],
      ["AI MODELS", ev.ai_forensics?.status || "—", ev.fusion?.status || "—"],
      ["CROSS-VAL", ev.cross_validation?.overall || "—", ev.cross_validation?.status || "—"],
    ];
    cards.forEach(([title, status, meta]) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "result-card";
      btn.innerHTML = `<p class="rc-title">${title}</p><p class="rc-status">${status}</p><p class="rc-meta">${meta} · VIEW DETAILS</p>`;
      btn.addEventListener("click", () => {
        openDrawer(title, `<pre class="mono tiny" style="white-space:pre-wrap">${escapeHtml(JSON.stringify(pickEvidence(title, ev, data), null, 2))}</pre>`);
      });
      wrap.appendChild(btn);
    });
  }

  function pickEvidence(title, ev, data) {
    switch (title) {
      case "DOCUMENT": return data.document_type || ev.document_type;
      case "OCR": return { ocr: ev.ocr, fields: ev.fields, mrz: ev.mrz };
      case "FORENSICS": return { tampering: ev.tampering, forensics: ev.forensics, text_visual: ev.text_visual };
      case "IDENTITY": return ev.face;
      case "AI MODELS": return { ai: ev.ai_forensics, fusion: ev.fusion };
      case "CROSS-VAL": return ev.cross_validation;
      default: return ev;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* —— Officer-facing verdict (prefers backend officer_verdict; never invents evidence) —— */
  function mapOfficerVerdict(decision, evidence, message, officer) {
    if (officer && officer.label) {
      const code = officer.code || "";
      let tone = "muted";
      let icon = "○";
      if (code.startsWith("PASS") || decision === "PASS") { tone = "ok"; icon = "✓"; }
      else if (code === "TAMPERING_INDICATED" || code.includes("TAMPER")) { tone = "danger"; icon = "⚠"; }
      else if (code === "SUSPICIOUS") { tone = "danger"; icon = "✕"; }
      else if (code.includes("REVIEW") || decision === "REVIEW") { tone = "warn"; icon = "⚠"; }
      else if (decision === "HIGH-RISK") { tone = "danger"; icon = "⚠"; }
      return {
        key: code || decision,
        label: officer.label,
        icon,
        tone,
        blurb: officer.summary || message || "",
        officer,
      };
    }
    const tamp = (evidence && evidence.tampering) || {};
    const elevated = tamp.severity === "ELEVATED_REVIEW" || (tamp.flags || []).length > 0;
    if (decision === "PASS") {
      return {
        key: "PASS_AUTHENTIC",
        label: "PASS — AUTHENTIC",
        icon: "✓",
        tone: "ok",
        blurb: message || "Document evidence is consistent. Screening assist only.",
      };
    }
    if (decision === "INCONCLUSIVE") {
      return {
        key: "INCONCLUSIVE",
        label: "INCONCLUSIVE",
        icon: "○",
        tone: "muted",
        blurb: message || "Insufficient evidence for a reliable screening decision.",
      };
    }
    if (decision === "REVIEW") {
      return {
        key: "INCONCLUSIVE_REVIEW",
        label: "INCONCLUSIVE — REVIEW REQUIRED",
        icon: "⚠",
        tone: "warn",
        blurb: message || "Evidence requires officer assessment before release.",
      };
    }
    if (decision === "HIGH-RISK") {
      if (elevated) {
        return {
          key: "TAMPERING_INDICATED",
          label: "TAMPERING INDICATED",
          icon: "⚠",
          tone: "danger",
          blurb: message || "Evidence indicates suspicious regions requiring review.",
        };
      }
      return {
        key: "SUSPICIOUS",
        label: "SUSPICIOUS / FAKE INDICATION",
        icon: "✕",
        tone: "danger",
        blurb: message || "Elevated screening risk. Officer review required.",
      };
    }
    return {
      key: "PENDING",
      label: decision || "PENDING",
      icon: "●",
      tone: "info",
      blurb: message || "Awaiting analysis.",
    };
  }

  function renderEvidenceDecisionPanel(officer) {
    const panel = $("evidence-decision-panel");
    if (!panel) return;
    if (!officer) {
      panel.classList.add("is-hidden");
      return;
    }
    panel.classList.remove("is-hidden");
    const docList = $("edp-document");
    const idList = $("edp-identity");
    const conflictList = $("edp-conflicts");
    const renderChecks = (el, checks) => {
      if (!el) return;
      el.innerHTML = "";
      (checks || []).forEach((c) => {
        const li = document.createElement("li");
        const mark = c.status === "OK" ? "✓" : c.status === "FAIL" ? "✕" : c.status === "WARN" ? "⚠" : "○";
        li.innerHTML = `<span>${mark} ${escapeHtml(c.item)}</span><strong>${escapeHtml(c.detail || c.status)}</strong>`;
        el.appendChild(li);
      });
      if (!(checks || []).length) el.innerHTML = `<li><span>○</span><strong>NOT AVAILABLE</strong></li>`;
    };
    renderChecks(docList, officer.document_checks);
    renderChecks(idList, officer.identity_checks);
    if (conflictList) {
      conflictList.innerHTML = "";
      const conflicts = officer.conflicts || [];
      if (!conflicts.length) {
        conflictList.innerHTML = `<li><span>✓</span><strong>None</strong></li>`;
      } else {
        conflicts.forEach((c) => {
          const li = document.createElement("li");
          li.innerHTML = `<span>⚠</span><strong>${escapeHtml(c)}</strong>`;
          conflictList.appendChild(li);
        });
      }
    }
    if ($("edp-doc-status")) $("edp-doc-status").textContent = officer.document_status || "—";
    if ($("edp-id-status")) $("edp-id-status").textContent = officer.identity_status || "—";
    if ($("edp-id-cal")) $("edp-id-cal").textContent = officer.identity_calibration || "NOT_ASSESSED";
  }

  function buildWhyItems(data) {
    const a = data.assessment || {};
    const ev = data.evidence || {};
    const items = [];
    const q = data.image_quality || {};
    if (q.insufficient_for_analysis || (q.overall_quality != null && q.overall_quality < 0.45)) {
      items.push({ sev: "REVIEW", title: "Image quality may affect analysis", detail: q.reason || "Low evidence quality for reliable screening.", tone: "review" });
    }
    if (q.blur === "HIGH" || q.resolution === "POOR") {
      const bits = [];
      if (q.resolution === "POOR") bits.push("LOW RESOLUTION");
      if (q.blur === "HIGH") bits.push("EXCESSIVE BLUR");
      items.push({ sev: "REVIEW", title: bits.join(" · "), detail: "Do not treat low quality as proof of tampering.", tone: "review" });
    }
    const tamp = ev.tampering || {};
    if (tamp.severity === "ELEVATED_REVIEW" || (tamp.flags || []).length) {
      items.push({
        sev: "HIGH",
        title: "Forensic / photo anomaly signals",
        detail: `Severity ${tamp.severity || "—"}${tamp.flags?.length ? ` · flags: ${tamp.flags.join(", ")}` : ""} · review signals only.`,
        tone: "high",
      });
    }
    const loc = tamp.localization || {};
    if (loc.status === "LOCALIZED" && (loc.hotspot_count || 0) > 0) {
      items.push({
        sev: "HIGH",
        title: `${loc.hotspot_count} localized hotspot region(s)`,
        detail: "Heatmap is heuristic review only — not calibrated fraud proof.",
        tone: "high",
      });
    }
    const typo = (ev.text_visual || {}).typography || {};
    if (typo.status === "ANALYZED") {
      const label = typo.consistency_label || ((typo.anomalies || []).length ? "ANOMALY_SIGNAL" : "CONSISTENT");
      if (label === "CONSISTENT") {
        items.push({
          sev: "PASS",
          title: "Text scale / spacing consistent",
          detail: `Typography review: ${typo.region_count || 0} regions · alignment σ=${typo.left_alignment_std ?? "n/a"} · no relative-height anomalies.`,
          tone: "pass",
        });
      } else {
        items.push({
          sev: "MEDIUM",
          title: "Text scale / spacing anomaly signal",
          detail: `${(typo.anomalies || []).length} region(s) differ from median height — review signal, not font-family proof.`,
          tone: "medium",
        });
      }
    } else if (typo.status && typo.status !== "NOT_ASSESSED") {
      items.push({
        sev: "INFO",
        title: `Typography ${typo.status}`,
        detail: typo.note || "Text visual metrics limited for this image.",
        tone: "info",
      });
    }
    const splice = (ev.text_visual || {}).text_splicing || {};
    const anomalies = (splice.regions || []).filter((r) => r.status === "ANOMALY_DETECTED");
    if (anomalies.length) {
      items.push({
        sev: "MEDIUM",
        title: "Text-region visual inconsistency",
        detail: `${anomalies.length} region(s) flagged by text forensics (review signal).`,
        tone: "medium",
      });
    }
    const photo = ev.photo_splicing || {};
    if (photo.status && photo.status !== "NOT_APPLICABLE" && photo.status !== "NOT_ASSESSED" && photo.detection === "ANOMALY_DETECTED") {
      items.push({ sev: "HIGH", title: "Photo-region anomaly", detail: photo.note || "Photo forensics reported an anomaly.", tone: "high" });
    }
    const cv = ev.cross_validation || {};
    if (cv.overall === "FAIL" || cv.overall === "WARNING") {
      items.push({
        sev: "REVIEW",
        title: `Cross-validation ${cv.overall}`,
        detail: `Pass rate ${cv.pass_rate != null ? pct(cv.pass_rate) : "N/A"} · review only (not authenticity).`,
        tone: "review",
      });
    }
    const face = ev.face || {};
    if (face.status === "DETECTED" || face.status === "MULTIPLE_DETECTED") {
      const ver = face.verification || {};
      items.push({
        sev: ver.status === "MATCH" ? "PASS" : "INFO",
        title: `Document face ${face.status}`,
        detail: `Verification: ${ver.status || "NOT_ASSESSED"} · age-robustness ${(ver.age_robustness || "NOT_VALIDATED")}`,
        tone: ver.status === "MATCH" ? "pass" : "info",
      });
    } else if (face.status === "NOT_DETECTED") {
      items.push({ sev: "INFO", title: "No document face detected", detail: "Identity 1:1 compare not assessed on this image.", tone: "info" });
    }
    if (!items.length && (a.reasons || []).length) {
      (a.reasons || []).slice(0, 5).forEach((r) => {
        items.push({ sev: "INFO", title: "Assessment note", detail: r, tone: "info" });
      });
    }
    if (!items.length) {
      items.push({ sev: "INFO", title: "No structured flag items", detail: a.recommended_action || data.message || "See technical details.", tone: "info" });
    }
    return items;
  }

  function renderVerdict(data) {
    const a = data.assessment || {};
    const ev = data.evidence || {};
    const decision = a.decision || "PENDING";
    const officer = a.officer_verdict || (ev.decision_engine || {}).officer_verdict;
    const v = mapOfficerVerdict(decision, ev, data.message || officer?.summary, officer);
    const hero = $("verdict-hero");
    if (hero) {
      hero.classList.remove("is-ok", "is-warn", "is-danger", "is-muted", "is-info");
      hero.classList.add(`is-${v.tone === "info" ? "muted" : v.tone}`);
    }
    if ($("verdict-icon")) $("verdict-icon").textContent = v.icon;
    if ($("verdict-label")) {
      $("verdict-label").textContent = v.label;
      const toneClass =
        decision === "PASS" ? "is-pass" :
        decision === "REVIEW" ? "is-review" :
        decision === "HIGH-RISK" ? "is-high-risk" :
        decision === "INCONCLUSIVE" ? "is-inconclusive" : "is-pending";
      $("verdict-label").className = `verdict-label decision ${toneClass}`;
    }
    if ($("verdict-blurb")) $("verdict-blurb").textContent = v.blurb;
    if ($("verdict-backend")) $("verdict-backend").textContent = decision;
    if ($("verdict-case")) $("verdict-case").textContent = data.case_id || "—";
    if ($("verdict-action")) $("verdict-action").textContent = a.recommended_action || "";
    if ($("out-decision-icon")) $("out-decision-icon").textContent = v.icon;
    if ($("out-backend-decision")) $("out-backend-decision").textContent = decision;
    renderEvidenceDecisionPanel(officer || v.officer);
    $("result-empty")?.classList.add("is-hidden");
    $("result-dashboard")?.classList.remove("is-hidden");
  }

  function renderWhy(data) {
    const list = $("why-list");
    if (!list) return;
    list.innerHTML = "";
    const decision = (data.assessment || {}).decision;
    const title = $("why-title");
    if (title) {
      title.textContent = decision === "PASS"
        ? "WHY THIS LOOKS AUTHENTIC (SCREENING ASSIST)"
        : decision === "INCONCLUSIVE" || decision === "REVIEW"
          ? "WHY REVIEW IS REQUIRED"
          : "WHY THIS RESULT";
    }
    buildWhyItems(data).forEach((it) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="why-sev is-${it.tone}">${escapeHtml(it.sev)}</span>
        <div><p class="why-title">${escapeHtml(it.title)}</p><p class="why-detail">${escapeHtml(it.detail)}</p></div>`;
      list.appendChild(li);
    });
  }

  function renderTypographyEvidence(data) {
    const tv = (data.evidence || {}).text_visual || {};
    const typo = tv.typography || {};
    const splice = tv.text_splicing || {};
    const summary = $("typo-result-summary");
    const bars = $("typo-result-bars");
    const list = $("typo-result-list");
    const note = $("typo-result-note");
    if (!summary) return;

    const label = typo.consistency_label
      || ((typo.anomalies || []).length ? "ANOMALY_SIGNAL" : typo.status === "ANALYZED" ? "CONSISTENT" : "NOT_ASSESSED");

    const chips = [
      ["Consistency", label],
      ["Regions", typo.region_count != null ? String(typo.region_count) : "NOT AVAILABLE"],
      ["Median relative height", typo.median_relative_height != null ? String(typo.median_relative_height) : "NOT AVAILABLE"],
      ["Alignment σ", typo.left_alignment_std != null ? String(typo.left_alignment_std) : "NOT AVAILABLE"],
      ["Mean sharpness", typo.mean_local_sharpness != null ? String(typo.mean_local_sharpness) : "NOT AVAILABLE"],
      ["Mean contrast", typo.mean_local_contrast != null ? String(typo.mean_local_contrast) : "NOT AVAILABLE"],
      ["Text splicing anomalies", splice.anomaly_count != null ? String(splice.anomaly_count) : "NOT AVAILABLE"],
    ];
    summary.innerHTML = chips.map(([k, v]) => `<div class="summary-chip"><span>${k}</span><strong>${escapeHtml(v)}</strong></div>`).join("");

    if (bars) {
      bars.innerHTML = "";
      const regions = typo.regions || [];
      const med = typo.median_relative_height || 0;
      regions.slice(0, 10).forEach((r) => {
        const rel = Number(r.relative_height || 0);
        const ratio = med > 0 ? Math.min(100, Math.round((rel / med) * 50)) : Math.min(100, Math.round(rel * 2000));
        const row = document.createElement("div");
        row.className = "typo-row";
        const anom = (typo.anomalies || []).some((a) => a.text === r.text);
        row.innerHTML = `<span class="mono">${escapeHtml((r.text || "region").slice(0, 14))}</span>
          <div class="bar" aria-hidden="true"><i style="width:${ratio}%;background:${anom ? "var(--warn)" : "var(--info)"}"></i></div>
          <strong class="mono">${rel ? rel.toFixed(4) : "n/a"}</strong>`;
        bars.appendChild(row);
      });
      if (!regions.length) {
        bars.innerHTML = `<p class="muted tiny">${typo.status === "ANALYZED" ? "No region samples returned." : typo.status || "NOT_ASSESSED"}</p>`;
      }
    }

    if (list) {
      fillKv(list, [
        ["Status", typo.status || "NOT_ASSESSED"],
        ["Mean relative width", typo.mean_relative_width ?? "NOT AVAILABLE"],
        ["Anomalies", String((typo.anomalies || []).length)],
        ["Splicing status", splice.status || "NOT_ASSESSED"],
      ]);
      (typo.anomalies || []).slice(0, 6).forEach((a) => {
        const li = document.createElement("li");
        li.innerHTML = `<span>${escapeHtml(a.text || "region")}</span><strong>${escapeHtml(a.status || "ANOMALOUS")} · ${escapeHtml(a.reason || "")}</strong>`;
        list.appendChild(li);
      });
    }
    if (note) {
      note.textContent = typo.note
        || "These measurements explain visual text consistency for officer review. They are not legal proof of authenticity or forgery.";
    }
  }

  function renderSummaryMeters(data) {
    const a = data.assessment || {};
    const ev = data.evidence || {};
    const q = data.image_quality || {};
    const face = ev.face || {};
    const ver = face.verification || {};
    const sg = $("summary-grid");
    if (sg) {
      const chips = [
        ["Identity", ver.status || face.status || "NOT_ASSESSED"],
        ["OCR", ev.ocr?.status || "NOT_AVAILABLE"],
        ["Forensics", ev.tampering?.severity || ev.tampering?.status || "NOT_ASSESSED"],
        ["Cross-validation", ev.cross_validation?.overall || "NOT_ASSESSED"],
        ["Backend decision", a.decision || "—"],
        ["Evidence level", a.decision === "HIGH-RISK" ? "HIGH" : a.decision === "REVIEW" ? "MEDIUM" : a.decision === "PASS" ? "LOW" : "INCONCLUSIVE"],
      ];
      sg.innerHTML = chips.map(([k, val]) => `<div class="summary-chip"><span>${k}</span><strong>${escapeHtml(String(val))}</strong></div>`).join("");
    }
    const mg = $("meter-grid");
    if (!mg) return;
    const meters = [
      ["Document quality", q.overall_quality, "Image suitability for analysis"],
      ["OCR reliability", ev.ocr?.mean_confidence ?? ev.ocr?.avg_confidence ?? null, "Transcription confidence ≠ authenticity"],
      ["Evidence quality", a.evidence_quality, "How complete usable evidence is"],
      ["Model confidence", a.confidence, "Internal confidence — not accuracy %"],
      ["Authenticity estimate", a.authenticity_estimate, "Proxy / calibrated only when backend provides it"],
      ["Forensic review score", ev.tampering?.review_score ?? null, "Heuristic review signal"],
    ];
    mg.innerHTML = "";
    meters.forEach(([name, val, note]) => {
      const div = document.createElement("div");
      div.className = "meter";
      const label = val == null ? "NOT AVAILABLE" : pct(val);
      const w = val == null ? 0 : Math.max(0, Math.min(100, Math.round(Number(val) * 100)));
      div.innerHTML = `<div class="meter-head"><span>${name}</span><strong>${label}</strong></div>
        ${val == null ? "" : `<div class="bar" aria-hidden="true"><i style="width:${w}%"></i></div>`}
        <p class="meter-note">${note}</p>`;
      mg.appendChild(div);
    });
    const faceMeter = document.createElement("div");
    faceMeter.className = "meter";
    faceMeter.innerHTML = `<div class="meter-head"><span>Face verification</span><strong>${escapeHtml(ver.status || face.status || "NOT_ASSESSED")}</strong></div>
      <p class="meter-note">MATCH / REVIEW / NO_MATCH / INCONCLUSIVE / NOT_ASSESSED</p>`;
    mg.appendChild(faceMeter);
  }

  function faceMark(face) {
    if (!face || !face.status) return "na";
    if (face.status === "DETECTED" || face.status === "MULTIPLE_DETECTED") return "ok";
    if (face.status === "NOT_DETECTED") return "warn";
    return "na";
  }
  function cvMark(cv) {
    if (!cv || !cv.overall) return "na";
    if (cv.overall === "PASS") return "ok";
    if (cv.overall === "WARNING") return "warn";
    if (cv.overall === "FAIL") return "fail";
    return "na";
  }

  function renderTimeline(data) {
    const ev = data.evidence || {};
    const q = data.image_quality || {};
    const steps = [
      ["UPLOAD", "ok", "Document received"],
      ["QUALITY CHECK", q.insufficient_for_analysis ? "warn" : "ok", q.overall_quality != null ? `overall ${pct(q.overall_quality)}` : (q.reason || "assessed")],
      ["CLASSIFICATION", data.document_type?.label ? "ok" : "na", data.document_type?.label || "NOT AVAILABLE"],
      ["OCR", ev.ocr?.status ? "ok" : "na", ev.ocr?.status || "NOT AVAILABLE"],
      ["FORENSICS", ev.tampering?.status === "ANALYZED" ? (ev.tampering?.severity === "ELEVATED_REVIEW" ? "warn" : "ok") : "na", ev.tampering?.severity || ev.tampering?.status || "NOT AVAILABLE"],
      ["FACE", faceMark(ev.face), ev.face?.status || "NOT AVAILABLE"],
      ["CROSS VALIDATION", cvMark(ev.cross_validation), ev.cross_validation?.overall || "NOT AVAILABLE"],
      ["EVIDENCE FUSION", ev.fusion?.status ? "ok" : "na", ev.fusion?.status || "NOT AVAILABLE"],
      ["FINAL RESULT", data.assessment?.decision === "HIGH-RISK" ? "fail" : data.assessment?.decision === "INCONCLUSIVE" ? "warn" : "ok", data.assessment?.decision || "—"],
    ];
    const ol = $("timeline-list");
    if (!ol) return;
    ol.innerHTML = steps.map(([name, mark, detail]) => {
      const sym = mark === "ok" ? "✓" : mark === "warn" ? "⚠" : mark === "fail" ? "✕" : "○";
      return `<li><span class="tl-mark is-${mark}" aria-hidden="true">${sym}</span><span><strong>${name}</strong> · ${escapeHtml(String(detail))}</span></li>`;
    }).join("");
  }

  function openRegionDrawer(i, h, ev) {
    const tamp = ev.tampering || {};
    openDrawer(`REGION ${String(i + 1).padStart(2, "0")}`, `
      <ul class="kv-list">
        <li><span>Type</span><strong>Forensic hotspot</strong></li>
        <li><span>Location</span><strong>x=${h.x}, y=${h.y}</strong></li>
        <li><span>Size</span><strong>${h.w} × ${h.h}</strong></li>
        <li><span>Score</span><strong>${h.score != null ? Number(h.score).toFixed(3) : "NOT AVAILABLE"}</strong></li>
        <li><span>Severity</span><strong>${escapeHtml(tamp.severity || "NOT AVAILABLE")}</strong></li>
        <li><span>Source</span><strong>Image forensics / localization</strong></li>
        <li><span>Flags</span><strong>${escapeHtml((tamp.flags || []).join(", ") || "none")}</strong></li>
      </ul>
      <p class="muted tiny">Heuristic review signal — not calibrated fraud proof.</p>`);
  }

  function applyMapMode() {
    const mode = state.mapMode || "combined";
    const heatOver = $("result-heat-overlay");
    const heatBase = $("result-heat-base");
    const note = $("map-mode-note");
    const loc = ((state.lastResponse || {}).evidence || {}).tampering?.localization || {};
    const comps = loc.components || {};
    document.querySelectorAll("[data-map]").forEach((b) => {
      b.classList.toggle("is-map-active", b.dataset.map === mode);
    });
    if (!state.forensicOverlay) {
      heatOver?.classList.add("is-hidden");
      if (note) note.textContent = "No localization overlay from backend.";
      return;
    }
    if (mode === "original") {
      heatOver?.classList.add("is-hidden");
      if (note) note.textContent = "Showing original only.";
      return;
    }
    const separate = ["ela", "noise", "copymove", "text", "photo", "stamp"];
    if (separate.includes(mode)) {
      heatOver?.classList.add("is-hidden");
      const keyHints = {
        ela: comps.ela_weight,
        noise: comps.noise_weight,
        copymove: "see flags / copy_move signal",
        text: "see text forensics panel",
        photo: "see photo forensics panel",
        stamp: "see stamp analysis",
      };
      if (note) {
        note.textContent = `Separate ${mode.toUpperCase()} raster map NOT AVAILABLE from backend. Component hint: ${keyHints[mode] ?? "N/A"}. Use COMBINED for actual overlay.`;
      }
      return;
    }
    if (heatOver) {
      heatOver.src = state.forensicOverlay;
      heatOver.classList.remove("is-hidden");
      heatOver.style.opacity = String((Number($("heatmap-opacity")?.value || 70) / 100));
    }
    heatBase?.classList.remove("is-hidden");
    if (note) note.textContent = "COMBINED heuristic localization overlay from backend.";
  }

  function renderCompareAndRegions(data) {
    const ev = data.evidence || {};
    const loc = (ev.tampering || {}).localization || {};
    const docImg = $("result-doc-img");
    const empty = $("result-doc-empty");
    const heatBase = $("result-heat-base");
    const na = $("forensic-map-na");
    const overlay = $("region-overlay");
    const regionList = $("region-list");
    const stage = $("compare-original");
    const src = state.objectUrl || preview?.getAttribute("src") || $("doc-img")?.getAttribute("src") || "";

    if (src && docImg) {
      docImg.src = src;
      docImg.classList.remove("is-hidden");
      empty?.classList.add("is-hidden");
      if (heatBase) {
        heatBase.src = src;
        heatBase.classList.remove("is-hidden");
      }
    } else {
      docImg?.classList.add("is-hidden");
      empty?.classList.remove("is-hidden");
    }

    const hasMap = loc.status === "LOCALIZED" && loc.overlay_png_base64;
    state.forensicOverlay = hasMap ? `data:image/png;base64,${loc.overlay_png_base64}` : null;
    state.mapMode = state.mapMode || "combined";
    applyMapMode();

    const hotspots = loc.hotspots || [];
    if (overlay) overlay.innerHTML = "";
    if (regionList) regionList.innerHTML = "";
    const placeBoxes = () => {
      if (!overlay || !docImg || !docImg.naturalWidth || !stage) return;
      const rect = docImg.getBoundingClientRect();
      const stageRect = stage.getBoundingClientRect();
      const scaleX = rect.width / docImg.naturalWidth;
      const scaleY = rect.height / docImg.naturalHeight;
      const offsetL = rect.left - stageRect.left;
      const offsetT = rect.top - stageRect.top;
      overlay.innerHTML = "";
      hotspots.slice(0, 12).forEach((h, i) => {
        const box = document.createElement("button");
        box.type = "button";
        box.className = "rbox";
        box.style.left = `${offsetL + h.x * scaleX}px`;
        box.style.top = `${offsetT + h.y * scaleY}px`;
        box.style.width = `${Math.max(8, h.w * scaleX)}px`;
        box.style.height = `${Math.max(8, h.h * scaleY)}px`;
        box.innerHTML = `<span class="rnum">${String(i + 1).padStart(2, "0")}</span>`;
        box.addEventListener("click", () => openRegionDrawer(i, h, ev));
        overlay.appendChild(box);
      });
    };
    if (docImg) {
      if (docImg.complete && docImg.naturalWidth) placeBoxes();
      else docImg.onload = placeBoxes;
    }
    hotspots.slice(0, 12).forEach((h, i) => {
      if (!regionList) return;
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "region-chip";
      chip.textContent = `Region ${String(i + 1).padStart(2, "0")} · score ${h.score != null ? Number(h.score).toFixed(2) : "N/A"}`;
      chip.addEventListener("click", () => openRegionDrawer(i, h, ev));
      regionList.appendChild(chip);
    });
    if (!hotspots.length && regionList) {
      regionList.innerHTML = `<p class="muted tiny">No localized suspicious regions reported.</p>`;
    }
    if (na) {
      if (hasMap) na.classList.add("is-hidden");
      else {
        na.classList.remove("is-hidden");
        na.textContent = "FORENSIC MAP NOT AVAILABLE";
      }
    }
  }

  function setWorkflow(step) {
    document.querySelectorAll(".wf-step").forEach((el) => {
      const order = ["document", "identity", "forensics", "evidence", "result"];
      const idx = order.indexOf(el.dataset.wf);
      const cur = order.indexOf(step);
      el.classList.toggle("is-active", el.dataset.wf === step);
      el.classList.toggle("is-done", idx >= 0 && idx < cur);
    });
  }

  function updateDocViewer(file, ev) {
    const img = $("doc-img");
    const overlay = $("doc-overlay");
    if (!img || !file) return;
    if (state.objectUrl) img.src = state.objectUrl;
    img.classList.remove("is-hidden");
    if (!overlay) return;
    overlay.innerHTML = "";
    // overlays require natural dimensions after load
    img.onload = () => {
      const stage = $("doc-stage");
      const scaleX = img.clientWidth / (img.naturalWidth || 1);
      const scaleY = img.clientHeight / (img.naturalHeight || 1);
      const boxes = (ev?.ocr?.text_boxes || []).slice(0, 40);
      boxes.forEach((b) => {
        const bb = b.bbox || [];
        if (bb.length < 4) return;
        const div = document.createElement("div");
        div.className = "obox";
        div.style.left = `${bb[0] * scaleX + img.offsetLeft}px`;
        div.style.top = `${bb[1] * scaleY + img.offsetTop}px`;
        div.style.width = `${bb[2] * scaleX}px`;
        div.style.height = `${bb[3] * scaleY}px`;
        div.title = `${b.text || ""} · ${b.status || ""}`;
        div.addEventListener("click", () => openDrawer("OCR region", `<p class="mono">${escapeHtml(b.text || "")}</p><p class="tiny muted">conf ${pct(b.confidence)} · pass ${b.source_pass || "—"}</p>`));
        overlay.appendChild(div);
      });
      const face = (ev?.face?.faces || [])[0];
      const fb = face?.bbox;
      if (fb) {
        const x = fb.x ?? fb[0];
        const y = fb.y ?? fb[1];
        const w = fb.w ?? fb[2];
        const h = fb.h ?? fb[3];
        const div = document.createElement("div");
        div.className = "obox is-face";
        div.style.left = `${x * scaleX + img.offsetLeft}px`;
        div.style.top = `${y * scaleY + img.offsetTop}px`;
        div.style.width = `${w * scaleX}px`;
        div.style.height = `${h * scaleY}px`;
        div.title = "Document face";
        overlay.appendChild(div);
      }
    };
  }

  document.querySelectorAll("[data-zoom]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const img = $("doc-img");
      if (!img) return;
      if (btn.dataset.zoom === "in") state.docZoom *= 1.15;
      if (btn.dataset.zoom === "out") state.docZoom /= 1.15;
      if (btn.dataset.zoom === "reset") state.docZoom = 1;
      img.style.transform = `scale(${state.docZoom})`;
    });
  });

  function applyScreeningResult(data, opts) {
    const options = opts || {};
    state.lastResponse = data;
    state.caseId = data.case_id;
    if ($("case-chip")) $("case-chip").textContent = `CASE: ${data.case_id || "—"}`;
    const a = data.assessment || {};
    const ev = data.evidence || {};
    const dt = data.document_type || {};
    const q = data.image_quality || {};

    emptyState?.classList.add("is-hidden");
    resultCard?.classList.remove("is-hidden");
    $("out-case").textContent = data.case_id || "—";
    $("out-type").textContent = `${dt.label || "UNKNOWN"}${dt.confidence != null ? ` · ${(dt.confidence * 100).toFixed(1)}%` : ""}`;
    $("out-sih-cat").textContent = dt.sih_category || dt.label || "—";
    $("out-clf-method").textContent = [dt.method, dt.model_version].filter(Boolean).join(" · ") || "—";
    $("out-auth").textContent = a.authenticity_estimate == null ? "NOT AVAILABLE" : `${pct(a.authenticity_estimate)} · proxy`;
    $("out-conf").textContent = a.confidence == null ? "NOT AVAILABLE" : pct(a.confidence);
    $("out-eq").textContent = a.evidence_quality == null ? "NOT AVAILABLE" : pct(a.evidence_quality);

    const decision = a.decision || "PENDING";
    const officer = a.officer_verdict || (ev.decision_engine || {}).officer_verdict;
    const verdict = mapOfficerVerdict(decision, ev, data.message || officer?.summary, officer);
    const decisionEl = $("out-decision");
    decisionEl.textContent = verdict.label;
    decisionEl.className = "decision";
    if (decision === "PASS") decisionEl.classList.add("is-pass");
    else if (decision === "REVIEW") decisionEl.classList.add("is-review");
    else if (decision === "HIGH-RISK") decisionEl.classList.add("is-high-risk");
    else if (decision === "INCONCLUSIVE") decisionEl.classList.add("is-inconclusive");
    else decisionEl.classList.add("is-pending");
    $("out-message").textContent = verdict.blurb || data.message || "";
    $("out-action").textContent = a.recommended_action || "";
    $("reasons-list").innerHTML = "";
    (a.reasons || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      $("reasons-list").appendChild(li);
    });
    $("pipeline").innerHTML = "";
    (data.pipeline || []).forEach((step) => {
      const li = document.createElement("li");
      li.textContent = step;
      $("pipeline").appendChild(li);
    });
    const techPipe = $("tech-pipeline");
    if (techPipe) {
      techPipe.innerHTML = "";
      (data.pipeline || []).forEach((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        techPipe.appendChild(li);
      });
    }

    fillQuality(q);
    fillFields(ev.fields, ev.mrz, ev.ocr);
    fillVision(ev.ai_forensics, ev.fusion);
    fillTampering(ev.tampering);
    fillFace(ev.face);
    fillCrossValidation(ev.cross_validation);
    fillCalibration(ev.calibration);
    fillConfidence(ev.confidence_engine);
    fillExplain(ev.explainability);
    fillForensics(ev.forensics, ev.text_visual, ev.stamp, ev.metadata, ev.photo_splicing);
    buildResultCards(data);
    buildResultCards(data, "result-cards-advanced");
    updateDocViewer(state.selectedFile, ev);

    renderVerdict(data);
    renderWhy(data);
    renderTypographyEvidence(data);
    renderSummaryMeters(data);
    renderTimeline(data);
    renderCompareAndRegions(data);

    const conflict = $("conflict-panel");
    const cl = $("conflict-list");
    if (conflict && cl) {
      const rows = [];
      if (ev.cross_validation?.overall === "FAIL" || ev.cross_validation?.overall === "WARNING") {
        rows.push(["Cross-validation", ev.cross_validation.overall]);
      }
      if (ev.tampering?.severity === "ELEVATED_REVIEW") rows.push(["Forensics", "SUSPICIOUS / ELEVATED_REVIEW"]);
      if ((ev.face?.status === "NOT_DETECTED") && decision !== "INCONCLUSIVE") rows.push(["Identity", "NO DOCUMENT FACE"]);
      if (rows.length) {
        conflict.classList.remove("is-hidden");
        fillKv(cl, rows.concat([["Final", decision], ["Officer label", verdict.label]]));
      } else conflict.classList.add("is-hidden");
    }

    const face = ev.face || {};
    const faceDetected = face.status === "DETECTED" || face.status === "MULTIPLE_DETECTED";
    const hasMatch = (face.verification || {}).result === "MATCH" || (face.verification || {}).result === "NO_MATCH";
    const skipGate = options.skipIdentityGate === true || state.identityComplete || hasMatch || options.finalize === true;

    if (liveBanner) {
      if (faceDetected && !skipGate) liveBanner.classList.remove("is-hidden");
      else liveBanner.classList.add("is-hidden");
    }

    window.DocshieldThree?.setScanActive(false);
    const cap = $("scanner-caption");
    if (cap) cap.textContent = `COMPLETE · ${verdict.label}`;
    window.DocshieldThree?.setGraphData([
      "CASE", "DOC", "OCR", "MRZ", "FORENSICS", "EffNet", "ViT", "CLIP", "FACE", "FUSION", "DECISION",
    ]);

    // Gate: require live face capture when document face exists
    if (faceDetected && !skipGate) {
      state.identityGateRequired = true;
      state.identityComplete = false;
      $("identity-gate")?.classList.remove("is-hidden");
      if ($("identity-gate-status")) {
        $("identity-gate-status").textContent = "Document face found — start camera to capture and match before final verdict.";
      }
      setWorkflow("identity");
      showView("identity");
      setStatus("Document analyzed · live face capture required for final verdict.", "is-ok");
      // Auto-prompt camera
      setTimeout(() => {
        if (state.identityGateRequired && !state.cameraStream) $("btn-cam-start")?.click();
      }, 350);
      return;
    }

    state.identityGateRequired = false;
    $("identity-gate")?.classList.add("is-hidden");
    setWorkflow("result");
    showView("result");
  }

  /* —— Upload / analyze —— */
  function clearPreview() {
    state.selectedFile = null;
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    preview?.removeAttribute("src");
    previewWrap?.classList.add("is-hidden");
    if (fileName) fileName.textContent = "";
    if (fileMeta) fileMeta.textContent = "";
    if (btnAnalyze) btnAnalyze.disabled = true;
    if (btnRemove) btnRemove.disabled = true;
    if (fileInput) fileInput.value = "";
  }

  function acceptFile(file) {
    if (!file) return;
    const extOk = /\.(png|jpe?g|webp|pdf)$/i.test(file.name);
    if (!ALLOWED.has(file.type) && !extOk) {
      setStatus("Only PNG, JPG, JPEG, WEBP, or PDF are accepted.", "is-error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setStatus("File exceeds 10 MB limit.", "is-error");
      return;
    }
    clearPreview();
    state.selectedFile = file;
    state.lastDocumentFile = file;
    state.objectUrl = URL.createObjectURL(file);
    if (preview) preview.src = state.objectUrl;
    previewWrap?.classList.remove("is-hidden");
    if (fileName) fileName.textContent = file.name;
    if (fileMeta) fileMeta.textContent = `${(file.size / 1024).toFixed(1)} KB · ${file.type || "image"}`;
    if (btnAnalyze) btnAnalyze.disabled = false;
    if (btnRemove) btnRemove.disabled = false;
    setStatus("DOCUMENT RECEIVED — starting analysis…", "is-ok");
    setWorkflow("document");
    window.DocshieldThree?.setScanActive(true);
    if ($("scanner-caption")) $("scanner-caption").textContent = "ANALYZING";
    showView("screening");
    setTimeout(() => {
      if (state.selectedFile === file) btnAnalyze?.click();
    }, 40);
  }

  dropzone?.addEventListener("click", () => fileInput?.click());
  dropzone?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput?.click();
    }
  });
  fileInput?.addEventListener("change", () => acceptFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone?.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone?.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-drag");
    });
  });
  dropzone?.addEventListener("drop", (e) => acceptFile(e.dataTransfer.files?.[0]));
  btnRemove?.addEventListener("click", () => {
    clearPreview();
    setStatus("File removed.");
  });
  btnClear?.addEventListener("click", () => {
    clearPreview();
    setStatus("Cleared.");
  });

  btnAnalyze?.addEventListener("click", async () => {
    if (!state.selectedFile) return;
    btnAnalyze.disabled = true;
    setStatus("ANALYSIS STARTED", "is-ok");
    showProgress([
      "[✓] Image loaded",
      "[●] Quality / classify / OCR / forensics / face…",
      "[ ] Decision",
    ]);
    window.DocshieldThree?.setScanActive(true);

    const body = new FormData();
    body.append("file", state.selectedFile, state.selectedFile.name);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        const msg = (detail && detail.error) || (typeof detail === "string" ? detail : null) || "Upload rejected.";
        setStatus(msg, "is-error");
        showProgress([`[✗] ${msg}`]);
        window.DocshieldThree?.setScanActive(false);
        btnAnalyze.disabled = false;
        return;
      }
      applyScreeningResult(data);
      const ev = data.evidence || {};
      showProgress([
        "[✓] Image loaded",
        `[✓] Quality · ${pct(data.image_quality?.overall_quality)}`,
        `[✓] Type · ${data.document_type?.label || "—"}`,
        `[${ev.ocr?.status === "OK" ? "✓" : "○"}] OCR · ${ev.ocr?.status || "—"}`,
        `[${ev.tampering?.status === "ANALYZED" ? "✓" : "○"}] Forensics`,
        `[✓] Face · ${ev.face?.status || "—"}`,
        `[✓] Decision · ${data.assessment?.decision || "—"}`,
      ]);
      setStatus(`Case ${data.case_id}: ${data.assessment?.decision}`, data.assessment?.decision === "INCONCLUSIVE" ? "is-error" : "is-ok");
    } catch {
      setStatus("Network error contacting API.", "is-error");
      showProgress(["[✗] Network error"]);
      window.DocshieldThree?.setScanActive(false);
    } finally {
      btnAnalyze.disabled = !state.selectedFile;
    }
  });

  async function ensureVideoPlaying(camVideo) {
    if (!camVideo) return false;
    try {
      if (camVideo.readyState < 2) {
        await new Promise((resolve, reject) => {
          const onReady = () => { cleanup(); resolve(); };
          const onErr = () => { cleanup(); reject(new Error("video_error")); };
          const cleanup = () => {
            camVideo.removeEventListener("loadeddata", onReady);
            camVideo.removeEventListener("error", onErr);
          };
          camVideo.addEventListener("loadeddata", onReady, { once: true });
          camVideo.addEventListener("error", onErr, { once: true });
          setTimeout(() => { cleanup(); resolve(); }, 2500);
        });
      }
      await camVideo.play();
    } catch (_) { /* autoplay may be blocked briefly; muted+playsinline usually OK */ }
    return Boolean(camVideo.videoWidth);
  }

  async function openCameraStream() {
    const attempts = [
      { video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: "user" }, audio: false },
      { video: true, audio: false },
    ];
    let lastErr = null;
    for (const constraints of attempts) {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        lastErr = err;
      }
    }
    throw lastErr || new Error("Camera unavailable");
  }

  /* —— Camera —— */
  function setCamStatus(msg) {
    const el = $("camera-status");
    if (el) el.textContent = msg || "";
  }

  function stopQualityLoop() {
    if (state.qualityTimer) {
      clearInterval(state.qualityTimer);
      state.qualityTimer = null;
    }
    state.readyStreak = 0;
    state.autoCaptureArmed = false;
  }

  async function stopCamera() {
    stopQualityLoop();
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
    }
    const camVideo = $("camera-video");
    if (camVideo) {
      camVideo.srcObject = null;
      camVideo.classList.add("is-hidden");
    }
    $("face-quality-hud")?.classList.add("is-hidden");
    if ($("btn-cam-capture")) $("btn-cam-capture").disabled = true;
    if ($("btn-cam-stop")) $("btn-cam-stop").disabled = true;
    if (state.cameraSessionId) {
      try {
        await fetch(`/api/camera/session/end?session_id=${encodeURIComponent(state.cameraSessionId)}`, { method: "POST" });
      } catch { /* network */ }
      state.cameraSessionId = null;
    }
    setCamStatus("Camera stopped.");
    setCameraState("OFF");
  }

  function sampleFrameBlob() {
    const camVideo = $("camera-video");
    const camCanvas = $("camera-canvas");
    if (!camVideo || !camCanvas || !camVideo.videoWidth) return null;
    camCanvas.width = Math.min(320, camVideo.videoWidth);
    camCanvas.height = Math.round(camCanvas.width * (camVideo.videoHeight / camVideo.videoWidth));
    camCanvas.getContext("2d").drawImage(camVideo, 0, 0, camCanvas.width, camCanvas.height);
    return new Promise((resolve) => camCanvas.toBlob(resolve, "image/jpeg", 0.7));
  }

  function frameBrightness(canvas) {
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    if (!width || !height) return { mean: 0, std: 0 };
    const data = ctx.getImageData(0, 0, width, height).data;
    let sum = 0;
    let sumSq = 0;
    const step = 16;
    let n = 0;
    for (let i = 0; i < data.length; i += 4 * step) {
      const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      sum += y;
      sumSq += y * y;
      n += 1;
    }
    const mean = sum / Math.max(1, n);
    const std = Math.sqrt(Math.max(0, sumSq / Math.max(1, n) - mean * mean));
    return { mean, std };
  }

  function startQualityLoop() {
    stopQualityLoop();
    state.autoCaptureArmed = true;
    $("face-quality-hud")?.classList.remove("is-hidden");
    let busy = false;
    state.qualityTimer = setInterval(async () => {
      if (busy || !state.autoCaptureArmed || !state.cameraStream) return;
      const camCanvas = $("camera-canvas");
      const camVideo = $("camera-video");
      if (!camVideo?.videoWidth || !camCanvas) return;
      camCanvas.width = Math.min(320, camVideo.videoWidth);
      camCanvas.height = Math.round(camCanvas.width * (camVideo.videoHeight / Math.max(1, camVideo.videoWidth)));
      camCanvas.getContext("2d").drawImage(camVideo, 0, 0, camCanvas.width, camCanvas.height);
      const light = frameBrightness(camCanvas);
      const lightOk = light.mean > 45 && light.mean < 210 && light.std > 18;
      if ($("fq-light")) {
        $("fq-light").textContent = `LIGHT: ${Math.round(light.mean)}`;
        $("fq-light").classList.toggle("is-ok", lightOk);
        $("fq-light").classList.toggle("is-warn", !lightOk);
      }
      busy = true;
      try {
        const blob = await sampleFrameBlob();
        if (!blob) return;
        const body = new FormData();
        body.append("file", blob, "preview.jpg");
        const det = await fetch("/api/face/detect", { method: "POST", body }).then((r) => r.json()).catch(() => null);
        const faceOk = det && det.status === "DETECTED" && Number(det.face_count || 0) === 1;
        const q0 = (((det?.faces || [])[0] || {}).quality || {});
        const qualityOk = !q0.label || ["GOOD", "ACCEPTABLE", "FAIR"].includes(q0.label);
        if ($("fq-face")) {
          $("fq-face").textContent = `FACE: ${det?.status || "—"} (${det?.face_count ?? 0})`;
          $("fq-face").classList.toggle("is-ok", !!faceOk);
          $("fq-face").classList.toggle("is-warn", !faceOk);
        }
        const ready = lightOk && faceOk && qualityOk;
        if ($("fq-ready")) {
          $("fq-ready").textContent = ready ? "STATUS: READY" : "STATUS: ADJUST";
          $("fq-ready").classList.toggle("is-ok", ready);
          $("fq-ready").classList.toggle("is-warn", !ready);
        }
        if ($("identity-gate-status")) {
          $("identity-gate-status").textContent = ready
            ? "READY — capturing automatically…"
            : "Adjust face/lighting until READY (one clear face).";
        }
        setCameraState(ready ? "READY" : (faceOk ? "FACE_DETECTED" : "ACTIVE"));
        if (ready) {
          state.readyStreak += 1;
          if (state.readyStreak >= 2 && state.autoCaptureArmed) {
            state.autoCaptureArmed = false;
            stopQualityLoop();
            $("btn-cam-capture")?.click();
          }
        } else state.readyStreak = 0;
      } finally {
        busy = false;
      }
    }, 900);
  }

  $("btn-cam-start")?.addEventListener("click", async () => {
    try {
      setCameraState("REQUESTING_PERMISSION");
      setCamStatus("Requesting camera permission…");
      showView("identity");
      $("identity-gate")?.classList.remove("is-hidden");
      if ($("identity-gate-status")) $("identity-gate-status").textContent = "Requesting camera permission…";
      const sessRes = await fetch("/api/camera/session/start", { method: "POST" });
      const sess = await sessRes.json();
      if (!sessRes.ok || !sess.session_id) {
        setCamStatus((sess.detail && sess.detail.error) || "Session start failed.");
        setCameraState("UNAVAILABLE");
        return;
      }
      state.cameraSessionId = sess.session_id;
      if (state.lastDocumentFile) {
        const body = new FormData();
        body.append("file", state.lastDocumentFile, state.lastDocumentFile.name);
        await fetch(`/api/camera/session/attach-document?session_id=${encodeURIComponent(state.cameraSessionId)}`, { method: "POST", body });
      }
      state.cameraStream = await openCameraStream();
      const camVideo = $("camera-video");
      camVideo.srcObject = state.cameraStream;
      camVideo.muted = true;
      camVideo.setAttribute("playsinline", "true");
      camVideo.classList.remove("is-hidden");
      $("camera-snap")?.classList.add("is-hidden");
      const ready = await ensureVideoPlaying(camVideo);
      $("btn-cam-capture").disabled = false;
      $("btn-cam-stop").disabled = false;
      if ($("btn-cam-retake")) $("btn-cam-retake").disabled = true;
      setCameraState(ready ? "ACTIVE" : "ACTIVE_WAITING_FRAME");
      setCamStatus(
        ready
          ? `Session ${state.cameraSessionId.slice(0, 8)}… · document attached · seeking clear face`
          : "Camera opened — waiting for first frame. Click Capture & match when your face is visible."
      );
      if ($("identity-gate-status")) {
        $("identity-gate-status").textContent = ready
          ? "Camera active — center one face with even lighting, or press Capture & match."
          : "Camera started — if preview is blank, wait a moment then press Capture & match.";
      }
      startQualityLoop();
    } catch (err) {
      const denied = err && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError");
      setCameraState(denied ? "PERMISSION_DENIED" : "UNAVAILABLE");
      setCamStatus(denied ? "CAMERA: PERMISSION_DENIED" : "Camera unavailable.");
      if ($("identity-gate-status")) {
        $("identity-gate-status").textContent = denied
          ? "Camera permission denied — enable camera in the browser, then retry."
          : "Camera unavailable.";
      }
      if ($("btn-cam-capture")) $("btn-cam-capture").disabled = true;
    }
  });

  $("btn-cam-stop")?.addEventListener("click", () => stopCamera());
  $("btn-cam-retake")?.addEventListener("click", async () => {
    $("camera-snap")?.classList.add("is-hidden");
    $("camera-video")?.classList.remove("is-hidden");
    if ($("btn-cam-retake")) $("btn-cam-retake").disabled = true;
    if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;
    state.identityComplete = false;
    if (state.cameraStream) startQualityLoop();
    else $("btn-cam-start")?.click();
  });
  $("btn-live-verify")?.addEventListener("click", () => {
    liveBanner?.classList.add("is-hidden");
    showView("identity");
    $("identity-gate")?.classList.remove("is-hidden");
    $("btn-cam-start")?.click();
  });
  $("btn-live-skip")?.addEventListener("click", () => {
    liveBanner?.classList.add("is-hidden");
    state.identityGateRequired = false;
    state.identityComplete = true;
    $("identity-gate")?.classList.add("is-hidden");
    if (state.lastResponse) applyScreeningResult(state.lastResponse, { skipIdentityGate: true, finalize: true });
    setCamStatus("Continued without live face — identity NOT_ASSESSED in final fusion.");
  });

  $("btn-cam-capture")?.addEventListener("click", async () => {
    const camVideo = $("camera-video");
    const camCanvas = $("camera-canvas");
    if (!camVideo || !camCanvas) {
      setCamStatus("Camera UI missing — refresh the page.");
      return;
    }
    if (!state.cameraSessionId) {
      setCamStatus("No camera session — press Start camera first.");
      return;
    }
    if (!state.cameraStream) {
      setCamStatus("Camera not running — press Start camera.");
      return;
    }
    stopQualityLoop();
    setCameraState("VERIFICATION_PROCESSING");
    setCamStatus("Capturing and matching to document face…");
    if ($("identity-gate-status")) $("identity-gate-status").textContent = "Matching live face to document face…";
    if ($("btn-cam-capture")) $("btn-cam-capture").disabled = true;

    const ready = await ensureVideoPlaying(camVideo);
    if (!ready || !camVideo.videoWidth) {
      setCameraState("ACTIVE");
      setCamStatus("No camera frame yet — wait for preview, then try Capture & match again.");
      if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;
      startQualityLoop();
      return;
    }

    camCanvas.width = camVideo.videoWidth;
    camCanvas.height = camVideo.videoHeight;
    camCanvas.getContext("2d").drawImage(camVideo, 0, 0);
    const blob = await new Promise((resolve) => camCanvas.toBlob(resolve, "image/png"));
    if (!blob) {
      setCamStatus("Could not capture frame from camera.");
      if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;
      startQualityLoop();
      return;
    }
    const body = new FormData();
    body.append("file", blob, "capture.png");
    let payload = null;
    try {
      const res = await fetch(
        `/api/camera/capture?session_id=${encodeURIComponent(state.cameraSessionId)}&compare_to_document=true`,
        { method: "POST", body }
      );
      payload = await res.json().catch(() => null);
      if (!res.ok || !payload || payload.status === "SESSION_NOT_FOUND" || payload.camera?.status === "ERROR") {
        const detail = (payload && (payload.detail?.error || payload.detail || payload.note || payload.status)) || `HTTP ${res.status}`;
        setCameraState("ACTIVE");
        setCamStatus(`Capture failed: ${typeof detail === "string" ? detail : "server error"}. Try again.`);
        if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;
        startQualityLoop();
        return;
      }
    } catch (err) {
      setCameraState("ACTIVE");
      setCamStatus("Network error during capture — check the API and retry.");
      if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;
      startQualityLoop();
      return;
    }
    const snap = $("camera-snap");
    if (snap) {
      snap.src = URL.createObjectURL(blob);
      snap.classList.remove("is-hidden");
    }
    camVideo.classList.add("is-hidden");
    if ($("btn-cam-retake")) $("btn-cam-retake").disabled = false;
    if ($("btn-cam-capture")) $("btn-cam-capture").disabled = false;

    const face = payload.face || {};
    const ver = face.verification || {};
    const L = payload.liveness || {};
    // Hard UI rule: detection without MATCH can never display PASS
    const comparedOk = ver.status === "COMPARED_UNCALIBRATED" || ver.status === "COMPARED";
    const isMatch = comparedOk && ver.result === "MATCH" && ver.similarity != null;
    const isNoMatch = comparedOk && ver.result === "NO_MATCH";
    fillKv($("camera-result"), [
      ["Capture", payload.status || "—"],
      ["Document face", (face.document_face || {}).status || face.document_face_status || "—"],
      ["Live face", (face.live_face || {}).status || face.status || "—"],
      ["Face count", String((face.live_face || {}).face_count ?? face.face_count ?? "—")],
      ["Comparison", comparedOk ? "RAN (doc↔live)" : "NOT RUN"],
      ["Verification", ver.result || "NOT_AVAILABLE"],
      ["Similarity", ver.similarity != null ? pct(ver.similarity) : "—"],
      ["Embedding sources", ver.embedding_sources || "—"],
      ["Calibration", ver.calibration || "UNCALIBRATED"],
      ["Liveness", L.decision || L.status || "NOT_ASSESSED"],
      ["Rule", "DETECTED ≠ MATCH ≠ PASS"],
    ]);

    const cmp = $("identity-compare-strip");
    if (cmp) {
      const docSrc = state.objectUrl || preview?.src || "";
      const liveSrc = snap?.src || "";
      const vres = ver.result || "NOT_AVAILABLE";
      cmp.classList.remove("is-hidden");
      cmp.innerHTML = `
        <div class="id-compare-col"><p class="tiny muted">DOCUMENT FACE</p><img alt="Document" src="${docSrc || ""}" /><p class="tiny mono">${escapeHtml((face.document_face || {}).status || face.document_face_status || "—")}</p></div>
        <div class="id-compare-mid mono">${escapeHtml(vres)}<br/><span class="tiny muted">sim ${ver.similarity != null ? pct(ver.similarity) : "n/a"}</span></div>
        <div class="id-compare-col"><p class="tiny muted">LIVE CAPTURE</p><img alt="Live capture" src="${liveSrc || ""}" /><p class="tiny mono">${escapeHtml((face.live_face || {}).status || face.status || "—")}</p></div>`;
    }

    if (state.lastResponse && state.lastResponse.evidence && state.lastResponse.evidence.decision_engine) {
      const priorFace = state.lastResponse.evidence.face || {};
      const mergedFace = {
        ...priorFace,
        ...face,
        // Preserve document detection from analyze; live capture supplies verification
        status: face.status,
        verification: ver,
        verification_debug: face.verification_debug || payload.verification_debug,
        document_face: face.document_face || priorFace.document_face,
        live_face: face.live_face,
        document_face_status: face.document_face_status || (face.document_face || {}).status || priorFace.status,
        liveness: L,
        camera: face.camera || payload.camera,
        reference: face.reference,
        face_verification: face.face_verification || ver.result,
      };
      state.lastResponse.evidence.face = mergedFace;
      fillFace(mergedFace);
      try {
        const refined = await fetch("/api/officer/refine-verdict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            decision_engine: state.lastResponse.evidence.decision_engine,
            face: mergedFace,
          }),
        }).then((r) => r.json());
        if (refined && refined.decision_engine) {
          // Never keep PASS unless compare produced MATCH
          if (!isMatch && refined.decision === "PASS") {
            refined.decision = "INCONCLUSIVE";
            if (refined.officer_verdict) {
              refined.officer_verdict.code = "INCONCLUSIVE";
              refined.officer_verdict.label = "INCONCLUSIVE";
              refined.officer_verdict.summary =
                "Face detected but not a real MATCH — INCONCLUSIVE.";
            }
            if (refined.decision_engine) refined.decision_engine.decision = "INCONCLUSIVE";
          }
          state.lastResponse.evidence.decision_engine = refined.decision_engine;
          state.lastResponse.assessment = {
            ...(state.lastResponse.assessment || {}),
            decision: refined.decision,
            officer_verdict: refined.officer_verdict,
            recommended_action: refined.decision_engine.recommended_action,
          };
        }
      } catch (_) { /* keep prior */ }
    }

    state.identityComplete = true;
    state.identityGateRequired = false;
    $("identity-gate")?.classList.add("is-hidden");
    setCameraState("COMPLETE");
    setCamStatus(
      isMatch
        ? `MATCH ${pct(ver.similarity)} · doc↔live compare OK · opening final result`
        : isNoMatch
          ? `NO_MATCH ${ver.similarity != null ? pct(ver.similarity) : "n/a"} → INCONCLUSIVE`
          : `No real MATCH (result=${ver.result || "NOT_AVAILABLE"}) → INCONCLUSIVE`
    );
    if (state.lastResponse) applyScreeningResult(state.lastResponse, { finalize: true, skipIdentityGate: true });
    else {
      setWorkflow("result");
      showView("result");
    }
  });

  /* —— Reports / audit —— */
  $("btn-load-report")?.addEventListener("click", async () => {
    if (!state.caseId) {
      $("report-preview").textContent = "No case yet.";
      return;
    }
    const r = await fetch(`/api/report/${state.caseId}?persist=false`).then((x) => x.json());
    $("report-preview").textContent = r.markdown || JSON.stringify(r, null, 2);
  });
  $("btn-verify-audit")?.addEventListener("click", async () => {
    const v = await fetch("/api/audit/chain/verify").then((x) => x.json());
    $("audit-integrity").textContent = `AUDIT INTEGRITY: ${v.valid ? "VERIFIED (local)" : v.status} · blockchain=${v.blockchain}`;
  });
  $("btn-load-audit")?.addEventListener("click", async () => {
    if (!state.caseId) return;
    const a = await fetch(`/api/audit/case/${state.caseId}`).then((x) => x.json());
    const list = $("audit-list");
    list.innerHTML = "";
    (a.events || []).forEach((e) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${(e.timestamp || "").slice(11, 19)} · ${e.action}</span><strong>${(e.entry_hash || "").slice(0, 10)}…</strong>`;
      list.appendChild(li);
    });
  });

  document.querySelectorAll("[data-map]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mapMode = btn.dataset.map;
      applyMapMode();
    });
  });
  $("heatmap-opacity")?.addEventListener("input", (e) => {
    const v = e.target.value;
    if ($("opacity-val")) $("opacity-val").textContent = `${v}%`;
    const heatOver = $("result-heat-overlay");
    if (heatOver) heatOver.style.opacity = String(Number(v) / 100);
  });
  document.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.goto));
  });
  $("btn-jump-heatmap")?.addEventListener("click", () => {
    showView("result");
    requestAnimationFrame(() => {
      $("compare-forensic")?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  /* —— Three.js boot —— */
  function bootThree() {
    if (!window.DocshieldThree?.ready) return;
    window.DocshieldThree.initHero($("hero-canvas"));
    window.DocshieldThree.initUpload($("upload-canvas"));
    window.DocshieldThree.initScanner($("scanner-canvas"));
    window.DocshieldThree.initGraph($("graph-canvas"));
    window.DocshieldThree.initIdentity($("identity-canvas"));
    window.DocshieldThree.setGraphData(["CASE", "DOC", "OCR", "FORENSICS", "FACE", "FUSION"]);
  }

  setCameraState("OFF");
  refreshHealth();
  setInterval(refreshHealth, 30000);
  bootThree();

  // Test / automation hook — applies a real /api/analyze payload (never fabricates fields)
  window.DocshieldApp = {
    applyScreeningResult,
    showView,
  };
})();
