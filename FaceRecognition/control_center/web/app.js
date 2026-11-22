(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const clientId = sessionStorage.getItem("niclaControlCenterClient") || (() => {
    const value = globalThis.crypto && typeof crypto.randomUUID === "function" ? crypto.randomUUID() : `cc-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    sessionStorage.setItem("niclaControlCenterClient", value);
    return value;
  })();
  const state = { mode: "replay", paused: false, polling: false, current: null, timer: null, history: [], hostDiagnosticEnabled: false, hostDiagnosticLastObservation: 0, hostDiagnosticCurrentObservation: 0, hostDiagnosticInFlight: false, hostRgbReferenceEnabled: false, hostRgbReferenceLastObservation: 0, hostRgbReferenceCurrentObservation: 0, hostRgbReferenceInFlight: false };
  let toastTimer = null;

  function toast(message, isError = false) {
    const box = $("toast");
    box.textContent = message;
    box.classList.toggle("error", isError);
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.hidden = true; }, 4500);
  }
  function setStatus(message, isError = false) {
    $("actionStatus").textContent = message;
    $("actionStatus").style.color = isError ? "#ff9b93" : "";
  }
  async function request(path, options = {}) {
    const response = await fetch(path, { cache: "no-store", ...options, headers: { "Content-Type": "application/json", "X-Control-Center-Client": clientId, ...(options.headers || {}) } });
    const type = response.headers.get("content-type") || "";
    const body = type.includes("application/json") ? await response.json() : await response.blob();
    if (!response.ok) throw new Error(body && body.error ? body.error : `HTTP ${response.status}`);
    return body;
  }
  async function post(path, body = {}) {
    return request(path, { method: "POST", body: JSON.stringify(body) });
  }
  function setMode(mode) {
    state.mode = mode;
    $("modeReplay").classList.toggle("primary", mode === "replay");
    $("modeLive").classList.toggle("primary", mode === "live");
    $("modeReplay").setAttribute("aria-pressed", String(mode === "replay"));
    $("modeLive").setAttribute("aria-pressed", String(mode === "live"));
    $("caseSelect").disabled = mode !== "replay";
    $("captureOnce").disabled = mode !== "live";
  }
  function badge(text, className) {
    const el = $("modeBadge");
    el.textContent = text;
    el.className = `badge ${className}`;
  }
  function showImage(url, alt) {
    const image = $("cameraImage");
    const empty = $("emptyFrame");
    image.onload = () => { image.hidden = false; empty.hidden = true; };
    image.onerror = () => { image.hidden = true; empty.hidden = false; };
    image.alt = alt;
    image.src = url;
  }
  function clearHostRgbReference(message = "Off. Preview-only mode remains active; the YuNet + SFace worker starts only after opt-in.") {
    state.hostRgbReferenceEnabled = false;
    state.hostRgbReferenceLastObservation = 0;
    state.hostRgbReferenceCurrentObservation = 0;
    $("hostRgbReferenceToggle").disabled = true;
    $("hostRgbReferenceToggle").textContent = "Enable HOST RGB REFERENCE";
    $("hostRgbReferenceState").textContent = message;
    $("hostRgbReferenceResult").hidden = true;
    $("rgbReferenceOverlay").replaceChildren();
    const crop = $("hostRgbReferenceCrop");
    crop.removeAttribute("src");
    crop.hidden = true;
  }
  function showLiveBlob(blob, observation) {
    const image = $("cameraImage");
    const previous = image.dataset.objectUrl;
    if (previous) URL.revokeObjectURL(previous);
    const objectUrl = URL.createObjectURL(blob);
    image.dataset.objectUrl = objectUrl;
    showImage(objectUrl, "Current Nicla camera preview through the existing serial owner; no detection overlay or decision is attached");
    $("frameId").textContent = `Host observation ${observation || "received"}`;
    $("frameSource").textContent = "LIVE CAMERA PREVIEW";
  }
  function renderReplay(event) {
    setMode("replay");
    state.hostDiagnosticEnabled = false;
    state.hostDiagnosticLastObservation = 0;
    state.hostDiagnosticCurrentObservation = 0;
    clearHostRgbReference("Off. Recorded replay does not run HOST RGB REFERENCE.");
    $("hostDiagnosticToggle").disabled = true;
    $("hostDiagnosticToggle").textContent = "Enable HOST RECONSTRUCTION";
    $("hostDiagnosticState").textContent = "Off. Recorded replay does not run host reconstruction.";
    $("hostDiagnosticResult").hidden = true;
    for (const id of ["hostDiagnosticDetectorOutputHash", "hostDiagnosticClassifierOutputHash", "hostDiagnosticCropCoords", "hostDiagnosticModelHashes"]) $(id).textContent = "—";
    for (const id of ["hostDiagnosticDetectorOutputBytes", "hostDiagnosticClassifierOutputBytes"]) $(id).value = "Not produced";
    $("frameOverlay").replaceChildren();
    $("connectLive").disabled = false;
    $("disconnectLive").disabled = true;
    badge("RECORDED REPLAY", "replay");
    $("connectionState").textContent = "No live device attached";
    $("cameraTitle").textContent = `Recorded scene preview · ${event.case}`;
    $("frameFreshness").textContent = "REPLAY";
    $("frameFreshness").className = "small-status";
    $("frameCaption").textContent = "Recorded scene preview saved before a historical board probe. The exact RGB565 frame and crop consumed by the board were not retained.";
    $("frameId").textContent = `Replay case ${event.frame_id}`;
    $("frameSource").textContent = "RECORDED REPLAY";
    $("frameAge").textContent = "Not applicable";
    showImage(`/api/v1/replay/frame?case=${encodeURIComponent(event.case)}&v=${Date.now()}`, `Recorded camera scene reference for ${event.case}; not the exact board tensor input`);
    $("decisionStatus").textContent = event.decision.status;
    $("decisionSubline").textContent = `${event.role} · one volatile probe · development evidence`;
    $("predictedClass").textContent = event.predicted_class || "No class emitted";
    $("thresholdValue").textContent = "Not retained";
    $("decisionWarning").textContent = "Historical board outcome; no live inference is attached to this preview.";
    $("reasonText").textContent = event.decision.reason;
    const clusterText = event.detection.status === "face" ? "FOMO accepted one cluster." : event.detection.status === "no_face" ? "FOMO accepted no clusters." : `FOMO state: ${event.detection.status}.`;
    const trace = [clusterText, "No device crop/tensor bytes, logits, or model scores are available for this row.", "Probe timing includes camera reset/warm-up and is not steady-state latency."];
    const list = $("traceList");
    list.replaceChildren(...trace.map(text => { const li = document.createElement("li"); li.textContent = text; return li; }));
    $("inputSource").textContent = event.model_input.source;
    $("inputShape").textContent = "Not retained";
    $("inputQuant").textContent = "Not retained";
    $("inputHash").textContent = "Not retained";
    $("modelHash").textContent = `enc ${event.model_sha256 || "—"} / head ${event.head_sha256 || "—"}`;
    $("previewFps").textContent = "Not measured in this probe";
    $("inferenceRate").textContent = "Not measured";
    $("probeTime").textContent = event.elapsed_ms == null ? "Not retained" : `${Number(event.elapsed_ms).toFixed(1)} ms*`;
    $("powerValue").textContent = "Not measured";
    $("memoryValue").textContent = "Not retained in this row";
    $("imuValue").textContent = "Not available";
    $("telemetryNote").textContent = "*Probe includes reset and warm-up. It is not steady-state capture-to-decision latency.";
    $("controlExplain").textContent = "Legacy v1 replay panel only. Use the frame-linked v2 Nicla operator controls above.";
    $("caseSelect").value = event.case;
    for (const row of $("historyRows").querySelectorAll("tr[data-case]")) {
      row.setAttribute("aria-current", String(row.dataset.case === event.case));
      row.tabIndex = 0;
    }
    state.current = event;
    state.history.unshift({ mode: "replay", case: event.case, status: event.decision.status, received: new Date().toLocaleTimeString() });
    state.history = state.history.slice(0, 50);
  }
  function renderLive(event) {
    setMode("live");
    const connected = event.connection.state === "connected";
    const fresh = connected && event.frame.stale === false;
    badge(fresh ? "LEGACY PREVIEW · READ ONLY" : (connected ? "LEGACY PREVIEW · STALE" : "PREVIEW OFFLINE"), fresh ? "nicla" : "offline");
    $("connectionState").textContent = fresh ? "Legacy preview connected · separate from v2 device events" : (connected ? "Legacy preview age unknown or stale" : "Legacy preview unavailable");
    $("cameraTitle").textContent = fresh ? "Live Nicla preview" : (connected ? "Nicla preview · stale" : "Live preview unavailable");
    $("frameFreshness").textContent = fresh ? "LIVE PREVIEW" : (connected ? "STALE / AGE UNKNOWN" : "STALE / OFFLINE");
    $("frameFreshness").className = "small-status";
    $("frameCaption").textContent = "Read-only preview from the existing serial owner. The dashboard does not control the camera and has no frame-matched FOMO/identity result.";
    $("frameId").textContent = event.frame_id ? `Host observation ${event.frame_id}` : "No frame received";
    $("frameSource").textContent = fresh ? "LIVE CAMERA PREVIEW" : (connected ? "STALE / UNKNOWN" : "NO FRAME");
    $("frameAge").textContent = event.frame.age_ms == null ? "—" : `${Number(event.frame.age_ms).toFixed(0)} ms`;
    $("captureOnce").disabled = !fresh;
    if (!fresh) {
      $("cameraImage").hidden = true;
      $("emptyFrame").hidden = false;
      $("emptyFrame").querySelector("strong").textContent = connected ? "Preview age unknown or stale" : "No live preview attached";
      $("emptyFrame").querySelector("span:last-child").textContent = connected
        ? "The current frame is not shown as live because its age is missing, invalid, or over 2 seconds."
        : "Optional legacy preview at 127.0.0.1:8766. Use explicit v2 Connect above for the Nicla broker.";
    }
    $("decisionStatus").textContent = "NOT RUN";
    $("decisionSubline").textContent = "No live detector or identity event for this frame.";
    $("predictedClass").textContent = "No decision";
    $("thresholdValue").textContent = "Not available";
    $("decisionWarning").textContent = "This legacy v1 panel has no frame-linked result. Inspect the v2 operator workspace above for broker events.";
    $("reasonText").textContent = event.decision.reason;
    const list = $("traceList");
    list.replaceChildren(...event.policy.trace.map(text => { const li = document.createElement("li"); li.textContent = text; return li; }));
    $("inputSource").textContent = "none";
    $("inputShape").textContent = "Not available";
    $("inputQuant").textContent = "Not available";
    $("inputHash").textContent = "Not available";
    $("modelHash").textContent = "Not available";
    $("previewFps").textContent = event.telemetry.preview_fps == null ? "—" : `${event.telemetry.preview_fps} fps`;
    $("inferenceRate").textContent = "Not available";
    $("probeTime").textContent = "Not available";
    $("powerValue").textContent = "Not measured";
    $("memoryValue").textContent = "Not available";
    $("imuValue").textContent = "Not available";
    $("telemetryNote").textContent = "Preview service cumulative average since start; no inference, IMU, memory, power, or device ACK telemetry is connected.";
    $("controlExplain").textContent = "Legacy v1 preview is read-only and separate from the v2 NiclaBroker. No device command is sent from this pane.";
    $("updatedAt").textContent = new Date().toLocaleTimeString();
    $("hostDiagnosticToggle").disabled = !fresh && !state.hostDiagnosticEnabled;
    $("hostDiagnosticToggle").textContent = state.hostDiagnosticEnabled ? "Disable HOST RECONSTRUCTION" : "Enable HOST RECONSTRUCTION";
    $("hostRgbReferenceToggle").disabled = !fresh && !state.hostRgbReferenceEnabled;
    $("hostRgbReferenceToggle").textContent = state.hostRgbReferenceEnabled ? "Disable HOST RGB REFERENCE" : "Enable HOST RGB REFERENCE";
    if (state.hostRgbReferenceEnabled && !fresh) {
      $("hostRgbReferenceState").textContent = "Latest preview is stale/unknown; HOST RGB REFERENCE is suppressed until a fresh JPEG arrives.";
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceResult").hidden = true;
    } else if (!fresh) {
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceResult").hidden = true;
    } else if (!state.hostRgbReferenceEnabled && connected) {
      $("hostRgbReferenceState").textContent = "Off. Preview-only mode remains active; no YuNet + SFace worker is running.";
    }
    if (state.hostDiagnosticEnabled && !fresh) {
      $("hostDiagnosticState").textContent = "Latest preview is stale/unknown; no host reconstruction runs until a fresh JPEG arrives.";
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticResult").hidden = true;
    } else if (!fresh) {
      $("hostDiagnosticResult").hidden = true;
    } else if (!state.hostDiagnosticEnabled && connected && fresh) {
      $("hostDiagnosticState").textContent = "Off. Legacy preview remains read-only; no host model worker is running.";
    }
    state.current = event;
  }
  function renderHostDiagnostic(result) {
    const source = result.source || {};
    const identity = result.identity || {};
    const detector = result.detector || {};
    $("hostDiagnosticResult").hidden = false;
    $("hostDiagnosticObservation").textContent = result.host_observation_index == null ? "—" : `Host observation ${result.host_observation_index}`;
    const snapshotAge = source.snapshot_age_ms_at_finish == null ? "unknown" : `${source.snapshot_age_ms_at_finish} ms host snapshot age`;
    const previewAge = source.preview_service_age_ms_at_finish == null ? "unknown" : `${source.preview_service_age_ms_at_finish} ms preview age`;
    $("hostDiagnosticAge").textContent = `${snapshotAge}; ${previewAge}`;
    $("hostDiagnosticJpegHash").textContent = source.jpeg_sha256 || "—";
    $("hostDiagnosticModelHashes").textContent = Object.entries(result.models || {}).map(([name, hash]) => `${name}: ${hash}`).join(" · ") || "—";
    $("hostDiagnosticDecision").textContent = String(identity.status || "not_run").replaceAll("_", " ").toUpperCase();
    $("hostDiagnosticCandidate").textContent = identity.candidate_class ? `${identity.candidate_class} · analysis only` : "No candidate class";
    $("hostDiagnosticReason").textContent = identity.reason || detector.status || "No result";
    $("hostDiagnosticManifestNote").textContent = result.manifest_runtime_config_matches === false
      ? "Runtime-config manifest digest differs from the pinned current config. Individual model/label hashes were checked; this host result remains experimental."
      : "Model and runtime-config hashes were checked before inference.";

    const regions = $("hostDiagnosticRegions");
    regions.replaceChildren();
    for (const item of detector.regions || []) {
      const li = document.createElement("li");
      li.textContent = `FOMO region xywh ${JSON.stringify(item.region_xywh_window_px)} px · center ${JSON.stringify(item.center_xy_window_px)} px · score ${Number(item.score).toFixed(4)} · grid ${JSON.stringify(item.grid_xy)}`;
      regions.appendChild(li);
    }
    if (!regions.children.length) {
      const li = document.createElement("li");
      li.textContent = detector.status === "no_face_regions" ? "No host FOMO regions passed the historical 0.7 gate." : "No regions returned.";
      regions.appendChild(li);
    }

    const scores = $("hostDiagnosticScores");
    scores.replaceChildren();
    if (Array.isArray(identity.raw_scores_int8) && Array.isArray(identity.raw_logits)) {
      const probabilities = identity.class_probabilities || [];
      (identity.class_order || []).forEach((label, index) => {
        const tr = document.createElement("tr");
        for (const value of [label, identity.raw_scores_int8[index], Number(identity.raw_logits[index]).toFixed(6), probabilities[index] == null ? "—" : Number(probabilities[index]).toFixed(6)]) {
          const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
        }
        scores.appendChild(tr);
      });
      $("hostDiagnosticThreshold").textContent = `Historical alpha0.25 max-known softmax threshold ${identity.threshold} ± ${identity.uncertainty_band}; comparison is experimental only. No identity is emitted.`;
    } else {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 4;
      td.textContent = `Classifier not run: ${identity.reason || "not reached"}`;
      tr.appendChild(td); scores.appendChild(tr);
      $("hostDiagnosticThreshold").textContent = `Historical alpha0.25 threshold ${identity.threshold}; classifier was not run.`;
    }

    const crop = identity.crop;
    const canvas = $("hostDiagnosticCrop");
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (crop && crop.bytes_base64 && crop.shape && crop.shape[0] === 96 && crop.shape[1] === 96) {
      const bytes = Uint8Array.from(atob(crop.bytes_base64), ch => ch.charCodeAt(0));
      if (bytes.length === 96 * 96) {
        const pixels = context.createImageData(96, 96);
        for (let i = 0; i < bytes.length; i++) {
          const j = i * 4; pixels.data[j] = bytes[i]; pixels.data[j + 1] = bytes[i]; pixels.data[j + 2] = bytes[i]; pixels.data[j + 3] = 255;
        }
        context.putImageData(pixels, 0, 0);
      }
    }
    $("hostDiagnosticCropHash").textContent = crop?.sha256 || "—";
    $("hostDiagnosticCropCoords").textContent = crop?.xywh_window_px ? JSON.stringify(crop.xywh_window_px) : "—";
    $("hostDiagnosticDetectorHash").textContent = detector.input_tensor?.sha256 || "—";
    $("hostDiagnosticDetectorOutputHash").textContent = detector.raw_output_tensor?.sha256 || "—";
    $("hostDiagnosticClassifierHash").textContent = identity.input_tensor?.sha256 || "—";
    $("hostDiagnosticClassifierOutputHash").textContent = identity.raw_output_tensor?.sha256 || "—";
    $("hostDiagnosticDetectorBytes").value = detector.input_tensor?.bytes_base64 || "Not produced";
    $("hostDiagnosticDetectorOutputBytes").value = detector.raw_output_tensor?.bytes_base64 || "Not produced";
    $("hostDiagnosticCropBytes").value = crop?.bytes_base64 || "Not produced";
    $("hostDiagnosticClassifierBytes").value = identity.input_tensor?.bytes_base64 || "Not produced";
    $("hostDiagnosticClassifierOutputBytes").value = identity.raw_output_tensor?.bytes_base64 || "Not produced";
    const timingList = $("hostDiagnosticTimings");
    timingList.replaceChildren();
    for (const [label, value] of Object.entries(result.timings_ms || {})) {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label.replaceAll("_", " "); dd.textContent = `${Number(value).toFixed(3)} ms`;
      div.append(dt, dd); timingList.appendChild(div);
    }
    renderHostRegionsOverlay(detector.regions || [], crop?.xywh_window_px || null);
    $("hostDiagnosticState").textContent = `HOST RECONSTRUCTION completed for observation ${result.host_observation_index}; JPEG-derived, experimental, not a device result.`;
  }
  function renderHostRgbReference(result) {
    const source = result.source || {};
    const detector = result.detector || {};
    const recognition = result.recognition || {};
    const hashes = result.models || {};
    const observation = result.host_observation_index;
    $("hostRgbReferenceResult").hidden = false;
    $("hostRgbReferenceObservation").textContent = Number.isSafeInteger(observation) ? `Host observation ${observation} · no device frame ID` : "Unknown host observation";
    $("hostRgbReferenceAge").textContent = `snapshot ${source.snapshot_age_ms_at_finish ?? "unknown"} ms · preview ${source.preview_service_age_ms_at_finish ?? "unknown"} ms`;
    $("hostRgbReferenceJpegHash").textContent = source.jpeg_sha256 || "—";
    const detectorScore = detector.score == null ? "not available" : Number(detector.score).toFixed(4);
    $("hostRgbReferenceDetector").textContent = `${detector.name || "YuNet"}: ${detector.status || "unknown"} · score ${detectorScore} · threshold ${detector.score_threshold ?? "—"}`;
    $("hostRgbReferenceBbox").textContent = Array.isArray(detector.bbox_xywh_240_px) ? JSON.stringify(detector.bbox_xywh_240_px) : "No single-face box";
    const rawCandidate = recognition.raw_top1_known_class;
    const candidateScore = rawCandidate == null ? null : recognition.known_cosines?.[rawCandidate];
    $("hostRgbReferenceCandidate").textContent = rawCandidate
      ? `${rawCandidate} · raw top-1 only · ${candidateScore == null ? "score unavailable" : Number(candidateScore).toFixed(6)}`
      : "No candidate (SFace suppressed)";
    const knownGate = recognition.known_cosine_gate || {};
    const marginGate = recognition.known_vs_other_margin_gate || {};
    const gateLabel = (gate) => gate.accepted === true ? "threshold passed" : gate.accepted === false ? "rejected" : "not run";
    const passCount = Number(knownGate.accepted === true) + Number(marginGate.accepted === true);
    $("hostRgbReferenceDecision").textContent = passCount === 2
      ? "Both host reference gates pass · still unqualified"
      : `Rejected or incomplete · ${passCount}/2 host gates pass · no identity emitted`;
    $("hostRgbReferenceReasons").textContent = `max cosine ${gateLabel(knownGate)} (${knownGate.reason || "not_run"}); known-vs-OTHER margin ${gateLabel(marginGate)} (${marginGate.reason || "not_run"})`;
    $("hostRgbReferenceModelHashes").textContent = `YuNet ${hashes.yunet_model || "—"} · SFace ${hashes.sface_model || "—"}`;

    const scoreRows = $("hostRgbReferenceScores");
    scoreRows.replaceChildren();
    const scores = recognition.known_cosines && typeof recognition.known_cosines === "object"
      ? [...Object.entries(recognition.known_cosines), ["OTHER", recognition.other_cosine]]
      : [];
    if (!scores.length) {
      const tr = document.createElement("tr"); const td = document.createElement("td");
      td.colSpan = 2; td.textContent = `SFace not run: ${recognition.reason || "no single YuNet detection"}`;
      tr.appendChild(td); scoreRows.appendChild(tr);
    } else {
      for (const [label, score] of scores) {
        const tr = document.createElement("tr");
        const name = document.createElement("td"); name.textContent = label === rawCandidate ? `${label} · raw top-1` : label;
        const value = document.createElement("td"); value.textContent = Number(score).toFixed(8);
        tr.append(name, value); scoreRows.appendChild(tr);
      }
    }
    $("hostRgbReferenceThresholds").textContent = `Known cosine gate: max(${rawCandidate || "known"}) ${knownGate.operator || ">="} ${knownGate.threshold ?? recognition.threshold_known_max_cosine ?? "—"} · ${gateLabel(knownGate)}. Known − OTHER margin ${marginGate.operator || ">="} ${marginGate.threshold ?? recognition.threshold_known_other_margin ?? "—"}; observed ${marginGate.observed ?? "—"} · ${gateLabel(marginGate)}. Both are historical host-only thresholds; they do not qualify recognition.`;

    const crop = recognition.aligned_crop;
    const cropImage = $("hostRgbReferenceCrop");
    if (crop?.bytes_base64 && crop.encoding === "PNG") {
      cropImage.src = `data:image/png;base64,${crop.bytes_base64}`;
      cropImage.hidden = false;
      $("hostRgbReferenceCropHash").textContent = `In-memory crop PNG SHA-256 ${crop.sha256} · ${JSON.stringify(crop.shape)} ${crop.color_order}`;
    } else {
      cropImage.removeAttribute("src"); cropImage.hidden = true;
      $("hostRgbReferenceCropHash").textContent = "No aligned crop produced for this observation.";
    }
    const timingList = $("hostRgbReferenceTimings");
    timingList.replaceChildren();
    for (const [label, value] of Object.entries(result.timings_ms || {})) {
      const div = document.createElement("div"); const dt = document.createElement("dt"); const dd = document.createElement("dd");
      dt.textContent = label.replaceAll("_", " "); dd.textContent = `${Number(value).toFixed(3)} ms`;
      div.append(dt, dd); timingList.appendChild(div);
    }
    const calibration = result.calibration || {};
    $("hostRgbReferenceCalibration").textContent = `${calibration.source || "—"} · usable unknown calibration people ${calibration.unknown_calibration_usable_people ?? "—"} · pretraining exposure ${calibration.pretraining_exposure || "unresolved"}`;
    $("hostRgbReferenceRunHash").textContent = hashes.run_result || "—";
    $("hostRgbReferenceGallery").textContent = JSON.stringify(calibration.gallery_source_groups || {});
    $("hostRgbReferenceScope").textContent = result.evidence_scope || "HOST REFERENCE ONLY; NOT ON NICLA";
    renderRgbReferenceOverlay(detector.bbox_xywh_240_px, detector.score);
    $("hostRgbReferenceState").textContent = `HOST RGB REFERENCE complete for observation ${observation}; lossy-JPEG-derived and unqualified. No identity is emitted.`;
  }
  function renderRgbReferenceOverlay(bbox, score) {
    const overlay = $("rgbReferenceOverlay");
    overlay.replaceChildren();
    if (!Array.isArray(bbox) || bbox.length !== 4 || !bbox.every(Number.isFinite)) return;
    const stage = $("imageStage").getBoundingClientRect();
    const square = Math.min(stage.width, stage.height);
    const leftPad = (stage.width - square) / 2;
    const topPad = (stage.height - square) / 2;
    const [x, y, w, h] = bbox;
    const rect = document.createElement("div");
    Object.assign(rect.style, { position: "absolute", left: `${leftPad + x / 240 * square}px`, top: `${topPad + y / 240 * square}px`, width: `${w / 240 * square}px`, height: `${h / 240 * square}px`, border: "2px solid #70d7ff", background: "rgba(80,190,255,.08)" });
    rect.title = `YuNet host reference bbox; score ${Number(score).toFixed(4)}; lossy preview JPEG only`;
    overlay.appendChild(rect);
  }
  function renderHostRegionsOverlay(regions, crop) {
    const overlay = $("frameOverlay");
    overlay.replaceChildren();
    const stage = $("imageStage").getBoundingClientRect();
    const square = Math.min(stage.width, stage.height);
    const leftPad = (stage.width - square) / 2;
    const topPad = (stage.height - square) / 2;
    for (const item of regions) {
      const rect = document.createElement("div");
      const [x, y, w, h] = item.region_xywh_window_px;
      Object.assign(rect.style, { position: "absolute", left: `${leftPad + x / 240 * square}px`, top: `${topPad + y / 240 * square}px`, width: `${w / 240 * square}px`, height: `${h / 240 * square}px`, border: "2px solid #ff8278", background: "rgba(255,130,120,.08)" });
      rect.title = `FOMO region, score ${Number(item.score).toFixed(3)}; not a face box`;
      overlay.appendChild(rect);
    }
    if (crop) {
      const [x, y, w, h] = crop;
      const rect = document.createElement("div");
      Object.assign(rect.style, { position: "absolute", left: `${leftPad + x / 240 * square}px`, top: `${topPad + y / 240 * square}px`, width: `${w / 240 * square}px`, height: `${h / 240 * square}px`, border: "2px dashed #73d2a2", background: "transparent" });
      rect.title = "Experimental host grayscale crop; derived from lossy JPEG";
      overlay.appendChild(rect);
    }
  }
  function renderHistory(cases) {
    const body = $("historyRows");
    const fragment = document.createDocumentFragment();
    for (const row of cases) {
      const tr = document.createElement("tr");
      tr.dataset.case = row.case;
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-label", `Replay ${row.case}: ${row.decision}; candidate ${row.predicted_class || "none"}`);
      for (const value of [row.case, row.role, row.decision, row.predicted_class || "—", `${Number(row.elapsed_ms).toFixed(1)} ms`]) {
        const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
      }
      const activate = () => selectCase(row.case);
      tr.addEventListener("click", activate);
      tr.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
      fragment.appendChild(tr);
    }
    body.replaceChildren(fragment);
  }
  async function selectCase(caseName) {
    try {
      const event = await post("/api/v1/replay/select", { case: caseName });
      renderReplay(event);
      setStatus(`Loaded recorded ${caseName} development row. No board command was sent.`);
    } catch (error) { setStatus(`Replay selection failed: ${error.message}`, true); toast(error.message, true); }
  }
  async function attachLive() {
    setStatus("Checking existing preview over loopback. Serial remains owned by live_camera.py.");
    try {
      const result = await post("/api/v1/live/connect");
      setMode("live");
      await refreshLive();
      $("disconnectLive").disabled = false;
      $("connectLive").disabled = true;
      setStatus(result.note || "Attached to the existing read-only preview.");
      toast("Live preview attached; this dashboard sent no serial command.");
    } catch (error) {
      try { await post("/api/v1/live/disconnect"); } catch (_) { /* service may be unreachable; visual state still resets */ }
      setMode("live");
      state.hostDiagnosticEnabled = false;
      state.hostDiagnosticLastObservation = 0;
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticResult").hidden = true;
      clearHostRgbReference("Off. Preview unavailable; HOST RGB REFERENCE did not run.");
      renderLive({
        connection: { state: "disconnected" },
        frame: { stale: true, age_ms: null },
        frame_id: null,
        decision: { status: "not_run", reason: "No live preview is attached; no current-frame inference exists." },
        policy: { trace: ["Preview attachment failed; no live image or result is shown."] },
        telemetry: { preview_fps: null },
      });
      setStatus(`${error.message} The dashboard did not open serial or change the device.`, true);
      toast(error.message, true);
    }
  }
  async function detachLive() {
    try {
      await post("/api/v1/live/disconnect");
      $("disconnectLive").disabled = true;
      $("connectLive").disabled = false;
      await loadReplay($("caseSelect").value);
      setStatus("Dashboard detached. Existing preview owner and camera state were unchanged.");
    } catch (error) { setStatus(error.message, true); toast(error.message, true); }
  }
  async function loadReplay(caseName = "K01") {
    try {
      if (state.mode === "live") {
        await post("/api/v1/live/disconnect");
        $("connectLive").disabled = false;
        $("disconnectLive").disabled = true;
      }
      const response = await request("/api/v1/replay/cases");
      renderHistory(response.cases);
      const event = await post("/api/v1/replay/select", { case: caseName });
      renderReplay(event);
      setStatus("Recorded development replay loaded. The run is historical; it is not live recognition.");
      $("updatedAt").textContent = new Date().toLocaleTimeString();
    } catch (error) { setStatus(`Replay unavailable: ${error.message}`, true); toast(error.message, true); }
  }
  async function refreshLive() {
    const before = await request("/api/v1/state");
    if (before.connection.state !== "connected" || before.frame.stale) {
      renderLive(before);
      return;
    }
    let blob;
    let observation;
    try {
      const response = await fetch(`/api/v1/live/snapshot.jpg?seq=${Date.now()}`, { cache: "no-store", headers: { "X-Control-Center-Client": clientId } });
      if (!response.ok) throw new Error(`Preview frame request failed: HTTP ${response.status}`);
      blob = await response.blob();
      observation = response.headers.get("X-Preview-Observation");
    } catch (error) {
      before.connection.state = "error";
      before.frame.stale = true;
      before.policy.trace = [...before.policy.trace, "Snapshot read failed; last image hidden/marked stale."];
      renderLive(before);
      throw error;
    }
    const event = await request("/api/v1/state");
    renderLive(event);
    if (event.connection.state !== "connected" || event.frame.stale !== false) return;
    const observationIndex = Number(observation);
    if (!Number.isSafeInteger(observationIndex) || observationIndex < 1 || observationIndex !== event.frame_id) {
      event.frame.stale = true;
      event.frame.age_ms = null;
      event.decision.reason = "Snapshot observation binding is missing or does not match the host event.";
      renderLive(event);
      throw new Error("preview snapshot observation could not be bound to a fresh host observation");
    }
    if (observationIndex !== state.hostDiagnosticCurrentObservation) {
      state.hostDiagnosticCurrentObservation = observationIndex;
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticResult").hidden = true;
      if (state.hostDiagnosticEnabled) $("hostDiagnosticState").textContent = `Fresh JPEG received for host observation ${observationIndex}; waiting for HOST RECONSTRUCTION.`;
    }
    if (observationIndex !== state.hostRgbReferenceCurrentObservation) {
      state.hostRgbReferenceCurrentObservation = observationIndex;
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceResult").hidden = true;
      if (state.hostRgbReferenceEnabled) $("hostRgbReferenceState").textContent = `Fresh color JPEG received for host observation ${observationIndex}; waiting for HOST RGB REFERENCE.`;
    }
    event.host_observation_index = observationIndex;
    state.current = event;
    showLiveBlob(blob, observationIndex);
    $("frameAge").textContent = event.frame.age_ms == null ? "—" : `${Number(event.frame.age_ms).toFixed(0)} ms`;
  }
  async function runHostDiagnosticForCurrent() {
    const event = state.current;
    const observation = event && event.host_observation_index;
    if (state.hostDiagnosticInFlight || !state.hostDiagnosticEnabled || state.mode !== "live" || !event || event.frame.stale !== false || !Number.isSafeInteger(observation)) return;
    if (observation <= state.hostDiagnosticLastObservation) return;
    state.hostDiagnosticInFlight = true;
    try {
      const status = await request("/api/v1/live/diagnostic/status");
      if (status.status !== "ready") {
        $("hostDiagnosticState").textContent = status.error
          ? `HOST RECONSTRUCTION unavailable: ${status.error}`
          : `HOST RECONSTRUCTION worker ${status.status || "starting"}; waiting for a fresh preview.`;
        return;
      }
      $("hostDiagnosticState").textContent = `Reconstructing from lossy JPEG for host observation ${observation}…`;
      const result = await post("/api/v1/live/diagnostic/run", { host_observation_index: observation });
      state.hostDiagnosticLastObservation = observation;
      if (result.host_observation_index !== observation || state.hostDiagnosticCurrentObservation !== observation) {
        $("hostDiagnosticResult").hidden = true;
        $("frameOverlay").replaceChildren();
        $("hostDiagnosticState").textContent = `Discarded result: observation ${observation} was superseded before display.`;
        return;
      }
      renderHostDiagnostic(result);
    } catch (error) {
      state.hostDiagnosticLastObservation = observation;
      $("hostDiagnosticResult").hidden = true;
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticState").textContent = `HOST RECONSTRUCTION suppressed/failed for observation ${observation}: ${error.message}`;
    } finally {
      state.hostDiagnosticInFlight = false;
    }
  }
  async function runHostRgbReferenceForCurrent() {
    const event = state.current;
    const observation = event && event.host_observation_index;
    if (state.hostRgbReferenceInFlight || !state.hostRgbReferenceEnabled || state.mode !== "live" || !event || event.frame.stale !== false || !Number.isSafeInteger(observation)) return;
    if (observation <= state.hostRgbReferenceLastObservation) return;
    state.hostRgbReferenceInFlight = true;
    try {
      const status = await request("/api/v1/live/rgb-reference/status");
      if (status.status !== "ready") {
        $("hostRgbReferenceState").textContent = status.error
          ? `HOST RGB REFERENCE unavailable: ${status.error}`
          : `HOST RGB REFERENCE worker ${status.status || "starting"}; waiting for a fresh preview.`;
        return;
      }
      $("hostRgbReferenceState").textContent = `Analyzing exact color JPEG for host observation ${observation} with YuNet + SFace…`;
      const result = await post("/api/v1/live/rgb-reference/run", { host_observation_index: observation });
      state.hostRgbReferenceLastObservation = observation;
      if (result.host_observation_index !== observation || state.hostRgbReferenceCurrentObservation !== observation || !state.current || state.current.frame.stale !== false) {
        $("hostRgbReferenceResult").hidden = true;
        $("rgbReferenceOverlay").replaceChildren();
        $("hostRgbReferenceState").textContent = `Discarded result: observation ${observation} was superseded or stale before display.`;
        return;
      }
      renderHostRgbReference(result);
    } catch (error) {
      state.hostRgbReferenceLastObservation = observation;
      $("hostRgbReferenceResult").hidden = true;
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceState").textContent = `HOST RGB REFERENCE suppressed/failed for observation ${observation}: ${error.message}`;
    } finally {
      state.hostRgbReferenceInFlight = false;
    }
  }
  async function poll() {
    if (!$("legacyWorkspace").open || state.polling || state.paused) return;
    state.polling = true;
    try {
      const health = await request("/api/v1/health");
      $("serviceVersion").textContent = `Service v${health.version} · ${health.bind}`;
      if (state.mode === "live") {
        await refreshLive();
        const event = state.current;
        $("connectLive").disabled = event.connection.state === "connected";
        $("disconnectLive").disabled = event.connection.state !== "connected";
        if (event.connection.state === "connected" && event.frame.stale === false) {
          if (state.hostDiagnosticEnabled) await runHostDiagnosticForCurrent();
          if (state.hostRgbReferenceEnabled) await runHostRgbReferenceForCurrent();
          setStatus("Live preview only. Optional host diagnostics are derived from the lossy JPEG; no device decision stream is connected.");
        }
        else if (event.connection.state === "connected") setStatus(event.frame.age_ms == null ? "Preview age unknown; no frame is shown as live." : "Preview is stale; no frame is shown as live.", true);
      }
    } catch (error) {
      $("connectionState").textContent = "Control-center service unavailable";
      setStatus(`Cannot reach local service: ${error.message}`, true);
    } finally { state.polling = false; }
  }

  $("modeReplay").addEventListener("click", () => loadReplay($("caseSelect").value));
  $("modeLive").addEventListener("click", async () => {
    if (state.mode === "live") return;
    if (!$("connectLive").disabled) {
      setStatus("Attach to the existing preview first. This dashboard will not open serial.");
      return;
    }
    setMode("live");
    try { await refreshLive(); setStatus("Live preview only. Decision/inference is not connected."); }
    catch (error) { setStatus(error.message, true); }
  });
  $("connectLive").addEventListener("click", attachLive);
  $("disconnectLive").addEventListener("click", detachLive);
  $("hostDiagnosticToggle").addEventListener("click", async () => {
    const button = $("hostDiagnosticToggle");
    button.disabled = true;
    if (state.hostDiagnosticEnabled) {
      try {
        await post("/api/v1/live/diagnostic/disable");
        state.hostDiagnosticEnabled = false;
        state.hostDiagnosticLastObservation = state.hostDiagnosticCurrentObservation;
        $("hostDiagnosticResult").hidden = true;
        $("frameOverlay").replaceChildren();
        $("hostDiagnosticState").textContent = "Off. Worker stopped; the legacy preview remains read-only.";
        setStatus("HOST RECONSTRUCTION disabled. No device command was sent.");
      } catch (error) {
        $("hostDiagnosticState").textContent = `Could not stop host worker: ${error.message}`;
        toast(error.message, true);
      }
      button.disabled = !state.hostDiagnosticEnabled && (!state.current || state.current.frame.stale !== false);
      button.textContent = state.hostDiagnosticEnabled ? "Disable HOST RECONSTRUCTION" : "Enable HOST RECONSTRUCTION";
      return;
    }
    state.hostDiagnosticEnabled = true;
    state.hostDiagnosticLastObservation = 0;
    $("hostDiagnosticState").textContent = "Starting local TF2.10 worker. Preview-only mode remains active; this may take several seconds.";
    button.textContent = "Disable HOST RECONSTRUCTION";
    try {
      const result = await post("/api/v1/live/diagnostic/enable");
      if (result.status !== "ready") throw new Error(result.error || `worker status ${result.status || "unknown"}`);
      $("hostDiagnosticState").textContent = "Worker ready. Each fresh JPEG observation will be reconstructed once; results are experimental/unqualified.";
      setStatus("HOST RECONSTRUCTION enabled for lossy preview JPEGs. No board inference or serial command is involved.");
      await runHostDiagnosticForCurrent();
    } catch (error) {
      try { await post("/api/v1/live/diagnostic/disable"); } catch (_) { /* local worker start already failed or service ended */ }
      state.hostDiagnosticEnabled = false;
      $("hostDiagnosticResult").hidden = true;
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticState").textContent = `HOST RECONSTRUCTION unavailable: ${error.message}`;
      setStatus(`Host model worker could not start: ${error.message}`, true);
      toast(error.message, true);
    } finally {
      button.disabled = !state.current || state.current.frame.stale !== false;
      button.textContent = "Enable HOST RECONSTRUCTION";
      if (state.hostDiagnosticEnabled) button.textContent = "Disable HOST RECONSTRUCTION";
    }
  });
  $("hostRgbReferenceToggle").addEventListener("click", async () => {
    const button = $("hostRgbReferenceToggle");
    button.disabled = true;
    if (state.hostRgbReferenceEnabled) {
      try {
        await post("/api/v1/live/rgb-reference/disable");
        state.hostRgbReferenceEnabled = false;
        state.hostRgbReferenceLastObservation = state.hostRgbReferenceCurrentObservation;
        $("hostRgbReferenceResult").hidden = true;
        $("rgbReferenceOverlay").replaceChildren();
        $("hostRgbReferenceState").textContent = "Off. HOST RGB REFERENCE worker stopped; preview remains preview-only.";
        setStatus("HOST RGB REFERENCE disabled. No device command was sent.");
      } catch (error) {
        $("hostRgbReferenceState").textContent = `Could not stop HOST RGB REFERENCE worker: ${error.message}`;
        toast(error.message, true);
      }
      button.disabled = !state.hostRgbReferenceEnabled && (!state.current || state.current.frame.stale !== false);
      button.textContent = state.hostRgbReferenceEnabled ? "Disable HOST RGB REFERENCE" : "Enable HOST RGB REFERENCE";
      return;
    }
    state.hostRgbReferenceEnabled = true;
    state.hostRgbReferenceLastObservation = 0;
    $("hostRgbReferenceState").textContent = "Starting separate local OpenCV worker; preview-only mode remains active.";
    button.textContent = "Disable HOST RGB REFERENCE";
    try {
      const result = await post("/api/v1/live/rgb-reference/enable");
      if (result.status !== "ready") throw new Error(result.error || `worker status ${result.status || "unknown"}`);
      $("hostRgbReferenceState").textContent = "Pinned YuNet + SFace worker ready. Results remain experimental and unqualified.";
      setStatus("HOST RGB REFERENCE enabled for the exact cached color preview JPEG. No board inference or serial command is involved.");
      await runHostRgbReferenceForCurrent();
    } catch (error) {
      try { await post("/api/v1/live/rgb-reference/disable"); } catch (_) { /* startup failure or service ended */ }
      state.hostRgbReferenceEnabled = false;
      $("hostRgbReferenceResult").hidden = true;
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceState").textContent = `HOST RGB REFERENCE unavailable: ${error.message}`;
      setStatus(`YuNet + SFace host worker could not start: ${error.message}`, true);
      toast(error.message, true);
    } finally {
      button.disabled = !state.current || state.current.frame.stale !== false;
      button.textContent = state.hostRgbReferenceEnabled ? "Disable HOST RGB REFERENCE" : "Enable HOST RGB REFERENCE";
    }
  });
  $("caseSelect").addEventListener("change", (event) => selectCase(event.target.value));
  $("captureOnce").addEventListener("click", async () => {
    if (state.polling) {
      setStatus("Wait for the current preview poll to finish before capturing another observation.");
      return;
    }
    state.polling = true;
    try {
      const response = await fetch("/api/v1/live/capture", { method: "POST", cache: "no-store", headers: { "Content-Type": "application/json", "X-Control-Center-Client": clientId }, body: "{}" });
      if (!response.ok) { const body = await response.json(); throw new Error(body.error || `HTTP ${response.status}`); }
      const blob = await response.blob();
      const observation = Number(response.headers.get("X-Preview-Observation"));
      const event = await request("/api/v1/state");
      if (!Number.isSafeInteger(observation) || observation < 1 || event.frame_id !== observation || event.frame.stale !== false) {
        throw new Error("captured preview observation became stale or failed host-event binding");
      }
      renderLive(event);
      event.host_observation_index = observation;
      state.current = event;
      state.hostDiagnosticCurrentObservation = observation;
      state.hostRgbReferenceCurrentObservation = observation;
      $("frameOverlay").replaceChildren();
      $("hostDiagnosticResult").hidden = true;
      $("rgbReferenceOverlay").replaceChildren();
      $("hostRgbReferenceResult").hidden = true;
      showLiveBlob(blob, observation);
      $("frameAge").textContent = "Single observation";
      setStatus(state.hostDiagnosticEnabled
        ? "Captured one read-only preview observation; optional JPEG-derived host reconstruction is starting. No board inference or command was sent."
        : "Saved one read-only preview observation in browser memory; no inference or board command was run.");
      if (state.hostDiagnosticEnabled) await runHostDiagnosticForCurrent();
      if (state.hostRgbReferenceEnabled) await runHostRgbReferenceForCurrent();
    } catch (error) { setStatus(error.message, true); toast(error.message, true); }
    finally { state.polling = false; }
  });
  $("pauseUpdates").addEventListener("click", () => {
    state.paused = !state.paused;
    $("pauseUpdates").textContent = state.paused ? "Resume UI updates" : "Pause UI updates";
    setStatus(state.paused ? "UI refresh paused; device and camera owner were not changed." : "UI refresh resumed.");
    if (!state.paused) poll();
  });
  $("shutdown").addEventListener("click", async () => {
    if (!window.confirm("Stop only the control-center web service? The separate camera preview owner will keep running.")) return;
    try { const result = await post("/api/v1/shutdown", {}); setStatus(result.message); $("shutdown").disabled = true; }
    catch (error) { setStatus(`Could not stop the dashboard: ${error.message}`, true); }
  });

  window.addEventListener("pagehide", () => {
    const headers = { "Content-Type": "application/json", "X-Control-Center-Client": clientId };
    if (state.hostDiagnosticEnabled) fetch("/api/v1/live/diagnostic/disable", { method: "POST", keepalive: true, headers, body: "{}" }).catch(() => {});
    if (state.hostRgbReferenceEnabled) fetch("/api/v1/live/rgb-reference/disable", { method: "POST", keepalive: true, headers, body: "{}" }).catch(() => {});
  });

  const legacyWorkspace = $("legacyWorkspace");
  let replayLoaded = false;
  function syncLegacyPolling() {
    clearInterval(state.timer);
    state.timer = null;
    if (!legacyWorkspace.open) return;
    state.timer = setInterval(poll, 900);
    if (replayLoaded) {
      poll();
    } else {
      replayLoaded = true;
      loadReplay().then(poll);
    }
  }
  legacyWorkspace.addEventListener("toggle", syncLegacyPolling);
  syncLegacyPolling();
})();
