(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const clientId = sessionStorage.getItem("niclaControlCenterClient") || (() => {
    const value = crypto.randomUUID ? crypto.randomUUID() : `cc-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    sessionStorage.setItem("niclaControlCenterClient", value);
    return value;
  })();
  const tabId = sessionStorage.getItem("niclaOperatorTab") || (() => {
    const value = crypto.randomUUID ? crypto.randomUUID() : `tab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    sessionStorage.setItem("niclaOperatorTab", value);
    return value;
  })();
  const query = new URLSearchParams(location.search);
  const fixture = query.get("fixture") === "1";
  const MAX_EVENTS = 50;
  const MAX_RECORDING = 100;
  const REQUEST_TIMEOUT_MS = 2200;
  const COMMAND_TERMINAL = new Set(["completed", "done", "error", "unsupported", "rejected", "timeout", "busy"]);
  const STALE_AFTER_MS = 2500;
  const LOCK_KEY = "nicla-control-center-controller";
  const VIDEO_PATH = "/api/v2/device/video";
  const VIDEO_STATES = ["idle", "starting", "recording", "saving", "ready", "error"];
  const SIGHTINGS_PATH = "/api/v2/device/sightings";
  const BATTERY_PATH = "/api/v2/device/battery";
  const BATTERY_BUTTONS = { arm: "batteryArm", disarm: "batteryDisarm", import: "batteryImport" };
  const SIGHTING_PAGE_SIZE = 12;
  const SIGHTING_PEOPLE = ["K01", "K02", "K03", "K04", "K05", "other"];
  const PERFORMANCE_WINDOW = 20;
  const sightingDate = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" });
  const clipReview = { manifest: null, index: 0, request: 0, playing: false, timer: null };
  const state = {
    fixture, paused: false, polling: false, status: null, capability: null, events: [], seen: new Set(),
    nextSeq: 0, selectedEvent: null, resultFrozen: false, latestFrame: null, latestReportedFrame: null, latestReceivedAt: null, resultReceivedAt: null, resultAgeBaseMs: 0,
    latestUrl: null, matchedUrl: null, matchedFrameVerified: null, imageRequest: { latest: 0, matched: 0 }, imageGeometry: { latest: null, matched: null }, zoom: { latest: 1, matched: 1 },
    controllerAuthority: false, controllerId: null, controllerLeaseTimer: null, heartbeatTimer: null, pending: new Map(), loopRunning: false, applyingSettings: false,
    sleeping: false, model: null, policy: null, modelDraft: null, policyDraft: null, measurementStageDraft: null, recording: false, recordingEvents: [], fixtureFrame: 0,
    hostReferenceKey: null, hostReferenceBusy: false, hostReference: null,
    video: null, videoAvailable: false, videoAction: null, videoRevision: 0, videoError: null,
    sightings: null, sightingsAvailable: null, sightingsBusy: false, sightingsPolling: false,
    sightingsRevision: 0, sightingsError: null, sightingCards: new Map(),
    sightingFilter: "all", sightingSource: "all", sightingLimit: SIGHTING_PAGE_SIZE, sightingGroupsKey: null,
    sightingGalleryResult: null, sightingGalleryView: null,
    openSightingId: null, sightingOpener: null, sightingAssigning: false, sightingClaiming: false, claimingController: false,
    battery: null, batteryAvailable: null, batteryAction: null, batteryPolling: false,
    batteryRevision: 0, batteryError: null, captureStarts: new Map(),
    fixtureLoop: null, fixturePendingTimers: new Map(), cancelled: new Set(), fixtureSeeds: new Set(), identityOutputs: new Map(), errorCount: 0, droppedCount: 0, packetRejectCount: 0, skippedEventCount: 0, fixtureScenario: "normal", ackText: "None", lastPollOk: null,
  };

  const cmdAliases = {
    connect: ["connect", "connect_device", "status"],
    one_shot: ["one_shot", "oneshot", "capture_infer_once", "infer_once"],
    start_loop: ["start", "start_loop", "start_processing", "loop_start"],
    stop_loop: ["stop", "stop_loop", "stop_processing", "loop_stop"],
    set_config: ["set_config", "set_model_policy", "configure"],
    set_model: ["set_model", "model"],
    set_policy: ["set_policy", "policy"],
    sleep_camera: ["sleep", "sleep_camera", "camera_sleep"],
    wake_camera: ["wake", "wake_camera", "camera_wake"],
  };
  const wireCommand = { start_loop: "start", stop_loop: "stop", sleep_camera: "sleep", wake_camera: "wake" };
  function wireParams(cmd, params) {
    if (cmd === "set_model") return { model_slot: params.model_slot || params.model };
    if (cmd === "set_policy") return { policy_id: params.policy_id || params.policy };
    return params;
  }

  function setModeBadge(text, kind) {
    const badge = $("deviceMode");
    badge.textContent = text;
    badge.className = `badge ${kind || "offline"}`;
  }
  function setAction(message, isError = false) {
    $("deviceActionStatus").textContent = message;
    $("deviceActionStatus").style.color = isError ? "#ff9b93" : "";
  }
  function renderRuntimeState(status) {
    const node = $("deviceRuntimeState");
    if (!node) return;
    const connected = status?.connected === true || status?.connection?.state === "connected" || status?.state === "connected";
    if (!connected) {
      node.textContent = "Device runtime state unavailable; no live state is inferred from the last result.";
      node.dataset.state = "offline";
      return;
    }
    const applied = status.applied_state || {};
    const loop = typeof applied.processing === "boolean" ? applied.processing
      : typeof applied.running === "boolean" ? applied.running
      : typeof status.loop_running === "boolean" ? status.loop_running
      : typeof status.running === "boolean" ? status.running : null;
    const sleeping = typeof applied.camera_sleeping === "boolean" ? applied.camera_sleeping
      : typeof applied.sleeping === "boolean" ? applied.sleeping
      : typeof status.sleeping === "boolean" ? status.sleeping : null;
    const policy = applied.policy_id || applied.policy || status.policy || "not reported";
    const model = applied.model_override || (applied.model && applied.model !== "policy" ? applied.model : "policy selected");
    const loopText = loop === null ? "loop state not reported" : `loop ${loop ? "RUNNING" : "STOPPED"}`;
    const cameraText = sleeping === null ? "camera state not reported" : `camera ${sleeping ? "sleeping" : "awake"}`;
    node.textContent = `${loopText} · ${cameraText} · policy ${policy} · model ${model}`;
    node.dataset.state = loop === null || sleeping === null ? "unknown" : "reported";
  }
  function setConnection(message, isError = false) {
    $("deviceConnection").textContent = message;
    $("deviceConnection").style.color = isError ? "#ff9b93" : "";
  }
  function resultFrameId(event) {
    return event?.frame?.frame_id ?? event?.frame_id ?? null;
  }
  function eventBootId(event) { return event?.boot_id || event?.frame?.boot_id || "?"; }
  function eventKey(event) { return `${eventBootId(event)}:${resultFrameId(event) ?? "?"}`; }
  function compareObservationOrder(a, b) {
    // The status snapshot and event stream share the broker's receipt time.
    // Status snapshots may lack event_seq, and frame numbers restart at boot.
    const receiptOrder = (a.host_received_monotonic_ns ?? NaN) - (b.host_received_monotonic_ns ?? NaN);
    if (Number.isFinite(receiptOrder) && receiptOrder !== 0) return receiptOrder;
    if (eventBootId(a) === eventBootId(b)) return resultFrameId(a) - resultFrameId(b);
    const utcOrder = Date.parse(a.host_received_utc) - Date.parse(b.host_received_utc);
    if (Number.isFinite(utcOrder) && utcOrder !== 0) return utcOrder;
    const seqOrder = (a.event_seq ?? NaN) - (b.event_seq ?? NaN);
    return Number.isFinite(seqOrder) ? seqOrder : 0;
  }
  function toInt(value) { return Number.isSafeInteger(Number(value)) ? Number(value) : null; }
  function numberText(value, decimals = 1, suffix = "") {
    return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(decimals)}${suffix}` : "Not measured";
  }
  function showAge(value) { return value == null ? "Age unknown" : `${Math.max(0, Math.round(value))} ms old`; }
  function rememberCaptureStart(event) {
    if (state.fixture || (event.event !== "capture_start" && event.record?.event !== "capture_start")) return;
    const record = event.record || event;
    const boot = record.boot_id || record.run_id || event.boot_id;
    const frame = record.frame_index ?? record.frame_id ?? event.frame_id;
    const { ticks_ms: ticks, ticks_modulus_ms: modulus } = record;
    if (record.command_id || !boot || !Number.isSafeInteger(frame) || frame < 0 ||
        !Number.isSafeInteger(modulus) || modulus < 2 || !Number.isSafeInteger(ticks) || ticks < 0 || ticks >= modulus) return;
    state.captureStarts.set(`${boot}:${frame}`, { boot, frame, ticks, modulus,
      policy: record.policy_id, stage: record.measurement_stage_id });
    if (state.captureStarts.size > MAX_EVENTS * 2) state.captureStarts.delete(state.captureStarts.keys().next().value);
  }
  function recentCaptureRates(boot) {
    const starts = [...state.captureStarts.values()].filter((row) => row.boot === boot)
      .sort((left, right) => left.frame - right.frame).slice(-(PERFORMANCE_WINDOW + 1));
    const latest = starts.at(-1);
    const rates = [];
    for (let index = 1; index < starts.length; index++) {
      const previous = starts[index - 1], current = starts[index];
      if (current.frame - previous.frame !== 1 || current.modulus !== previous.modulus ||
          current.policy !== previous.policy || current.stage !== previous.stage ||
          current.policy !== latest.policy || current.stage !== latest.stage) continue;
      const interval = (current.ticks - previous.ticks + current.modulus) % current.modulus;
      if (interval > 0 && interval < current.modulus / 2) rates.push(1000 / interval);
    }
    return rates;
  }
  function median(values) {
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    if (!sorted.length) return null;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  function performanceTiming(event, field) {
    const value = (event.timings_ms || event.telemetry?.timings_ms)?.[field];
    return Number.isFinite(value) && value >= 0 ? value : null;
  }
  function faceActivity(event) {
    const status = String(event.detection?.status || event.detector?.status || "").toLowerCase();
    if (["face", "one_face", "one_face_duplicate_peaks_merged"].includes(status)) return "face";
    if (["multiple", "multiple_or_ambiguous", "multiple_faces"].includes(status)) return "multiple";
    return status === "no_face" ? "clear" : "unchecked";
  }
  function renderPerformanceBars(id, bars) {
    const chart = $(id);
    chart.hidden = bars.length === 0;
    const scale = Math.max(1, ...bars.map((bar) => bar.value));
    chart.replaceChildren(...bars.map(({ value, label, kind }) => {
      const bar = document.createElement("span");
      bar.style.setProperty("--bar-height", `${Math.max(8, 100 * value / scale)}%`);
      if (kind) bar.dataset.state = kind;
      bar.title = label;
      return bar;
    }));
  }
  function renderPerformance() {
    const boot = state.status?.boot_id || state.status?.device?.boot_id || eventBootId(state.events.at(-1));
    const events = state.fixture ? [] : state.events.filter((event) => eventBootId(event) === boot).slice(-PERFORMANCE_WINDOW);
    const rates = state.fixture ? [] : recentCaptureRates(boot);
    const totals = events.map((event) => performanceTiming(event, "total")).filter((value) => value !== null);
    const identity = events.filter((event) => event.crop?.status === "valid" && event.identity?.status === "ran")
      .map((event) => performanceTiming(event, "identity")).filter((value) => value !== null);
    const activity = events.map(faceActivity);
    const checked = activity.filter((kind) => kind !== "unchecked").length;
    const faceFrames = activity.filter((kind) => kind === "face" || kind === "multiple").length;
    const crops = events.filter((event) => event.crop?.status === "valid").length;
    const haveSamples = events.length > 0 || rates.length > 0;
    const context = state.fixture ? "Live metrics unavailable in test mode" : state.paused ? "Paused · last samples shown"
      : state.lastPollOk === false ? "Broker offline · last samples shown"
      : !state.loopRunning && haveSamples ? "Loop stopped · recent samples"
      : events.length ? `Last ${events.length} completed ${events.length === 1 ? "frame" : "frames"}`
      : rates.length ? "Recent capture intervals" : "Waiting for live events";
    $("sessionPerformanceStatus").textContent = context;
    $("sessionCaptureRate").textContent = rates.length ? numberText(median(rates), 1, " FPS") : "—";
    $("sessionCaptureNote").textContent = rates.length
      ? `Median of ${rates.length} capture ${rates.length === 1 ? "interval" : "intervals"}`
      : "Start processing to measure time between captures on the Nicla.";
    renderPerformanceBars("sessionCaptureChart", rates.map((value) => ({ value, label: numberText(value, 1, " FPS") })));
    $("sessionDecisionTime").textContent = totals.length ? numberText(median(totals), 0, " ms") : "—";
    $("sessionDecisionNote").textContent = totals.length
      ? `Median of ${totals.length} ${totals.length === 1 ? "response" : "responses"}${identity.length ? ` · identity ${numberText(median(identity), 0, " ms")} median` : ""}`
      : "Waiting for a frame with a reported response time.";
    renderPerformanceBars("sessionDecisionChart", totals.map((value) => ({ value, label: numberText(value, 0, " ms") })));
    $("sessionFaceCount").textContent = checked ? `${faceFrames} / ${checked}` : "—";
    $("sessionFaceNote").textContent = checked
      ? `Frames with a detection · ${crops} ${crops === 1 ? "valid crop" : "valid crops"}`
      : "Waiting for the detector to run.";
    const marks = { face: [3, "One face"], multiple: [4, "Multiple or ambiguous"], clear: [2, "No face"], unchecked: [1, "Detector did not report a check"] };
    renderPerformanceBars("sessionFaceChart", checked ? activity.map((kind) => ({ value: marks[kind][0], label: marks[kind][1], kind })) : []);
  }
  function updateAges() {
    const framePending = state.latestReportedFrame && state.latestReportedFrame !== state.latestFrame;
    if (state.latestReceivedAt != null) {
      const age = (state.latestFrame?.age_ms ?? 0) + (performance.now() - state.latestReceivedAt);
      $("latestMeta").textContent = `Frame ${state.latestFrame?.frame_id ?? "—"} · ${showAge(age)}`;
      $("latestMeta").title = state.latestFrame?.boot_id || state.latestFrame?.session_id || "";
      if (state.paused) {
        $("latestState").textContent = state.latestFrame?.stale === true || age > STALE_AFTER_MS ? "PAUSED · STALE" : "PAUSED";
        $("latestState").className = state.latestFrame?.stale === true || age > STALE_AFTER_MS ? "small-status stale" : "small-status";
        $("latestCaption").textContent = "UI updates are paused. Device acquisition is unchanged; age is still advancing.";
      } else if (state.latestFrame?.stale === true || age > STALE_AFTER_MS) {
        $("latestState").textContent = "STALE";
        $("latestState").className = "small-status stale";
        $("latestCaption").textContent = "The last received frame is stale. No green live state is shown until a fresh acquisition arrives.";
      } else if (state.lastPollOk !== false) {
        $("latestState").textContent = framePending ? "FRAME DELAYED" : "FRESH";
        $("latestState").className = framePending ? "small-status stale" : "small-status fresh";
        const { width, height } = state.imageGeometry.latest || state.latestFrame;
        $("latestCaption").textContent = framePending ? "Showing the last camera image while the next one arrives."
          : `${width} × ${height} px · Nicla camera`;
      }
    } else if (framePending && state.lastPollOk !== false) {
      $("latestState").textContent = "WAITING FOR IMAGE";
      $("latestState").className = "small-status";
      $("latestCaption").textContent = "The camera reported a frame. Waiting for its image.";
    }
    if (state.resultReceivedAt != null && state.selectedEvent) {
      const age = state.resultAgeBaseMs + (performance.now() - state.resultReceivedAt);
      const frame = resultFrameId(state.selectedEvent);
      const newest = state.latestFrame?.frame_id;
      const resultBoot = eventBootId(state.selectedEvent);
      const currentBoot = state.status?.boot_id || state.latestFrame?.boot_id;
      const stale = age > 10000 || (currentBoot && currentBoot !== resultBoot) ||
        (state.latestFrame?.boot_id === resultBoot && Number.isSafeInteger(newest) && newest - frame > 50);
      $("matchedFreshness").textContent = state.paused ? `PAUSED${stale ? " · STALE RESULT" : ""} · ${showAge(age)}` : stale ? `STALE RESULT · ${showAge(age)}` : `COMPLETED · ${showAge(age)}`;
      if (stale && !$("resultStatus").textContent.startsWith("STALE")) {
        $("resultStatus").textContent = `STALE · ${String(state.selectedEvent.decision?.status || "RESULT").replaceAll("_", " ")}`;
        $("resultBadge").textContent = "EXPERIMENTAL · STALE";
        $("resultBadge").className = "badge offline";
      }
    }
    if (state.paused) {
      $("deviceFreshness").textContent = `PAUSED · ${state.latestReceivedAt == null ? "no frame age" : showAge((state.latestFrame?.age_ms ?? 0) + performance.now() - state.latestReceivedAt)}`;
    } else if (state.lastPollOk === false) {
      $("deviceFreshness").textContent = "Service unavailable · last state retained as stale";
    } else {
      $("deviceFreshness").textContent = state.resultReceivedAt == null ? "No matched inference yet" : `Last completed result · ${showAge(state.resultAgeBaseMs + performance.now() - state.resultReceivedAt)}`;
    }
  }
  async function request(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(new Error("request deadline exceeded")), timeoutMs);
    try {
      const response = await fetch(path, {
        cache: "no-store", signal: abort.signal, ...options,
        headers: { "X-Control-Center-Client": clientId, "X-Nicla-Viewer-ID": tabId, "X-Viewer-ID": tabId, ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
      });
      const type = response.headers.get("content-type") || "";
      const data = type.includes("application/json") ? await response.json() : await response.arrayBuffer();
      if (!response.ok) throw new Error(data?.error || data?.message || data?.reason || `HTTP ${response.status}`);
      return { response, data };
    } catch (error) {
      if (error?.name === "AbortError") throw new Error(`Request timed out after ${timeoutMs} ms`);
      throw error;
    } finally { clearTimeout(timeout); }
  }
  async function getJson(path, timeoutMs) { return (await request(path, {}, timeoutMs)).data; }
  async function postJson(path, body, timeoutMs) {
    return (await request(path, { method: "POST", body: JSON.stringify(body) }, timeoutMs)).data;
  }
  function rawFrameUrl(frame) {
    const explicit = frame?.frame_ref || frame?.url || frame?.ref;
    if (typeof explicit === "string" && explicit.startsWith("/")) return explicit;
    const boot = frame?.boot_id || state.status?.boot_id;
    const id = frame?.frame_id;
    if (boot == null || id == null) return null;
    return `/api/v2/device/frame?boot_id=${encodeURIComponent(boot)}&frame_id=${encodeURIComponent(id)}`;
  }
  function imageUrl(frame) {
    if (fixture) {
      const item = state.fixtureScenario === "no_face" ? "BLANK" : state.fixtureScenario === "multiple" ? "OTHER" : "K01";
      return `/api/v1/replay/frame?case=${item}&fixture_frame=${state.fixtureFrame}`;
    }
    return rawFrameUrl(frame);
  }
  async function unavailablePayloadMessage(response, kind) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body?.reason_code || body?.error || body?.message || "";
    } catch (_) {}
    const availability = response.status === 410 ? "expired" : response.status === 404 ? "not retained or expired" : "API error";
    return `${kind} payload ${availability} (HTTP ${response.status})${detail ? `: ${String(detail).slice(0, 180)}` : ""}`;
  }
  function revokeImage(slot) {
    const key = slot === "latest" ? "latestUrl" : "matchedUrl";
    if (state[key]) URL.revokeObjectURL(state[key]);
    state[key] = null;
  }
  const recentFrameReads = new Map();
  function readFrame(url) {
    if (recentFrameReads.has(url)) return recentFrameReads.get(url);
    const pending = (async () => {
      const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
      if (!response.ok) throw new Error(await unavailablePayloadMessage(response, "Frame"));
      const blob = await response.blob();
      if (!blob.type.startsWith("image/") || blob.size > 1_000_000) {
        throw new Error("Frame is not a supported image or exceeds the 1 MB bound");
      }
      const sha256 = response.headers.get("X-Frame-SHA256");
      if (sha256 && globalThis.crypto?.subtle) {
        const bytes = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
        const actual = [...new Uint8Array(bytes)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
        if (actual !== sha256) throw new Error("Frame payload SHA-256 does not match broker metadata");
      }
      return { blob, sha256, boot: response.headers.get("X-Device-Boot-ID"),
        frame: response.headers.get("X-Device-Frame-ID") };
    })();
    recentFrameReads.set(url, pending);
    if (recentFrameReads.size > 6) recentFrameReads.delete(recentFrameReads.keys().next().value);
    pending.catch(() => {
      if (recentFrameReads.get(url) === pending) recentFrameReads.delete(url);
    });
    return pending;
  }
  async function loadFrameImage(slot, frame, event = null, observedAt = performance.now()) {
    const requestId = ++state.imageRequest[slot];
    const current = () => state.imageRequest[slot] === requestId &&
      (slot === "latest" ? state.latestReportedFrame === frame
        : !event || eventKey(state.selectedEvent) === eventKey(event));
    const url = imageUrl(frame);
    const image = $(slot === "latest" ? "latestImage" : "matchedImage");
    const empty = $(slot === "latest" ? "latestEmpty" : "matchedEmpty");
    const viewport = $(slot === "latest" ? "latestViewport" : "matchedViewport");
    if (!url) { image.hidden = true; empty.hidden = false; return false; }
    try {
      if (!url.startsWith("/")) throw new Error("external frame URLs are not accepted");
      const { blob, sha256, boot: responseBoot, frame: responseFrame } = await readFrame(url);
      let verified = false;
      if (event) {
        if (responseBoot && responseBoot !== eventBootId(event)) throw new Error("frame response boot ID does not match selected event");
        if (responseFrame && Number(responseFrame) !== Number(resultFrameId(event))) throw new Error("frame response ID does not match selected inference");
        verified = state.fixture ? true : Boolean(responseBoot && responseFrame);
      }
      const expectedHash = event?.frame?.sha256 || event?.frame?.jpeg_sha256;
      if (expectedHash && sha256 && expectedHash !== sha256) throw new Error("frame hash header does not match event metadata");
      if (!current()) return false;
      const objectUrl = URL.createObjectURL(blob);
      const showFrame = (width, height) => {
        if (!current()) return;
        image.hidden = false; empty.hidden = true;
        const geometry = { width, height };
        state.imageGeometry[slot] = geometry;
        const svg = $(slot === "latest" ? "latestOverlay" : "matchedOverlay");
        svg.setAttribute("viewBox", `0 0 ${geometry.width} ${geometry.height}`);
        viewport.style.transform = `scale(${state.zoom[slot]})`;
        renderOverlay(slot, event || frame, geometry);
        if (slot === "matched" && event) renderCropAndInput(event, image, current).catch((error) => { if (current()) $("tensorSummary").textContent = `Input inspection failed: ${error.message}`; });
      };
      if (slot === "latest") {
        // Decode before replacing the displayed frame so an unavailable or
        // invalid JPEG leaves the prior image, age and overlays together.
        const decoded = new Image();
        try {
          decoded.src = objectUrl;
          await decoded.decode();
        } catch (error) {
          URL.revokeObjectURL(objectUrl);
          throw error;
        }
        if (!current()) { URL.revokeObjectURL(objectUrl); return false; }
        revokeImage(slot);
        state.latestUrl = objectUrl;
        image.src = objectUrl;
        state.latestFrame = frame;
        state.latestReceivedAt = observedAt;
        showFrame(decoded.naturalWidth, decoded.naturalHeight);
        return true;
      }
      state.matchedFrameVerified = verified;
      revokeImage(slot);
      state.matchedUrl = objectUrl;
      image.onload = () => showFrame(image.naturalWidth, image.naturalHeight);
      image.onerror = () => { if (current()) { image.hidden = true; empty.hidden = false; empty.querySelector("strong").textContent = "Frame could not be decoded"; } };
      image.src = objectUrl;
      image.alt = `Frozen frame ${event?.boot_id || "?"}:${resultFrameId(event) ?? "?"} linked to this inference`;
      return true;
    } catch (error) {
      if (!current()) return false;
      if (slot === "latest") return false;
      image.hidden = true; empty.hidden = false;
      empty.querySelector("strong").textContent = "Frame unavailable";
      const small = empty.querySelector("span");
      if (small) small.textContent = `Frame ${event ? `${eventBootId(event)}:${resultFrameId(event)}` : `${frame?.boot_id || "?"}:${frame?.frame_id ?? "?"}`} — ${error.message}. The image payload is unavailable; event provenance remains visible.`;
      state.errorCount++;
      renderErrorCounts();
      return false;
    }
  }
  function asRect(value) {
    if (Array.isArray(value) && value.length >= 4) {
      const [x, y, width, height] = value.slice(0, 4).map(Number);
      return [x, y, width, height].every(Number.isFinite) ? { x, y, width, height } : null;
    }
    if (!value || typeof value !== "object") return null;
    const r = value.rect_px || value.xywh || value.xywh_px || value.bbox_xywh || value;
    if (Array.isArray(r)) return asRect(r);
    const x = Number(r.x ?? r.left), y = Number(r.y ?? r.top), width = Number(r.width ?? r.w), height = Number(r.height ?? r.h);
    return [x, y, width, height].every(Number.isFinite) ? { x, y, width, height } : null;
  }
  function rectInFrame(value, stage) {
    const rect = asRect(value);
    if (!rect) return null;
    const space = String(value?.coordinate_space || value?.space || stage?.coordinate_space || "sensor_px").toLowerCase();
    if (["sensor_px", "frame_px", "camera_px", "sensor", "frame"].includes(space)) return rect;
    const map = stage?.transform_to_sensor || stage?.to_sensor || stage?.sensor_transform;
    if (!map) return null;
    const sx = Number(map.scale_x ?? map.sx), sy = Number(map.scale_y ?? map.sy);
    const ox = Number(map.offset_x ?? map.ox ?? 0), oy = Number(map.offset_y ?? map.oy ?? 0);
    if (![sx, sy, ox, oy].every(Number.isFinite) || sx <= 0 || sy <= 0) return null;
    return { x: rect.x * sx + ox, y: rect.y * sy + oy, width: rect.width * sx, height: rect.height * sy };
  }
  function renderOverlay(slot, event, geometry) {
    const svg = $(slot === "latest" ? "latestOverlay" : "matchedOverlay");
    svg.replaceChildren();
    const detector = event?.detection || event?.detector || {};
    const regions = detector.regions_xywh || detector.regions || detector.boxes || [];
    for (const [index, item] of regions.entries()) {
      const rect = rectInFrame(item, detector);
      if (!rect || rect.width <= 0 || rect.height <= 0) continue;
      const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      box.setAttribute("x", rect.x); box.setAttribute("y", rect.y);
      box.setAttribute("width", rect.width); box.setAttribute("height", rect.height);
      svg.appendChild(box);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", rect.x + 3); label.setAttribute("y", Math.max(12, rect.y - 4));
      const backend = String(detector.backend || detector.type || detector.kind || "").toLowerCase();
      label.textContent = detector.coordinate_space === "fomo_centroid" || backend.includes("fomo") || backend.includes("centroid") ? `FOMO region ${index + 1} · not a face box` : `Nicla face box ${index + 1}`;
      svg.appendChild(label);
    }
    const centers = detector.centers_xy || detector.centroids_xy || [];
    for (const center of centers) {
      let x = Array.isArray(center) ? Number(center[0]) : Number(center.x), y = Array.isArray(center) ? Number(center[1]) : Number(center.y);
      const space = String(detector.coordinate_space || "sensor_px").toLowerCase();
      if (!["sensor_px", "frame_px", "camera_px", "sensor", "frame"].includes(space)) {
        const map = detector.transform_to_sensor || detector.to_sensor || detector.sensor_transform;
        if (!map) continue;
        const sx = Number(map.scale_x ?? map.sx), sy = Number(map.scale_y ?? map.sy);
        const ox = Number(map.offset_x ?? map.ox ?? 0), oy = Number(map.offset_y ?? map.oy ?? 0);
        if (![sx, sy, ox, oy].every(Number.isFinite) || sx <= 0 || sy <= 0) continue;
        x = x * sx + ox; y = y * sy + oy;
      }
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const mark = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      mark.setAttribute("cx", x); mark.setAttribute("cy", y); mark.setAttribute("r", 4);
      mark.setAttribute("class", "detector-center"); svg.appendChild(mark);
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", x + 7); label.setAttribute("y", y - 5); label.textContent = "FOMO center · not a face box"; svg.appendChild(label);
    }
    const crop = rectInFrame(event?.crop?.rect_px || event?.crop?.rectangle || event?.crop, event?.crop);
    if (crop && crop.width > 0 && crop.height > 0) {
      const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      box.setAttribute("class", "crop-box"); box.setAttribute("x", crop.x); box.setAttribute("y", crop.y);
      box.setAttribute("width", crop.width); box.setAttribute("height", crop.height);
      svg.appendChild(box);
    }
    const reference = state.hostReference;
    if (slot === "matched" && reference?.key === eventKey(event) && reference.face) {
      const { x, y, width, height } = reference.face;
      const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      box.setAttribute("class", "host-face-box");
      for (const [name, value] of Object.entries({ x, y, width, height })) box.setAttribute(name, value);
      svg.appendChild(box);
    }
  }
  function formatScore(decision) {
    const candidate = decision?.candidate ?? decision?.raw_candidate ?? decision?.status;
    const scores = decision?.scores || {};
    let score = decision?.score;
    if (score == null && candidate != null && typeof scores[candidate] === "number") score = scores[candidate];
    const semantics = ({ cosine_similarity_of_l2_normalized_embeddings: "cosine similarity",
      max_known_softmax_probability_from_logits: "known-class probability" })[decision?.score_semantics] || decision?.score_semantics || "score";
    return score == null ? "Score not retained" : `${Number(score).toFixed(4)} ${semantics}`;
  }
  function identityName(value, roster = "device") {
    const names = { OTHER: "OTHER / unknown", UNCERTAIN: "Uncertain" };
    if (roster === "lfw" && value === "K01") return "K01 — George W. Bush";
    return names[value] || String(value || "No candidate").replaceAll("_", " ");
  }
  function renderErrorCounts({ eventErrors = 0, eventDrops = null } = {}) {
    const drops = eventDrops == null ? state.droppedCount : eventDrops;
    $("deviceErrors").textContent = String(drops) + " drops · " + state.errorCount + " device/transport errors · " + state.packetRejectCount + " malformed events rejected · " + eventErrors + " errors in selected event";
  }
  function renderTelemetry(event) {
    const telemetry = event?.telemetry || {};
    const timings = event?.timings_ms || telemetry.timings_ms || (event?.timings_us ? Object.fromEntries(Object.entries(event.timings_us).map(([key, value]) => [key, Number(value) / 1000])) : {});
    const compact = (keys) => keys.map(([key, label]) => timings[key] == null ? null : `${label} ${numberText(Number(timings[key]), 1, " ms")}`).filter(Boolean).join(" · ") || "Not measured";
    $("stageTimings").textContent = compact([["capture", "capture"], ["preprocess", "preprocess"], ["detection", "detect"], ["identity", "identity"], ["model_load", "load"]]);
    $("decisionTiming").textContent = compact([["decision", "decision"], ["total", "total"]]);
    const rates = [telemetry.capture_hz != null ? `capture ${numberText(Number(telemetry.capture_hz), 2, " Hz")}` : null, telemetry.inference_hz != null ? `inference ${numberText(Number(telemetry.inference_hz), 2, " Hz")}` : null].filter(Boolean);
    $("deviceRates").textContent = rates.join(" · ") || "Not measured";
    const mem = [telemetry.heap_bytes != null ? `heap ${telemetry.heap_bytes} B` : null, telemetry.arena_bytes != null ? `arena ${telemetry.arena_bytes} B` : null, telemetry.framebuffer_bytes != null ? `frame ${telemetry.framebuffer_bytes} B` : null].filter(Boolean);
    $("deviceMemory").textContent = mem.join(" · ") || "Not measured";
    const camera = event?.camera || {};
    const imu = telemetry.imu || event?.imu;
    const imuText = imu ? `${imu.accel_m_s2 ? `accel ${imu.accel_m_s2.join(", ")} m/s²` : "IMU present"}${camera.sleeping == null ? "" : ` · camera ${camera.sleeping ? "sleeping" : "awake"}`}` : camera.sleeping == null ? "Not measured" : `Camera ${camera.sleeping ? "sleeping" : "awake"} · IMU not measured`;
    $("cameraImu").textContent = imuText;
    $("devicePower").textContent = telemetry.power_w == null ? "Not measured" : `${Number(telemetry.power_w).toFixed(4)} W`;
    // This panel belongs to the selected immutable event. Do not borrow the
    // currently applied model from a newer status poll when viewing history.
    const applied = event?.applied_state?.model;
    const identity = event?.identity || {};
    const inferred = identity.model_id || event?.identity_model_id || event?.model_id || event?.decision?.model_id;
    const reportedModelHash = identity.model_sha256 || identity.model_hash || event?.model_sha256 || null;
    const modelHash = typeof reportedModelHash === "string" && /^[0-9a-f]{64}$/i.test(reportedModelHash)
      ? reportedModelHash : null;
    const identityStatus = String(identity.status || "").trim().toLowerCase();
    const identitySkipped = ["skipped", "not_run", "not-run", "not run", "not attempted"].includes(identityStatus);
    const identityFailed = ["error", "failed", "failure"].includes(identityStatus);
    const identityRan = identity.executed === true || identityStatus === "simulated" || ["ran", "completed", "success", "ok"].includes(identityStatus);
    const inferenceLabel = identitySkipped
      ? "identity skipped"
      : identityFailed
        ? `identity stage error${inferred ? ` · ${inferred}` : ""}`
        : identityRan && inferred
          ? `inference ran ${inferred}`
          : inferred
            ? `model reported ${inferred}; execution not reported`
            : "inference model not reported";
    $("deviceModelObserved").textContent = [
      applied ? `applied ${applied}` : "applied state not reported",
      inferenceLabel,
      modelHash && (identityRan || identityFailed) ? `SHA-256 ${String(modelHash).slice(0, 12)}…` : null,
      state.fixture ? "synthetic fixture" : null,
    ].filter(Boolean).join(" · ");
    const errors = telemetry.errors || event?.errors || [];
    renderErrorCounts({ eventErrors: Array.isArray(errors) ? errors.length : 0, eventDrops: telemetry.dropped_frames ?? event?.dropped_frames ?? null });
  }
  async function loadExactInput(event, kind, descriptor) {
    if (descriptor?.payload_status === "not_exported_live") return descriptor;
    if (state.fixture || descriptor?.values || descriptor?.bytes || descriptor?.preview || descriptor?.visualization) return descriptor || null;
    if (kind === "identity" && event?.identity?.status === "skipped") return null;
    const boot = eventBootId(event), frame = resultFrameId(event);
    const url = descriptor.input_ref || `/api/v2/device/input?boot_id=${encodeURIComponent(boot)}&frame_id=${encodeURIComponent(frame)}&kind=${encodeURIComponent(kind)}`;
    if (typeof url !== "string" || !url.startsWith("/")) throw new Error("external input URLs are not accepted");
    const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
    if (!response.ok) throw new Error(await unavailablePayloadMessage(response, `Exact ${kind} input`));
    const responseBoot = response.headers.get("X-Device-Boot-ID"), responseFrame = response.headers.get("X-Device-Frame-ID");
    if (responseBoot && responseBoot !== boot || responseFrame && Number(responseFrame) !== Number(frame)) throw new Error(`${kind} input response does not match selected boot/frame`);
    const raw = await response.arrayBuffer();
    if (!raw.byteLength || raw.byteLength > 262_144) throw new Error(`${kind} input exceeds the 256 KiB inspector limit`);
    const headerHash = response.headers.get("X-Input-SHA256");
    const actualHash = globalThis.crypto?.subtle ? [...new Uint8Array(await crypto.subtle.digest("SHA-256", raw))].map((b) => b.toString(16).padStart(2, "0")).join("") : null;
    if (descriptor.sha256 && actualHash && descriptor.sha256 !== actualHash) throw new Error(`${kind} tensor bytes do not match the event SHA-256`);
    if (headerHash && actualHash && headerHash !== actualHash) throw new Error(`${kind} tensor bytes do not match the broker SHA-256 header`);
    let shape = descriptor.shape;
    if (!shape) { try { shape = JSON.parse(response.headers.get("X-Input-Shape") || "[]"); } catch (_) { shape = []; } }
    const wireDtype = response.headers.get("X-Input-DType") || descriptor.dtype || "int8";
    // Older broker processes can return OpenMV's byte-valued dtype code even
    // though the event metadata already carries its normalized name. Keep the
    // visualization faithful across a broker refresh/restart boundary.
    const dtype = ({ "98": "int8", "66": "uint8", "104": "int16", "72": "uint16", "102": "float32", b: "int8", B: "uint8", h: "int16", H: "uint16", f: "float32" })[String(wireDtype)] || String(wireDtype).toLowerCase();
    const viewTypes = { int8: Int8Array, uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array, float32: Float32Array };
    const ViewType = viewTypes[dtype];
    if (!ViewType || raw.byteLength % ViewType.BYTES_PER_ELEMENT !== 0) throw new Error(`unsupported or truncated exact input dtype ${String(wireDtype)}`);
    const values = new ViewType(raw);
    const scaleHeader = response.headers.get("X-Input-Scale");
    const zeroHeader = response.headers.get("X-Input-Zero-Point");
    return { ...descriptor, shape, dtype,
      scale: scaleHeader == null ? (descriptor.scale ?? null) : Number(scaleHeader),
      zero_point: zeroHeader == null ? (descriptor.zero_point ?? null) : Number(zeroHeader),
      sha256: descriptor.sha256 || headerHash || actualHash || null,
      source: descriptor.source || `Exact ${kind} bytes from Nicla · ${boot}:${frame}`,
      values, verified_sha256: actualHash || headerHash || null };
  }
  function base64OfBytes(bytes) {
    const data = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength);
    let binary = "";
    for (let i = 0; i < data.length; i += 0x8000) binary += String.fromCharCode(...data.subarray(i, i + 0x8000));
    return btoa(binary);
  }
  async function renderCropAndInput(event, image, current = () => eventKey(state.selectedEvent) === eventKey(event)) {
    const cropCanvas = $("cropCanvas"), cropCtx = cropCanvas.getContext("2d");
    const r = rectInFrame(event?.crop?.rect_px || event?.crop?.rectangle || event?.crop, event?.crop);
    cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
    if (r && image.complete && image.naturalWidth && r.width > 0 && r.height > 0) {
      cropCtx.drawImage(image, r.x, r.y, r.width, r.height, 0, 0, cropCanvas.width, cropCanvas.height);
      const diagnostic = String(event.crop?.status || "").includes("diagnostic");
      $("cropSummary").textContent = `${Math.round(r.width)}×${Math.round(r.height)} px · ${diagnostic ? "diagnostic crop" : "source crop"}`;
      $("cropRect").textContent = `${r.x}, ${r.y}, ${r.width}, ${r.height} px · ${event.crop.coordinate_space || "frame coordinates"}`;
    } else {
      const detectionStatus = event?.detection?.status || event?.detector?.status;
      const reason = event?.crop?.rejection_reason || event?.crop?.rejectionReason;
      const skipped = detectionStatus === "no_face" ? "Skipped · detector reported no face" :
        detectionStatus === "multiple" ? "Skipped · multiple or ambiguous faces" :
        reason ? `Rejected · ${reason}` : "Crop metadata was not retained";
      $("cropSummary").textContent = skipped;
      $("cropRect").textContent = event?.crop?.status || "Not reported";
    }
    let input = event?.identity_input || event?.model_input || {};
    let detectorInput = event?.detector_input || event?.detector?.input || event?.detector?.input_metadata || {};
    let detectorPayloadError = null, identityPayloadError = null;
    try { detectorInput = await loadExactInput(event, "detector", detectorInput); }
    catch (error) { detectorPayloadError = error.message; }
    try { input = await loadExactInput(event, "identity", input); }
    catch (error) { identityPayloadError = error.message; }
    if (!current()) return;
    input = input || {};
    $("inputProvenance").textContent = input.source || (event?.identity?.status === "skipped" ? "NOT RUN · no identity inference" : "NOT RETAINED");
    $("tensorShape").textContent = Array.isArray(input.shape) ? input.shape.join(" × ") : [input.width, input.height, input.channels].filter((v) => v != null).join(" × ") || "—";
    $("tensorQuant").textContent = `${input.dtype || "dtype unknown"} · scale ${input.scale ?? "—"} · zero ${input.zero_point ?? "—"}`;
    $("tensorHash").textContent = input.sha256 || "Not retained";
    $("detectorHash").textContent = detectorInput.sha256 || "Not retained";
    $("modelHashes").textContent = `model ${event.model_sha256 || input.model_sha256 || "—"} · config ${event.config_sha256 || "—"} · firmware ${event.firmware_sha256 || event.firmware?.sha256 || "—"}`;
    const detectorCanvas = $("detectorCanvas"), detectorCtx = detectorCanvas.getContext("2d"); detectorCtx.clearRect(0, 0, detectorCanvas.width, detectorCanvas.height);
    if (detectorInput?.payload_status === "not_exported_live") $("detectorSummary").textContent = "Run one shot to capture exact detector bytes.";
    else if (detectorPayloadError) $("detectorSummary").textContent = `Exact detector bytes unavailable · ${detectorPayloadError}`;
    else if (detectorInput && (Array.isArray(detectorInput.values) || ArrayBuffer.isView(detectorInput.values))) {
      paintTensor(detectorCtx, detectorInput.values, detectorInput); $("detectorSummary").textContent = `${detectorInput.dtype || "tensor"} · exact bytes`;
    } else $("detectorSummary").textContent = detectorInput?.sha256 ? "Exact hash retained; visualization bytes unavailable" : "Detector input not captured or retained";
    const tensorCanvas = $("tensorCanvas"), ctx = tensorCanvas.getContext("2d");
    ctx.clearRect(0, 0, tensorCanvas.width, tensorCanvas.height);
    const preview = input.preview || input.visualization || input.preview_base64;
    if (preview && typeof preview === "string") {
      const img = new Image(); img.onload = () => { if (current()) { ctx.imageSmoothingEnabled = false; ctx.drawImage(img, 0, 0, tensorCanvas.width, tensorCanvas.height); } };
      img.src = preview.startsWith("data:") ? preview : `data:image/png;base64,${preview}`;
      $("tensorSummary").textContent = "Device-supplied tensor visualization";
    } else if (Array.isArray(input.values) || ArrayBuffer.isView(input.values) || Array.isArray(input.bytes) || ArrayBuffer.isView(input.bytes)) {
      paintTensor(ctx, input.values || input.bytes, input);
      $("tensorSummary").textContent = input.verified_sha256 ? `Exact bytes verified · ${input.verified_sha256.slice(0, 16)}…` : "Rendered from retained input values";
    } else if (input.payload_status === "not_exported_live") {
      $("tensorSummary").textContent = "Run one shot to capture exact identity bytes.";
    } else if (identityPayloadError) {
      $("tensorSummary").textContent = `Exact identity bytes unavailable · ${identityPayloadError}`;
    } else {
      $("tensorSummary").textContent = input.sha256 ? "Exact bytes hashed; visualization unavailable" : event?.identity?.status === "skipped" ? "Identity input not run for this result" : "Exact input not retained";
    }
    if (input.values) $("tensorBytes").value = base64OfBytes(input.values);
    else if (input.bytes) $("tensorBytes").value = base64OfBytes(input.bytes);
    else $("tensorBytes").value = "Exact bytes not retained by this event.";
    const rawOutput = event?.identity?.raw_output;
    if (Array.isArray(rawOutput) && rawOutput.length <= 512) {
      const outputModel = event.identity.raw_output_model_id || event.identity.model_id || event.model_id || "model unknown";
      const outputHash = event.identity.raw_output_model_sha256 || event.identity.model_sha256 || "hash unavailable";
      $("identityOutputMeta").textContent = `${outputModel} · ${rawOutput.length} raw values · ${event.identity.output_representation || "representation unspecified"} · model SHA-256 ${outputHash}`;
      $("identityOutputVector").textContent = JSON.stringify(rawOutput.map(Number), null, 2);
    } else {
      $("identityOutputMeta").textContent = "No frame-linked raw model output retained. A missing vector is not evidence of a score.";
      $("identityOutputVector").textContent = "Not available";
    }
  }
  function paintTensor(ctx, values, input) {
    const dims = Array.isArray(input.shape) ? input.shape.map(Number) : [Number(input.height), Number(input.width), Number(input.channels || 1)];
    let h, w, c;
    if (dims.length === 4 && dims[0] === 1) [h, w, c] = dims.slice(1);
    else if (dims.length === 3 && dims[0] === 1) [h, w, c] = [dims[1], dims[2], 1];
    else if (dims.length === 3) [h, w, c] = dims;
    else [h, w, c] = [dims[0], dims[1], Number(input.channels || 1)];
    if (!Number.isSafeInteger(w) || !Number.isSafeInteger(h) || !Number.isSafeInteger(c) || w < 1 || h < 1 || w * h * c !== values.length) return;
    const pixels = new Uint8ClampedArray(w * h * 4);
    // Real gray image contracts use a fixed full-range display mapping so
    // previews are comparable across frames and don't invent contrast.
    let lo = c === 1 && input.dtype === "uint8" && Number(input.scale) === 1 && Number(input.zero_point) === 0 ? 0 :
      c === 1 && input.dtype === "int8" && Number(input.zero_point) === -128 && Math.abs(Number(input.scale) - 1 / 255) < 1e-6 ? -128 : Infinity;
    let hi = Number.isFinite(lo) ? lo + 255 : -Infinity;
    if (!Number.isFinite(lo)) {
      for (const value of values) { const n = Number(value); if (Number.isFinite(n)) { lo = Math.min(lo, n); hi = Math.max(hi, n); } }
    }
    const scale = hi > lo ? 255 / (hi - lo) : 1;
    for (let i = 0, p = 0; i < values.length; i += c, p += 4) {
      const bgr = String(input.channels || input.channel_order || "").toLowerCase().includes("bgr");
      const channels = c === 1 ? [values[i], values[i], values[i]] : bgr ? [values[i + 2], values[i + 1], values[i]] : [values[i], values[i + 1], values[i + 2]];
      pixels[p] = Math.max(0, Math.min(255, (Number(channels[0]) - lo) * scale));
      pixels[p + 1] = Math.max(0, Math.min(255, (Number(channels[1]) - lo) * scale));
      pixels[p + 2] = Math.max(0, Math.min(255, (Number(channels[2]) - lo) * scale));
      pixels[p + 3] = 255;
    }
    const offscreen = document.createElement("canvas"); offscreen.width = w; offscreen.height = h;
    // putImageData requires destination coordinates as well as the pixel buffer.
    offscreen.getContext("2d").putImageData(new ImageData(pixels, w, h), 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(offscreen, 0, 0, 128, 128);
  }
  function attachFrameIdentityOutput(event) {
    const output = state.identityOutputs.get(eventKey(event));
    if (!output) return event;
    return { ...event, identity: { ...(event.identity || {}),
      raw_output: output.raw_output, output_dtype: output.output_dtype,
      output_scale: output.output_scale, output_zero_point: output.output_zero_point,
      output_representation: output.output_representation,
      raw_output_model_id: output.model_id, raw_output_model_sha256: output.model_sha256 } };
  }
  async function renderEvent(event, { select = true } = {}) {
    if (!event || typeof event !== "object") return;
    event = attachFrameIdentityOutput(event);
    const key = eventKey(event);
    if (key !== state.hostReferenceKey) {
      state.hostReferenceKey = null;
      state.hostReference = null;
      $("deviceHostReference").disabled = true;
      $("deviceHostReferenceStatus").textContent = "YuNet and SFace run on the Mac using the linked camera frame.";
      $("deviceHostCropFigure").hidden = true;
      $("deviceHostCrop").removeAttribute("src");
    }
    if (select) {
      state.selectedEvent = event;
      state.resultReceivedAt = performance.now();
      const receivedUnix = Number(event.host_received_unix_ms);
      const receivedUtc = Date.parse(event.host_received_utc || "");
      const timestamp = Number.isFinite(receivedUnix) && receivedUnix > 0 ? receivedUnix : Number.isFinite(receivedUtc) ? receivedUtc : Date.now();
      state.resultAgeBaseMs = Math.max(0, Date.now() - timestamp);
    }
    const frame = event.frame || { boot_id: event.boot_id, frame_id: event.frame_id, ...event.camera };
    $("resultFrameId").textContent = `${event.boot_id || "boot?"} / ${resultFrameId(event) ?? "frame?"}`;
    const origin = event.connection?.execution_location || event.source;
    $("matchedMeta").textContent = `Frame ${resultFrameId(event) ?? "—"} · ${origin === "on_nicla" ? "Nicla" : String(origin || "device event").replaceAll("_", " ")}`;
    $("matchedMeta").title = eventBootId(event) || "";
    const decision = event.decision || {};
    const detection = event.detection || event.detector || {};
    const status = String(decision.status || "ERROR").replaceAll("_", " ");
    $("resultStatus").textContent = status;
    $("resultCandidate").textContent = identityName(decision.candidate || decision.raw_candidate || event.predicted_class || (status.startsWith("K0") ? status : "No known identity accepted"));
    const cutoff = decision.acceptance_cutoff ?? decision.threshold;
    $("resultScore").textContent = `${formatScore(decision)}${cutoff == null ? "" : ` · ${decision.acceptance_cutoff == null ? "threshold" : "acceptance cutoff"} ${Number(cutoff).toFixed(4)}`}${decision.margin == null ? "" : ` · margin ${Number(decision.margin).toFixed(4)}`}`;
    const reason = decision.reason || event.reason;
    const sceneRejected = String(reason || "").startsWith("INSUFFICIENT_SCENE_QUALITY:");
    const sceneMessage = reason === "INSUFFICIENT_SCENE_QUALITY:low_contrast_or_blank"
      ? "The frame had too little contrast for detection. Try more light or aim at a visible subject."
      : "The Nicla skipped detection because the frame did not meet its image quality check.";
    const reasonText = {
      no_face_detected: "The detector found no face, so identity inference was skipped.",
      below_gallery_threshold_band: "The identity score was below the match threshold.",
      multiple_faces_detected: "The frame has multiple faces. Identity inference was skipped.",
      known_identity_acceptance_disabled_unqualified_crop: "The identity model ranked a candidate. This build reports experimental results only.",
    };
    $("resultReason").textContent = sceneRejected ? sceneMessage :
      reasonText[reason] || reason || "The device reported no additional decision details.";
    const skippedNoFace = event.identity?.status === "skipped" && (decision.status === "NO_FACE" || detection.status === "no_face");
    const diagnosticDetectorBypass = detection.status === "not_run_diagnostic_stage" ||
      String(detection.region_semantics || "").includes("intentionally bypassed");
    $("traceSummary").textContent = sceneRejected ? sceneMessage : event.trace_summary || (diagnosticDetectorBypass
      ? "Identity center-crop diagnostic: detector was intentionally bypassed; this is not the full pipeline."
      : skippedNoFace
      ? "Detection found no accepted face region; crop and identity inference were skipped."
      : `Detection: ${detection.status || "unknown"}. Crop: ${event.crop ? "reported" : "not reported"}. Identity: ${event.identity_input || event.model_input ? "input present" : "input unavailable"}.`);
    let trace = event.trace || event.policy?.trace || [
      detection.status ? `Detector reported ${detection.status}${detection.coordinate_space ? ` in ${detection.coordinate_space}` : ""}.` : "Detector status was not included.",
      event.crop ? "Crop metadata belongs to this boot/frame pair." : "No crop box was reported; centroid regions are not treated as face boxes.",
      event.identity_input?.sha256 || event.model_input?.sha256 ? "Identity stage ran with the exact input hash shown below." : "Exact identity model input was not retained.",
      decision.reason || "No decision reason was supplied by the runtime.",
    ];
    // Older firmware versions describe every skipped identity stage as
    // "no model selected", even when the selected model is applied and the
    // detector found no face. State the observable gating reason accurately.
    if (sceneRejected) trace = [
      `Captured a new Nicla frame (${eventBootId(event)}:${resultFrameId(event)}).`,
      "The image quality check stopped this frame before detection and identity inference.",
      `Device reason: ${reason}.`,
    ];
    else if (diagnosticDetectorBypass) trace = [
      `Captured a new Nicla frame (${eventBootId(event)}:${resultFrameId(event)}).`,
      "Detector: not run; this identity center-crop diagnostic intentionally bypassed detection.",
      `Crop: ${event.crop?.status || "not reported"}; fixed center crop is diagnostic, not a detected face box.`,
      `Identity: ${event.identity?.status === "ran" ? `ran ${event.identity.model_id || "the selected model"}` : "not run"}; acceptance disabled.`,
      `Decision: ${decision.status || "UNCERTAIN"} — ${decision.reason || "diagnostic only"}.`,
    ];
    else if (skippedNoFace) trace = [
      `Captured a new Nicla frame (${eventBootId(event)}:${resultFrameId(event)}).`,
      `Detector: ${detection.backend || event.detector?.backend || "reported backend"}; 0 accepted face region(s).`,
      "Crop: skipped because no accepted face region was available.",
      "Identity: skipped because no crop/input was produced; the selected model was not invoked.",
      `Decision: ${decision.status} — ${decision.reason || "no_face_detected"}.`,
    ];
    $("executionTrace").replaceChildren(...(Array.isArray(trace) ? trace : [String(trace)]).map((line) => { const li = document.createElement("li"); li.textContent = typeof line === "string" ? line : JSON.stringify(line); return li; }));
    const experimental = decision.experimental !== false || event.qualification?.statistically_qualified !== true;
    $("resultBadge").textContent = experimental ? "EXPERIMENTAL" : "QUALIFIED";
    $("resultBadge").title = experimental ? "Recognition accuracy has not been qualified." : "Statistically qualified result";
    $("resultBadge").className = experimental ? "badge experimental" : "badge nicla";
    if (state.fixture) { $("resultBadge").textContent = "TEST FIXTURE · SYNTHETIC"; $("resultBadge").className = "badge fixture"; }
    if (!frame.frame_id) frame.frame_id = resultFrameId(event);
    if (!frame.boot_id) frame.boot_id = event.boot_id;
    const loaded = await loadFrameImage("matched", frame, event);
    if (eventKey(state.selectedEvent) !== key) return;
    state.hostReferenceKey = loaded && state.matchedFrameVerified && event.source === "on_nicla" ? key : null;
    $("deviceHostReference").disabled = state.hostReferenceBusy || !state.hostReferenceKey;
    renderLinkage(event);
    if (!loaded) {
      $("matchedEmpty").querySelector("strong").textContent = "Matched frame unavailable";
      const detail = $("matchedEmpty").querySelector("span");
      if (detail && !detail.textContent.trim()) detail.textContent = `No frame reference was retained for ${eventBootId(event)}:${resultFrameId(event) ?? "?"}; result metadata remains available.`;
    }
    renderTelemetry(event);
    if (state.recording && !state.recordingEvents.some((row) => eventKey(row) === key)) {
      state.recordingEvents.push(event);
      state.recordingEvents = state.recordingEvents.slice(-MAX_RECORDING);
      $("deviceExport").disabled = state.recordingEvents.length === 0;
    }
    updateAges();
  }
  function validateEvent(event) {
    const frame = event?.frame || {};
    const id = resultFrameId(event);
    if (typeof (event?.boot_id || frame.boot_id) !== "string" || !event.boot_id && !frame.boot_id) return "missing boot/session identifier";
    if (!Number.isSafeInteger(Number(id)) || Number(id) < 0) return "missing or invalid frame identifier";
    if (event?.boot_id && frame.boot_id && event.boot_id !== frame.boot_id) return "root/frame boot IDs do not match";
    if (event?.frame_id != null && frame.frame_id != null && Number(event.frame_id) !== Number(frame.frame_id)) return "root/frame IDs do not match";
    for (const stage of [event?.detection, event?.detector, event?.decision]) {
      if (stage?.frame_id != null && Number(stage.frame_id) !== Number(id)) return "detection/decision frame ID does not match event frame";
    }
    const input = event?.identity_input || event?.model_input;
    if (input?.frame_id != null && Number(input.frame_id) !== Number(id)) return "identity input frame ID does not match event frame";
    if (input?.boot_id && input.boot_id !== (event?.boot_id || frame.boot_id)) return "identity input boot ID does not match event frame";
    if (event?.crop?.frame_id != null && Number(event.crop.frame_id) !== Number(id)) return "crop frame ID does not match event frame";
    return null;
  }
  function renderLinkage(event) {
    const id = resultFrameId(event);
    const detection = event.detection || event.detector || {};
    const input = event.identity_input || event.model_input || null;
    const detectId = detection.frame_id;
    const cropId = event.crop?.frame_id;
    const inputId = input?.frame_id;
    const decisionId = event.decision?.frame_id;
    const skipped = detection.status === "no_face" || detection.status === "multiple" || event.identity?.status === "skipped";
    const cropState = event.crop ? (cropId == null ? "?" : cropId) : skipped ? `skipped@${id}` : "not retained";
    const inputState = input ? (inputId == null ? "?" : inputId) : skipped ? `skipped@${id}` : "not retained";
    const proof = $("linkageProof");
    proof.textContent = `JPEG ${eventBootId(event)}#${id} · detect ${detectId ?? "?"} · crop ${cropState} · identity input ${inputState} · decision ${decisionId ?? "?"}${state.matchedFrameVerified ? " · frame bytes ID-verified" : " · frame bytes not independently ID-verified"}`;
    const complete = state.matchedFrameVerified && Number(detectId) === Number(id) && Number(decisionId) === Number(id)
      && (event.crop == null ? skipped : Number(cropId) === Number(id))
      && (!input || Number(inputId) === Number(id) || skipped);
    proof.className = `linkage-proof ${complete ? "passed" : "partial"}`;
    if (state.matchedFrameVerified && (Number(detectId) !== Number(id) || Number(decisionId) !== Number(id) || event.crop && Number(cropId) !== Number(id) || input && Number(inputId) !== Number(id))) proof.className = "linkage-proof failed";
  }
  function renderTimeline() {
    const body = $("deviceTimeline");
    const rows = [...state.events].reverse();
    body.replaceChildren();
    if (!rows.length) {
      const row = document.createElement("tr"), cell = document.createElement("td"); cell.colSpan = 6; cell.textContent = "No device events received."; row.appendChild(cell); body.appendChild(row);
    }
    for (const event of rows) {
      const row = document.createElement("tr"); row.tabIndex = 0; row.dataset.key = eventKey(event);
      const cells = [
        `${event.boot_id || "?"} / ${resultFrameId(event) ?? "?"}`,
        event.detection?.status || event.detector?.status || "—",
        event.decision?.status || "ACK",
        identityName(event.decision?.candidate || event.decision?.raw_candidate || event.predicted_class || "—"),
        event.timings_ms?.total != null ? `${event.timings_ms.total} ms` : event.telemetry?.timings_ms?.total != null ? `${event.telemetry.timings_ms.total} ms` : "—",
        event.decision?.reason || event.message || "",
      ];
      for (const value of cells) { const cell = document.createElement("td"); cell.textContent = String(value); row.appendChild(cell); }
      if (state.selectedEvent && eventKey(state.selectedEvent) === eventKey(event)) row.setAttribute("aria-current", "true");
      row.addEventListener("click", () => {
        state.resultFrozen = true;
        renderEvent(event);
        setAction(`Showing frozen event ${eventKey(event)}; latest camera updates continue independently.`);
        renderTimeline();
      });
      row.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); row.click(); } });
      body.appendChild(row);
    }
    $("timelineCount").textContent = `${state.events.length} recent events${state.skippedEventCount ? ` · ${state.skippedEventCount} older broker events omitted (not a lost-frame count)` : ""}`;
    $("returnToLive").disabled = !state.resultFrozen || !state.events.length;
  }
  function normalizeCapabilityList(caps) {
    const commands = caps?.commands || caps?.supported_commands || caps?.actions || {};
    return (cmd) => {
      const aliases = cmdAliases[cmd] || [cmd];
      if (Array.isArray(commands)) return aliases.some((name) => commands.includes(name));
      return aliases.some((name) => commands[name] === true || commands[name]?.available === true) || caps?.[cmd] === true;
    };
  }
  function populateChoices(select, values, current, placeholder) {
    values = Array.isArray(values) && values.length ? values : [{ id: "", label: placeholder }];
    const reportedPolicyIds = new Set(values.map((item) => typeof item === "string" ? item : String(item.id ?? item.value ?? "")));
    const choices = [];
    for (const item of values) {
      const value = typeof item === "string" ? item : String(item.id ?? item.value ?? "");
      const raw = typeof item === "string" ? item : String(item.label ?? item.name ?? item.id ?? value);
      // G and H currently resolve to the same deployed policy fields; expose
      // one truthful choice instead of pretending the alias is a separate arm.
      if (reportedPolicyIds.has("H") && reportedPolicyIds.has("G") &&
          (value === "G" && current !== "G" || value === "H" && current === "G")) continue;
      const policyLabels = {
        H: reportedPolicyIds.has("G") ? "H / G alias — fixed high · 250 ms · large model · no temporal votes" : "H — fixed high · large model",
        L: "L — fixed low · 2,000 ms · large model",
        S: "S — fixed high · small model",
        G: reportedPolicyIds.has("H") ? "G / H alias — fixed high · 250 ms · large model · no temporal votes" : "G — fixed high · large model",
        T: "T — scene-change trigger · detector fallback every 2,000 ms",
        V: "V — spatial scene-change trigger",
        U: "U — fixed 2,000 ms · quality gate disabled · experimental",
        W: "W — faster live loop · small model",
        F: "F — fast live · small model",
        "A-acquire": "A-acquire — periodic capture only",
        "A-sense": "A-sense — IMU motion adaptive",
        "A-model": "A-model — small then large on uncertainty",
        "A-risk": "A-risk — risk calibrated escalation",
        "A-cost": "A-cost — cost aware escalation",
        "A-full": "A-full — adaptive sensing · escalation · sleep request",
      };
      const label = policyLabels[value] || raw;
      choices.push({ value, label });
    }
    // Preserve native dropdowns while device status is polled several times a second.
    if (select.children.length !== choices.length || choices.some(({ value, label }, index) => {
      const option = select.children[index];
      return option.value !== value || option.textContent !== label;
    })) {
      select.replaceChildren(...choices.map(({ value, label }) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        return option;
      }));
    }
    if (current != null && select.value !== String(current) && choices.some(({ value }) => value === String(current))) {
      select.value = String(current);
    }
    if (!choices[0]?.value) select.disabled = true;
  }
  function policyModelSlot(policy) {
    if (["S", "T", "V", "W", "F"].includes(policy)) return "small";
    if (["H", "G", "L"].includes(policy)) return "large";
    return null;
  }
  function populateModelChoices(select, models, current, status) {
    const policy = status?.applied_state?.policy || status?.policy || "?";
    const modelOverride = status?.applied_state?.model_override;
    const policyModel = policyModelSlot(policy) || "varies";
    const policyLabel = modelOverride == null
      ? `Policy-selected · ${policy} → ${policyModel}`
      : `Use policy selection · ${policy} → ${policyModel}`;
    const values = [{ id: "policy", label: policyLabel }, ...(Array.isArray(models) ? models : [])];
    const known = new Set(values.map((item) => typeof item === "string" ? item : String(item.id ?? item.value ?? "")));
    if (current != null && !known.has(String(current))) values.unshift({ id: String(current), label: `Current model setting · ${current}` });
    populateChoices(select, values, current, "No model choices reported");
  }
  function populateMeasurementStages(select, stageIds, current) {
    const labels = {
      idle: "idle · no camera capture",
      camera_capture_only: "camera capture only",
      detector_only: "detector only",
      identity_center_crop_diagnostic: "identity · fixed center crop diagnostic",
      identity_small_only: "identity · small model only",
      identity_large_only: "identity · large model only",
      cold_model_load: "cold model load",
      warm_inference: "warm inference",
      model_swap_and_gc: "model swap and garbage collection",
      full_pipeline: "full capture → detector → identity pipeline",
    };
    const values = Array.isArray(stageIds) ? stageIds.map((id) => ({ id, label: labels[id] || id })) : [];
    populateChoices(select, values, current, "Measurement stages not reported by firmware");
  }
  function videoLocked() {
    return Boolean(state.videoAction || ["starting", "recording", "saving"].includes(state.video?.state));
  }
  function syncVideoControls(connected, trialActive = false) {
    const button = $("deviceVideo");
    const modeChoice = $("deviceVideoMode");
    const download = $("deviceVideoDownload");
    const manifest = $("deviceVideoManifest");
    const review = $("deviceVideoReview");
    const notice = $("deviceVideoStatus");
    const { state: phase, frames, max_duration_ms: maxDuration,
      missed_frames: missed, reason, error, live_max_duration_ms: liveDuration,
      matched_observations: matched, mode, owns_trial: ownsTrial, draining } = state.video || {};
    const saved = state.video?.last_clip || (phase === "ready" ? state.video : null);
    const url = saved?.download_url, manifestUrl = saved?.manifest_url;
    const duration = (phase === "recording" ? ownsTrial === false : state.loopRunning) ? liveDuration : maxDuration;
    const seconds = Number(duration) > 0 ? Number(duration) / 1000 : 12;
    const frameCount = Number.isSafeInteger(frames) && frames >= 0 ? `${frames} ${frames === 1 ? "frame" : "frames"}` : "";
    const canControl = connected && !state.fixture && state.controllerAuthority && state.controllerId && state.videoAvailable && state.lastPollOk !== false;
    const stopping = phase === "recording";
    const desiredStage = modeChoice.value === "inference" ? "full_pipeline" : "camera_capture_only";
    const currentStage = state.status?.applied_state?.measurement_stage_id;
    const blocked = !state.videoAvailable ? "Video capture is unavailable." : !connected ? "Connect the Nicla to record video."
      : !state.controllerAuthority ? "Claim control to record video." : state.paused ? "Resume UI updates to record video."
      : trialActive ? "Wait for the bounded trial to finish."
      : state.loopRunning && currentStage !== desiredStage ? "Stop processing to change video mode."
      : state.pending.size ? "Wait for the device command to finish." : "";
    button.textContent = phase === "starting" ? "Starting video…" : draining && stopping ? "Finishing video…"
      : stopping || state.videoAction === "stop" ? "Stop video" : "Record video";
    button.className = stopping ? "button danger" : "button";
    button.disabled = !canControl || Boolean(state.videoAction) || phase === "starting" || phase === "saving"
      || (stopping ? Boolean(draining) : Boolean(blocked));
    modeChoice.disabled = !canControl || videoLocked();
    button.title = phase === "saving" ? "The video file is being saved." : stopping && canControl ? "Stop recording and save the frames captured so far."
      : blocked || `Record a local AVI clip for up to ${seconds} seconds.`;
    const hasDownload = typeof url === "string" && url.startsWith(VIDEO_PATH + "/file?");
    const clipActions = $("deviceClipActions");
    clipActions.hidden = !hasDownload;
    if (!hasDownload) clipActions.open = false;
    download.hidden = !hasDownload;
    if (hasDownload) download.href = url;
    else download.removeAttribute("href");
    const hasManifest = hasDownload && typeof manifestUrl === "string" && manifestUrl.startsWith(VIDEO_PATH + "/manifest?");
    manifest.hidden = !hasManifest;
    review.hidden = !hasManifest;
    if (hasManifest) manifest.href = manifestUrl;
    else manifest.removeAttribute("href");
    const results = mode === "inference" && Number.isSafeInteger(matched) ? ` · ${matched} linked results` : "";
    const counts = frameCount ? ` · ${frameCount}${results}` : "";
    let message = blocked || `Records up to ${seconds} seconds.`;
    if (phase === "starting" || state.videoAction === "start") message = "Starting video capture…";
    else if (phase === "saving" || state.videoAction === "stop") message = `Saving AVI…${counts}`;
    else if (phase === "recording") message = `${draining ? "Finishing recording…" : "Recording…"}${counts}`;
    else if (phase === "ready" && !blocked) {
      const interrupted = ["disconnected", "device_restarted", "timeout"].includes(reason);
      const gaps = missed > 0 ? ` · ${missed} missing` : "";
      message = `${interrupted ? "Partial video saved" : "Video ready"}${counts}${gaps}.`;
    }
    else if (phase === "error") message = String(error || "Video capture failed.");
    if (state.videoError) message = state.videoError;
    if (notice.textContent !== message) notice.textContent = message;
    notice.dataset.state = state.videoError || phase === "error" ? "error" : phase || "idle";
  }
  async function pollVideoStatus() {
    if (state.fixture || state.videoAction) return;
    const revision = ++state.videoRevision;
    try {
      const result = await getJson(VIDEO_PATH);
      if (state.videoRevision !== revision) return;
      state.videoAvailable = VIDEO_STATES.includes(result?.state);
      if (state.videoAvailable) state.video = result;
    } catch (_) {
      if (state.videoRevision === revision) state.videoAvailable = false;
    }
  }
  async function toggleVideo() {
    const button = $("deviceVideo");
    if (button.disabled || state.videoAction || !state.controllerAuthority || !state.controllerId) return;
    const action = ["starting", "recording"].includes(state.video?.state) ? "stop" : "start";
    if (action === "start" && (state.pending.size || state.paused)) return;
    state.videoAction = action;
    state.videoRevision++;
    state.videoError = null;
    syncControls(state.status, state.capability);
    try {
      const body = { controller_id: state.controllerId, viewer_id: tabId };
      if (action === "start") body.mode = $("deviceVideoMode").value;
      const result = await postJson(`${VIDEO_PATH}/${action}`, body);
      if (VIDEO_STATES.includes(result?.state)) state.video = result;
    } catch (error) {
      state.videoError = `Could not ${action} video: ${error.message}`;
    } finally {
      state.videoAction = null;
      await pollVideoStatus();
      syncControls(state.status, state.capability);
    }
  }
  function setSightings(result) {
    if (typeof result?.enabled !== "boolean" || !Array.isArray(result.items) ||
        !Number.isSafeInteger(result.total_encounters) || result.total_encounters < 0 ||
        !Number.isSafeInteger(result.retained_count) || result.retained_count < 0) {
      throw new Error("Invalid sightings response.");
    }
    state.sightings = result;
    state.sightingsAvailable = true;
  }
  function sightingPerson(item) {
    const legacyPerson = item.decision_status === "KNOWN_IDENTITY" &&
      item.known_acceptance_enabled === true ? item.identity_id : null;
    const person = Object.hasOwn(item, "person_id") ? item.person_id : legacyPerson;
    return SIGHTING_PEOPLE.includes(person) ? person : "other";
  }
  function sightingPersonLabel(person) {
    return person === "all" ? "All people" : person === "other" ? "Other" : `Person ${SIGHTING_PEOPLE.indexOf(person) + 1}`;
  }
  function sightingTimestamp(item) {
    const imported = item.source === "on_nicla_battery";
    const timestamp = imported ? item.imported_utc : item.utc;
    const date = new Date(typeof timestamp === "string" ? timestamp : NaN);
    return Number.isFinite(date.getTime()) ? {
      iso: date.toISOString(), text: (imported ? "Imported " : "") + sightingDate.format(date),
    } : { iso: null, text: "Time unavailable" };
  }
  function sightingOrigin(item) {
    return item.source === "on_nicla_battery" ? "On-board capture" : "USB capture";
  }
  function sightingProvenance(item) {
    if (item.person_assignment === "manual") return "Assigned manually";
    if (item.person_assignment === "device" ||
        (item.person_assignment == null && sightingPerson(item) !== "other")) return "Identified by Nicla";
    return "Unassigned";
  }
  function sightingPhotoUrl(item) {
    return item.image_url === `${SIGHTINGS_PATH}/frame?id=${item.id}` ? item.image_url : null;
  }
  function sightingCard(item) {
    const card = document.createElement("li");
    card.className = "sighting-card";
    const review = document.createElement("button");
    review.type = "button";
    review.className = "sighting-review";
    const stamp = sightingTimestamp(item);
    review.setAttribute("aria-label", `Review ${sightingPersonLabel(sightingPerson(item))} encounter. ${stamp.text}`);
    review.addEventListener("click", () => openSighting(item.id, review));
    const url = sightingPhotoUrl(item);
    if (url) {
      const photo = document.createElement("img");
      photo.loading = "lazy";
      photo.decoding = "async";
      photo.width = 240;
      photo.height = 240;
      photo.alt = "";
      photo.src = url;
      review.appendChild(photo);
    }
    const origin = document.createElement("strong");
    origin.textContent = sightingOrigin(item);
    review.appendChild(origin);
    const time = document.createElement("time");
    if (stamp.iso) time.dateTime = stamp.iso;
    time.textContent = stamp.text;
    review.appendChild(time);
    const provenance = document.createElement("span");
    provenance.textContent = sightingProvenance(item);
    review.appendChild(provenance);
    card.appendChild(review);
    return card;
  }
  function syncSightingGallery(result) {
    const view = `${state.sightingFilter}:${state.sightingSource}:${state.sightingLimit}:${state.sightingsAvailable}`;
    if (result === state.sightingGalleryResult && view === state.sightingGalleryView) {
      syncSightingAssignment();
      return;
    }
    state.sightingGalleryResult = result;
    state.sightingGalleryView = view;
    const list = $("sightingsList");
    const retained = (result?.items || []).filter((item) => Number.isSafeInteger(item?.id) && item.id > 0);
    const isImported = (item) => item.source === "on_nicla_battery";
    $("batteryReview").hidden = !retained.some(isImported);
    const items = retained.filter((item) => state.sightingSource === "all" ||
      (state.sightingSource === "onboard" ? isImported(item) : !isImported(item)))
      .sort((a, b) => b.id - a.id);
    const totals = Object.fromEntries(SIGHTING_PEOPLE.map((person) => [person, 0]));
    for (const item of items) totals[sightingPerson(item)]++;
    for (const option of $("sightingsPerson").options) {
      const count = option.value === "all" ? items.length : totals[option.value];
      if (count == null) continue;
      const label = `${sightingPersonLabel(option.value)} (${count.toLocaleString()})`;
      if (option.textContent !== label) option.textContent = label;
    }
    const matches = state.sightingFilter === "all" ? items : items.filter((item) => sightingPerson(item) === state.sightingFilter);
    const cards = new Map();
    const groups = new Map();
    const photosPerGroup = state.sightingFilter === "all"
      ? Math.ceil(state.sightingLimit / 2) : state.sightingLimit;
    for (const item of matches) {
      const person = sightingPerson(item);
      if (!groups.has(person)) groups.set(person, []);
      const group = groups.get(person);
      if (group.length >= photosPerGroup) continue;
      const key = JSON.stringify([item.id, person, item.person_assignment, item.utc, item.imported_utc, item.source, item.image_url]);
      const card = state.sightingCards.get(key) || sightingCard(item);
      cards.set(key, card);
      group.push(card);
    }
    const groupsKey = JSON.stringify([state.sightingFilter, [...cards.keys()], totals]);
    if (groupsKey !== state.sightingGroupsKey) {
      const focused = list.contains?.(document.activeElement) ? document.activeElement : null;
      const sections = [];
      for (const person of SIGHTING_PEOPLE) {
        const photos = groups.get(person);
        if (!photos) continue;
        const section = document.createElement("section");
        section.className = "sightings-group";
        const heading = document.createElement("h3");
        heading.textContent = `${sightingPersonLabel(person)} · ${totals[person].toLocaleString()}`;
        section.appendChild(heading);
        const gallery = document.createElement("ul");
        gallery.className = "sightings-list";
        gallery.replaceChildren(...photos);
        section.appendChild(gallery);
        sections.push(section);
      }
      list.replaceChildren(...sections);
      if (focused?.isConnected) focused.focus({ preventScroll: true });
      state.sightingGroupsKey = groupsKey;
      state.sightingCards = cards;
    }
    list.hidden = cards.size === 0;
    $("sightingsFilters").hidden = !result;
    const more = $("sightingsMore");
    const remaining = matches.length - cards.size;
    more.hidden = remaining <= 0;
    more.textContent = remaining > 0 ? `Show more photos (${remaining.toLocaleString()})` : "Show more photos";
    const empty = $("sightingsEmpty");
    empty.hidden = !state.sightingsAvailable || !result || cards.size > 0;
    empty.textContent = state.sightingFilter !== "all"
      ? `No matching photos for ${sightingPersonLabel(state.sightingFilter)}.`
      : state.sightingSource === "onboard" ? "No imported photos are currently retained."
        : state.sightingSource === "usb" ? "No live USB photos are currently retained."
          : result?.total_encounters ? "No photos are currently retained." : "No encounters saved yet.";
    const kept = result?.retained_count || 0;
    $("sightingsRetention").textContent = result ? state.sightingSource !== "all"
      ? `${matches.length.toLocaleString()} matching ${matches.length === 1 ? "photo" : "photos"}`
      : retained.length < kept
      ? `${retained.length.toLocaleString()} of ${kept.toLocaleString()} retained photos available`
      : `${kept.toLocaleString()} ${kept === 1 ? "photo" : "photos"} kept locally` : "";
    syncSightingAssignment();
  }
  function selectedSighting() {
    return state.sightings?.items.find((item) => item.id === state.openSightingId);
  }
  function syncSightingAssignment() {
    if (!$("sightingDialog").open) return;
    const item = selectedSighting();
    const ownsControl = state.controllerAuthority && Boolean(state.controllerId);
    const waiting = state.sightingAssigning || state.sightingClaiming;
    const select = $("sightingAssignPerson");
    const canChange = Boolean(item) && state.sightingsAvailable && state.lastPollOk !== false && ownsControl && !waiting;
    select.disabled = !canChange;
    $("sightingAssignSave").disabled = !canChange ||
      (item?.person_assignment === "manual" && sightingPerson(item) === select.value);
    $("sightingAssignSave").textContent = state.sightingAssigning ? "Saving…" : "Save assignment";
    $("sightingClaim").hidden = ownsControl || !item;
    const lease = state.status?.controller_authority;
    const owner = lease?.viewer_id || lease?.owner_viewer_id;
    const anotherController = owner && owner !== tabId && Number(lease.lease_remaining_ms) > 0;
    $("sightingClaim").disabled = !state.sightingsAvailable || state.lastPollOk === false || waiting ||
      state.claimingController || Boolean(anotherController);
    if (!item) $("sightingAssignmentStatus").textContent = "This photo is no longer retained.";
    else if (!ownsControl && !$("sightingAssignmentStatus").textContent) {
      $("sightingAssignmentStatus").textContent = anotherController ? "Another tab holds control."
        : "Take control to assign this photo. The Nicla can stay disconnected.";
    }
  }
  function openSighting(id, opener) {
    state.openSightingId = id;
    const item = selectedSighting();
    if (!item) return;
    state.sightingOpener = opener;
    const personLabel = sightingPersonLabel(sightingPerson(item));
    $("sightingDialogTitle").textContent = personLabel;
    const image = $("sightingDialogPhoto");
    const original = $("sightingOriginal");
    const url = sightingPhotoUrl(item);
    image.hidden = original.hidden = !url;
    if (url) {
      image.src = url;
      original.href = url;
    }
    const frame = Number.isSafeInteger(item.frame_id) ? `Frame ${item.frame_id}` : null;
    $("sightingDialogDetails").textContent = [
      sightingOrigin(item), sightingTimestamp(item).text, frame, sightingProvenance(item),
    ].filter(Boolean).join(" · ");
    $("sightingAssignPerson").value = sightingPerson(item);
    $("sightingAssignmentStatus").textContent = "";
    $("sightingDialog").showModal();
    syncSightingAssignment();
  }
  async function saveSightingAssignment(event) {
    event.preventDefault();
    const item = selectedSighting(), person = $("sightingAssignPerson").value;
    if (!item || !SIGHTING_PEOPLE.includes(person) || $("sightingAssignSave").disabled) return;
    state.sightingAssigning = true;
    syncSightingAssignment();
    const status = $("sightingAssignmentStatus");
    status.textContent = "Saving assignment…";
    try {
      const personId = person === "other" ? null : person;
      const response = await postJson(`${SIGHTINGS_PATH}/assign`, {
        sighting_id: item.id, person_id: personId, controller_id: state.controllerId, viewer_id: tabId,
      });
      if (response?.sighting_id !== item.id || response.person_id !== personId ||
          response.person_assignment !== "manual") throw new Error("The saved assignment could not be confirmed.");
      state.sightingsRevision++;
      state.sightings = { ...state.sightings, items: state.sightings.items.map((entry) => entry.id === item.id
        ? { ...entry, person_id: personId, assigned_person_id: personId, person_assignment: "manual" } : entry) };
      if (state.openSightingId === item.id) {
        const personLabel = sightingPersonLabel(person);
        $("sightingDialogTitle").textContent = personLabel;
        $("sightingDialogDetails").textContent = [
          sightingOrigin(item), sightingTimestamp(item).text,
          Number.isSafeInteger(item.frame_id) ? `Frame ${item.frame_id}` : null, "Assigned manually",
        ].filter(Boolean).join(" · ");
        status.textContent = `Saved to ${personLabel}.`;
      }
      syncSightingGallery(state.sightings);
    } catch (error) {
      if (state.openSightingId === item.id) status.textContent = `Could not assign the photo. ${error.message}`;
    } finally {
      state.sightingAssigning = false;
      syncSightingAssignment();
      pollSightings();
    }
  }
  function syncSightingsControls(status) {
    const button = $("sightingsToggle");
    const notice = $("sightingsStatus");
    const count = $("sightingsCount");
    const connected = state.lastPollOk !== false && (status?.connected === true || status?.connection?.state === "connected" || status?.state === "connected");
    const authority = state.controllerAuthority && Boolean(state.controllerId);
    const result = state.sightings;
    const enabled = result?.enabled === true;
    button.disabled = state.fixture || !state.sightingsAvailable || !authority || state.sightingsBusy || (!connected && !enabled);
    button.textContent = state.sightingsBusy ? "Updating…" : enabled ? "Stop saving" : "Start saving";
    button.title = !state.sightingsAvailable ? "The local sightings service is unavailable."
      : !authority ? "Claim control to change photo saving." : !connected && !enabled ? "Connect the Nicla to start saving photos."
      : enabled ? "Stop saving new encounters." : "Save one photo per detected encounter to this computer.";
    let message = "Checking local sightings…";
    if (state.fixture) message = "Sightings are unavailable in test fixture mode.";
    else if (state.sightingsAvailable === false) message = result ? "Sightings service unavailable. Showing the last loaded list." : "Sightings service unavailable.";
    else if (enabled) message = connected ? "Saving new face encounters to this computer." : "Saving is on. Waiting for the Nicla to reconnect.";
    else if (result) message = connected ? "Saving is off. Turn it on to keep photos on this computer." : "Saving is off. Connect the Nicla to start saving.";
    const storageError = state.sightingsAvailable && typeof result?.error === "string" && result.error;
    if (storageError) message = `Photo storage error. ${storageError}`;
    if (state.sightingsError) message = state.sightingsError;
    if (notice.textContent !== message) notice.textContent = message;
    notice.dataset.state = storageError || state.sightingsError || state.sightingsAvailable === false ? "error" : "ready";
    count.textContent = result ? `${result.total_encounters.toLocaleString()} total ${result.total_encounters === 1 ? "encounter" : "encounters"}` : "Total encounters —";
    syncSightingGallery(result);
  }
  async function pollSightings() {
    if (state.fixture || state.sightingsBusy || state.sightingsPolling) return;
    state.sightingsPolling = true;
    const revision = ++state.sightingsRevision;
    try {
      const result = await getJson(SIGHTINGS_PATH);
      if (revision === state.sightingsRevision) {
        const priorEnabled = state.sightings?.enabled;
        setSightings(result);
        if (priorEnabled !== result.enabled) state.sightingsError = null;
      }
    } catch (_) {
      if (revision === state.sightingsRevision) state.sightingsAvailable = false;
    } finally {
      state.sightingsPolling = false;
      syncSightingsControls(state.status);
    }
  }
  async function toggleSightings() {
    if ($("sightingsToggle").disabled || !state.controllerAuthority || !state.controllerId) return;
    state.sightingsBusy = true;
    state.sightingsRevision++;
    state.sightingsError = null;
    syncSightingsControls(state.status);
    try {
      setSightings(await postJson(SIGHTINGS_PATH, {
        enabled: !state.sightings.enabled, controller_id: state.controllerId, viewer_id: tabId,
      }));
    } catch (error) {
      state.sightingsError = `Could not update photo saving. ${error.message}`;
    } finally {
      state.sightingsBusy = false;
      syncSightingsControls(state.status);
    }
  }
  function setBatteryStatus(response) {
    const battery = response?.battery || response;
    if (typeof battery?.supported !== "boolean") throw new Error("Battery capture status was not reported.");
    const previous = state.battery;
    state.battery = battery;
    state.batteryAvailable = true;
    if (!battery.busy && (previous?.active_action === "import" ||
        Number.isSafeInteger(battery.imported_count) && battery.imported_count > (previous?.imported_count ?? 0))) {
      pollSightings();
    }
  }
  function syncBatteryControls(status) {
    const connected = state.lastPollOk !== false && (status?.connected === true ||
      status?.connection?.state === "connected" || status?.state === "connected");
    const battery = state.batteryAvailable === true ? state.battery : null;
    const pending = battery?.pending_photos ?? battery?.record_count;
    const count = connected && battery?.supported && Number.isSafeInteger(pending) && pending >= 0 ? pending : null;
    const busy = Boolean(state.batteryAction || battery?.busy);
    const activeCapture = videoLocked() || state.loopRunning || status?.applied_state?.trial_active === true;
    const ownsControl = state.controllerAuthority && Boolean(state.controllerId);
    const canAct = connected && !state.fixture && battery?.supported === true && battery.available !== false &&
      ownsControl && !busy && !activeCapture;
    const allowed = { arm: battery?.armed === false, disarm: battery?.armed === true, import: count > 0 };
    for (const [action, id] of Object.entries(BATTERY_BUTTONS)) {
      $(id).disabled = !canAct || !allowed[action] || battery.actions?.[action] === false;
    }
    const importing = (state.batteryAction || battery?.active_action) === "import";
    $("batteryImport").textContent = importing ? "Importing photos…" : "Import photos";
    const limit = battery?.max_records;
    $("batteryPending").textContent = count === null ? "—" :
      Number.isSafeInteger(limit) ? `${count.toLocaleString()} / ${limit.toLocaleString()}` : count.toLocaleString();
    const imported = battery?.imported_count ?? state.sightings?.battery_import_count;
    $("batteryImported").textContent = Number.isSafeInteger(imported) ? imported.toLocaleString() : "—";
    const prepare = $("batteryPrepare");
    const nextAction = !connected ? "connect" : !ownsControl ? "control" :
      state.loopRunning && !videoLocked() && status?.applied_state?.trial_active !== true ? "stop" : null;
    const actionLabel = { connect: "Connect Nicla", control: "Take control", stop: "Stop processing" };
    prepare.hidden = !nextAction || state.fixture || (connected && battery?.supported === false && !battery.checking);
    prepare.disabled = state.lastPollOk === false || busy || state.claimingController ||
      (nextAction === "stop" ? $("deviceLoop").disabled : $("deviceConnect").disabled);
    prepare.textContent = actionLabel[nextAction] || "Connect Nicla";
    prepare.dataset.action = nextAction || "";
    let label = "CHECKING", message = "Checking whether the Nicla can store photos while unplugged…";
    if (state.fixture) {
      label = "TEST MODE";
      message = "Offline capture is unavailable in test mode.";
    } else if (state.batteryAvailable === false) {
      label = "UNAVAILABLE";
      message = "The dashboard could not read offline capture support.";
    } else if (!connected) {
      label = "CONNECT NICLA";
      message = "Plug the Nicla into this computer with a USB data cable to check for saved photos.";
    } else if (battery?.checking) {
      message = "Waiting for the Nicla to report its offline capture capabilities.";
    } else if (battery?.supported === false) {
      label = "UNSUPPORTED";
      message = typeof battery.reason === "string" ? battery.reason : "The connected firmware does not report offline photo capture support.";
    } else if (battery?.supported) {
      label = importing ? "IMPORTING" : busy ? "WORKING" : battery.armed === true ? "ARMED" : battery.armed === false ? "DISARMED" : "STATUS UNKNOWN";
      if (busy) {
        message = importing ? "Copying and verifying photos from the Nicla…" : battery.reason || "Waiting for the Nicla to finish the requested action…";
      } else if (!ownsControl) {
        message = "Take camera control to change offline capture or import photos.";
      } else if (videoLocked()) {
        message = "Finish the video recording to manage offline capture or import photos.";
      } else if (activeCapture) {
        message = battery.armed === true
          ? "Ready to take on a power bank. Stop processing after reconnecting to check for new photos."
          : "Stop processing to check the saved photo count, import photos, or arm offline capture.";
      } else if (battery.reason) {
        message = battery.reason;
      } else if (battery.available === false || count === null) {
        message = "Waiting for the Nicla to report its offline capture status.";
      } else {
        const unimported = battery.unimported_count;
        message = count > 0 ? Number.isSafeInteger(unimported) && unimported === 0
          ? "All listed photos have already been imported. Importing again skips duplicates."
          : "Import photos to add them to Saved sightings."
          : battery.armed === true ? "Offline capture is armed. There are no photos to import."
            : "Arm offline capture before switching to battery power.";
      }
    }
    const notice = $("batteryStatus");
    const shown = state.batteryError || battery?.error || message;
    if (notice.textContent !== shown) notice.textContent = shown;
    notice.dataset.state = state.batteryError || battery?.error || state.batteryAvailable === false ? "error" : "ready";
    if ($("batteryState").textContent !== label) $("batteryState").textContent = label;
    $("batteryState").closest?.(".battery-panel")?.setAttribute("data-armed", String(connected && battery?.armed === true));
  }
  async function pollBattery() {
    if (state.fixture || state.batteryPolling || state.batteryAction) return;
    state.batteryPolling = true;
    const revision = ++state.batteryRevision;
    try {
      const result = await getJson(BATTERY_PATH);
      if (revision === state.batteryRevision) setBatteryStatus(result);
    } catch (_) {
      if (revision === state.batteryRevision) state.batteryAvailable = false;
    } finally {
      state.batteryPolling = false;
      syncBatteryControls(state.status);
    }
  }
  async function updateBattery(action) {
    if ($(BATTERY_BUTTONS[action]).disabled || !state.controllerAuthority || !state.controllerId) return;
    state.batteryAction = action;
    state.batteryRevision++;
    state.batteryError = null;
    syncBatteryControls(state.status);
    try {
      const result = await postJson(BATTERY_PATH, { action, controller_id: state.controllerId, viewer_id: tabId },
        action === "import" ? 30000 : 5000);
      setBatteryStatus(result);
      if (action === "import") pollSightings();
    } catch (error) {
      const attempted = action === "import" ? "import offline photos" : `${action} offline capture`;
      state.batteryError = `Could not ${attempted}. ${error.message}`;
    } finally {
      state.batteryAction = null;
      syncBatteryControls(state.status);
    }
  }
  function pauseClip() {
    clipReview.playing = false;
    clearTimeout(clipReview.timer);
    $("deviceClipPlay").textContent = "Play clip";
  }
  function closeClip(restoreFocus = false) {
    pauseClip();
    clipReview.request++;
    clipReview.manifest = null;
    $("deviceClipReview").hidden = true;
    if (restoreFocus) $("deviceClipActions").querySelector?.("summary")?.focus();
  }
  async function openClip() {
    const url = (state.video?.last_clip || state.video)?.manifest_url;
    if (!url?.startsWith(VIDEO_PATH + "/manifest?")) return;
    closeClip();
    const request = clipReview.request;
    const panel = $("deviceClipReview");
    panel.hidden = false;
    $("deviceClipClose").focus?.();
    $("deviceClipSummary").textContent = "Loading saved frames…";
    $("deviceClipPosition").disabled = $("deviceClipPlay").disabled = true;
    const canvas = $("deviceClipCanvas");
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    try {
      const manifest = await getJson(url);
      if (request !== clipReview.request) return;
      if (!Array.isArray(manifest.frames) || !manifest.frames.length ||
          !/^\d{8}T\d{6}Z-[0-9a-f]{8}$/.test(manifest.clip_id)) throw new Error("The clip manifest is incomplete.");
      clipReview.manifest = manifest;
      $("deviceClipPosition").max = String(manifest.frames.length);
      $("deviceClipPosition").disabled = $("deviceClipPlay").disabled = manifest.frames.length < 2;
      await showClipFrame(0);
    } catch (error) {
      if (request === clipReview.request) $("deviceClipSummary").textContent = error.message;
    }
  }
  async function showClipFrame(index) {
    const manifest = clipReview.manifest;
    const frame = manifest?.frames[index];
    if (!frame) return;
    const request = ++clipReview.request;
    clipReview.index = index;
    $("deviceClipPosition").value = String(index + 1);
    const image = new Image();
    image.src = `${VIDEO_PATH}/frame?id=${encodeURIComponent(manifest.clip_id)}&number=${index + 1}`;
    try {
      await image.decode();
      if (request !== clipReview.request) return;
      const canvas = $("deviceClipCanvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0);
      ctx.lineWidth = 2;
      const observation = frame.observation;
      ctx.strokeStyle = "#5ee8a4";
      for (const region of observation?.detection?.regions_xywh || []) {
        if (Array.isArray(region.rect_px) && region.rect_px.length === 4) ctx.strokeRect(...region.rect_px);
      }
      const crop = observation?.crop?.rect_px;
      ctx.strokeStyle = "#ffd76c";
      if (Array.isArray(crop) && crop.length === 4) ctx.strokeRect(...crop);
      const decision = observation?.decision?.status?.replaceAll("_", " ");
      const detail = decision ? `${decision} · detection ${observation.detection?.status || "unavailable"} · identity ${observation.identity?.status || "unavailable"}`
        : manifest.recording_mode === "inference" ? "No matching result saved" : "Camera-only frame";
      $("deviceClipSummary").textContent = `Frame ${index + 1} / ${manifest.frames.length} · ${detail}`;
      $("deviceClipPosition").setAttribute("aria-valuetext", `Frame ${index + 1} of ${manifest.frames.length}`);
      if (!clipReview.playing) return;
      const next = manifest.frames[index + 1];
      if (!next) return pauseClip();
      let delay = (next.host_received_monotonic_ns - frame.host_received_monotonic_ns) / 1e6;
      const modulus = frame.ticks_modulus_ms;
      if (modulus && next.ticks_modulus_ms === modulus && frame.capture_ticks_ms != null && next.capture_ticks_ms != null) {
        const ticks = (next.capture_ticks_ms - frame.capture_ticks_ms + modulus) % modulus;
        if (ticks > 0 && ticks < modulus / 2) delay = ticks;
      }
      if (!(delay > 0)) delay = (manifest.timing?.frame_interval_us || 1000000) / 1000;
      clipReview.timer = setTimeout(() => showClipFrame(index + 1), Math.max(20, delay));
    } catch (error) {
      if (request !== clipReview.request) return;
      pauseClip();
      $("deviceClipSummary").textContent = "The saved JPEG could not be opened.";
    }
  }
  function toggleClipPlayback() {
    if (clipReview.playing) return pauseClip();
    if (!clipReview.manifest) return;
    clipReview.playing = true;
    $("deviceClipPlay").textContent = "Pause clip";
    showClipFrame(clipReview.index + 1 >= clipReview.manifest.frames.length ? 0 : clipReview.index);
  }
  function syncControls(status, caps) {
    const can = normalizeCapabilityList(caps);
    const connected = status?.connected === true || status?.connection?.state === "connected" || status?.state === "connected";
    const lease = status?.controller_authority && typeof status.controller_authority === "object" ? status.controller_authority : {};
    const authority = lease.expired !== true && (status?.controller_authority === true || lease.owned_by_viewer === true ||
      status?.control?.authority === true || status?.controller?.is_owner === true || state.fixture && state.controllerAuthority);
    const leaseOwner = lease.viewer_id || lease.owner_viewer_id || null;
    const leaseHeldByOther = !state.fixture && Boolean(leaseOwner && leaseOwner !== tabId && Number(lease.lease_remaining_ms || 0) > 0);
    let fixtureLockHeldByOther = false;
    if (state.fixture) {
      try {
        const fixtureLease = JSON.parse(localStorage.getItem(LOCK_KEY) || "null");
        fixtureLockHeldByOther = Boolean(fixtureLease && fixtureLease.client !== tabId && Number(fixtureLease.expires) > Date.now());
      } catch (_) {}
    }
    state.controllerAuthority = authority;
    const trialActive = status?.applied_state?.trial_active === true;
    state.controllerId = authority ? status?.controller_id || lease.controller_id || status?.control?.controller_id ||
      status?.controller?.controller_id || state.controllerId : null;
    state.loopRunning = status?.applied_state?.processing === true || status?.applied_state?.loop_running === true || status?.loop_running === true || status?.running === true;
    state.sleeping = status?.applied_state?.camera_sleeping === true || status?.camera?.sleeping === true || status?.sleeping === true;
    state.model = appliedModelChoice(status) || state.model;
    state.policy = status?.applied_state?.policy || status?.policy || state.policy;
    renderRuntimeState(status);
    const deviceOnline = connected || state.fixture;
    const commandAllowed = (cmd) => deviceOnline && authority && can(cmd) &&
      (cmd === "stop_loop" || !state.pending.size && !state.applyingSettings);
    $("deviceOneShot").disabled = !commandAllowed("one_shot") || state.sleeping || trialActive;
    $("deviceOneShot").title = !can("one_shot") ? "One-shot inference is not reported as supported by this firmware." : !authority ? "This viewer is read-only; connect from the authoritative controller tab." : state.sleeping ? "Wake the camera before requesting a new frame." : "Capture and run the available detector and identity route once.";
    $("deviceProbeNoFace").disabled = !commandAllowed("one_shot") || state.sleeping || trialActive;
    $("deviceProbeNoFace").title = !can("one_shot") ? "One-shot inference is not reported as supported by this firmware." : !authority ? "This viewer is read-only; connect from the authoritative controller tab." : state.sleeping ? "Wake the camera before requesting a new frame." : "Capture a new frame; if FOMO reports NO_FACE, run one selected identity model on a fixed center crop. The decision stays NO_FACE and acceptance is disabled.";
    $("deviceLoop").disabled = !((state.loopRunning && commandAllowed("stop_loop")) || (!state.loopRunning && commandAllowed("start_loop")));
    $("deviceLoop").textContent = state.loopRunning ? "Stop processing" : "Start processing";
    $("deviceLoop").title = !(state.loopRunning ? can("stop_loop") : can("start_loop")) ? "Firmware does not report the requested processing-loop action." : !authority ? "This viewer is read-only; only the controller can change the processing loop." : state.loopRunning ? "Stop the device loop at its safe boundary." : "Start the device processing loop.";
    $("deviceSleep").disabled = !(state.sleeping ? commandAllowed("wake_camera") : commandAllowed("sleep_camera")) || trialActive;
    $("deviceSleep").textContent = state.sleeping ? "Wake camera" : "Sleep camera";
    $("deviceSleep").title = !(state.sleeping ? can("wake_camera") : can("sleep_camera")) ? "Firmware does not report the requested camera sleep or wake action." : !authority ? "This viewer is read-only; only the controller can change camera power state." : state.sleeping ? "Wake the camera and report its settling state." : "Request camera sleep where the firmware supports it.";
    $("deviceApplySettings").disabled = !deviceOnline || !authority || trialActive || state.pending.size > 0 ||
      state.applyingSettings || !(can("set_model") || can("set_policy"));
    $("deviceModel").disabled = !deviceOnline || !authority || trialActive || !Array.isArray(caps?.models || caps?.model_choices) || !(caps.models || caps.model_choices).length;
    $("devicePolicy").disabled = !deviceOnline || !authority || trialActive || !Array.isArray(caps?.policies || caps.policy_choices) || !(caps.policies || caps.policy_choices).length;
    const stages = Array.isArray(caps?.measurement_stage_ids) ? caps.measurement_stage_ids : [];
    const trialReady = commandAllowed("trial") && stages.length > 0 && !state.loopRunning;
    $("deviceMeasurementStage").disabled = !deviceOnline || !authority || !can("trial") || !stages.length || state.loopRunning;
    $("deviceTrialFrames").disabled = !trialReady;
    $("deviceTrialDuration").disabled = !trialReady;
    $("deviceTrialId").disabled = !trialReady;
    $("deviceTrial").disabled = !trialReady || trialActive;
    $("deviceTrial").title = !can("trial") ? "Firmware does not advertise bounded trial commands." : !stages.length ? "Firmware did not report measurement stages." : !authority ? "This viewer is read-only; the controller must start a trial." : state.loopRunning ? "Stop the current loop before starting a bounded trial." : trialActive ? "A bounded trial is already active." : "Run the selected measurement stage for a bounded number of frames or time.";
    $("deviceConnect").textContent = status?.connecting === true ? "Connecting…" :
      leaseHeldByOther ? "View only" : !connected ? "Connect to device" : !authority ? "Take control" : "Refresh device status";
    $("deviceConnect").disabled = status?.connecting === true || leaseHeldByOther || fixtureLockHeldByOther;
    $("deviceConnect").title = leaseHeldByOther ? "Release control in the other dashboard tab to take control here."
      : fixtureLockHeldByOther ? "Another test fixture tab holds control."
      : authority && connected ? "Refresh settings and firmware capabilities." : "Connect to the Nicla and take control.";
    const ownership = $("deviceOwnership");
    if (ownership) {
      ownership.textContent = leaseHeldByOther ? "View only. Release control in the other dashboard tab."
        : connected && !authority ? "Take control to operate the camera."
        : authority && connected ? "This tab has camera control." : "";
    }
    $("deviceRelease").disabled = !authority || state.fixture;
    $("deviceDisconnect").disabled = !(connected || status?.serial_open === true) || !authority || state.fixture;
    $("deviceDisconnect").title = status?.serial_open && !connected
      ? "Releases an open serial handle after a boot/hello timeout; the device loop is not stopped."
      : "Closes the host broker connection; it does not stop the device processing loop.";
    const recordingVideo = videoLocked();
    if (recordingVideo) {
      for (const id of ["deviceDisconnect", "deviceRelease", "deviceOneShot", "deviceProbeNoFace",
        "deviceLoop", "deviceSleep", "deviceModel", "devicePolicy", "deviceApplySettings", "deviceMeasurementStage",
        "deviceTrialFrames", "deviceTrialDuration", "deviceTrialId", "deviceTrial"]) {
        $(id).disabled = true;
      }
      if (authority || !connected) $("deviceConnect").disabled = true;
    }
    $("devicePause").disabled = recordingVideo && !state.paused;
    syncVideoControls(connected, trialActive);
    syncSightingsControls(status);
    syncBatteryControls(status);
    updateCancelControl();
    if (!deviceOnline) $("deviceCapabilityNote").textContent = "No connected device or test fixture. Controls stay unavailable until broker status and explicit capabilities are received.";
    else if (!authority) {
      const owner = status?.controller_authority?.owner_viewer_id || status?.controller?.viewer_id || status?.controller?.id;
      $("deviceCapabilityNote").textContent = `This tab is a read-only viewer${owner ? `; controller viewer: ${owner}` : ""}. Only the authoritative controller can issue device commands.`;
    }
    else $("deviceCapabilityNote").textContent = `Controller connected. One-shot ${can("one_shot") ? "available" : "unsupported"}; processing loop ${can("start_loop") || can("stop_loop") ? "available" : "unsupported"}; camera sleep/wake ${can("sleep_camera") && can("wake_camera") ? "available" : "unsupported"}. Unsupported actions are disabled.`;
  }
  function modeFrom(status) {
    const locationName = status?.connection?.execution_location || status?.execution_location || status?.mode;
    if (state.fixture || locationName === "simulation" || locationName === "test_fixture") { setModeBadge("TEST FIXTURE", "fixture"); return; }
    if (locationName === "on_nicla" || status?.connected === true || status?.connection?.state === "connected") { setModeBadge("ON NICLA", "nicla"); return; }
    if (locationName === "host_reference") { setModeBadge("HOST REFERENCE", "host"); return; }
    if (locationName === "recorded_replay" || locationName === "replay") { setModeBadge("RECORDED REPLAY", "replay"); return; }
    setModeBadge("OFFLINE", "offline");
  }
  function appliedModelChoice(status) {
    const applied = status?.applied_state || {};
    if (Object.prototype.hasOwnProperty.call(applied, "model_override")) return applied.model_override || "policy";
    return applied.model || status?.model;
  }
  function modelOverrideHelp(status) {
    const applied = status?.applied_state || {};
    if (!Object.prototype.hasOwnProperty.call(applied, "model_override") || applied.model_override == null) {
      const policy = applied.policy || status?.policy || "?";
      return policy === "S" ? "The policy selects the small model. Choose a named model to override the policy." : ["H", "G", "L"].includes(policy) ? "The policy selects the large model. Choose a named model to override the policy." : "The active policy selects the identity model; choose a named model to override it.";
    }
    return `Explicit model override: ${applied.model_override}.`;
  }
  async function updateLatestFrame(frame, observedAt = performance.now()) {
    if (state.paused || !frame || frame.frame_id == null) return;
    const boot = frame.boot_id || state.status?.boot_id || "?";
    const prior = state.latestReportedFrame;
    const id = Number(frame.frame_id);
    if (prior && prior.boot_id === boot && Number(prior.frame_id) > id) return;
    if (prior && prior.boot_id !== boot) {
      revokeImage("latest");
      $("latestImage").hidden = true;
      $("latestImage").removeAttribute("src");
      $("latestEmpty").hidden = false;
      $("latestOverlay").replaceChildren();
      $("latestMeta").textContent = "Frame identity and age not available";
      $("latestMeta").title = "";
      state.latestFrame = null;
      state.latestReceivedAt = null;
      state.imageGeometry.latest = null;
    }
    const candidate = { ...frame, boot_id: boot, frame_id: id };
    state.latestReportedFrame = candidate;
    if (state.latestFrame?.boot_id === boot && state.latestFrame.frame_id === id) {
      state.latestFrame = candidate;
      state.latestReceivedAt = observedAt;
    } else if (frame.payload_ready !== false) {
      await loadFrameImage("latest", candidate, candidate, observedAt);
    }
    updateAges();
  }
  async function updateLivePreview(status, events, observedAt) {
    const frame = status.latest_frame || status.preview_frame || status.frame;
    const boot = status.boot_id || frame?.boot_id;
    const packet = events.findLast((event) => event.type === "frame_payload" &&
      event.kind === "frame_jpeg" && event.boot_id === boot && Number.isSafeInteger(event.frame_id));
    const statusFrameId = frame?.frame_id ?? -1;
    if (packet && (packet.frame_id > statusFrameId ||
        (packet.frame_id === statusFrameId && frame.payload_ready === false))) {
      const receivedMs = Date.parse(packet.host_received_utc);
      const age = Number.isFinite(receivedMs) ? Math.max(0, Date.now() - receivedMs) : 0;
      await updateLatestFrame({ ...packet.metadata, boot_id: packet.boot_id,
        frame_id: packet.frame_id, age_ms: age, payload_ready: true }, performance.now());
    } else if (frame?.frame_id != null) {
      await updateLatestFrame(frame, observedAt);
    }
  }
  function updateCommand(commandId, status, message) {
    const pending = state.pending.get(commandId);
    if (!pending) return;
    const terminal = COMMAND_TERMINAL.has(status);
    const succeeded = status === "completed" || status === "done";
    if (terminal) {
      clearTimeout(pending.timeout);
      state.pending.delete(commandId);
      pending.resolve({ status, command_id: commandId, message });
      if (state.fixture) updateFixtureLock();
      else if (state.status) syncControls(state.status, state.capability);
    }
    state.ackText = `${commandId.slice(0, 8)} · ${status}`;
    $("commandAck").textContent = state.ackText;
    setAction(`${pending.name.replaceAll("_", " ")} ${status}${terminal ? "" : " · awaiting device confirmation"}${message ? ` — ${message}` : ""}`, terminal && !succeeded);
    updateCancelControl();
  }
  function ackEvent(event) {
    // A completed observation carries the originating command_id for traceability.
    // Only protocol ACK records may be consumed by this handler; otherwise a
    // successful one-shot is silently dropped after its HTTP command response.
    if (event?.type !== "command_ack" && event?.event !== "control_ack") return false;
    if (!event?.command_id) return false;
    const status = event.control_status || event.ack?.status || event.command_status || event.status || "completed";
    updateCommand(event.command_id, status, event.message || event.reason_code);
    return true;
  }
  function updateCancelControl() {
    const pending = [...state.pending.entries()].find(([, value]) => ["one_shot", "start_loop", "set_config", "set_model", "set_policy"].includes(value.name));
    const supported = normalizeCapabilityList(state.capability)("cancel");
    $("deviceCancel").disabled = !pending || !state.controllerAuthority || !supported || videoLocked();
    $("deviceCancel").title = !supported ? "Firmware does not advertise safe cancellation." : !pending ? "No command is waiting for completion." : `Cancel pending ${pending[1].name} at the supported safe boundary.`;
  }
  async function ingestEvent(event) {
    if (!event || typeof event !== "object") { state.packetRejectCount++; renderErrorCounts(); return; }
    rememberCaptureStart(event);
    if (ackEvent(event)) return;
    if (event.event === "identity_model_output" && event.boot_id && Number.isSafeInteger(Number(event.frame_id))) {
      const vector = event.raw_output;
      if (Array.isArray(vector) && vector.length > 0 && vector.length <= 512 && vector.every((value) => Number.isFinite(Number(value)))) {
        const key = eventKey(event);
        const output = { raw_output: vector.map(Number), output_dtype: event.output_dtype || null,
          output_scale: event.output_scale ?? null, output_zero_point: event.output_zero_point ?? null,
          output_representation: event.output_representation || null,
          model_id: event.model_id || null, model_sha256: event.model_sha256 || null };
        state.identityOutputs.set(key, output);
        while (state.identityOutputs.size > MAX_EVENTS) state.identityOutputs.delete(state.identityOutputs.keys().next().value);
        const index = state.events.findIndex((row) => eventKey(row) === key);
        if (index >= 0) {
          const observation = state.events[index];
          const enriched = { ...observation, identity: { ...(observation.identity || {}),
            raw_output: output.raw_output, output_dtype: output.output_dtype,
            output_scale: output.output_scale, output_zero_point: output.output_zero_point,
            output_representation: output.output_representation,
            raw_output_model_id: output.model_id, raw_output_model_sha256: output.model_sha256 } };
          state.events[index] = enriched;
          if (!state.paused && state.selectedEvent && eventKey(state.selectedEvent) === key) {
            state.selectedEvent = enriched;
            await renderEvent(enriched, { select: false });
          }
        }
      }
      return;
    }
    // The broker event stream includes transport and progress records that
    // intentionally have no observation frame key. Only completed inference
    // observations belong in the result timeline or frame-linkage validator.
    if (event.type !== "observation_complete" && event.schema_id !== "nicla-observation-v2") {
      if (event.type === "connection_warning") setConnection(`Connection warning · ${event.reason_code || "device hello/frame timeout"}`, true);
      else if (event.type === "connection_error") setConnection(`Connection error · ${event.error || "serial read failed"}`, true);
      return;
    }
    const bad = validateEvent(event);
    if (bad) {
      state.packetRejectCount++;
      renderErrorCounts();
      setConnection(`Invalid device event: ${bad}`, true);
      return;
    }
    if ($("deviceConnection").textContent.startsWith("Invalid device event:")) {
      const device = state.status?.device?.name || state.status?.device || "Device broker";
      setConnection(state.fixture ? "Synthetic event simulator · no hardware commands" : String(device));
    }
    const key = eventKey(event);
    if (state.seen.has(key)) return;
    event = attachFrameIdentityOutput(event);
    state.seen.add(key);
    state.events.push(event);
    state.events.sort(compareObservationOrder);
    state.events = state.events.slice(-MAX_EVENTS);
    if (state.recording && !state.recordingEvents.some((row) => eventKey(row) === key)) {
      state.recordingEvents.push(event); state.recordingEvents = state.recordingEvents.slice(-MAX_RECORDING);
    }
    const newest = state.events[state.events.length - 1];
    if (!state.paused && !state.resultFrozen && eventKey(state.selectedEvent) !== eventKey(newest)) {
      await renderEvent(newest);
    }
    renderTimeline();
  }
  async function pollDevice() {
    if (state.polling || state.fixture) return;
    state.polling = true;
    const receivedAt = performance.now();
    try {
      const [status] = await Promise.all([getJson("/api/v2/device/status"), pollVideoStatus()]);
      state.status = status;
      const lease = status.controller_authority || {};
      if (lease.owned_by_viewer === true && lease.expired !== true && lease.controller_id) {
        state.controllerId = lease.controller_id;
        state.controllerAuthority = true;
        if (!state.heartbeatTimer) startControllerHeartbeat();
      }
      state.lastPollOk = true;
      state.errorCount = Number(status.error_count || 0);
      state.droppedCount = Number(status.dropped_frames || 0);
      renderErrorCounts();
      modeFrom(status);
      const connected = status.connected === true || status.connection?.state === "connected" || status.state === "connected";
      const deviceName = typeof status.device === "string" ? status.device : (status.device?.product || status.device?.name || status.serial_descriptor?.product || "Nicla Vision");
      const firmwareVersion = status.firmware?.version || (typeof status.firmware === "string" ? status.firmware : null);
      setConnection(connected ? "USB connected" : `Device ${status.connection?.state || status.state || "disconnected"}`);
      $("deviceConnection").title = connected ? [deviceName, firmwareVersion].filter(Boolean).join(" · ") : "";
      let caps = status.capabilities;
      if (!caps || !Array.isArray(caps.models) || caps.models.length === 0 || !Array.isArray(caps.commands) || caps.commands.length === 0) {
        const discovered = await getJson("/api/v2/device/capabilities");
        caps = { ...discovered, ...(caps || {}) };
        if (!Array.isArray(status.capabilities?.models) || status.capabilities.models.length === 0) caps.models = discovered.models || [];
        if (!Array.isArray(status.capabilities?.policies) || status.capabilities.policies.length === 0) caps.policies = discovered.policies || [];
        if (!Array.isArray(status.capabilities?.commands) || status.capabilities.commands.length === 0) caps.commands = discovered.commands || discovered.supported_commands || [];
      }
      state.capability = caps;
      const appliedModel = appliedModelChoice(status) || state.model;
      const appliedPolicy = status.applied_state?.policy || status.policy || state.policy;
      if (state.modelDraft === appliedModel) state.modelDraft = null;
      if (state.policyDraft === appliedPolicy) state.policyDraft = null;
      populateModelChoices($("deviceModel"), caps.models || caps.model_choices, state.modelDraft ?? appliedModel, status);
      populateChoices($("devicePolicy"), caps.policies || caps.policy_choices, state.policyDraft ?? appliedPolicy, "No policy choices reported");
      const appliedStage = status.applied_state?.measurement_stage_id || "full_pipeline";
      if (state.measurementStageDraft === appliedStage) state.measurementStageDraft = null;
      populateMeasurementStages($("deviceMeasurementStage"), caps.measurement_stage_ids, state.measurementStageDraft ?? appliedStage);
      const maxTrialFrames = Number(caps.max_trial_frames) || 50;
      const maxTrialDuration = Number(caps.max_trial_duration_ms) || 30000;
      $("deviceTrialFrames").max = String(maxTrialFrames);
      $("deviceTrialDuration").max = String(maxTrialDuration);
      $("deviceModel").title = modelOverrideHelp(status);
      syncControls(status, caps);
      if (Number.isSafeInteger(status.event_seq) && status.event_seq < state.nextSeq) state.nextSeq = 0;
      const after = state.nextSeq;
      const eventResult = await getJson(`/api/v2/device/events?after=${encodeURIComponent(after)}`);
      const events = Array.isArray(eventResult) ? eventResult : (eventResult.events || eventResult.items || []);
      // The preview follows JPEGs even when the device cannot serialize an
      // inference result. Matched inference still requires an observation.
      await updateLivePreview(status, events, receivedAt);
      // Rehydrate the last completed result before processing the event backlog.
      if (status.last_observation && typeof status.last_observation === "object") {
        await ingestEvent(status.last_observation);
      }
      if (Number(eventResult.skipped_count) > 0) {
        state.skippedEventCount += Number(eventResult.skipped_count);
        renderTimeline();
      }
      for (const event of events) {
        const seq = Number(event.event_seq ?? event.seq);
        if (Number.isSafeInteger(seq)) state.nextSeq = Math.max(state.nextSeq, seq);
        await ingestEvent(event);
      }
      const latestSeq = Number(eventResult.next_seq ?? eventResult.next_after);
      if (Number.isSafeInteger(latestSeq)) state.nextSeq = Math.max(state.nextSeq, latestSeq);
      $("commandAck").textContent = state.ackText;
      state.lastPollOk = true;
    } catch (error) {
      state.lastPollOk = false;
      state.errorCount++;
      state.status = null;
      syncControls(null, state.capability);
      setModeBadge("OFFLINE", "offline");
      renderRuntimeState(null);
      setConnection(`Device broker unavailable · ${error.message}`, true);
      $("deviceCapabilityNote").textContent = "No live state is inferred from the last cached event. Refresh to reconnect; last frame/result ages continue advancing.";
      $("latestState").textContent = "STALE / OFFLINE";
      $("latestState").className = "small-status stale";
      renderErrorCounts();
      if (state.selectedEvent && state.selectedEvent.decision?.status !== "not_run") {
        const prior = String(state.selectedEvent.decision?.status || "RESULT").replaceAll("_", " ");
        if (!$("resultStatus").textContent.startsWith("STALE")) $("resultStatus").textContent = `STALE · ${prior}`;
        $("resultBadge").textContent = "LAST RESULT · OFFLINE";
        $("resultBadge").className = "badge offline";
      }
    } finally { state.polling = false; renderPerformance(); updateAges(); }
  }

  function updateFixtureLock(allowTakeover = false) {
    let lease = null;
    try { lease = JSON.parse(localStorage.getItem(LOCK_KEY) || "null"); } catch (_) { lease = null; }
    const now = Date.now();
    const expired = !lease || Number(lease.expires) < now;
    if (!lease || lease.client === tabId || expired && allowTakeover) {
      lease = { client: tabId, expires: now + 15000 };
      localStorage.setItem(LOCK_KEY, JSON.stringify(lease));
      state.controllerAuthority = true;
    } else state.controllerAuthority = false;
    const caps = { controller_authority: state.controllerAuthority, commands: ["connect", "one_shot", "start", "stop", "set_model", "set_policy", "sleep", "wake", "cancel"], models: ["mobilefacenet_experimental"], policies: ["fixed_high", "fixed_low", "face_triggered", "adaptive"] };
    const status = { connected: true, execution_location: "simulation", controller_authority: state.controllerAuthority, controller_id: state.controllerAuthority ? `fixture-controller-${tabId.slice(0,8)}` : null, applied_state: { processing: state.loopRunning, camera_sleeping: state.sleeping, model: state.model || "mobilefacenet_experimental", policy: state.policy || "fixed_high" }, device: "TEST FIXTURE", firmware: "fixture-v1" };
    state.capability = caps; state.status = status; syncControls(status, caps);
    if (!state.controllerAuthority) $("deviceCapabilityNote").textContent = "TEST FIXTURE is read-only in this tab because another fixture tab holds controller authority.";
  }
  function nextFixtureSeed() {
    let seed;
    do {
      const words = new Uint32Array(1);
      if (globalThis.crypto?.getRandomValues) crypto.getRandomValues(words);
      else words[0] = Math.floor(Math.random() * 0x100000000);
      seed = words[0] >>> 0;
    } while (state.fixtureSeeds.has(seed));
    state.fixtureSeeds.add(seed);
    return seed;
  }
  function seededFixtureTensor(seed, length) {
    let value = seed >>> 0 || 0x6d2b79f5;
    return Array.from({ length }, () => {
      value ^= value << 13; value ^= value >>> 17; value ^= value << 5;
      return (value & 0x7f) - 64;
    });
  }
  function fixtureBaseEvent(scenario, frameId, seed = nextFixtureSeed()) {
    const boot = `fixture-${tabId.slice(0, 8)}`;
    const legacySkippedModel = scenario === "legacy_skipped_model";
    const noFace = scenario === "no_face" || legacySkippedModel;
    const status = noFace ? "NO_FACE" : scenario === "multiple" ? "MULTIPLE" : scenario === "error" ? "ERROR" : "UNCERTAIN";
    const identitySkipped = noFace || scenario === "multiple";
    const identityFailed = scenario === "error";
    const regions = noFace ? [] : scenario === "multiple" ? [[34, 38, 60, 61], [143, 41, 56, 59]] : [[61, 49, 92, 94]];
    const score = scenario === "rejected" ? 0.69 : 0.74;
    const fixtureAgeMs = scenario === "stale" ? 12000 : 0;
    const fixtureReceivedAt = Date.now() - fixtureAgeMs;
    const frame = { boot_id: boot, frame_id: frameId, width: 240, height: 240, orientation: "landscape", source: "fixture", encoding: "JPEG fixture", age_ms: fixtureAgeMs, stale: fixtureAgeMs > STALE_AFTER_MS, frame_ref: `/api/v1/replay/frame?case=${noFace ? "BLANK" : scenario === "multiple" ? "OTHER" : "K01"}` };
    return {
      schema_id: "nicla-observation-v2", source: "test_fixture", test_fixture: { scenario, randomized_property_seed: "0x" + seed.toString(16).padStart(8, "0"), no_device_data: true }, session_id: `fixture-session-${tabId.slice(0, 8)}`, boot_id: boot, frame_id: frameId,
      event_seq: frameId, host_received_monotonic_ns: Math.round(performance.now() * 1e6), host_received_utc: new Date(fixtureReceivedAt).toISOString(), host_received_unix_ms: fixtureReceivedAt, frame,
      applied_state: { model: state.model || "mobilefacenet_experimental", policy: state.policy || "fixed_high", processing: state.loopRunning, camera_sleeping: state.sleeping },
      camera: { width: 240, height: 240, orientation: "landscape", sleeping: state.sleeping },
      detection: { frame_id: frameId, coordinate_space: "sensor_px", status: noFace ? "no_face" : scenario === "multiple" ? "multiple" : "face", regions_xywh: regions, backend: "synthetic fixture" },
      crop: identitySkipped ? null : { frame_id: frameId, rect_px: [61, 49, 92, 94], coordinate_space: "sensor_px", quality: { accepted: true, sharpness: 0.81 } },
      identity_input: identitySkipped ? null : { frame_id: frameId, source: "synthetic_fixture_tensor", shape: [96, 96, 1], channels: "gray", dtype: "int8", scale: 0.03125, zero_point: -1, sha256: null, values: seededFixtureTensor(seed, 96 * 96) },
      identity: { execution: identitySkipped ? "skipped" : "simulated", model_id: legacySkippedModel ? "fixture_mobilefacenet" : identitySkipped ? null : "fixture_mobilefacenet", model_sha256: null, status: identitySkipped ? "not_run" : identityFailed ? "error" : "simulated", error: identityFailed ? "fixture_injected_error" : null },
      decision: { frame_id: frameId, status, candidate: noFace || scenario === "multiple" || scenario === "error" ? null : "K01", scores: noFace || scenario === "multiple" || scenario === "error" ? {} : { K01: score, OTHER: 0.32 }, score_semantics: "cosine_similarity", score: noFace || scenario === "multiple" || scenario === "error" ? null : score, threshold: 0.82, margin: 0.42, reason: scenario === "rejected" ? "Candidate K01 did not meet the frozen 0.82 acceptance threshold; no known identity accepted." : noFace ? "Detector reported no face; identity model was skipped." : scenario === "multiple" ? "Two regions were present; identity was skipped as ambiguous." : scenario === "error" ? "Injected fixture inference error." : "Synthetic K01 candidate is below the acceptance threshold; this fixture never asserts a real identification.", experimental: true },
      trace: (noFace ? ["Fixture camera generated frame.", "Detector returned no face.", "Identity stage skipped by no-face gate.", ...(legacySkippedModel ? ["Legacy-shaped event retained a reported model ID despite identity status not_run."] : [])] : scenario === "multiple" ? ["Fixture camera generated frame.", "Detector returned two regions.", "Identity stage skipped by ambiguity gate."] : scenario === "error" ? ["Fixture camera generated frame.", "Fault injection: identity stage error.", "Decision retained as ERROR."] : ["Fixture frame acquired.", "Detector supplied one face region.", "Identity input generated in fixture only.", "Candidate K01 rejected below threshold; no actual identity claim."]).concat(["Randomized property seed 0x" + seed.toString(16).padStart(8, "0") + " · generated synthetic tensor only."]),
      timings_ms: { capture: 5.2, preprocess: 1.8, detection: 14.6, identity: identitySkipped ? null : 23.1, decision: 0.7, total: scenario === "slow" ? 6000 : 46.3 },
      telemetry: { heap_bytes: 72000, arena_bytes: 49000, framebuffer_bytes: 115200, capture_hz: 2, inference_hz: 0.8, power_w: null, dropped_frames: 0, errors: scenario === "error" ? ["fixture_injected_error"] : [], imu: null },
      connection: { state: "connected", execution_location: "simulation", device: "TEST FIXTURE", firmware: "fixture-v1" },
      model_sha256: "fixture-not-hardware", config_sha256: "fixture-config", firmware_sha256: "fixture-firmware", qualification: { statistically_qualified: false },
    };
  }
  function addFixtureEvent(event) {
    if (state.fixtureScenario === "malformed") event = { ...event, decision: { ...event.decision, frame_id: event.frame_id + 1 } };
    if (state.fixtureScenario === "malformed") { ingestEvent(event); return; }
    renderLatestFixture(event.frame);
    ingestEvent(event);
  }
  async function renderLatestFixture(frame) {
    state.fixtureFrame++;
    await updateLatestFrame({ ...frame, frame_id: frame.frame_id }, performance.now());
  }
  function startFixture() {
    $("fixtureScenarioLabel").hidden = false; $("fixtureScenario").hidden = false;
    setModeBadge("TEST FIXTURE", "fixture");
    setConnection("Synthetic event simulator · no hardware commands");
    $("deviceCapabilityNote").textContent = "TEST FIXTURE mode is synthetic and never sends commands to the Nicla. Choose a scenario, then operate the simulated controls.";
    state.fixtureScenario = $("fixtureScenario").value;
    updateFixtureLock();
    const initial = { boot_id: `fixture-${tabId.slice(0, 8)}`, frame_id: 0, width: 240, height: 240, source: "fixture", encoding: "JPEG fixture", age_ms: 0, frame_ref: "/api/v1/replay/frame?case=K01" };
    updateLatestFrame(initial);
    $("deviceFreshness").textContent = "TEST FIXTURE · no live hardware";
    window.addEventListener("storage", (event) => { if (event.key === LOCK_KEY) updateFixtureLock(); });
    state.controllerLeaseTimer = setInterval(() => { updateFixtureLock(); }, 1500);
  }
  async function fixtureCommand(cmd, params = {}, commandId = null) {
    updateFixtureLock();
    if (!state.controllerAuthority) throw new Error("This fixture tab is a read-only viewer; another fixture tab has control.");
    if (cmd === "connect") return { status: "completed", message: "Synthetic fixture broker connected." };
    if (cmd === "one_shot") {
      const scenario = state.fixtureScenario;
      const id = ++state.fixtureFrame;
      setAction("TEST FIXTURE · one-shot accepted; waiting for simulated completion…");
      const finish = async () => {
        state.fixturePendingTimers.delete(commandId);
        if (state.cancelled.has(commandId)) {
          state.cancelled.delete(commandId);
          ackEvent({ type: "command_ack", command_id: commandId, control_status: "completed", message: "Cancelled at the synthetic safe boundary; no result was attached." });
          return;
        }
        if (scenario === "timeout") {
          ackEvent({ type: "command_ack", command_id: commandId, control_status: "timeout", message: "Injected 2200 ms device-command timeout; no late result was attached." });
          return;
        }
        const seed = nextFixtureSeed();
        const event = fixtureBaseEvent(scenario, id, seed);
        if (params.identity_probe_on_no_face === true && scenario === "no_face") {
          event.crop = { frame_id: id, rect_px: [72, 72, 96, 96], coordinate_space: "sensor_px", status: "valid_diagnostic_only", semantics: "synthetic fixture center crop; never eligible for identity acceptance" };
          event.identity_input = { frame_id: id, source: "synthetic_fixture_center_crop_tensor", shape: [1, 96, 96, 1], channels: "gray", dtype: "int8", scale: 0.00392157, zero_point: -128, sha256: null, values: seededFixtureTensor(seed, 96 * 96) };
          event.identity = { execution: "simulated", model_id: params.probe_model_slot === "large" ? "fixture_classifier" : "fixture_mobilefacenet", model_sha256: null, status: "simulated", diagnostic_result: { status: "INFERENCE_ONLY", reason: "fixture detector returned no face; synthetic center-crop inference; acceptance disabled", identity_acceptance_enabled: false } };
          event.timings_ms.identity = 23.1;
          event.trace = ["TEST FIXTURE generated a synthetic frame.", "Synthetic detector returned no face.", "Synthetic center-crop identity probe was simulated; it has no hardware input or model result.", "Final decision remains NO_FACE; identity acceptance is disabled.", "Randomized property seed 0x" + seed.toString(16).padStart(8, "0") + "."];
        }
        if (scenario === "malformed") {
          await ingestEvent({ ...event, decision: { ...event.decision, frame_id: id + 1 } });
          ackEvent({ type: "command_ack", command_id: commandId, control_status: "error", message: "Malformed frame linkage rejected by UI validator." });
          return;
        }
        await updateLatestFrame(event.frame, performance.now());
        await ingestEvent(event);
        ackEvent({ type: "command_ack", command_id: commandId, control_status: "completed", completed_frame_id: id, message: "Fixture " + scenario + " event " + eventKey(event) + " shown; randomized property seed 0x" + seed.toString(16).padStart(8, "0") + "." });
      };
      const delay = scenario === "timeout" ? REQUEST_TIMEOUT_MS : scenario === "slow" ? 6000 : 100;
      state.fixturePendingTimers.set(commandId, setTimeout(finish, delay));
      return { status: "accepted", command_id: commandId, message: "Synthetic one-shot queued." };
    }
    if (cmd === "start_loop") {
      state.loopRunning = true;
      clearInterval(state.fixtureLoop);
      state.fixtureLoop = setInterval(async () => {
        if (!state.sleeping) {
          const id = ++state.fixtureFrame;
          const event = fixtureBaseEvent(state.fixtureScenario, id);
          await updateLatestFrame(event.frame, performance.now());
          await ingestEvent(event);
        }
      }, 1600);
      return { status: "completed", message: "Synthetic processing loop started." };
    }
    if (cmd === "stop_loop") { clearInterval(state.fixtureLoop); state.fixtureLoop = null; state.loopRunning = false; return { status: "completed", message: "Synthetic processing loop stopped." }; }
    if (cmd === "sleep_camera") { state.sleeping = true; return { status: "completed", message: "Synthetic camera state: sleeping." }; }
    if (cmd === "wake_camera") { state.sleeping = false; return { status: "completed", message: "Synthetic camera state: awake; settling is simulated." }; }
    if (cmd === "cancel") {
      const target = params.target_command_id;
      const timer = target && state.fixturePendingTimers.get(target);
      if (timer) {
        clearTimeout(timer); state.fixturePendingTimers.delete(target); state.cancelled.add(target);
        setTimeout(() => {
          state.cancelled.delete(target);
          ackEvent({ type: "command_ack", command_id: target, control_status: "completed", message: "Cancelled at the synthetic safe boundary; no result was attached." });
        }, 25);
      }
      return { status: "completed", message: "Cancellation requested for " + (target || "pending fixture operation") + "." };
    }
    if (cmd === "set_model") { state.model = params.model_slot || params.model || state.model; return { status: "completed", message: `Synthetic model applied: ${state.model}.` }; }
    if (cmd === "set_policy") { state.policy = params.policy_id || params.policy || state.policy; return { status: "completed", message: `Synthetic policy applied: ${state.policy}.` }; }
    if (cmd === "set_config") { state.model = params.model || state.model; state.policy = params.policy || state.policy; return { status: "completed", message: `Synthetic settings applied: ${state.model} / ${state.policy}.` }; }
    return { status: "rejected", message: `Fixture does not support ${cmd}.` };
  }
  async function sendCommand(cmd, params = {}) {
    if (videoLocked()) return { status: "rejected", message: "Video capture is active." };
    const commandId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
    const command = { schema: "nicla-command-v2", controller_id: state.controllerId, viewer_id: tabId, command_id: commandId, cmd: wireCommand[cmd] || cmd, ...wireParams(cmd, params) };
    const completion = new Promise((resolve) => state.pending.set(commandId, { name: cmd, resolve }));
    if (state.fixture) updateFixtureLock();
    else if (state.status?.controller_authority?.owned_by_viewer) syncControls(state.status, state.capability);
    updateCancelControl();
    $("commandAck").textContent = `${commandId.slice(0, 8)} · sending`;
    setAction(`${cmd.replaceAll("_", " ")} sent · waiting for matching device ACK…`);
    try {
      const result = state.fixture ? await fixtureCommand(cmd, params, commandId) : await postJson("/api/v2/device/commands", command);
      if (result.command_id && result.command_id !== commandId) throw new Error("ACK command ID did not match request");
      const status = result.status || result.state || "accepted";
      const pending = state.pending.get(commandId);
      if (pending && !COMMAND_TERMINAL.has(status)) {
        pending.timeout = setTimeout(() => {
          updateCommand(commandId, "timeout", "No matching firmware acknowledgment. Check the applied state before retrying.");
        }, 17000);
      }
      updateCommand(commandId, status, result.message || result.reason_code);
    } catch (error) {
      if (state.pending.has(commandId)) {
        state.errorCount++;
        renderErrorCounts();
        updateCommand(commandId, "error", error.message);
      }
    }
    if (!state.fixture) void pollDevice();
    return completion;
  }
  function setupZoom(slot, delta) {
    const next = Math.max(1, Math.min(4, state.zoom[slot] + delta));
    state.zoom[slot] = next;
    $(`${slot}Viewport`).style.transform = `scale(${next})`;
  }
  function updatePause() {
    state.paused = !state.paused;
    $("devicePause").textContent = state.paused ? "Resume UI" : "Pause UI";
    $("devicePause").setAttribute("aria-pressed", String(state.paused));
    if (state.paused) {
      $("deviceFreshness").textContent = "PAUSED · ages continue to advance";
      $("latestState").textContent = "PAUSED";
      $("latestState").className = "small-status";
      setAction("Live images paused. Device controls and telemetry remain available.");
    } else {
      setAction("Live images resumed.");
      const newest = state.events.at(-1);
      if (!state.resultFrozen && newest) void renderEvent(newest);
      if (!state.fixture) pollDevice();
    }
    renderPerformance();
    updateAges();
  }
  async function applySettings() {
    if (state.applyingSettings || state.pending.size) return;
    const can = normalizeCapabilityList(state.capability);
    const model = $("deviceModel").value || null;
    const policy = $("devicePolicy").value || null;
    const commands = [];
    if (model && model !== state.model) {
      if (!can("set_model")) { setAction("Selected model cannot be changed on this firmware.", true); return; }
      commands.push(["set_model", { model_slot: model }]);
    }
    if (policy && policy !== state.policy) {
      if (!can("set_policy")) { setAction("Selected policy cannot be changed on this firmware.", true); return; }
      commands.push(["set_policy", { policy_id: policy }]);
    }
    if (!commands.length) { setAction("Selections already match the reported applied model and policy."); return; }
    state.applyingSettings = true;
    syncControls(state.status, state.capability);
    try {
      for (const [cmd, params] of commands) {
        const result = await sendCommand(cmd, params);
        if (!["completed", "done"].includes(result.status)) break;
      }
    } finally {
      state.applyingSettings = false;
      syncControls(state.status, state.capability);
    }
  }
  function blobToBase64(blob) {
    return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(",")[1]); reader.onerror = reject; reader.readAsDataURL(blob); });
  }
  async function sha256Hex(bytes) {
    if (!globalThis.crypto?.subtle) throw new Error("Web Crypto SHA-256 is unavailable; payload omitted");
    const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    return [...new Uint8Array(await crypto.subtle.digest("SHA-256", view))].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  async function exportInputPayload(event, kind) {
    if (kind === "identity" && event?.identity?.status === "skipped") return null;
    const descriptor = (kind === "identity"
      ? (event.identity_input || event.model_input)
      : (event.detector_input || event.detector?.input || event.detector?.input_metadata)) || {};
    if (descriptor.payload_status === "not_exported_live") return null;
    if (typeof descriptor !== "object") return null;
    let bytes;
    let provenance = "nicla_packet";
    let shape = descriptor.shape || null;
    let dtype = descriptor.dtype || "int8";
    let scale = descriptor.scale ?? null;
    let zeroPoint = descriptor.zero_point ?? null;
    if (Array.isArray(descriptor.values) || ArrayBuffer.isView(descriptor.values) || Array.isArray(descriptor.bytes) || ArrayBuffer.isView(descriptor.bytes)) {
      const values = descriptor.values || descriptor.bytes;
      const dtype = String(descriptor.dtype || "int8").toLowerCase();
      bytes = dtype === "uint8" ? Uint8Array.from(values) : new Uint8Array(Int8Array.from(values).buffer);
      provenance = state.fixture || descriptor.source === "synthetic_fixture_tensor" ? "synthetic_fixture" : "event_inline_bytes";
    } else {
      const boot = eventBootId(event), frameId = resultFrameId(event);
      const url = descriptor.input_ref || `/api/v2/device/input?boot_id=${encodeURIComponent(boot)}&frame_id=${encodeURIComponent(frameId)}&kind=${encodeURIComponent(kind)}`;
      if (typeof url !== "string" || !url.startsWith("/")) throw new Error("External tensor references are not included in local export");
      const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
      if (!response.ok) throw new Error(`Exact ${kind} input unavailable (HTTP ${response.status})`);
      if (!response.headers.get("Content-Type")?.startsWith("application/octet-stream")) throw new Error(`${kind} input response is not a bounded binary tensor`);
      const responseBoot = response.headers.get("X-Device-Boot-ID");
      const responseFrame = response.headers.get("X-Device-Frame-ID");
      if (event.source !== "test_fixture" && (!responseBoot || !responseFrame)) throw new Error(`${kind} input response lacks matching boot/frame headers`);
      if (responseBoot && responseBoot !== boot || responseFrame && Number(responseFrame) !== Number(frameId)) throw new Error(`${kind} input is linked to a different frame`);
      bytes = new Uint8Array(await response.arrayBuffer());
      const headerHash = response.headers.get("X-Input-SHA256");
      if (event.source !== "test_fixture" && !headerHash) throw new Error(`${kind} input response has no SHA-256 header`);
      const responseKind = response.headers.get("X-Input-Kind");
      if (responseKind && responseKind !== kind) throw new Error(`${kind} input response kind mismatch`);
      const digest = await sha256Hex(bytes);
      if (descriptor.sha256 && digest !== descriptor.sha256 || headerHash && digest !== headerHash) throw new Error(`${kind} input SHA-256 mismatch`);
      try {
        const headerShape = JSON.parse(response.headers.get("X-Input-Shape") || "null");
        if (Array.isArray(descriptor.shape) && Array.isArray(headerShape) && JSON.stringify(descriptor.shape) !== JSON.stringify(headerShape)) throw new Error(`${kind} input shape differs from the observation`);
        shape = Array.isArray(descriptor.shape) ? descriptor.shape : headerShape;
      } catch (error) {
        if (error instanceof SyntaxError) throw new Error(`${kind} input shape header is malformed`);
        throw error;
      }
      dtype = response.headers.get("X-Input-DType") || dtype;
      if (response.headers.get("X-Input-Scale") != null) scale = Number(response.headers.get("X-Input-Scale"));
      if (response.headers.get("X-Input-Zero-Point") != null) zeroPoint = Number(response.headers.get("X-Input-Zero-Point"));
    }
    if (!bytes.length || bytes.length > 262_144) throw new Error(`${kind} input exceeds the 256 KiB per-tensor cap`);
    const digest = await sha256Hex(bytes);
    if (provenance !== "synthetic_fixture" && descriptor.sha256 && digest !== descriptor.sha256) throw new Error(`${kind} input does not match the event SHA-256`);
    return {
      kind: `${kind}_input`, event_key: eventKey(event), boot_id: eventBootId(event), frame_id: resultFrameId(event),
      provenance, sha256: digest, declared_sha256: descriptor.sha256 || null,
      shape, dtype, scale, zero_point: zeroPoint,
      byte_count: bytes.byteLength, base64: base64OfBytes(bytes),
    };
  }
  async function exportBundle() {
    const events = state.recordingEvents.slice(-20);
    if (!events.length) { setAction("Log session events and collect at least one completed event before export.", true); return; }
    $("deviceExport").disabled = true;
    setAction(`Preparing local bundle for ${events.length} bounded events…`);
    const media = [];
    const omitted = [];
    let rawBytes = 0;
    let encodedChars = 0;
    const maxBytes = 10_000_000;
    const eventRows = events.slice();
    let eventsJsonl = eventRows.map((event) => JSON.stringify(event)).join("\n");
    while (new TextEncoder().encode(eventsJsonl).byteLength > 4_000_000 && eventRows.length > 1) {
      eventRows.shift();
      eventsJsonl = eventRows.map((event) => JSON.stringify(event)).join("\n");
    }
    const eventJsonBytes = new TextEncoder().encode(eventsJsonl).byteLength;
    if (eventRows.length < events.length) omitted.push({ kind: "events", count: events.length - eventRows.length, reason: "4 MB JSONL event cap" });
    const reserve = 128_000;
    const fits = (row) => rawBytes + row.byte_count <= maxBytes && eventJsonBytes + encodedChars + row.base64.length + reserve <= maxBytes;
    async function addPayload(row) {
      if (!row) return;
      if (!fits(row)) { omitted.push({ event_key: row.event_key, kind: row.kind, reason: "10 MB bundle cap reached" }); return; }
      rawBytes += row.byte_count;
      encodedChars += row.base64.length;
      media.push(row);
    }
    for (const event of eventRows) {
      const frame = event.frame || { boot_id: event.boot_id, frame_id: event.frame_id };
      const url = rawFrameUrl(frame);
      if (url) {
        try {
          const response = await fetch(url, { cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
          if (response.ok && (!response.headers.get("Content-Type") || response.headers.get("Content-Type").startsWith("image/"))) {
            const responseBoot = response.headers.get("X-Device-Boot-ID");
            const responseFrame = response.headers.get("X-Device-Frame-ID");
            if (event.source !== "test_fixture" && (!responseBoot || !responseFrame)) throw new Error("Matched frame response lacks boot/frame identity headers");
            if (responseBoot && responseBoot !== eventBootId(event) || responseFrame && Number(responseFrame) !== Number(resultFrameId(event))) throw new Error("Matched frame response does not match event boot/frame ID");
            const blob = await response.blob();
            if (!blob.type.startsWith("image/") || blob.size > 1_000_000) throw new Error("Matched frame is not an image under 1 MB");
            const frameBytes = new Uint8Array(await blob.arrayBuffer());
            const digest = await sha256Hex(frameBytes);
            const headerHash = response.headers.get("X-Frame-SHA256");
            const expectedHash = event.frame?.sha256 || event.frame?.jpeg_sha256;
            if (headerHash && digest !== headerHash || expectedHash && digest !== expectedHash) throw new Error("Matched frame SHA-256 mismatch");
            await addPayload({ kind: "frame", event_key: eventKey(event), boot_id: eventBootId(event), frame_id: resultFrameId(event), provenance: event.source === "test_fixture" ? "synthetic_fixture_image" : "broker_matched_frame", mime_type: blob.type, sha256: digest, byte_count: frameBytes.byteLength, base64: await blobToBase64(blob) });
          } else {
            omitted.push({ event_key: eventKey(event), kind: "frame", reason: `HTTP ${response.status}` });
          }
        } catch (error) { omitted.push({ event_key: eventKey(event), kind: "frame", reason: String(error.message || error).slice(0, 160) }); }
      }
      for (const kind of ["detector", "identity"]) {
        try { await addPayload(await exportInputPayload(event, kind)); }
        catch (error) { omitted.push({ event_key: eventKey(event), kind: `${kind}_input`, reason: String(error.message || error).slice(0, 160) }); }
      }
    }
    const eventDigest = await sha256Hex(new TextEncoder().encode(eventsJsonl));
    const modelHashes = [...new Set(eventRows.map((event) => event.identity?.model_sha256 || event.model_sha256).filter(Boolean))];
    const configHashes = [...new Set(eventRows.map((event) => event.config_sha256).filter(Boolean))];
    const firmwareHashes = [...new Set(eventRows.map((event) => event.firmware_sha256 || event.firmware?.sha256).filter(Boolean))];
    const bundle = {
      manifest: { schema: "nicla-local-session-bundle-v2", created_utc: new Date().toISOString(), source: state.fixture ? "test_fixture" : "local_device_broker", event_count: eventRows.length, original_selected_event_count: events.length, event_jsonl_sha256: eventDigest, payload_count: media.length, payload_bytes: rawBytes, estimated_utf8_bytes: eventJsonBytes + encodedChars + reserve, exact_utf8_bytes: null, max_bundle_bytes: maxBytes, model_sha256s: modelHashes, config_sha256s: configHashes, firmware_sha256s: firmwareHashes, payloads_omitted: omitted, models_included: false, private_frames_local_only: true, limitations: state.fixture ? ["Synthetic fixture; no hardware bytes or real identity predictions.", "Synthetic tensor/image payloads are labeled and hashed as fixture bytes."] : ["Exact tensors are included only when the local broker retains matching frame/kind packet bytes.", "Model/config/firmware hashes are references; model files are not bundled.", "This local download is not uploaded or shared."] },
      events_jsonl: eventsJsonl, payloads: media,
    };
    const encoder = new TextEncoder();
    let serialized = "";
    let exactBytes = 0;
    for (let attempt = 0; attempt < 4; attempt++) {
      serialized = JSON.stringify(bundle, null, 2);
      exactBytes = encoder.encode(serialized).byteLength;
      if (bundle.manifest.exact_utf8_bytes === exactBytes) break;
      bundle.manifest.exact_utf8_bytes = exactBytes;
    }
    if (exactBytes > maxBytes) {
      $("deviceExport").disabled = false;
      setAction(`Export stopped: exact bundle size ${exactBytes} B exceeds the ${maxBytes} B cap.`, true);
      return;
    }
    serialized = JSON.stringify(bundle, null, 2);
    exactBytes = encoder.encode(serialized).byteLength;
    if (exactBytes > maxBytes) {
      $("deviceExport").disabled = false;
      setAction(`Export stopped: exact bundle size ${exactBytes} B exceeds the ${maxBytes} B cap.`, true);
      return;
    }
    const blob = new Blob([serialized], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `nicla-session-${new Date().toISOString().replaceAll(":", "-")}.json`; a.click(); URL.revokeObjectURL(a.href);
    setAction(`Downloaded bounded local bundle with ${eventRows.length} events, ${media.length} payloads and ${rawBytes} raw payload bytes.`);
    $("deviceExport").disabled = !state.recordingEvents.length;
  }
  async function claimDeviceController() {
    if (state.controllerAuthority && state.controllerId) return;
    if (state.claimingController) throw new Error("Control is already being requested.");
    state.claimingController = true;
    try {
      const result = await postJson("/api/v2/device/controller/claim", { viewer_id: tabId });
      state.controllerId = result.controller_id || null;
      if (!state.controllerId) throw new Error(result.message || "Controller lease was not granted.");
      state.controllerAuthority = true;
      $("deviceRelease").disabled = false;
      setAction("This tab has control.");
      startControllerHeartbeat();
      await pollDevice();
    } finally {
      state.claimingController = false;
      syncSightingAssignment();
      syncBatteryControls(state.status);
    }
  }
  function setupControls() {
    $("deviceVideo").addEventListener("click", toggleVideo);
    $("sightingsToggle").addEventListener("click", toggleSightings);
    $("sightingsPerson").addEventListener("change", (event) => {
      state.sightingFilter = event.target.value;
      state.sightingLimit = SIGHTING_PAGE_SIZE;
      syncSightingGallery(state.sightings);
    });
    $("sightingsSource").addEventListener("change", (event) => {
      state.sightingSource = event.target.value;
      state.sightingLimit = SIGHTING_PAGE_SIZE;
      syncSightingGallery(state.sightings);
    });
    $("batteryReview").addEventListener("click", () => {
      state.sightingFilter = $("sightingsPerson").value = "all";
      state.sightingSource = $("sightingsSource").value = "onboard";
      state.sightingLimit = SIGHTING_PAGE_SIZE;
      syncSightingGallery(state.sightings);
      $("sightingsSource").focus({ preventScroll: true });
    });
    $("sightingsMore").addEventListener("click", () => {
      state.sightingLimit += SIGHTING_PAGE_SIZE;
      syncSightingGallery(state.sightings);
    });
    $("sightingDialog").addEventListener("close", () => {
      state.openSightingId = null;
      if (state.sightingOpener && !state.sightingOpener.isConnected) $("sightingsPerson").focus();
      state.sightingOpener = null;
    });
    $("sightingAssignment").addEventListener("submit", saveSightingAssignment);
    $("sightingAssignPerson").addEventListener("change", syncSightingAssignment);
    $("sightingClaim").addEventListener("click", async () => {
      if ($("sightingClaim").disabled || !selectedSighting()) return;
      state.sightingClaiming = true;
      const status = $("sightingAssignmentStatus");
      status.textContent = "Requesting control…";
      syncSightingAssignment();
      try {
        await claimDeviceController();
        status.textContent = state.controllerAuthority ? "Choose the person for this encounter." : "Could not claim control.";
      } catch (error) {
        status.textContent = `Could not claim control. ${error.message}`;
      } finally {
        state.sightingClaiming = false;
        syncSightingAssignment();
      }
    });
    $("batteryPrepare").addEventListener("click", () => {
      const prepare = $("batteryPrepare");
      if (prepare.disabled || prepare.hidden) return;
      if (prepare.dataset.action === "stop" && state.loopRunning) $("deviceLoop").click();
      else if (["connect", "control"].includes(prepare.dataset.action)) $("deviceConnect").click();
    });
    for (const [action, id] of Object.entries(BATTERY_BUTTONS)) {
      $(id).addEventListener("click", () => updateBattery(action));
    }
    $("deviceVideoMode").addEventListener("change", () => syncControls(state.status, state.capability));
    $("deviceVideoReview").addEventListener("click", () => {
      $("deviceClipActions").open = false;
      openClip();
    });
    $("deviceClipClose").addEventListener("click", () => closeClip(true));
    $("deviceClipPlay").addEventListener("click", toggleClipPlayback);
    $("deviceClipPosition").addEventListener("input", (event) => {
      pauseClip();
      showClipFrame(Number(event.target.value) - 1);
    });
    $("deviceHostReference").addEventListener("click", async () => {
      const event = state.selectedEvent;
      const key = state.hostReferenceKey;
      if (!key || key !== eventKey(event) || state.hostReferenceBusy) return;
      const output = $("deviceHostReferenceStatus");
      state.hostReferenceBusy = true;
      $("deviceHostReference").disabled = true;
      output.textContent = "Analyzing the linked Nicla frame on the Mac…";
      try {
        const boot = eventBootId(event);
        const frame = Number(resultFrameId(event));
        const result = await postJson("/api/v2/device/host-reference", { boot_id: boot, frame_id: frame }, 60000);
        if (state.hostReferenceKey !== key) return;
        if (result.source?.device_boot_id !== boot || result.source?.device_frame_id !== frame)
          throw new Error("Mac result does not match the selected frame.");
        const faceCount = result.detector?.face_count;
        const recognition = result.recognition || {};
        const candidate = recognition.raw_top1_known_class;
        const gatesPass = recognition.known_cosine_gate?.accepted && recognition.known_vs_other_margin_gate?.accepted;
        const face = asRect(result.detector?.bbox_xywh_240_px);
        state.hostReference = { key, face };
        if (state.imageGeometry.matched) renderOverlay("matched", event, state.imageGeometry.matched);
        const crop = asRect(event.crop?.rect_px);
        let caption = "Mac aligned crop · blue box shows the Mac detection";
        if (face && face.width > 0 && face.height > 0 && crop) {
          const width = Math.max(0, Math.min(crop.x + crop.width, face.x + face.width) - Math.max(crop.x, face.x));
          const height = Math.max(0, Math.min(crop.y + crop.height, face.y + face.height) - Math.max(crop.y, face.y));
          caption += ` · ${Math.round(100 * width * height / (face.width * face.height))}% of that box is inside the yellow Nicla crop`;
        }
        const png = recognition.aligned_crop?.bytes_base64;
        if (typeof png === "string" && png.length <= 180000 && /^iVBORw0KGgo[A-Za-z0-9+/=]+$/.test(png)) {
          $("deviceHostCrop").src = `data:image/png;base64,${png}`;
          $("deviceHostCropCaption").textContent = caption;
          $("deviceHostCropFigure").hidden = false;
        }
        output.textContent = candidate
          ? `Mac reference · frame ${frame} · ${identityName(candidate, "lfw")} · ${gatesPass ? "both match gates pass" : "match gates did not both pass"}`
          : `Mac reference · frame ${frame} · ${faceCount ?? "unknown number of"} face(s) · no identity candidate`;
      } catch (error) {
        if (state.hostReferenceKey === key) output.textContent = `Mac reference unavailable: ${error.message}`;
      } finally {
        state.hostReferenceBusy = false;
        $("deviceHostReference").disabled = !state.hostReferenceKey;
      }
    });
    $("deviceConnect").addEventListener("click", async () => {
      if (state.fixture) {
        updateFixtureLock(true);
        setAction(state.controllerAuthority ? "TEST FIXTURE control lease held by this tab; no device was contacted." : "TEST FIXTURE remains read-only because the controller lease has not expired.");
        return;
      }
      if (!state.controllerAuthority || !state.controllerId) {
        try {
          await claimDeviceController();
        } catch (error) { setConnection(`Read-only · ${error.message}`, true); setAction(`Could not claim device control; no serial connection was opened: ${error.message}`, true); return; }
      }
      const currentlyConnected = state.status?.connected === true || state.status?.connection?.state === "connected" || state.status?.state === "connected";
      if (!currentlyConnected) {
        try {
          const connected = await postJson("/api/v2/device/connect", { schema: "nicla-connect-v2", viewer_id: tabId, controller_id: state.controllerId }, 12000);
          if (!connected.connected) throw new Error(connected.connection_warning || connected.connection_error || "Nicla did not complete its boot handshake.");
          setAction(`Device broker connected: ${connected.device?.name || connected.device || "Nicla Vision"}. Discovering capabilities…`);
          await sendCommand("config");
          await pollDevice();
        } catch (error) { setConnection(`Connection failed · ${error.message}`, true); setAction(`Device connection failed: ${error.message}`, true); return; }
      }
      else {
        // Refresh the firmware-reported model/policy/command contract as well
        // as cached broker status; a serial reconnect must not inherit guessed
        // browser defaults or leave supported controls blank.
        await sendCommand("config");
        await pollDevice();
      }
    });
    $("deviceDisconnect").addEventListener("click", async () => {
      try {
        const result = await postJson("/api/v2/device/disconnect", { schema: "nicla-disconnect-v2", viewer_id: tabId, controller_id: state.controllerId });
        setAction(`${result.message || "Local broker disconnected."} Device processing is not stopped.`);
        await pollDevice();
      } catch (error) { setAction(`Broker disconnect failed: ${error.message}`, true); }
    });
    $("deviceRelease").addEventListener("click", releaseController);
    $("deviceModel").addEventListener("change", () => { state.modelDraft = $("deviceModel").value; });
    $("devicePolicy").addEventListener("change", () => { state.policyDraft = $("devicePolicy").value; });
    $("deviceMeasurementStage").addEventListener("change", () => { state.measurementStageDraft = $("deviceMeasurementStage").value; });
    $("deviceOneShot").addEventListener("click", () => sendCommand("one_shot"));
    $("deviceProbeNoFace").addEventListener("click", () => {
      const modelChoice = $("deviceModel").value || state.model;
      const policyChoice = $("devicePolicy").value || state.policy;
      const slot = modelChoice === "large" ? "large" : modelChoice === "small" ? "small" :
        ["H", "G", "L"].includes(policyChoice) ? "large" : "small";
      sendCommand("one_shot", { identity_probe_on_no_face: true, probe_model_slot: slot });
    });
    $("deviceCancel").addEventListener("click", () => {
      const pending = [...state.pending.entries()].find(([, value]) => ["one_shot", "start_loop", "set_config", "set_model", "set_policy"].includes(value.name));
      if (pending) sendCommand("cancel", { target_command_id: pending[0] });
    });
    $("deviceLoop").addEventListener("click", () => sendCommand(state.loopRunning ? "stop_loop" : "start_loop"));
    $("deviceSleep").addEventListener("click", () => sendCommand(state.sleeping ? "wake_camera" : "sleep_camera"));
    $("deviceApplySettings").addEventListener("click", applySettings);
    $("deviceTrial").addEventListener("click", () => {
      const frames = Number($("deviceTrialFrames").value);
      const duration_ms = Number($("deviceTrialDuration").value);
      const maxFrames = Number($("deviceTrialFrames").max) || 50;
      const maxDuration = Number($("deviceTrialDuration").max) || 30000;
      if (!Number.isInteger(frames) || frames < 1 || frames > maxFrames ||
          !Number.isInteger(duration_ms) || duration_ms < 1 || duration_ms > maxDuration) {
        setAction(`Trial bounds must be whole numbers: 1–${maxFrames} frames and 1–${maxDuration} ms.`, true);
        return;
      }
      const modelChoice = $("deviceModel").value || state.model;
      const policyChoice = $("devicePolicy").value || state.policy;
      const model_slot = modelChoice === "large" || modelChoice === "small" ? modelChoice :
        ["H", "G", "L"].includes(policyChoice) ? "large" : "small";
      const trial_id = $("deviceTrialId").value.trim() || `trial-${Date.now().toString(36)}`;
      sendCommand("trial", { frames, duration_ms, trial_id,
        measurement_stage_id: $("deviceMeasurementStage").value || "full_pipeline",
        model_slot, policy_id: policyChoice || "S" });
    });
    $("devicePause").addEventListener("click", updatePause);
    $("deviceRecord").addEventListener("click", () => {
      state.recording = !state.recording;
      $("deviceRecord").textContent = state.recording ? "Stop logging events" : "Log session events";
      $("deviceRecord").setAttribute("aria-pressed", String(state.recording));
      if (!state.recording) state.recordingEvents = state.recordingEvents.slice(-MAX_RECORDING);
      $("deviceExport").disabled = state.recordingEvents.length === 0;
      setAction(state.recording ? `Session event logging started · up to ${MAX_RECORDING} events in memory. Use Export session JSON to save them.` : `Session event logging stopped · ${state.recordingEvents.length} events retained in memory.`);
    });
    $("deviceExport").addEventListener("click", exportBundle);
    $("fixtureScenario").addEventListener("change", () => { state.fixtureScenario = $("fixtureScenario").value; });
    for (const slot of ["latest", "matched"]) {
      $(`${slot}ZoomIn`).addEventListener("click", () => setupZoom(slot, 0.25));
      $(`${slot}ZoomOut`).addEventListener("click", () => setupZoom(slot, -0.25));
      $(`${slot}ZoomReset`).addEventListener("click", () => { state.zoom[slot] = 1; $(`${slot}Viewport`).style.transform = "scale(1)"; });
    }
    $("returnToLive").addEventListener("click", async () => {
      const event = state.events[state.events.length - 1];
      if (!event) return;
      state.resultFrozen = false;
      await renderEvent(event);
      renderTimeline();
      if (!state.resultFrozen) setAction("Returned to latest completed event; camera preview continues independently.");
    });
  }
  async function releaseController() {
    if (!state.controllerId) return;
    const controllerId = state.controllerId;
    state.controllerId = null; state.controllerAuthority = false;
    clearInterval(state.heartbeatTimer); state.heartbeatTimer = null;
    $("deviceRelease").disabled = true;
    if (state.fixture) { updateFixtureLock(); setAction("TEST FIXTURE control released; its simulated loop remains unchanged."); return; }
    try {
      const result = await postJson("/api/v2/device/controller/release", { controller_id: controllerId, viewer_id: tabId });
      setAction(`Control released (${result.status || "released"}); device processing and camera state were not changed.`);
    } catch (error) { setAction(`Controller release request failed: ${error.message}`, true); }
    await pollDevice();
  }
  function startControllerHeartbeat() {
    clearInterval(state.heartbeatTimer);
    state.heartbeatTimer = setInterval(async () => {
      const controllerId = state.controllerId;
      if (!controllerId || state.fixture) return;
      try {
        const result = await postJson("/api/v2/device/controller/heartbeat", { controller_id: controllerId, viewer_id: tabId }, 1500);
        if (result.status === "rejected") throw new Error(result.message || result.reason || "Heartbeat rejected");
      } catch (error) {
        if (state.controllerId !== controllerId) return;
        state.controllerId = null; state.controllerAuthority = false;
        clearInterval(state.heartbeatTimer); state.heartbeatTimer = null;
        $("deviceRelease").disabled = true;
        setConnection(`Control lease lost · ${error.message}`, true);
        setAction("Controller lease expired; device state is read-only. Reconnect to claim control again.", true);
        await pollDevice();
      }
    }, 5000);
  }
  function initialize() {
    setupControls();
    if (fixture) {
      document.body.classList.add("test-fixture-mode");
      syncSightingsControls(null);
      syncBatteryControls(null);
      renderPerformance();
      $("deviceOneShot").disabled = false; $("deviceLoop").disabled = false; $("deviceSleep").disabled = false;
      $("deviceModel").disabled = false; $("devicePolicy").disabled = false; $("deviceApplySettings").disabled = false;
      populateChoices($("deviceModel"), ["mobilefacenet_experimental"], "mobilefacenet_experimental", "No model choices reported");
      populateChoices($("devicePolicy"), ["fixed_high", "fixed_low", "face_triggered", "adaptive"], "fixed_high", "No policy choices reported");
      startFixture();
    } else {
      setModeBadge("CONNECTING", "offline");
      setConnection("Looking for the device broker…");
      pollDevice();
      let lastDevicePoll = performance.now();
      setInterval(() => {
        const now = performance.now();
        const interval = state.loopRunning && state.policy === "F" ? 180 : 600;
        if (now - lastDevicePoll >= interval && !state.polling) {
          lastDevicePoll = now;
          pollDevice();
        }
      }, 60);
      pollSightings();
      setInterval(pollSightings, 1800);
      pollBattery();
      setInterval(pollBattery, 2500);
    }
    setInterval(updateAges, 100);
    window.addEventListener("resize", () => {
      for (const slot of ["latest", "matched"]) {
        const image = $(slot === "latest" ? "latestImage" : "matchedImage");
        if (!image.naturalWidth) continue;
        const geometry = { width: image.naturalWidth, height: image.naturalHeight };
        state.imageGeometry[slot] = geometry;
        const svg = $(slot === "latest" ? "latestOverlay" : "matchedOverlay");
        svg.setAttribute("viewBox", `0 0 ${geometry.width} ${geometry.height}`);
        renderOverlay(slot, slot === "latest" ? state.latestFrame : state.selectedEvent, geometry);
      }
    });
    window.addEventListener("pagehide", () => {
      clearInterval(state.controllerLeaseTimer); clearInterval(state.fixtureLoop);
      if (fixture) {
        try { const lease = JSON.parse(localStorage.getItem(LOCK_KEY) || "null"); if (lease?.client === tabId) localStorage.removeItem(LOCK_KEY); } catch (_) { /* private storage may be unavailable */ }
      } else if (state.controllerId) {
        const body = JSON.stringify({ controller_id: state.controllerId, viewer_id: tabId });
        navigator.sendBeacon("/api/v2/device/controller/release", new Blob([body], { type: "application/json" }));
      }
      clearInterval(state.heartbeatTimer);
    });
  }
  initialize();
})();
