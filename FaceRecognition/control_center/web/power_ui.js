(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const panel = $("powerMeter");
  if (!panel || new URLSearchParams(location.search).get("fixture") === "1") return;

  const API = "/api/v2/power";
  const ACTIVE_POLL_MS = 800;
  const IDLE_POLL_MS = 5000;
  const TREND_SECONDS = 20;
  const MAX_POINTS = 512;
  const STATUS_NAMES = {
    unsupported: "Windows only", dependency_missing: "HID support unavailable", unknown: "Ready to scan",
    absent: "Meter absent", available: "Meter available", ambiguous: "Choose an interface", busy: "Meter busy",
    connecting: "Connecting", receiving: "Receiving", stale: "Stale", error: "Meter error", disconnecting: "Disconnecting",
    unreachable: "Dashboard unavailable",
  };
  const RECORD_NAMES = { running: "Recording", complete: "Saved", failed: "Capture failed", interrupted: "Capture interrupted" };
  const METRICS = [
    { id: "powerVoltage", key: "voltage_v", unit: "V", digits: 3, color: "#91ded1" },
    { id: "powerCurrent", key: "current_ma", unit: "mA", digits: 1, color: "#a4c7f0" },
    { id: "powerWatts", key: "power_w", unit: "W", digits: 3, color: "#e7c77f" },
  ];
  const QUALITY = [
    ["powerCrc", "crc_fail_count"], ["powerGaps", "gap_count"], ["powerInvalid", "invalid_count"],
    ["powerTimeouts", "timeout_count"], ["powerDuplicates", "duplicate_count"], ["powerClockJumps", "clock_jump_count"],
  ];
  const state = {
    status: null, session: null, cursor: 0, points: [], descriptors: "", action: null, revision: 0,
    timer: null, polling: false, supportedPlatform: false, visible: !("IntersectionObserver" in window), trendDirty: true,
  };

  function setText(element, text) {
    if (element.textContent !== text) element.textContent = text;
  }

  async function request(path, body) {
    const clientId = sessionStorage.getItem("niclaControlCenterClient") || (crypto.randomUUID?.() ?? `cc-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    sessionStorage.setItem("niclaControlCenterClient", clientId);
    const headers = { "X-Control-Center-Client": clientId };
    const options = { cache: "no-store", headers, signal: AbortSignal.timeout(6000) };
    if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      options.method = "POST";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${API}/${path}`, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Power meter request failed (${response.status}).`);
    return result;
  }

  function actionMessage(text, error = false) {
    const element = $("powerMeterAction");
    element.toggleAttribute("data-error", error);
    setText(element, text);
  }

  function updateInterfaces(interfaces) {
    const select = $("powerInterface");
    const signature = JSON.stringify(interfaces);
    if (signature !== state.descriptors) {
      const selectedId = select.value;
      const options = interfaces.map((item, index) => {
        const label = interfaces.length > 1 ? [`Candidate ${index + 1}`] : [];
        label.push(item.product || item.manufacturer || "USB meter");
        if (Number.isInteger(item.interface_number) && item.interface_number >= 0) label.push(`interface ${item.interface_number}`);
        if (Number.isInteger(item.usage_page) && Number.isInteger(item.usage)) {
          label.push(`usage 0x${item.usage_page.toString(16)}/0x${item.usage.toString(16)}`);
        }
        return new Option(label.join(" · "), item.id);
      });
      if (interfaces.length > 1) options.unshift(new Option("Choose a meter interface", ""));
      select.replaceChildren(...options);
      if (interfaces.some((item) => item.id === selectedId)) select.value = selectedId;
      state.descriptors = signature;
    }
    $("powerInterfaceField").hidden = !interfaces.length;
  }

  function updateTopology() {
    const topology = $("powerTopology");
    let label = "Topology unconfirmed";
    if (state.status?.topology_confirmed) label = "Operator-confirmed inline topology";
    else if (topology.checked) label = "Confirmation applies when recording starts";
    setText($("powerTopologyStatus"), label);
  }

  function renderControls() {
    const status = state.status || {};
    const code = status.state;
    const supported = Boolean(code) && !["unsupported", "dependency_missing", "unreachable"].includes(code);
    const transition = ["connecting", "disconnecting"].includes(code);
    const connected = status.connected === true;
    const recording = status.recording?.state === "running";
    const pending = state.action !== null;
    panel.hidden = $("powerMeterLink").hidden = !state.supportedPlatform;
    $("powerMeterReadout").hidden = !connected;
    $("powerRecord").hidden = !connected && !status.recording;
    $("powerTopologyRow").hidden = !connected;
    $("powerConnect").hidden = !status.interfaces?.length || connected || transition;
    $("powerDisconnect").hidden = !connected && !transition;
    $("powerRecordStart").hidden = !connected;
    $("powerRecordStop").hidden = !connected && !recording;
    $("powerScan").disabled = !supported || pending || transition || connected || Boolean(status.scan_in_progress);
    $("powerInterface").disabled = !supported || pending || transition || connected || Boolean(status.scan_in_progress);
    $("powerConnect").disabled = $("powerInterface").disabled || !$("powerInterface").value;
    $("powerDisconnect").disabled = pending || code === "disconnecting" || (!connected && code !== "connecting");
    $("powerRecordStart").disabled = pending || transition || !connected || recording;
    $("powerRecordStop").disabled = pending || !recording;
    $("powerTopology").disabled = pending || transition || !connected || recording;
  }

  function updateRecording(recording) {
    const summary = $("powerRecordingSummary");
    const duration = $("powerRecordingDuration");
    const location = $("powerRecordingLocation");
    summary.hidden = !recording;
    duration.hidden = !Number.isFinite(recording?.duration_s);
    location.hidden = !recording?.output_dir;
    setText(summary, recording ? `${RECORD_NAMES[recording.state] || "Capture"} · ${recording.run_id}` : "");
    setText(duration, duration.hidden ? "" : `${recording.duration_s.toFixed(1)} s`);
    setText($("powerRecordingPath"), recording?.output_dir || "");
  }

  function updateStatus(status) {
    if ((status.session_id || null) !== state.session) {
      state.session = status.session_id || null;
      state.cursor = 0;
      state.points = [];
      state.trendDirty = true;
      $("powerTopology").checked = status.topology_confirmed === true;
    }
    state.status = status;
    if (status.state !== "unreachable") state.supportedPlatform = status.state !== "unsupported";
    panel.dataset.state = status.state;
    const badge = $("powerMeterState");
    badge.dataset.state = status.state;
    setText(badge, status.scan_in_progress ? "Scanning" : STATUS_NAMES[status.state] || "Power meter unavailable");
    setText($("powerMeterStatus"), status.message || "");
    updateInterfaces(status.interfaces || []);
    renderControls();
    updateTopology();
    updateRecording(status.recording);

    for (const metric of METRICS) {
      const value = status.connected ? status.latest?.[metric.key] : null;
      const available = Number.isFinite(value);
      const element = $(metric.id);
      setText(element, available ? `${value.toFixed(metric.digits)} ${metric.unit}` : "Unavailable");
      element.toggleAttribute("data-unavailable", !available);
    }
    const age = status.connected ? status.sample_age_ms : null;
    const ageLabel = age < 1000 ? `${Math.round(age)} ms` : `${(age / 1000).toFixed(1)} s`;
    setText($("powerSampleAge"), Number.isFinite(age) ? ageLabel : "—");
    const rate = status.connected ? status.report_rate_hz : null;
    setText($("powerReportCadence"), Number.isFinite(rate) ? `${rate.toFixed(1)} reports/s` : "—");
    for (const [id, key] of QUALITY) {
      const element = $(id);
      const value = status[key];
      setText(element, Number.isInteger(value) ? String(value) : "—");
      element.toggleAttribute("data-attention", value > 0);
    }
  }

  function drawChart(metric, points, end) {
    const canvas = $(`${metric.id}Trend`);
    const range = $(`${metric.id}Range`);
    canvas.hidden = range.hidden = !points.length;
    if (!points.length) return;
    const values = points.map((point) => point[metric.key]);
    const low = Math.min(...values);
    const high = Math.max(...values);
    setText(range, `${low.toFixed(metric.digits)}–${high.toFixed(metric.digits)} ${metric.unit}`);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(canvas.clientWidth * pixelRatio);
    const height = Math.round(canvas.clientHeight * pixelRatio);
    if (!width || !height) return;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    const pad = 4 * pixelRatio;
    const start = end - TREND_SECONDS;
    const yPad = Math.max((high - low) * 0.08, 0.5 * 10 ** -metric.digits);
    const dots = [];
    context.clearRect(0, 0, width, height);
    context.beginPath();
    context.lineWidth = 1.5 * pixelRatio;
    context.strokeStyle = metric.color;
    points.forEach((point, index) => {
      const x = pad + ((point.elapsed_s - start) / TREND_SECONDS) * (width - 2 * pad);
      const y = height - pad - ((values[index] - low + yPad) / (high - low + 2 * yPad)) * (height - 2 * pad);
      if (point.gap_before || index === 0) {
        context.moveTo(x, y);
        dots.push([x, y]);
      } else context.lineTo(x, y);
      if (index === points.length - 1) dots.push([x, y]);
    });
    context.stroke();
    context.fillStyle = metric.color;
    for (const [x, y] of dots) {
      context.beginPath();
      context.arc(x, y, 1.5 * pixelRatio, 0, 2 * Math.PI);
      context.fill();
    }
  }

  function drawTrends() {
    if (!state.visible || !state.trendDirty) return;
    const end = state.points.at(-1)?.elapsed_s;
    const points = state.points.filter((point) => point.elapsed_s >= end - TREND_SECONDS);
    for (const metric of METRICS) drawChart(metric, points, end);
    const label = $("powerTrendWindow");
    label.hidden = !points.length;
    setText(label, `Past ${TREND_SECONDS} s through the latest valid report`);
    state.trendDirty = false;
  }

  function applySamples(result) {
    if (result.session_id !== state.session) return;
    const points = result.samples.filter((point) => Number.isFinite(point.elapsed_s) &&
      METRICS.every((metric) => Number.isFinite(point[metric.key])));
    if (result.reset) state.points = [];
    state.cursor = result.next_cursor;
    if (points.length || result.reset) {
      state.points = state.points.concat(points).slice(-MAX_POINTS);
      state.trendDirty = true;
    }
    drawTrends();
  }

  function schedule(delay) {
    clearTimeout(state.timer);
    if (delay !== null && !document.hidden) state.timer = setTimeout(refresh, delay);
  }

  function pollInterval(status) {
    if (status?.state === "unsupported") return null;
    if (status?.connected || status?.scan_in_progress || ["connecting", "disconnecting"].includes(status?.state)) return ACTIVE_POLL_MS;
    return IDLE_POLL_MS;
  }

  async function refresh() {
    if (state.polling || state.action || document.hidden) return;
    state.polling = true;
    const revision = state.revision;
    try {
      const status = await request("status");
      if (revision !== state.revision) return;
      updateStatus(status);
      if (status.connected && state.session && status.latest) {
        try {
          const samples = await request(`samples?after=${state.cursor}&limit=100`);
          if (revision !== state.revision) return;
          applySamples(samples);
        } catch (error) {
          if (revision === state.revision) setText($("powerMeterStatus"), `Trend temporarily unavailable. ${error.message}`);
        }
      }
      drawTrends();
    } catch (error) {
      if (revision === state.revision) {
        updateStatus({ state: "unreachable", message: error.message, recording: state.status?.recording });
        drawTrends();
      }
    } finally {
      state.polling = false;
      if (!state.action) schedule(pollInterval(state.status));
    }
  }

  async function act(action, body = {}) {
    if (state.action) return;
    state.revision += 1;
    state.action = action;
    clearTimeout(state.timer);
    actionMessage("");
    renderControls();
    try {
      updateStatus(await request(action, body));
      drawTrends();
    } catch (error) {
      actionMessage(error.message, true);
    } finally {
      state.action = null;
      renderControls();
      schedule(0);
    }
  }

  $("powerInterface").addEventListener("change", renderControls);
  $("powerTopology").addEventListener("change", updateTopology);
  $("powerScan").addEventListener("click", () => act("scan"));
  $("powerConnect").addEventListener("click", () => act("connect", { interface_id: $("powerInterface").value }));
  $("powerDisconnect").addEventListener("click", () => act("disconnect"));
  $("powerRecordStart").addEventListener("click", () => act("start-recording", { topology_confirmed: $("powerTopology").checked }));
  $("powerRecordStop").addEventListener("click", () => act("stop-recording"));
  $("powerCopyPath").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("powerRecordingPath").textContent);
      actionMessage("Capture folder copied.");
    } catch {
      actionMessage("Select the capture folder path to copy it.");
    }
  });
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      state.visible = entry.isIntersecting;
      drawTrends();
    }).observe(panel);
  }
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => {
      state.trendDirty = true;
      drawTrends();
    }).observe($("powerMeterReadout"));
  }
  document.addEventListener("visibilitychange", () => schedule(0));
  schedule(0);
})();
