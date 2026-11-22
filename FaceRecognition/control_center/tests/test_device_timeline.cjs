const assert = require("node:assert/strict");
const { webcrypto } = require("node:crypto");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { test } = require("node:test");
const { runInNewContext } = require("node:vm");

class Element {
  constructor() {
    this.attributes = {};
    this.children = [];
    this.dataset = {};
    this.handlers = {};
    this.selectors = new Map();
    this.style = {};
    this.textContent = "";
  }
  get options() { return this.children; }
  addEventListener(name, handler) { this.handlers[name] = handler; }
  appendChild(child) { this.children.push(child); }
  removeAttribute(name) { delete this.attributes[name]; }
  getContext() { return { clearRect() {} }; }
  querySelector(selector) {
    if (!this.selectors.has(selector)) this.selectors.set(selector, new Element());
    return this.selectors.get(selector);
  }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  click() { return this.handlers.click?.(); }
  focus() { this.focused = true; }
  showModal() { this.open = true; }
}

class FrameImage extends Element {
  naturalWidth = 240;
  naturalHeight = 240;
  hidden = true;

  get src() { return this.attributes.src || ""; }
  set src(value) {
    this.attributes.src = value;
    if (value) queueMicrotask(() => this.onload?.());
  }
  async decode() {}
}

function loadDashboard({ realFrameImages = false } = {}) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) {
      const isFrameImage = realFrameImages && (id === "latestImage" || id === "matchedImage");
      elements.set(id, isFrameImage ? new FrameImage() : new Element());
    }
    return elements.get(id);
  };
  const source = readFileSync(join(__dirname, "../web/device_ui.js"), "utf8");
  const startup = "  initialize();\n})();";
  assert.ok(source.includes(startup));
  const frameRequests = [];
  const frameResponses = new Map();
  const sandbox = {
    AbortSignal,
    clearInterval() {},
    clearTimeout,
    crypto: webcrypto,
    document: { getElementById: element, createElement: () => new Element() },
    fetch: async (path) => {
      frameRequests.push(path);
      const response = frameResponses.get(path);
      assert.ok(response, `Unexpected frame request: ${path}`);
      return response;
    },
    Image: FrameImage,
    location: { search: "" },
    performance,
    sessionStorage: { getItem: () => "test-viewer" },
    setInterval: () => 1,
    setTimeout,
    URL,
    URLSearchParams,
    responses: {},
    postResponses: {},
    posts: [],
    requests: [],
  };
  const injected = source.replace(startup, `
    getJson = async (path) => {
      requests.push(path);
      const response = responses[path.split("?")[0]];
      if (response instanceof Error) throw response;
      return response;
    };
    postJson = async (path, body) => {
      posts.push({ path, body });
      if (postResponses[path]) return postResponses[path](body);
      return { status: "accepted", command_id: body.command_id };
    };
    ${realFrameImages ? "" : "loadFrameImage = async () => true;"}
    globalThis.dashboard = { applySettings, ingestEvent, pollBattery, pollDevice, pollSightings, state, syncControls, updateLatestFrame };
    setupControls();
  })();`);
  runInNewContext(injected, sandbox, { filename: "device_ui.js" });
  return { ...sandbox.dashboard, element, responses: sandbox.responses, posts: sandbox.posts, requests: sandbox.requests,
    postResponses: sandbox.postResponses, frameRequests, frameResponses };
}

function observation(boot, frame, receiveNs, seq) {
  return {
    type: "observation_complete",
    boot_id: boot,
    frame_id: frame,
    host_received_monotonic_ns: receiveNs,
    event_seq: seq,
    decision: { frame_id: frame, status: "OTHER" },
  };
}

function framePath(boot, id) {
  return `/api/v2/device/frame?boot_id=${boot}&frame_id=${id}`;
}

function frameStatus(boot, id, ready) {
  return { boot_id: boot, frame_id: id, payload_ready: ready, frame_ref: ready ? framePath(boot, id) : null,
    width: 240, height: 240, age_ms: 20, stale: false };
}

function jpeg(boot, id) {
  return new Response(new Uint8Array([0xff, 0xd8, 0xff, 0xd9]), {
    headers: { "Content-Type": "image/jpeg", "X-Device-Boot-ID": boot, "X-Device-Frame-ID": String(id) },
  });
}

test("fresh status snapshot survives historical event catch-up across a boot", async () => {
  const { element, pollDevice, responses } = loadDashboard();
  const latest = observation("new-boot", 1, 4e9);
  responses["/api/v2/device/status"] = {
    boot_id: "new-boot",
    connected: true,
    last_observation: latest,
    capabilities: { commands: ["one_shot"], models: ["small"], policies: ["S"] },
  };
  responses["/api/v2/device/events"] = {
    next_seq: 82,
    events: [observation("prior-boot", 63, 2e9, 79), { ...latest, event_seq: 82 }],
  };
  await pollDevice();
  const rows = element("deviceTimeline").children;
  assert.equal(element("resultFrameId").textContent, "new-boot / 1");
  assert.deepEqual(rows.map((row) => row.dataset.key), ["new-boot:1", "prior-boot:63"]);
  assert.equal(rows[0].attributes["aria-current"], "true");
  assert.equal(element("returnToLive").disabled, true);
});

test("clicking the newest event freezes it through the next capture until returning live", async () => {
  const { element, ingestEvent } = loadDashboard();
  await ingestEvent(observation("boot", 1, 1e9, 20));
  element("deviceTimeline").children[0].click();
  await ingestEvent(observation("boot", 2, 2e9, 21));
  assert.equal(element("resultFrameId").textContent, "boot / 1");
  assert.equal(element("returnToLive").disabled, false);
  await element("returnToLive").click();
  assert.equal(element("resultFrameId").textContent, "boot / 2");
  assert.equal(element("returnToLive").disabled, true);
  await ingestEvent(observation("boot", 3, 3e9, 22));
  assert.equal(element("resultFrameId").textContent, "boot / 3");
});

test("live preview holds the last JPEG through a missing frame and retries when it is ready", async () => {
  const { element, frameRequests, frameResponses, state, updateLatestFrame } = loadDashboard({ realFrameImages: true });
  const image = element("latestImage");
  const badge = element("latestState");
  const metadata = element("latestMeta");
  const first = framePath("boot", 1);
  const second = framePath("boot", 2);

  frameResponses.set(first, jpeg("boot", 1));
  await updateLatestFrame(frameStatus("boot", 1, true));
  const firstImage = image.src;
  const firstReceivedAt = state.latestReceivedAt;
  assert.ok(firstImage);
  assert.equal(image.hidden, false);
  assert.match(metadata.textContent, /^Frame 1\b/);

  await updateLatestFrame(frameStatus("boot", 2, false));
  assert.deepEqual(frameRequests, [first]);
  assert.equal(badge.textContent, "FRAME DELAYED");
  assert.equal(image.hidden, false);
  assert.equal(image.src, firstImage);
  assert.equal(state.latestReceivedAt, firstReceivedAt);
  assert.match(metadata.textContent, /^Frame 1\b/);

  frameResponses.set(second, Response.json({ error: "Frame unavailable" }, { status: 404 }));
  await updateLatestFrame(frameStatus("boot", 2, true));
  assert.equal(badge.textContent, "FRAME DELAYED");
  assert.equal(image.hidden, false);
  assert.equal(image.src, firstImage);
  assert.equal(state.latestReceivedAt, firstReceivedAt);
  assert.match(metadata.textContent, /^Frame 1\b/);

  frameResponses.set(second, jpeg("boot", 2));
  await updateLatestFrame(frameStatus("boot", 2, true));
  assert.deepEqual(frameRequests, [first, second, second]);
  assert.equal(image.hidden, false);
  assert.notEqual(image.src, firstImage);
  assert.match(metadata.textContent, /^Frame 2\b/);

  await updateLatestFrame(frameStatus("new-boot", 1, false));
  assert.deepEqual(frameRequests, [first, second, second]);
  assert.equal(image.hidden, true);
  assert.equal(badge.textContent, "WAITING FOR IMAGE");
  assert.doesNotMatch(metadata.textContent, /^Frame 2\b/);
});

test("live preview follows the current boot JPEG when the same-frame inference record is missing", async () => {
  const { element, frameRequests, frameResponses, pollDevice, responses } = loadDashboard({ realFrameImages: true });
  const completed = {
    ...observation("boot", 1, 1e9, 1),
    detector_input: { payload_status: "not_exported_live" },
    identity: { status: "skipped" },
  };
  responses["/api/v2/device/status"] = {
    ...deviceStatus(), boot_id: "boot", event_seq: 1, last_observation: completed, latest_frame: frameStatus("boot", 1, true),
  };
  responses["/api/v2/device/events"] = { next_seq: 1, events: [] };
  frameResponses.set(framePath("boot", 1), jpeg("boot", 1));
  await pollDevice();
  const latest = element("latestImage");
  const matched = element("matchedImage");
  const firstImage = latest.src;
  const firstMatched = matched.src;
  assert.ok(firstImage && firstMatched);
  assert.match(element("latestMeta").textContent, /^Frame 1\b/);
  assert.equal(element("resultFrameId").textContent, "boot / 1");

  responses["/api/v2/device/status"].event_seq = 5;
  const payload = (boot, id, kind, seq) => ({
    type: "frame_payload", kind, boot_id: boot, frame_id: id, event_seq: seq,
    host_received_utc: new Date().toISOString(), metadata: { width: 240, height: 240 },
  });
  responses["/api/v2/device/events"] = { next_seq: 5, events: [
    payload("boot", 2, "frame_jpeg", 2),
    { type: "telemetry", event: "telemetry_encode_error", boot_id: "boot", event_seq: 3,
      record: { failed_event: "observation_complete", error_type: "MemoryError" } },
    payload("boot", 3, "detector_input", 4),
    payload("prior-boot", 90, "frame_jpeg", 5),
  ] };
  frameResponses.set(framePath("boot", 2), jpeg("boot", 2));
  await pollDevice();
  assert.deepEqual(frameRequests, [framePath("boot", 1), framePath("boot", 2)]);
  assert.equal(latest.hidden, false);
  assert.notEqual(latest.src, firstImage);
  assert.match(element("latestMeta").textContent, /^Frame 2\b/);
  assert.equal(matched.src, firstMatched);
  assert.equal(element("resultFrameId").textContent, "boot / 1");

  responses["/api/v2/device/events"] = { next_seq: 5, events: [] };
  await pollDevice();
  assert.match(element("latestMeta").textContent, /^Frame 2\b/);
});

function deviceStatus() {
  return {
    connected: true,
    capabilities: { commands: ["one_shot", "set_model", "set_policy"], models: ["small", "large"], policies: ["S", "F"] },
    controller_authority: { viewer_id: "test-viewer", owned_by_viewer: true, controller_id: "owned", lease_remaining_ms: 10000 },
    applied_state: { model_override: "small", policy: "S" },
  };
}

test("changing both settings waits for firmware confirmation, including while live images are paused", async () => {
  const { applySettings, element, pollDevice, posts, responses } = loadDashboard();
  responses["/api/v2/device/status"] = deviceStatus();
  responses["/api/v2/device/events"] = { next_seq: 0, events: [] };
  await pollDevice();
  const originalOption = element("devicePolicy").options[0];
  await pollDevice();
  assert.equal(element("devicePolicy").options[0], originalOption);
  element("devicePause").click();
  element("deviceModel").value = "large";
  element("devicePolicy").value = "F";
  const applied = applySettings();
  await new Promise(setImmediate);
  assert.deepEqual(posts.map(({ body }) => body.cmd), ["set_model"]);
  assert.equal(element("deviceApplySettings").disabled, true);
  responses["/api/v2/device/events"] = { next_seq: 1, events: [
    { type: "command_ack", command_id: posts[0].body.command_id, control_status: "completed", event_seq: 1 },
  ] };
  await pollDevice();
  assert.deepEqual(posts.map(({ body }) => body.cmd), ["set_model", "set_policy"]);
  responses["/api/v2/device/events"] = { next_seq: 2, events: [
    { type: "command_ack", command_id: posts[1].body.command_id, control_status: "completed", event_seq: 2 },
  ] };
  await pollDevice();
  await applied;
  assert.match(element("commandAck").textContent, /completed$/);
  assert.equal(element("deviceApplySettings").disabled, false);
});

test("broker restart rewinds the event cursor and expired ownership disables camera commands", async () => {
  const { element, pollDevice, requests, responses } = loadDashboard();
  responses["/api/v2/device/status"] = { ...deviceStatus(), event_seq: 32 };
  responses["/api/v2/device/events"] = { next_seq: 32, events: [] };
  await pollDevice();
  assert.equal(element("deviceOneShot").disabled, false);
  responses["/api/v2/device/status"] = {
    ...deviceStatus(),
    event_seq: 1,
    controller_authority: { expired: true, owned_by_viewer: false },
  };
  responses["/api/v2/device/events"] = { next_seq: 1, events: [] };
  await pollDevice();
  assert.equal(requests.at(-1), "/api/v2/device/events?after=0");
  assert.equal(element("deviceOneShot").disabled, true);
  assert.equal(element("deviceConnect").textContent, "Take control");
});

test("the last saved video remains available after a failed recording and battery controls explain what to do", () => {
  const { element, state, syncControls } = loadDashboard();
  const status = deviceStatus();
  state.batteryAvailable = true;
  state.battery = { supported: true, available: true, armed: true, pending_photos: 1 };
  state.video = {
    state: "error",
    error: "Capture failed",
    last_clip: {
      download_url: "/api/v2/device/video/file?id=good",
      manifest_url: "/api/v2/device/video/manifest?id=good",
    },
  };
  syncControls({ ...status, running: true }, status.capabilities);
  assert.equal(element("deviceClipActions").hidden, false);
  assert.equal(element("deviceVideoReview").hidden, false);
  assert.match(element("deviceVideoStatus").textContent, /Capture failed/);
  assert.equal(element("batteryImport").disabled, true);
  assert.match(element("batteryStatus").textContent, /Stop processing/);
  syncControls(status, status.capabilities);
  assert.equal(element("batteryImport").disabled, false);
  assert.match(element("batteryStatus").textContent, /Import photos/);
});

test("local gallery separates manual filing, accepted identity, and unknown results", async () => {
  const { element, pollSightings, responses } = loadDashboard();
  const items = [
    { id: 1, identity_id: "K01", person_id: "K01", person_assignment: "device", decision_status: "KNOWN_IDENTITY" },
    { id: 2, identity_id: "K01", person_id: null, person_assignment: "manual", decision_status: "KNOWN_IDENTITY" },
    { id: 3, identity_id: "K02", decision_status: "UNKNOWN", source: "on_nicla_battery",
      imported_utc: "2026-11-16T13:00:00Z", image_url: "/api/v2/device/sightings/frame?id=3" },
  ];
  responses["/api/v2/device/sightings"] = { items, enabled: false, total_encounters: 8, retained_count: 3 };
  await pollSightings();
  const groups = element("sightingsList").children;
  assert.deepEqual(groups.map((group) => group.children[0].textContent), ["Person 1 · 1", "Other · 2"]);
  assert.equal(groups[0].children[1].children[0].children[0].children.at(-1).textContent, "Identified by Nicla");
  assert.equal(groups[1].children[1].children[1].children[0].children.at(-1).textContent, "Assigned manually");
  const batteryPhoto = groups[1].children[1].children[0].children[0];
  batteryPhoto.click();
  assert.equal(element("sightingDialog").open, true);
  assert.equal(element("sightingDialogPhoto").src, "/api/v2/device/sightings/frame?id=3");
  assert.match(element("sightingDialogDetails").textContent, /^On-board capture · Imported /);
  assert.equal(element("sightingClaim").hidden, false);
  element("sightingsPerson").handlers.change({ target: { value: "K01" } });
  assert.deepEqual(element("sightingsList").children.map((group) => group.children[0].textContent), ["Person 1 · 1"]);
});

test("newer Other photos do not hide a person group", async () => {
  const { element, pollSightings, responses } = loadDashboard();
  const items = [{ id: 1, person_id: "K01", person_assignment: "manual" }];
  for (let id = 2; id <= 14; id++) items.push({ id, person_id: null, person_assignment: "unassigned" });
  responses["/api/v2/device/sightings"] = { items, enabled: false, total_encounters: 14, retained_count: 14 };
  await pollSightings();
  const gallery = element("sightingsList");
  assert.deepEqual(gallery.children.map((group) => group.children[0].textContent), ["Person 1 · 1", "Other · 13"]);
  assert.equal(gallery.children[1].children[1].children.length, 6);
  assert.equal(element("sightingsMore").hidden, false);
});

test("the battery review shortcut shows only imported photos and still groups by person", async () => {
  const { element, pollSightings, responses } = loadDashboard();
  const items = [
    { id: 1, person_id: "K01", person_assignment: "manual", source: "on_nicla_battery" },
    { id: 2, person_id: null, person_assignment: "unassigned" },
    { id: 3, person_id: "K02", person_assignment: "manual", source: "on_nicla_battery" },
  ];
  responses["/api/v2/device/sightings"] = { items, enabled: false, total_encounters: 3, retained_count: 3 };
  await pollSightings();
  assert.equal(element("batteryReview").hidden, false);
  await element("batteryReview").click();
  assert.equal(element("sightingsSource").value, "onboard");
  assert.equal(element("sightingsSource").focused, true);
  assert.deepEqual(element("sightingsList").children.map((group) => group.children[0].textContent),
    ["Person 1 · 1", "Person 2 · 1"]);
  element("sightingsSource").handlers.change({ target: { value: "usb" } });
  assert.deepEqual(element("sightingsList").children.map((group) => group.children[0].textContent), ["Other · 1"]);
});

test("taking control from a photo permits manual filing without opening serial", async () => {
  const { element, pollSightings, responses, postResponses, posts } = loadDashboard();
  responses["/api/v2/device/sightings"] = { items: [{ id: 4, person_id: null, person_assignment: "unassigned" }],
    enabled: false, total_encounters: 1, retained_count: 1 };
  responses["/api/v2/device/status"] = {
    ...deviceStatus(), connected: false, controller_authority: { owned_by_viewer: true, controller_id: "gallery-lease" },
  };
  responses["/api/v2/device/events"] = { events: [], next_seq: 0 };
  postResponses["/api/v2/device/controller/claim"] = () => ({ controller_id: "gallery-lease" });
  postResponses["/api/v2/device/sightings/assign"] = (body) => {
    const assigned = { sighting_id: body.sighting_id, person_id: body.person_id, person_assignment: "manual" };
    Object.assign(responses["/api/v2/device/sightings"].items[0], assigned);
    return assigned;
  };
  await pollSightings();
  element("sightingsList").children[0].children[1].children[0].children[0].click();
  await element("sightingClaim").click();
  element("sightingAssignPerson").value = "K03";
  element("sightingAssignPerson").handlers.change();
  await element("sightingAssignment").handlers.submit({ preventDefault() {} });
  assert.deepEqual(posts.map((entry) => entry.path), ["/api/v2/device/controller/claim", "/api/v2/device/sightings/assign"]);
  assert.equal(posts[1].body.person_id, "K03");
  assert.equal(posts[1].body.controller_id, "gallery-lease");
  assert.equal(element("sightingAssignmentStatus").textContent, "Saved to Person 3.");
  assert.equal(element("sightingsList").children[0].children[0].textContent, "Person 3 · 1");
});

test("battery preparation guides a rebooted running Nicla through stop and explicit import", async () => {
  const { element, pollBattery, responses, posts, state, syncControls, postResponses } = loadDashboard();
  const batteryPath = "/api/v2/device/battery";
  const ready = deviceStatus();
  state.status = { ...ready, running: true };
  syncControls(state.status, ready.capabilities);
  assert.equal(element("batteryPrepare").textContent, "Stop processing");
  assert.equal(element("batteryImport").disabled, true);
  assert.equal(posts.length, 0);
  responses[batteryPath] = { supported: false, checking: true };
  await pollBattery();
  assert.equal(element("batteryState").textContent, "CHECKING");
  responses[batteryPath] = { supported: true, available: true, armed: true, pending_photos: 3,
    imported_count: 1, max_records: 24, actions: { import: true } };
  await pollBattery();
  assert.equal(element("batteryPending").textContent, "3 / 24");
  assert.equal(element("batteryImported").textContent, "1");
  assert.equal(element("batteryImport").disabled, true);
  state.status = ready;
  syncControls(ready, ready.capabilities);
  postResponses[batteryPath] = () => ({ ...responses[batteryPath], busy: true, active_action: "import" });
  await element("batteryImport").click();
  assert.equal(posts[0].path, batteryPath);
  assert.equal(posts[0].body.action, "import");
  assert.equal(element("batteryState").textContent, "IMPORTING");
  responses[batteryPath] = { ...responses[batteryPath], unimported_count: 0, imported_count: 3, last_import_count: 2 };
  await pollBattery();
  assert.equal(element("batteryImported").textContent, "3");
  assert.equal(element("batteryImport").disabled, false);
  assert.match(element("batteryStatus").textContent, /skips duplicates/);
});
