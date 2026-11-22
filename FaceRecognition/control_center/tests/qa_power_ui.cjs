// Browser QA. Use Playwright with installed Chrome. Every HTTP response is a fixture.
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");
const { chromium } = require("playwright");

const origin = "http://127.0.0.1:8780";
const staticRoot = path.join(__dirname, "..", "web");
const output = process.argv[2];
const contentTypes = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const descriptors = [
  { id: "fixture-one", product: "FNB-C2 fixture A", interface_number: 1, usage_page: 0xff00, usage: 1 },
  { id: "fixture-two", product: "FNB-C2 fixture B", interface_number: 2, usage_page: 0xff00, usage: 1 },
];
const fixtureFrame = `<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">
  <defs><pattern id="tiles" width="40" height="40" patternUnits="userSpaceOnUse">
    <rect width="40" height="40" fill="#b8c6ca"/><path d="M0 0h20v20H0zM20 20h20v20H20z" fill="#233341"/>
  </pattern></defs><rect width="240" height="240" fill="url(#tiles)"/>
  <rect width="240" height="38" y="101" fill="#101b22"/>
  <text x="120" y="124" text-anchor="middle" fill="white" font-size="14" font-family="sans-serif">SYNTHETIC FRAME</text>
</svg>`;
const trace = Array.from({ length: 80 }, (_, i) => {
  const voltage_v = 5.008 + 0.014 * Math.sin(i / 7);
  const current_ma = 125.4 + 15.2 * Math.cos(i / 10);
  return {
    cursor: i + 1, elapsed_s: i / 4, voltage_v, current_ma, power_w: voltage_v * current_ma / 1000,
    host_receive_utc: new Date(Date.UTC(2026, 10, 11, 18) + 250 * i).toISOString(), gap_before: i === 32,
  };
});

async function run() {
  const browser = await chromium.launch({ channel: "chrome", headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await page.context().grantPermissions(["clipboard-read", "clipboard-write"], { origin });
    const pageErrors = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    const actions = [];
    let cameraFrames = 0;
    let powerReads = 0;
    let scans = 0;
    let attempts = 0;
    let status = { state: "unsupported", message: "TEST FIXTURE · Optional FNIRSI support is available on Windows." };
    await page.route("**/*", async (route) => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin !== origin) return route.abort();
      const resource = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
      if (Object.hasOwn(contentTypes, path.extname(resource)) && !resource.includes("/")) {
        return route.fulfill({ body: await fs.readFile(path.join(staticRoot, resource)), contentType: contentTypes[path.extname(resource)] });
      }
      const respond = (json, code = 200) => route.fulfill({ json, status: code });
      if (url.pathname === "/api/v1/replay/frame") return route.fulfill({ body: fixtureFrame, contentType: "image/svg+xml" });
      if (url.pathname.startsWith("/api/v2/device/")) {
        assert.equal(request.method(), "GET");
        const boot_id = "browser-qa-fixture";
        switch (url.pathname) {
          case "/api/v2/device/status":
            cameraFrames += 1;
            return respond({
              connected: true, execution_location: "test_fixture", boot_id,
              capabilities: { commands: ["one_shot"], models: ["small"], policies: ["S"] },
              latest_frame: { boot_id, frame_id: cameraFrames, age_ms: 20, width: 240, height: 240, payload_ready: true },
            });
          case "/api/v2/device/frame":
            return route.fulfill({ body: fixtureFrame, contentType: "image/svg+xml", headers: {
              "X-Device-Boot-ID": boot_id, "X-Device-Frame-ID": url.searchParams.get("frame_id"),
            } });
          case "/api/v2/device/events": return respond({ next_seq: 0, events: [] });
          case "/api/v2/device/video": return respond(status.recording?.state === "running" ? { state: "recording", frames: 3 } : { state: "idle" });
          default: return respond({}, 404);
        }
      }
      if (!url.pathname.startsWith("/api/v2/power/")) return respond({}, 404);
      const action = resource.split("/").at(-1);
      assert.ok(request.headers()["x-control-center-client"]);
      if (request.method() === "GET") {
        powerReads += 1;
        if (action === "status") return respond(status);
        assert.equal(action, "samples");
        assert.equal(status.connected, true);
        const after = Number(url.searchParams.get("after"));
        const limit = Number(url.searchParams.get("limit"));
        assert.ok(Number.isInteger(after) && after >= 0);
        assert.ok(limit >= 1 && limit <= 100);
        const samples = trace.filter((point) => point.cursor > after).slice(0, limit);
        return respond({ session_id: status.session_id, samples, next_cursor: samples.at(-1)?.cursor ?? after, reset: false });
      }
      assert.equal(request.method(), "POST");
      const body = request.postDataJSON();
      actions.push(action);
      if (action === "scan") {
        scans += 1;
        status = scans === 1 ? { state: "absent", message: "TEST FIXTURE · Meter unplugged" } :
          { state: "ambiguous", interfaces: descriptors, message: "TEST FIXTURE · Two matching descriptors" };
      } else if (action === "connect") {
        assert.equal(body.interface_id, "fixture-two");
        attempts += 1;
        if (attempts === 1) {
          status = { ...status, state: "busy", message: "TEST FIXTURE · CLI owns the meter" };
        } else {
          status = {
            ...status, state: "receiving", connected: true, session_id: `fixture-session-${attempts}`, latest: trace.at(-1),
            sample_age_ms: 82, report_rate_hz: 4, crc_fail_count: 1, gap_count: 1, invalid_count: 1,
            timeout_count: 0, duplicate_count: 0, clock_jump_count: 0, topology_confirmed: false, message: "TEST FIXTURE · Simulated power trace",
          };
        }
      } else if (action === "start-recording") {
        assert.equal(body.topology_confirmed, true);
        status = { ...status, topology_confirmed: true, recording: {
          state: "running", run_id: "fixture-run", duration_s: 2.5, output_dir: "C:\\fixture\\meter\\fixture-run",
        } };
      } else if (action === "stop-recording") {
        status = { ...status, recording: { ...status.recording, state: "complete", duration_s: 5.5 } };
      } else if (action === "disconnect") {
        status = { state: "available", interfaces: descriptors, message: "TEST FIXTURE · Meter released", recording: status.recording };
      } else {
        throw new Error(`Unexpected POST ${action}`);
      }
      return respond(status);
    });

    const visit = () => page.goto(origin);
    const id = (name) => page.locator(`#${name}`);
    const expectText = (name, text) => page.waitForFunction(([id, text]) => document.getElementById(id).textContent.includes(text), [name, text]);
    await page.goto(`${origin}/?fixture=1`);
    await expectText("deviceMode", "TEST FIXTURE");
    assert.equal(await id("powerMeter").isVisible(), false);
    assert.equal(powerReads, 0);
    await visit();
    await expectText("powerMeterState", "Windows only");
    assert.equal(await id("powerMeter").isVisible(), false);
    assert.equal(await id("powerMeterLink").isVisible(), false);
    await id("latestImage").waitFor({ state: "visible" });
    assert.deepEqual(actions, []);

    status = { state: "dependency_missing", message: "TEST FIXTURE · Optional hid module unavailable" };
    await visit();
    await expectText("powerMeterState", "HID support unavailable");
    assert.equal(await id("powerScan").isEnabled(), false);
    assert.deepEqual(actions, []);

    status = { state: "unknown", message: "TEST FIXTURE · Scan to find the meter" };
    await visit();
    await expectText("powerMeterState", "Ready to scan");
    assert.equal(await id("powerScan").isVisible(), true);
    assert.equal(await id("powerConnect").isVisible(), false);
    assert.equal(await id("powerMeterReadout").isVisible(), false);
    await id("powerScan").focus();
    await page.keyboard.press("Enter");
    await expectText("powerMeterState", "Meter absent");
    assert.equal(await id("powerConnect").isEnabled(), false);
    await id("powerScan").click();
    await expectText("powerMeterState", "Choose an interface");
    assert.equal(await id("powerConnect").isEnabled(), false);
    assert.equal(await id("powerInterface").locator("option").count(), 3);
    await id("powerInterface").selectOption("fixture-two");
    await id("powerConnect").click();
    await expectText("powerMeterState", "Meter busy");
    await id("powerConnect").click();
    await expectText("powerMeterState", "Receiving");
    await expectText("powerVoltage", `${trace.at(-1).voltage_v.toFixed(3)} V`);
    await expectText("powerCurrent", `${trace.at(-1).current_ma.toFixed(1)} mA`);
    await expectText("powerWatts", `${trace.at(-1).power_w.toFixed(3)} W`);
    await expectText("powerSampleAge", "82 ms");
    await expectText("powerReportCadence", "4.0 reports/s");
    await expectText("powerCrc", "1");
    await expectText("powerGaps", "1");
    await expectText("powerTrendWindow", "Past 20 s");
    assert.equal(await id("powerVoltageTrend").isVisible(), true);
    assert.equal(await id("latestImage").isVisible(), true);

    await id("powerTopology").check();
    await expectText("powerTopologyStatus", "Confirmation applies when recording starts");
    await id("powerRecordStart").click();
    await expectText("powerTopologyStatus", "Operator-confirmed inline topology");
    await expectText("powerRecordingSummary", "Recording · fixture-run");
    await expectText("powerRecordingDuration", "2.5 s");
    await expectText("deviceVideoStatus", "Recording…");
    const frameBefore = await id("latestMeta").textContent();
    await page.waitForFunction((previous) => document.getElementById("latestMeta").textContent.split(" · ")[0] !== previous.split(" · ")[0], frameBefore);
    assert.equal(await id("powerVoltageTrend").isVisible(), true);
    await id("powerRecordStop").click();
    await expectText("powerRecordingSummary", "Saved · fixture-run");
    await id("powerCopyPath").click();
    await expectText("powerMeterAction", "Capture folder copied");
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), status.recording.output_dir);

    for (const [width, height] of [[1440, 900], [1280, 800], [390, 844]]) {
      await page.setViewportSize({ width, height });
      await id("powerMeter").scrollIntoViewIfNeeded();
      await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.equal(await id("powerMeter").evaluate((panel) => panel.scrollWidth <= panel.clientWidth), true);
      assert.equal(await id("powerVoltageTrend").isVisible(), true);
      if (output) {
        await fs.mkdir(output, { recursive: true });
        await page.screenshot({ path: path.join(output, `fixture-power-${width}x${height}.png`), animations: "disabled" });
      }
    }

    status = { ...status, state: "stale", sample_age_ms: 4500 };
    await expectText("powerMeterState", "Stale");
    await expectText("powerSampleAge", "4.5 s");
    status = { state: "error", message: "TEST FIXTURE · Device removed", recording: status.recording };
    await expectText("powerMeterState", "Meter error");
    await expectText("powerWatts", "Unavailable");
    assert.equal(await id("powerVoltageTrend").isVisible(), false);
    assert.equal(await id("powerTopology").isChecked(), false);
    await id("powerScan").click();
    await id("powerInterface").selectOption("fixture-two");
    await id("powerConnect").click();
    await expectText("powerMeterState", "Receiving");
    await id("powerDisconnect").click();
    await expectText("powerMeterState", "Meter available");
    await expectText("powerVoltage", "Unavailable");

    assert.deepEqual(actions, ["scan", "scan", "connect", "connect", "start-recording", "stop-recording", "scan", "connect", "disconnect"]);
    assert.ok(cameraFrames > 1);
    assert.deepEqual(pageErrors, []);
    console.log(`PASS · ${await browser.version()} · deterministic power UI fixtures at 1440x900, 1280x800 and 390x844`);
  } finally {
    await browser.close();
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
