/*
 * Nano meter diagnostic workload, version 1.1.0.
 * Diagnostic USB load/serial traffic only; not Nicla firmware.
 * Commands: PING <token>, SEED <unsigned-integer>, START, HELP
 * Waits 30 s for host SEED/START commands, then auto-starts without a host.
 */
#include <Arduino.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

static const char *FW_VERSION = "nano-meter-diagnostic-1.1.0";
static const uint32_t AUTO_START_MS = 30000UL;
static const uint32_t PHASE_MS = 5000UL;
static const uint32_t SETTLE_MS = 1000UL;
static const uint32_t HEARTBEAT_MS = 1000UL;
static const size_t LINE_CAPACITY = 80;

static uint32_t rootSeed = 1;
static uint32_t rngState = 1;
static uint32_t sequenceNumber = 0;
static uint32_t bootTick = 0;
static uint32_t lastHeartbeat = 0;
static uint32_t cpuState = 0x12345678UL;
static bool experimentStarted = false;
static bool experimentDone = false;
static bool startRequested = false;
static char inputLine[LINE_CAPACITY];
static size_t inputLength = 0;
static bool inputOverflow = false;

struct Phase {
  const char *name;
  bool cpuActive;
  bool ledOn;
};

static Phase phases[] = {
  {"idle_led_off", false, false},
  {"active_led_off", true, false},
  {"idle_led_on", false, true},
  {"active_led_on", true, true},
};

static void emitEvent(const char *kind, const char *detail = "") {
  if (Serial.availableForWrite() < 32) return;
  Serial.print("EV,");
  Serial.print(sequenceNumber++);
  Serial.print(',');
  Serial.print((uint32_t)millis());
  Serial.print(',');
  Serial.print(kind);
  if (detail[0] != '\0') {
    Serial.print(',');
    Serial.print(detail);
  }
  Serial.println();
}

static void emitBanner() {
  if (Serial.availableForWrite() < 48) return;
  Serial.print("BOOT,");
  Serial.print(FW_VERSION);
  Serial.print(",seed,");
  Serial.print(rootSeed);
  Serial.print(",baud,115200,auto_start_ms,");
  Serial.print(AUTO_START_MS);
  Serial.println(",phase_ms,5000");
}

static uint32_t nextRandom() {
  uint32_t x = rngState ? rngState : 1UL;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rngState = x;
  return x;
}

static bool parseUnsigned(const char *text, uint32_t *out) {
  if (!text || !*text) return false;
  char *end = NULL;
  errno = 0;
  unsigned long value = strtoul(text, &end, 10);
  if (errno == ERANGE || *end != '\0' || end == text || value > 0xFFFFFFFFUL) return false;
  *out = (uint32_t)value;
  return true;
}

static void handleLine(char *line) {
  if (strcmp(line, "PING") == 0 || strncmp(line, "PING ", 5) == 0) {
    const char *token = (line[4] == ' ') ? line + 5 : "";
    if (Serial.availableForWrite() >= 24) {
      Serial.print("PONG,");
      Serial.print(sequenceNumber++);
      Serial.print(',');
      Serial.print((uint32_t)millis());
      Serial.print(',');
      Serial.println(token);
    }
    return;
  }

  if (strncmp(line, "SEED ", 5) == 0) {
    uint32_t candidate = 0;
    if (!experimentStarted && parseUnsigned(line + 5, &candidate)) {
      rootSeed = candidate ? candidate : 1UL;
      rngState = rootSeed ^ 0x9e3779b9UL;
      cpuState = rootSeed ^ 0x9e3779b9UL;
      emitEvent("seed_set", line + 5);
      emitBanner();
    } else {
      emitEvent("seed_rejected", experimentStarted ? "already_started" : "invalid_value");
    }
    return;
  }

  if (strcmp(line, "START") == 0) {
    startRequested = true;
    emitEvent("start_requested");
    return;
  }

  if (strcmp(line, "HELP") == 0) {
    emitEvent("help", "PING [token]; SEED integer before start; START");
    return;
  }
  emitEvent("unknown_command");
}

static void serviceSerial() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (inputOverflow) {
        emitEvent("serial_line_rejected", "too_long");
      } else {
        inputLine[inputLength] = '\0';
        if (inputLength > 0) handleLine(inputLine);
      }
      inputLength = 0;
      inputOverflow = false;
    } else if (inputLength + 1 < LINE_CAPACITY && !inputOverflow) {
      inputLine[inputLength++] = c;
    } else {
      inputOverflow = true;
    }
  }
}

static void emitHeartbeat() {
  const uint32_t now = (uint32_t)millis();
  if ((uint32_t)(now - lastHeartbeat) >= HEARTBEAT_MS) {
    lastHeartbeat = now;
    if (Serial.availableForWrite() >= 32) {
      Serial.print("STAT,");
      Serial.print(sequenceNumber++);
      Serial.print(',');
      Serial.print(now);
      Serial.print(",seed,");
      Serial.print(rootSeed);
      Serial.print(",running,");
      Serial.println(experimentStarted && !experimentDone ? 1 : 0);
    }
  }
}

static void runPhase(const Phase &phase) {
  char detail[48];
  snprintf(detail, sizeof(detail), "seed=%lu,phase=%s,settle_ms=%lu",
           (unsigned long)rootSeed, phase.name, (unsigned long)SETTLE_MS);
  digitalWrite(LED_BUILTIN, phase.ledOn ? HIGH : LOW);
  emitEvent("phase_start", detail);

  const uint32_t phaseStart = (uint32_t)millis();
  uint32_t lastProgress = phaseStart;
  while ((uint32_t)((uint32_t)millis() - phaseStart) < PHASE_MS) {
    serviceSerial();
    emitHeartbeat();
    const uint32_t now = (uint32_t)millis();
    if ((uint32_t)(now - lastProgress) >= SETTLE_MS) {
      lastProgress = now;
      emitEvent("phase_progress", phase.name);
    }
    if (phase.cpuActive) {
      cpuState = cpuState * 1664525UL + 1013904223UL;
      cpuState ^= (cpuState >> 13);
    } else {
      delay(5);
    }
  }
  digitalWrite(LED_BUILTIN, LOW);
  snprintf(detail, sizeof(detail), "seed=%lu,phase=%s,cpu_state=%lu",
           (unsigned long)rootSeed, phase.name, (unsigned long)cpuState);
  emitEvent("phase_end", detail);
}

static void runExperiment() {
  experimentStarted = true;
  uint8_t order[4] = {0, 1, 2, 3};
  for (int i = 3; i > 0; --i) {
    const int j = (int)(nextRandom() % (uint32_t)(i + 1));
    const uint8_t tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  char orderText[64];
  snprintf(orderText, sizeof(orderText), "%s|%s|%s|%s",
           phases[order[0]].name, phases[order[1]].name,
           phases[order[2]].name, phases[order[3]].name);
  emitEvent("experiment_start", orderText);
  for (uint8_t i = 0; i < 4; ++i) runPhase(phases[order[i]]);
  digitalWrite(LED_BUILTIN, LOW);
  experimentDone = true;
  emitEvent("experiment_end", "four randomized phases complete");
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  bootTick = (uint32_t)millis();
  cpuState = rootSeed ^ 0x9e3779b9UL;
  rngState = rootSeed ^ 0x9e3779b9UL;
  emitBanner();
  emitEvent("ready", "seed/start window is 30 s; autonomous auto-start remains enabled");
}

void loop() {
  serviceSerial();
  emitHeartbeat();
  const uint32_t elapsed = (uint32_t)((uint32_t)millis() - bootTick);
  if (!experimentStarted && (startRequested || elapsed >= AUTO_START_MS)) {
    runExperiment();
  }
}


