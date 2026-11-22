# Join the video trial to FNIRSI power

Use Python 3.9 or newer on Windows or macOS. The analyzer uses the standard library.

## Inputs

Pass the per-arm `RESULT.json` files produced by the video trial runner. A trial must have `measurement_ready: true` and finite `meter_window_start_unix_ms` and `meter_window_end_unix_ms` values from the render clock. `arm`, `physical_block_id` and `video_sha256` identify paired runs of the same clip across independent starts. Earlier receipts without a physical ID use their local `block` within one pilot. For optional episode integration, give `episodes` entries with `name`, `start_ms`, and `end_ms`, relative to the start of the same video window.

Supply the completed native FNIRSI logger's `decoded_samples.csv` with `--meter-csv`. A trial can instead name its own `meter_csv_path`, resolved relative to that trial JSON. If a CSV has more than one run ID, select `meter_run_id` in the trial or use `--meter-run-id`.

The FNIRSI `power-run-1.0` writer gives four rows per HID report. The analyzer reads `run_id`, `report_arrival_index`, `sample_index_in_report`, `host_receive_utc`, `host_receive_monotonic_ns`, `voltage_v`, `current_a`, and `crc_status`. It checks `power_w_calculated` against voltage times current when present. One timestamped point is the mean of the four V×I values. The device sample acquisition times are absent from this format.

## Clock and quality

Use `--same-host-clock` when the browser and logger ran on the same active operating-system clock. Windows and macOS runs after a reboot are separate clock observations. For separate clocks, supply a measured offset and its uncertainty.

```text
meter_clock_ms = render_clock_ms + meter_minus_render_ms
```

A per-trial clock receipt can override the command-line setting.

```json
{
  "clock_alignment": {
    "mode": "measured_offset",
    "meter_minus_render_ms": 12.5,
    "uncertainty_ms": 2.0,
    "method": "example only; replace with the actual clock comparison receipt"
  }
}
```

The report uses UTC for the join and trapezoidal integration. Windows monotonic timestamps can be coarser than UTC. The analyzer checks the UTC elapsed time against monotonic elapsed time to detect larger jumps. Every analyzed window must be bracketed by meter reports, including the offset-uncertainty envelope. The default report gap limit is 200 ms, and the default wall-versus-monotonic disagreement limit is 100 ms. Record any validated protocol-specific changes made with `--max-gap-ms` or `--max-clock-jump-ms`. Invalid trials carry a reason and a null energy value. The command exits with code 2 if any trial is invalid.

The endpoint energies in `clock_offset_endpoint_energy_j` show sensitivity to the two offset extremes. They are not a confidence interval. Episode joules include all load energy in that episode, including its idle component. Hardware topology, instrument qualification, and video frame onset accuracy come from the experiment's separate receipts.

## Run

One active host clock and a shared CSV:

```sh
python analyze_video_energy.py fixed/RESULT.json adaptive/RESULT.json \
  --meter-csv decoded_samples.csv --same-host-clock --baseline-arm fixed \
  --per-episode --output ENERGY.json
```

For Windows PowerShell:

```powershell
py -3 analyze_video_energy.py fixed/RESULT.json adaptive/RESULT.json --meter-csv decoded_samples.csv --same-host-clock --baseline-arm fixed --per-episode --output ENERGY.json
```

For separate clocks, replace `--same-host-clock` with these flags using the measured values:

```text
--clock-offset-ms OFFSET --clock-offset-uncertainty-ms UNCERTAINTY
--clock-offset-method "REFERENCE TO THE MEASURED CLOCK RECEIPT"
```

The report keeps input SHA-256 values, bracketing report IDs, largest contributing gaps, duration, time-weighted watts, and joules. Pairing uses one unique trial per arm in each physical block for the same video hash. The two arms must use the same meter run and meter CSV digest, board boot, release hashes, topology, seed, and planned order when present. Their meter windows must be disjoint. Check that each planned comparison appears and reports `paired` even when individual integration succeeds. Deltas are candidate minus baseline. Durations are recorded alongside each delta, and the summary averages joule deltas across pairs whose windows differ by less than 1 ms. Mean-power deltas use the full valid paired set.

## Synthetic verification

```sh
python test_analyze_video_energy.py -v
```

The fixture creates a linear power curve with four decoded samples per synthetic HID report. The two windows integrate to 0.060 and 0.030 J and produce a paired difference of −0.030 J. It also checks per-episode partitioning, measured offset sensitivity, and rejection of missing coverage, excessive report gaps, and a clock jump. Synthetic inputs and output use the explicit `--synthetic` scope. The archived Windows FNB-C2 commissioning CSV was parsed for format compatibility only.
