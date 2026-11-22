# Windows FNIRSI adapter

This package provides a Windows-only acquisition worker with TensorFlow-free imports.

- capture_windows.py is the maintained board-bound version of the supplied logger. It requires fresh meter interface selection and, for active boards, a board label/reset ID plus firmware source, binary and configuration files. It hashes the selected files and does not invent a Nano sketch hash.
- instrument_adapter.py exposes the Windows lock, serial capture, raw-first decoder, bounded reconnect matching and steady-window estimator.
- steady_average.py preserves the supplied report-center diagnostic and never assigns timestamps to the four co-received values or integrates energy.
- upstream/ preserves original files and patches. upstream/scripts/log_fnirsi_windows.py contains the historical hardcoded ESP32 sketch-hash field; do not use it for new measurements. Use capture_windows.py.
- Existing failed/rejected traces and the full source packet remain in work/windows-fnirsi-results-import-2026-10-28/ as immutable evidence.

Windows hardware qualification is pending. These tools do not qualify physical meter identity, measurement boundary, sample timing, absolute accuracy or Nicla power. Python 3.14.9 HID/serial stays separate from TensorFlow training.
