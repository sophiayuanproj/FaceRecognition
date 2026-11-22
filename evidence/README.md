# Reproduce the power study

The full accepted and excluded R19 study records are attached to the repository's [research assets release](https://github.com/sophiayuanproj/FaceRecognition/releases/tag/research-assets-2026-11-22). They include raw meter reports and the replay scripts. They are optional for starting the dashboard.

From the repository root, download and verify the evidence.

```sh
python3 tools/fetch_power_evidence.py
python3 -m zipfile -e assets/Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip assets/power-evidence
cd assets/power-evidence
python3 resume/replay_belkin_primary.py --runs resume --schedule prior-results/final-block-ledger.csv --expect-count 10
```

Expected result is ten qualified pairs with mean fixed-to-scene-gated power change approximately `-0.04954353 W`. The graph and paired CSV are in `work/project-research-v5/final-readiness-2026-11-12/`.

The original raw capture ZIP is 41,332,455 bytes with SHA-256 `66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356`. The measured board release is identified by `bench/RELEASE_HASHES.json`. The physical battery boot rehearsal remains pending.
