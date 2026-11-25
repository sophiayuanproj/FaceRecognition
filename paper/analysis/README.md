# R19 secondary analysis

This directory audits the same ten qualified physical starts and does not change the original inclusion decisions. Obtain `Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip` using `tools/fetch_power_evidence.py` from the [project checkout](https://github.com/sophiayuanproj/FaceRecognition), or download it from the [research assets release](https://github.com/sophiayuanproj/FaceRecognition/releases/tag/research-assets-2026-11-22). Its SHA-256 is `66f9d14dd4b75b87d79d974cd28dc17492c5f3625db6edbc3255911e347e2356`.

Python 3.11 or newer is sufficient. From this paper directory, run:

```sh
python3 analysis/verify_paper_claims.py \
  --power-archive /path/to/Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip

python3 analysis/replay_secondary_results.py \
  /path/to/Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip \
  --output regenerated
```

- `gate-path-by-boot.csv` joins scene-trigger events to the accepted per-trial observations using the boot/frame pair. This excludes preroll. It reconciles trigger counts with the primary trial totals.
- `gate-first-match-by-episode.csv` looks up the first spatial face match for each scripted episode and reports the trigger reason that led to it. A no-match row has a blank reason and latency.
- `order-sensitivity-by-boot.csv` joins the accepted block ledger to the primary power estimates by attempt ID. Power saved is the fixed-minus-gated difference, so positive numbers favor gating.
- `secondary-summary.json` gives the aggregate values reported in the paper.

The trigger split does not measure what would happen without periodic checks. The bench repeated the same face segments. The order split retains five starts per subgroup and other workload positions vary within those starts. The summaries are exploratory.

## Raw meter replay

Extract the raw archive and run from its extracted root:

```sh
python3 resume/replay_belkin_primary.py \
  --runs resume --schedule prior-results/final-block-ledger.csv --expect-count 10
```

This independently decodes raw AA04 packets and integrates four voltage/current slot products per report. Expected mean gated-minus-fixed power is approximately `-0.04954353 W`. The accepted power schedule seeds are **62001–62007, 62009, 62010, 62012**, distinct from the ten model-training root seeds **20261101–20261110**.

## Regenerate plots

These scripts require Matplotlib in addition to Python. From the paper directory:

```sh
python3 analysis/make_figures.py \
  /path/to/Nicla_R19_Mac_Return_Belkin_10_Pairs_Final.zip --out regenerated
python3 analysis/make_host_figures.py \
  /path/to/project-checkout/work/project-finish-v4 --out regenerated
```

Both verify source hashes before deriving plot tables. The host figure command reads `handoffs/D/CANDIDATE_PAIRWISE_FINAL_ANALYSIS.json` and `reports/E/E_ROOT_METRICS.csv`; both are in the public checkout. The paper's `data/` directory stores the exact plotted rows. `verify_paper_claims.py --host-source /path/to/project-checkout/work/project-finish-v4` additionally verifies the host source digests. No source images are included in this paper package.
