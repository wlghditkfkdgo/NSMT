# Experiment workflow

- Keep `main` as the reproducible reference implementation. Do not commit experimental work directly to `main`, push experimental changes to `main`, or force-push it. Integrate a validated experiment into `main` only when the user explicitly requests that integration.
- Before a new experiment, create `exp/<experiment-name>` from the chosen reference commit (normally current `origin/main`). Record the base commit. Continue an existing experiment on its existing branch.
- Preserve pre-existing local changes; do not reset or discard them when creating branches.
- Maintain one canonical, append-only experiment record at `docs/PROJECT_LOG.md` at the Git repository root. `NSMT/docs/PROJECT_LOG.md` links to that same file. Append new dated entries and corrections; do not rewrite historical entries.
- Each experiment entry records its purpose, branch and base commit, code/config changes, dataset and splits/preprocessing, environment, exact commands, seeds, hyperparameters, metrics and artifact locations, checks, limitations, conclusion, and commit/tag identifiers. Use “not run” for checks or experiments that were not performed.
- At experiment completion, commit the code and log, and create an annotated tag `exp/<experiment-name>-<YYYYMMDD>` (add a suffix for repeated runs). An entry in the tagged commit may identify its own commit by the tag because the commit hash cannot be embedded in itself. Snapshot tags must be identified as snapshots, not completed training runs.
- Push only the explicitly intended experiment branch and tags. Never use `git push --all`, `--mirror`, or force when publishing an experiment.
- Put NSMT task outputs under `<model>/<task>/log/` and `results/`, shell scripts under `<model>/<task>/scripts/`, and queue/lock files under `scripts/queues/`. Shared utilities live in `NSMT/scripts/`.
- Keep datasets, checkpoints, caches, archives, and raw event/console logs local unless the user requests an artifact storage workflow. Preserve code, configuration, documentation, and text experiment results in Git. Never delete local artifacts just to make a commit.
