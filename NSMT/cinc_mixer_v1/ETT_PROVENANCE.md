# ETT forecasting code copied from the other session (2026-10-31, exp/f-lif-pop-v3)

User decision (2026-10-31): the ETT forecasting wrapper of the other session is **copied with its provenance**, not imported
from the other tree and not rewritten. Source: branch `exp/nsmt-flif-neo`, commit `dcfdc4811`
(main tree `Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning`), extracted with `git show`; the
source tree was not modified. The first commit of these files is the verbatim copy; every later change is in later commits.

| file here (NSMT/cinc_mixer_v1/) | source path at dcfdc4811 | git blob | changed here |
|---|---|---|---|
| forecast_wrap.py | NSMT/nsmt_flif_v1/classification/forecast_wrap.py | 13795f87aa3b458619cbe151121bb59402080378 | yes (SpkMixer kind) |
| train_forecast.py | NSMT/nsmt_flif_v1/classification/train_forecast.py | 8c4ff92c5c91a06fe3f489c7c4933748d0f3dcb0 | no |
| energy_recount_ett.py | NSMT/nsmt_flif_v1/classification/energy_recount_ett.py | 48ac885616e9f299c2f9def6fda037e60deb807b | yes (SpkMixer kind, results path) |
| data_provider/augmentation.py | NSMT/nsmt_flif_v1/forecasting/data_provider/augmentation.py | 178d584f4b1110cb9d1986f87ad97adfb0ce3572 | no |
| data_provider/data_factory.py | NSMT/nsmt_flif_v1/forecasting/data_provider/data_factory.py | 24367ce2a5d7804c210cebf029716e679e21cbd7 | no |
| data_provider/data_loader.py | NSMT/nsmt_flif_v1/forecasting/data_provider/data_loader.py | 75b9bd5119a8c465ef653d1c930697fc863e5b32 | no |
| data_provider/download_data.py | NSMT/nsmt_flif_v1/forecasting/data_provider/download_data.py | c2a846e5131c88b86d44952256bba425f0e97d4f | no |
| data_provider/dtw.py | NSMT/nsmt_flif_v1/forecasting/data_provider/dtw.py | 941eae8e227c441da4e7f472e310d3f7bb999a5b | no |
| data_provider/m4.py | NSMT/nsmt_flif_v1/forecasting/data_provider/m4.py | 7a40d237ef6a9d3615558eda6dcf9a5c88b92990 | no |
| data_provider/timefeatures.py | NSMT/nsmt_flif_v1/forecasting/data_provider/timefeatures.py | 913ff08859a7a2f626a76953b183e4da9dfcb10b | no |
| data_provider/uea.py | NSMT/nsmt_flif_v1/forecasting/data_provider/uea.py | b43d26f81d0bbea0e8b3a1ec9fa2c267650195d7 | no |

Also used as templates (not copied verbatim): `NSMT/nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh` (blob
8b0a34926b9ff80464ad418f941eb59308bc304d) -> `scripts/run_ett_spk.sh`; the criterion of `NSMT/nsmt_flif_v1/classification/summarize_gen.py`
(blob 0ce13f7b235bcf58ad5ec746258ff8f764ed2f4c, lines 151-169) -> `summarize_ett_mux.py`.

`train_forecast.py` imports `data_provider` from its own folder first (the script folder is sys.path[0]); its extra
`sys.path.append('../forecasting')` finds no `data_provider` in this worktree. ETT csv files: `NSMT/forecasting/dataset/ETT-small/`
(a symlink of this worktree to the main tree's dataset folder; read only). `check_ett_mux.py` t0 re-checks the unchanged files
against the source blobs.
