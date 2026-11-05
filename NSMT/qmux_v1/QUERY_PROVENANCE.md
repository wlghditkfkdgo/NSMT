# Query-form model code copied from the other session (2026-11-05, exp/f-lif-pop-v3)

User request (2026-11-04): apply the bidirectional muxlin Neo-mixer to the Neocortex path of [query pop LIF NSMT T=4]. That model exists
only on branch `exp/nsmt-flif-neo`, so its code is **copied with its provenance**, not imported from the other tree and not
rewritten (same practice as `cinc_mixer_v1/ETT_PROVENANCE.md`). Source: commit `90d9371e6ee54bcb11754cda850c3a4fac6d7f00`, extracted with `git show`; the source
tree was not modified. Files: the import closure of train_forecast.py, train.py, train_rhythm.py and energy_recount_ett.py, the
forecasting data provider, and the two runners used as templates. The first commit of this folder is the verbatim copy; every later
change is in later commits ("changed here" below lists them).

| file here (NSMT/qmux_v1/) | source path | git blob | changed here |
|---|---|---|---|
| bix.py | NSMT/nsmt_flif_v1/classification/bix.py | e24d3d46fa8cdedfb77cf1528eed491369bbe66c | no |
| config.py | NSMT/nsmt_flif_v1/classification/config.py | bd70fe5b572710cdd3f16eb80600b32dc47c4a51 | yes (--neo_mux options, run-folder suffix) |
| dataloader.py | NSMT/nsmt_flif_v1/classification/dataloader.py | 2ef8f4215dc3b2059b4a90291668e45d8b1e6e4c | no |
| energy.py | NSMT/nsmt_flif_v1/classification/energy.py | 3388f498fe675a9e13c7e5e95d2bcf285e872200 | no |
| energy_recount_ett.py | NSMT/nsmt_flif_v1/classification/energy_recount_ett.py | 48ac885616e9f299c2f9def6fda037e60deb807b | yes (qmx_ conditions, results path) |
| forecast_wrap.py | NSMT/nsmt_flif_v1/classification/forecast_wrap.py | 13795f87aa3b458619cbe151121bb59402080378 | no |
| layers.py | NSMT/nsmt_flif_v1/classification/layers.py | d847e2d65890420be42f1186fa0fe5ec8c631e1c | no |
| load_model.py | NSMT/nsmt_flif_v1/classification/load_model.py | 7698343073bc1f759dc4b384bd850fdddcf41a02 | yes (passes the --neo_mux options) |
| model.py | NSMT/nsmt_flif_v1/classification/model.py | aee3a1709375c0bf3a79110f9c2f32e43ea69195 | yes (time_block replaced when --neo_mux is on) |
| neo_mixer.py | NSMT/nsmt_flif_v1/classification/neo_mixer.py | 2057b114c586f0f265dda12ceb81420c90baf501 | no |
| neurons.py | NSMT/nsmt_flif_v1/classification/neurons.py | ae435608563874ea4b1da5025eddc726a644a12b | no |
| positional.py | NSMT/nsmt_flif_v1/classification/positional.py | e217468e10a9e18be253ba87aafd1bf62dcb9d86 | no |
| rhythm_data.py | NSMT/nsmt_flif_v1/classification/rhythm_data.py | 59e261b9ff7d34cadef4d217117b3e324321ef73 | no |
| rhythm_eval.py | NSMT/nsmt_flif_v1/classification/rhythm_eval.py | 7d860bffbbf9f66e68966fc2cd95db3c7c860097 | no |
| stream_mixer.py | NSMT/nsmt_flif_v1/classification/stream_mixer.py | 2c385f229e2f13bf7ae449abc465489a6cbbf8b7 | no |
| summarize_rhythm.py | NSMT/nsmt_flif_v1/classification/summarize_rhythm.py | 760a4912cfb9c359fd930c85c9826972ed0fa3a2 | no |
| test.py | NSMT/nsmt_flif_v1/classification/test.py | 4ecfee3cee7d812040eefe115177101078f05f0f | no |
| token_order.py | NSMT/nsmt_flif_v1/classification/token_order.py | 3c69abf265f9f4bf024c949edc60a194abacab1f | no |
| train.py | NSMT/nsmt_flif_v1/classification/train.py | f34576e11fc82fa885fdf415a9eedfc2268acc7b | no |
| train_forecast.py | NSMT/nsmt_flif_v1/classification/train_forecast.py | 8c4ff92c5c91a06fe3f489c7c4933748d0f3dcb0 | no |
| train_rhythm.py | NSMT/nsmt_flif_v1/classification/train_rhythm.py | 0cf4853381eb8160cb7921ebf9ccbb343c4e11f7 | no |
| utils.py | NSMT/nsmt_flif_v1/classification/utils.py | a7383b8d0326e32fc155678232cb35c548331e3b | no |
| data_provider/augmentation.py | NSMT/nsmt_flif_v1/forecasting/data_provider/augmentation.py | 178d584f4b1110cb9d1986f87ad97adfb0ce3572 | no |
| data_provider/data_factory.py | NSMT/nsmt_flif_v1/forecasting/data_provider/data_factory.py | 24367ce2a5d7804c210cebf029716e679e21cbd7 | no |
| data_provider/data_loader.py | NSMT/nsmt_flif_v1/forecasting/data_provider/data_loader.py | 75b9bd5119a8c465ef653d1c930697fc863e5b32 | no |
| data_provider/download_data.py | NSMT/nsmt_flif_v1/forecasting/data_provider/download_data.py | c2a846e5131c88b86d44952256bba425f0e97d4f | no |
| data_provider/dtw.py | NSMT/nsmt_flif_v1/forecasting/data_provider/dtw.py | 941eae8e227c441da4e7f472e310d3f7bb999a5b | no |
| data_provider/m4.py | NSMT/nsmt_flif_v1/forecasting/data_provider/m4.py | 7a40d237ef6a9d3615558eda6dcf9a5c88b92990 | no |
| data_provider/timefeatures.py | NSMT/nsmt_flif_v1/forecasting/data_provider/timefeatures.py | 913ff08859a7a2f626a76953b183e4da9dfcb10b | no |
| data_provider/uea.py | NSMT/nsmt_flif_v1/forecasting/data_provider/uea.py | b43d26f81d0bbea0e8b3a1ec9fa2c267650195d7 | no |
| scripts/run_ett_qmx.sh | NSMT/nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh | 40b49b8ceab6a3ff43460c4311b8cdc383d6e08f | yes (paths, qmx_ conditions, LR, RESULT exp) |
| scripts/run_mitbih_qmx.sh | NSMT/nsmt_flif_v1/classification/scripts/run_mitbih4.sh | 158d7b36cbdb9bfe109315dd4e3882dcdf5ba217 | yes (data path, nsmt4q_lif / flif, qmx_ conditions, RESULT exp) |

New here (not copied): `neo_mux.py`, `check_qmx.py`, `summarize_qmx.py`, `scripts/make_queue_qmx.sh`.
