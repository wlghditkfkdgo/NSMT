# Look-back (input length) in SNN forecasting papers with Spikformer-type models (read 2026-11-19, exp/f-lif-pop-v3)

Read from the PDFs with `pdftotext -layout` (PDFs kept local, not committed). Quotes are verbatim; page = printed page number.

## SpikF (Wu, Huo, Chen, ICML 2025, PMLR 267:67342-67368; PDF github.com/mlresearch/v267 assets/wu25m/wu25m.pdf)
- Table 1 caption (p. 6): "All baseline models are configured with a uniform look-back window of L = 96 timesteps. Performance metrics are averaged over H = {96, 192, 336, 720} prediction lengths."
- Appendix B.1: "We fix the look-back window for all datasets except Solar-energy to 96, consistent with iTransformer (Liu et al., 2024) for long-term prediction tasks. The look-back window of Solar-energy is set to 128, following (Lv et al., 2024b)."
- Sec. 4.3 "Extension of Input Sequence": "we modify the look-back window across five temporal spans: 48, 96, 192, 336, and 720 time steps ... when the length of input sequence expands from 48 to 720, an averaged prediction promotion is 9.0% in terms of MSE and 2.4% in terms of MAE". Figure 5 caption: "MSE and MAE values for the ETTh2 (top) and ETTm2 (bottom) datasets across varying input and output lengths." (SpikF only; no Spikformer / iSpikformer at other lengths.)
- Appendix E.2, Table 12 "More benchmarks on ETT dataset" (SpikF, SpikeRNN, iSpikformer, FITS, FEDformer; "SNN-based models are equipped with SPE"), setting of B.1 (look-back 96). iSpikformer MSE: ETTh2 H96 0.308, H720 0.438; ETTm2 H96 0.178, H720 0.406. SpikF: ETTh2 0.290 / 0.416; ETTm2 0.175 / 0.405.
- Official code (github.com/WWJ-creator/SpikF, HEAD 041e398): `scripts/run_ETTh2.sh` uses `for seq_len in 96`. The local copy /home/yschoi/SpikF also has untracked `scripts/*iSpikformer*.sh` and `test_results/*_sl336_*` (seq_len 336); `git ls-files test_results` is empty, so these are local runs, not upstream.

## SeqSNN (Lv et al., ICML 2024, PMLR 235:33624-33637, arXiv 2402.01533; iSpikformer = Spikformer SSA on iTransformer)
- Datasets: Metr-la, Pems-bay, Solar, Electricity (no ETT). Appendix A.1 Table 4 "The statistics of datasets" column "Length": Metr-la 12, Pems-bay 12, Solar-energy 168, Electricity 168.

## Spikformer (Zhou et al., ICLR 2023)
- Image classification model; no look-back. Time-series versions are adaptations (iSpikformer above; this branch's Spikformer-style / SF-de).
