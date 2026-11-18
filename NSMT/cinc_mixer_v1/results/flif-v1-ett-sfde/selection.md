# SF-de learning-rate selection (validation only)

- ETTh2: selected 0.0005; score (mean over H96 / H720 of the seed-mean best validation L1 relative to lr 5e-4): 0.0001 1.00359, 0.00025 1.00240, 0.0005 1.00000, 0.001 1.00447, 0.002 1.01058
- ETTm2: selected 0.0001; score (mean over H96 / H720 of the seed-mean best validation L1 relative to lr 5e-4): 0.0001 0.99713, 0.00025 0.99846, 0.0005 1.00000, 0.001 1.00045, 0.002 1.00384
- MIT-BIH: selected 0.01; seed-mean last-epoch validation macro-F1 on fold 0: 0.0025 91.655, 0.005 91.454, 0.01 92.193, 0.02 91.523

flif-v1-ett-sfde/selected_lr.json: {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
flif-v1-mitbih-sfde/selected_lr.json: {'MITBIH': '0.01'}
