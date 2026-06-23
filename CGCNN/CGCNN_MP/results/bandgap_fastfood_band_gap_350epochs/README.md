# CGCNN MP band gap Fastfood results

350-epoch CGCNN band-gap Fastfood random-subspace sweep on the Materials Project dataset
(Adam, `main.py` cached-MP path).

- `summary_mae_runtime.csv` — validation/test MAE (eV) and runtime for every
  dimension/seed run (18 runs).
- `predictions/` — per-crystal test-set predictions for each run, one CSV per run, named by
  the run directory (`cgcnn_mp_bandgap_fastfood_dim<pct>_dataseed<split>_modelseed<model>`).
- `logs/` — the training log for each run (`..._train.log`).

Intrinsic-dimension fractions swept: 100, 80, 60, 50, 45, 20, 10, 5 %. Seeds: model 123/456
(+789 for 45 % and 60 %), split 1123/1456/1789. Model: n_conv=4, atom_fea_len=56,
h_fea_len=64, n_h=3; batch size 256.
