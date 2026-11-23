# Transolver-Experiments: Accuracy vs. Cost of a Lightweight DeepONet on ShapeNet-Car

Exploratory proof of concept (Assignment 5) building on my CyberShuttle AI4Science
Foundry contribution. The reproducible Transolver inference workflow lives in a separate
repo: [Magical452/Transolver-Car](https://github.com/Magical452/Transolver-Car)
(trained weights and inference code).

**Question:** how much of Transolver's accuracy on ShapeNet-Car airflow prediction can a
much simpler DeepONet retain, and what does it save in inference time, memory, and input
requirements?

This is experimental code, not merge-ready. It is organized so another researcher can
understand and attempt to re-run it.

## Variants tested

| Run | Branch input | Training target |
|---|---|---|
| `deeponet` | subsampled, x-sorted surface point cloud (2048 pts) | ground truth |
| `deeponet --distill --distill_alpha 0` | same | Transolver predictions |
| `deeponet --distill --distill_alpha 0.5` | same | 0.5 × ground truth + 0.5 × Transolver |
| `transolver` | n/a (evaluate trained checkpoint only) | n/a |
| `hybrid --embed slice` | frozen Transolver slice tokens | ground truth |
| `hybrid --embed meanmax` | frozen Transolver mean+max pooled features | ground truth |
| `hybrid --embed slice` (100-epoch encoder) | frozen slice tokens, earlier checkpoint | ground truth |

Distillation target: `α · y_true + (1 − α) · y_teacher` (α = 0 is pure distillation).
All DeepONet variants share the same trunk (Fourier-feature MLP on the 7-d node input
[position, SDF, normal]) and output 3 velocity components + pressure. Loss is MSE on
velocity plus `reg × MSE` on surface pressure (`reg = 1.0`).

## Results (fold 0, 100 held-out cars, single seed, 1× V100)

| Model | Pressure err | Speed err | Drag err | Spearman ρ (drag) | GPU ms/car | CPU ms/car | Extra GPU mem | Trainable params |
|---|---|---|---|---|---|---|---|---|
| Transolver (200 ep) | 0.0773 | 0.0207 | 0.0135 | 0.992 | 58.4 | 665 | 223 MB | 3.86M |
| DeepONet (plain) | 0.307 | 0.101 | 0.215 | 0.274 | 3.6 | 34.5 | 128 MB | 3.72M |
| DeepONet, distilled α=0 | 0.304 | 0.100 | 0.211 | 0.281 | 3.0 | 33.8 | 128 MB | 3.72M |
| DeepONet, distilled α=0.5 | 0.308 | 0.102 | 0.216 | 0.244 | 3.0 | 33.7 | 128 MB | 3.72M |
| Hybrid, slice (enc 200) | 0.137 | 0.067 | 0.027 | 0.961 | 61.7 | 656 | 223 MB | 4.77M (+3.86M frozen) |
| Hybrid, mean/max (enc 200) | 0.144 | 0.066 | 0.031 | 0.955 | 57.8 | 646 | 285 MB | 0.82M (+3.86M frozen) |
| Hybrid, slice (enc 100) | 0.139 | 0.067 | 0.033 | 0.953 | 59.4 | 640 | 223 MB | 4.77M (+3.86M frozen) |

Errors are relative L2 (pressure, velocity) and relative drag-coefficient error. CPU timing
uses 10 cars. Training takes about 2 min per DeepONet or hybrid run. The full job log is in
`logs/` and per-run JSON results are in `Car-Design-ShapeNetCar/results_poc/`.

Key findings: the DeepONet is about 16–20× faster per car with about 43% less inference
memory, but about 4× (pressure) to 16× (drag) less accurate, and distillation did not help.
Parameter counts are nearly equal (3.72M vs 3.86M), so the savings come from architecture,
not size. See the report for discussion and limitations.

## Repository layout

```
Car-Design-ShapeNetCar/
  hybrid_deeponet.py     # all models, training, evaluation, and the 7-run suite
  results_poc/           # per-run result JSON files
jobs/
  poc.sbatch             # Slurm job script (PACE ICE)
  submit_poc.sh          # submits the job with the required GPU constraint
logs/
  poc-suite_7246417.out  # job log for the results above
POC_README.md            # original run notes
```

The remaining files are the upstream
[thuml/Transolver](https://github.com/thuml/Transolver) code that `hybrid_deeponet.py` builds on.

## Environment

- Georgia Tech PACE ICE, partition `ice-gpu`, 1× Tesla V100 (16 GB), 8 CPUs, 32 GB RAM
- Python 3.8, PyTorch [version], timm [version], CUDA driver 13.0
- [Dependency file or install command, e.g. `pip install -r requirements.txt`]

## Data and checkpoints (not included in this repo)

1. **ShapeNet-Car** (`mlcfd_data/training_data`): download and preprocess following the upstream
   Transolver instructions (about 1.5 h one-time preprocessing). Set `--data_dir` and `--save_dir`.
2. **Transolver checkpoints** (needed for the `transolver`, `hybrid`, and distillation runs):
   `metrics/Transolver/0/200_0.5/model_200.pth` and `metrics/Transolver/0/100_0.5/model_100.pth`.
   Weights from my earlier training runs are in
   [Magical452/Transolver-Car](https://github.com/Magical452/Transolver-Car). Place them at
   those paths relative to `Car-Design-ShapeNetCar/`.

## Reproducing the experiment

On PACE (the `-C` constraint matters: a plain GPU request once gave me an AMD GPU, and PyTorch
saw no CUDA):

```bash
cd jobs
bash submit_poc.sh
```

Then check the job log early for `cuda available: True`. The job runs all 7 variants in
priority order and saves each result as it finishes. The full suite took about 31 minutes
(about 3 min of data loading plus about 1 min per variant of setup and training).

To run a single variant [verify flags against `--help`]:

```bash
cd Car-Design-ShapeNetCar
python hybrid_deeponet.py --model deeponet --epochs 40 \
    --data_dir <path>/mlcfd_data/training_data --save_dir <path>/preprocessed_data
```

Edit the `data_dir`/`save_dir` paths in `poc.sbatch` for your own environment; the ones in
the included log are specific to my account.

## Known issues and limitations

- Single seed and a single split (fold 0): small differences between variants are not conclusive.
- Final-epoch metrics are reported; no checkpoint selection.
- The plain DeepONet's validation pressure error rises as it trains (0.249 at epoch 10 → 0.307
  at epoch 40) while training loss falls, which suggests overfitting.
- [Note whether Transolver's per-car time includes input feature computation.]
- The 1.5 h Transolver preprocessing is not included in the timings.
- `POC_README.md` may mention a 107-car test set; the correct number is 100.

## References

- Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024.
- Lu et al., *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence, 2021.
