# DimeNet++ Dense/Fastfood Wrapper Debug History

## Purpose
This note documents the full debugging path for the DimeNet++ intrinsic-dimension wrapper work: what failed, why it failed, what was changed, and why each change mattered. The goal is to preserve the reasoning chain, not just the final state.

## Baseline context
The work started from two separate implementations:
- ALIGNN in PyTorch, where the wrapper approach around the base model was already working well.
- DimeNet++ in TensorFlow, where the active implementation had drifted toward a reconstructed layer-by-layer projected model.

The practical goals were:
- recover the earlier speed for Fastfood runs,
- make dense runs practical again,
- preserve comparability with the intrinsic-dimension construction used in the Uber work,
- and support a full-dimension orthonormal sanity test.

## 1. Problem: Fastfood became much slower in the newer DimeNet++ run
### What was observed
An older Fastfood run completed in roughly 140 s/epoch after warmup, while the newer run was around 500 s/epoch.

### Why this was suspicious
That magnitude of slowdown was too large to attribute confidently to wrapper overhead alone. It suggested either a major device-placement change or a severe graph regression.

### What was checked
The older and newer log files were compared directly.
- Older run: TensorFlow created GPU 0 successfully.
- Newer run: TensorFlow reported `Visible GPUs: []` and `Cannot dlopen some GPU libraries`.

### Root cause
The newer run was not using the GPU at all. It had fallen back to CPU.

### Fix
The TensorFlow runtime inside `/venv/pydimnet` was checked against the available NVIDIA libraries. The required CUDA libraries were present inside the environment, but the env-local NVIDIA library directories were not on `LD_LIBRARY_PATH`.

### Why the fix mattered
This established that the timing regression was primarily an environment/runtime problem, not proof that the wrapper math had become fundamentally 4-5x slower.

## 2. Problem: TensorFlow in `pydimnet` could not see the GPU consistently
### What was observed
Repeated DimeNet++ runs printed:
- `Cannot dlopen some GPU libraries`
- `Skipping registering GPU devices...`
- `Visible GPUs: []`

### What was checked
TensorFlow build info, installed library paths, and the env-local NVIDIA package directories were inspected. The missing load path was traced to env-local CUDA/cuDNN/cuSOLVER libraries under `/venv/pydimnet/lib/python3.12/site-packages/nvidia/...`.

### Root cause
The `pydimnet` environment had the needed libraries, but the dynamic linker was not searching those directories.

### Fix
The runtime was launched with `LD_LIBRARY_PATH` including the relevant `nvidia/.../lib` directories inside `/venv/pydimnet`.

### Why the fix mattered
Without this, every TensorFlow benchmark or memory test was invalid because the code was silently running on CPU. This had to be fixed before any serious wrapper comparison.

## 3. Problem: The active TensorFlow implementation was not structurally similar to the working ALIGNN wrapper
### What was observed
The current DimeNet++ code in `dimenet_uber.py` reconstructed the model layer by layer, with projected layers pulling slices from a global flattened parameter object.

### Why this was a problem
This design is much heavier than the simple wrapper pattern used in ALIGNN. It adds more places for:
- graph overhead,
- repeated parameter reconstruction,
- silent mismatches with the original base model,
- and memory pressure from a more explicit projected graph.

It also made comparisons to the original wrapper-based logic less clean.

### Fix
A wrapper-based TensorFlow path was reintroduced as:
- `wrapper_tensorflow_v2.py`
- `dimenet_run_v2.py`

This path keeps the base DimeNet++ model intact and updates its weights from `theta(z)` in the wrapper.

### Why the fix mattered
This brought the TensorFlow design closer to the simpler wrapper logic used in the PyTorch ALIGNN code, reducing architectural confounding when comparing behavior.

## 4. Problem: Dense 100% in TensorFlow was failing due to large initialization-time memory spikes
### What was observed
The dense `100%` TensorFlow path OOMed during initialization. The failure involved a `26.09 GiB` allocation and the allocator state showed multiple giant chunks already present.

### Initial interpretation
At first glance it seemed that a 96 GB GPU should have been enough, because a single dense `D x D` FP32 matrix is about 26 GiB when `D ≈ 83,685`.

### What deeper inspection showed
The issue was not just the size of the final stored matrix. The initialization path was creating large temporary tensors in addition to the final stored projection. In practice, the path was closer to:
- build a giant dense random matrix,
- normalize or transform it,
- copy again into a tracked TensorFlow variable,
- and hold multiple large allocations long enough to exceed practical headroom.

### Fix
For the non-orthonormal dense case, the projection was changed to an exact blockwise dense Gaussian storage scheme in `v2`.
- The distribution stayed the same.
- The final operator stayed dense.
- But the matrix was built and stored in column blocks instead of in one huge one-shot allocation.

### Why the fix mattered
This reduced initialization-time peak memory while preserving the intended dense Gaussian projection. It is the main reason dense `100%` non-orthonormal became workable.

## 5. Problem: Keras was counting the wrong parameters as trainable in `v2`
### What was observed
The model summary showed trainable parameter counts larger than expected. For full-dimensional runs, the wrapper should only train `z`, but the summary initially indicated that the base model parameters were still being treated as trainable too.

### Why this mattered
If Keras still treated the base model variables as ordinary trainable variables, then:
- the parameter accounting was misleading,
- optimizer state could be misunderstood,
- and the wrapper would no longer cleanly reflect the intended intrinsic-dimension optimization setup.

### Fix
The wrapper was adjusted so that its public trainable-variable interface exposed only `z`, while still using the base model weights internally to compute projected gradients.

### Why the fix mattered
This restored the correct conceptual model: the optimization lives in intrinsic coordinates `z`, not in the full ambient parameter space.

## 6. Problem: Training logs showed `loss: 0.0` and `val_loss: 0.0`
### What was observed
Runs appeared to complete with zero loss, which was clearly implausible for this task.

### What was checked
The cached target tensors were inspected and verified to be nonzero and nontrivial. This ruled out a degenerate dataset issue.

### Root cause
The custom wrapper training loop was not interacting cleanly with Keras’ default loss/metrics bookkeeping. The forward and backward pass could run, but the reported loss values were not trustworthy.

### Fix
The wrapper was changed to track loss explicitly using a dedicated `tf.keras.metrics.Mean` loss tracker and direct loss evaluation inside `train_step` and `test_step`.

### Why the fix mattered
This converted the run logs back into something interpretable. Without reliable loss reporting, successful execution would still not count as a meaningful training result.

## 7. Problem: Exact dense orthonormal `100%` still failed even after the non-orthonormal dense fix
### What was observed
The orthonormal branch stalled or OOMed, depending on whether QR was run on GPU or forced to CPU.

### Why this happened
Exact orthonormal dense `100%` means:
- build a full Gaussian matrix `A ∈ R^(D×D)`, then
- compute `Q` from QR, then
- store `Q` as the frozen projection.

For `D ≈ 83,685`, this is a very large linear algebra job. The problem is not only storage of the final matrix. Exact QR needs additional working memory and heavy compute.

### Important comparison against ALIGNN/PyTorch
The working ALIGNN/PyTorch code did **not** perform QR in the `d == D` case. At full dimension, it used:
- column-normalized dense Gaussian `A`, or
- the `full_rotation` permutation/sign-flip fallback.

QR in the PyTorch ALIGNN code was only used when `d != D` and `orthonormal=True`.

### Why this mattered
It clarified that the working TensorFlow dense `100%` path already matched the practical ALIGNN full-dimensional dense case more closely than the exact orthonormal QR path did.

## 8. Problem: CPU exact QR was theoretically possible but too slow for iteration
### What was observed
With QR forced off GPU, the process appeared to hang for a long time before training even started.

### Why this happened
Host RAM was large enough, but exact `QR(D×D)` at this size is still a very expensive CPU computation. Avoiding GPU OOM did not make the underlying QR cheap.

### Fix attempt
A separate CPU-based exact QR route was kept as a fallback option, but it was recognized as too slow to be the primary workflow.

### Why this mattered
This ruled out CPU QR as the practical default for repeated experimentation, even though it remained theoretically faithful.

## 9. Problem: External PyTorch QR on GPU still failed the first time
### What was observed
A first attempt was made to let a separate PyTorch process build the orthonormal `Q` and then hand the finished matrix to TensorFlow. That first attempt still OOMed.

### Root cause
TensorFlow had already initialized GPU memory before the external PyTorch QR helper ran. So PyTorch was trying to allocate for QR with reduced free VRAM.

### Fix
A new `v3` path was introduced. In `dimenet_run_v3.py`, the external orthonormal `Q` is precomputed and cached **before TensorFlow imports and initializes the GPU**.

### Why the fix mattered
Initialization order turned out to be decisive. Once PyTorch got first access to the GPU memory, exact GPU QR became feasible.

## 10. Final working result: exact orthonormal 100% through `v3`
### What was observed
With the `v3` ordering fix, the external PyTorch QR on GPU succeeded:
- the full `Q` was generated,
- moved to CPU,
- saved to disk,
- and then TensorFlow started afterward.

### What was changed
The exact orthonormal path is now:
1. launch `dimenet_run_v3.py`,
2. precompute `Q` externally through `torch_orthonormal_q_v3.py`,
3. cache `Q` under `orthonormal_q_cache_v3`,
4. initialize TensorFlow afterward,
5. load the cached `Q` into the wrapper and train.

### Why this mattered
This preserved the exact Gaussian+QR orthonormal construction while making it practical on the available hardware.

## Summary of what now works
### Fastfood
Working, provided the `pydimnet` environment is launched with the correct `LD_LIBRARY_PATH` so TensorFlow sees the GPU.

### Dense 100% non-orthonormal
Working through the `v2`/`v3` blockwise exact dense Gaussian storage path.

### Dense 100% orthonormal
Working through `v3`, where exact QR is computed externally in PyTorch before TensorFlow initializes the GPU.

## Main lessons
1. The largest timing regression was caused by silent CPU fallback, not just wrapper math.
2. For dense full-dimensional experiments, initialization order and temporary tensors matter as much as final matrix size.
3. The working ALIGNN full-dimensional dense path was not using exact QR, so comparisons had to distinguish between:
   - dense full dimension,
   - orthonormal subspace QR,
   - and full-rotation orthogonal surrogates.
4. Exact orthonormal `100%` was only made practical once the QR step was moved into a separate process that touched GPU memory before TensorFlow.

## Files involved
- `data_saving_formation_energy.py`
- `dimenet_run_v2.py`
- `wrapper_tensorflow_v2.py`
- `run_dense_sweep_v2.sh`
- `run_fastfood_sweep_v2.sh`
- `torch_orthonormal_q.py`
- `dimenet_run_v3.py`
- `wrapper_tensorflow_v3.py`
- `torch_orthonormal_q_v3.py`
- `run_dense_sweep_v3.sh`
- `run_fastfood_sweep_v3.sh`
- `run_dense_orthonormal_sweep_v3.sh`
