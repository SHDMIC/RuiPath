# RuiPath

The pathology image self-supervised training code is located in `pretrain/`. It supports datasets consisting of individual image files or patches stored in HDF5 files, pretraining and GRAM-stage training, and optional OD-space stain perturbation. Dependencies are specified in [conda.yaml](pretrain/conda.yaml).

Run the commands below from the code directory:

```bash
cd pretrain
```

Replace every `/path/to/...` placeholder before running. Edit the supplied YAML files to configure training; the YAML snippets below show the fields to update, rather than complete configuration files.

## Dataset formats

### RuiPathH5Dataset

Use a CSV or TSV manifest with the following three columns. Files ending in `.tsv` use tabs; other manifest files use commas.

```csv
h5_path,patch_count,cum_sum
slide_a.h5,100,100
subdir/slide_b.h5,60,160
```

- `h5_path`: the HDF5 file path relative to `root`. Relative paths make it easier to relocate the data directory.
- `patch_count`: the number of samples contributed by this file, which does not have to equal its stored patch count.
- `cum_sum`: the cumulative sum of `patch_count` through the current row. The example above contains 160 samples.

Each HDF5 file must contain an array named `patches`. Use `uint8` RGB arrays with shape `[N, H, W, 3]`; patches are converted to RGB images when read. The actual patch count comes from `patches.shape[0]`.

When the sampling quota is smaller than the stored patch count, the dataset samples deterministically from stratified intervals. Larger quotas reuse patches. Selection depends on the file path in the manifest and the logical sample index, and does not change between epochs. Each sample returns `(image, target, sample_id)`, with a placeholder target of `0`.

Set the dataset in the training YAML:

```yaml
train:
  dataset_path: "RuiPathH5Dataset:meta_csv=/path/to/metadata.tsv:root=/path/to/data"
```

### RuiPathPlainDataset

Use a TXT file without a header or empty lines, with one image path per line:

```text
/path/to/images/patch_0001.png
/path/to/images/patch_0002.jpg
```

Images readable by PIL are converted to RGB. Absolute paths are recommended. Relative image paths are resolved against the command's working directory, rather than the TXT file's directory. Each sample returns `(image, target)`, with a placeholder target of `0`.

To use this dataset instead, update the training YAML:

```yaml
train:
  dataset_path: "RuiPathPlainDataset:meta_file=/path/to/images.txt"
```

## pretraining

Edit `dinov3/configs/ruipath/ruipath_local.yaml`:

```yaml
train:
  batch_size_per_gpu: 128
  dataset_path: "RuiPathH5Dataset:meta_csv=/path/to/metadata.tsv:root=/path/to/data"
```

The launch example uses three nodes with eight GPU processes per node. Run it on every node, setting `NODE_RANK` to `0`, `1`, or `2` respectively. Replace `MASTER_ADDR` with a reachable address of node 0. Use the same node count, process count, master address, and port on all nodes. Each node must have access to the configured data and the same shared output directory.

Set these launch variables in each node's shell:

```bash
export NNODES=3
export NPROC_PER_NODE=8
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
export MASTER_PORT=29500
```

Launch pretraining on every node:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config-file dinov3/configs/ruipath/ruipath_local.yaml \
  --output-dir /path/to/shared/outputs/ruipath_local
```

Adjust the node count, GPU process count, and per-GPU batch size for your run. With the values above, the global batch size is `128 × 8 × 3 = 3072`.

## GRAM stage

Edit `dinov3/configs/ruipath/ruipath_local_gram.yaml`. Its initial `gram.ckpt` is a placeholder, and `train.dataset_path` is `null`; both must be replaced. The example initializes the student from a pretraining teacher checkpoint and supplies an initial Gram teacher:

```yaml
train:
  dataset_path: "RuiPathH5Dataset:meta_csv=/path/to/metadata.tsv:root=/path/to/data"
student:
  resume_from_teacher_chkpt: /path/to/pretrain_teacher_checkpoint.pth
gram:
  ckpt: /path/to/gram_teacher_checkpoint.pth
  it_first_update: 10000
```

The two checkpoint fields can refer to the same exported pretraining teacher file, whose top-level dictionary contains a `teacher` entry.

This example starts an independent GRAM stage at iteration 0. It sets `gram.it_first_update` to 10000 for that stage's iteration count. The supplied configuration retains 1010000; that value would not trigger a Gram teacher update during an independent stage with the default 75000 iterations. Set this field to match your training schedule.

Using the same launch variables as above, run on every node:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config-file dinov3/configs/ruipath/ruipath_local_gram.yaml \
  --output-dir /path/to/shared/outputs/ruipath_gram
```

## SMEKA

### Enable stain perturbation

Both supplied RuiPath configurations leave stain perturbation disabled. To enable it in either stage, add the following block to that stage's YAML file:

```yaml
smeka:
  perturbation:
    enabled: true
    sigma: 0.15
    iters: 100
    lr: 0.005
```

Then use the corresponding multi-node launch command above. The perturbation acts on the student's second global crop. `sigma` controls stain concentration variation; `iters` and `lr` control the iterative stain decomposition.

## Export a teacher checkpoint

Here, `--eval-only` loads a training checkpoint and exports the teacher. It does not automatically compute downstream task metrics. By default, it selects the numerically largest checkpoint directory under the output directory's `ckpt/`; directories ending in `_keep` must be selected manually as described below.

Use the same model configuration as training. With the launch variables set on each node, run:

```bash
torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config-file dinov3/configs/ruipath/ruipath_local.yaml \
  --output-dir /path/to/shared/outputs/ruipath_local \
  --eval-only
```

The default exported file is `eval/manual_<iteration>/teacher_checkpoint.pth` under the output directory. To export a GRAM-stage model, use the GRAM configuration with a valid `gram.ckpt` and that stage's output directory.

To select a fixed training checkpoint, locate the `if args.eval_only:` branch in `dinov3/train/train.py` and change:

```python
last_ckpt = find_latest_checkpoint(ckpt_dir)
# last_ckpt = ckpt_dir / "29999_keep"
```

to:

```python
# last_ckpt = find_latest_checkpoint(ckpt_dir)
last_ckpt = ckpt_dir / "29999_keep"
```

Replace `29999_keep` with the actual directory name on every node, then run the same export command. This selects a training checkpoint directory under `--output-dir/ckpt/`, rather than an exported `.pth` file. Restore automatic selection when finished.

## License

The code in `pretrain/` is based on [DINOv3](https://github.com/facebookresearch/dinov3) and retains the [DINOv3 License](pretrain/LICENSE.md).
