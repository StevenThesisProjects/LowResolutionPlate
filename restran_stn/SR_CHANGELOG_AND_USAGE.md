# Super Resolution integration for ResTranOCR + STN

## What changed

I added a lightweight super-resolution preprocessing step to the image pipeline so frames are enhanced **after** resize to model resolution (32×128) and **before** normalization. A separate **learnable deblur block** (`LearnableDeblurBlock`) can run inside the model before STN when `USE_LEARNABLE_SR=True`.

### Design choice
I applied SR **before both training and testing** instead of only one side.

Reasoning:
- The model benefits from seeing the same image enhancement at train and test time.
- For OCR, consistency matters more than using SR only during inference.
- The pipeline already works on low-resolution license plate crops, so a deterministic enhancement step is safer than a heavy learnable SR model that would require extra training data and more compute.

### SR method chosen
I used a **lightweight OpenCV-based enhancement** instead of a separate deep SR network.

This is a practical fit for `ResTranOCR + STN` because:
- it does not change the model architecture,
- it is fast enough for multi-frame OCR,
- it preserves the STN behavior,
- it is easy to turn off with a flag,
- it avoids introducing a second large network that may be hard to train jointly.

The SR step does:
- bicubic upsampling,
- bilateral filtering for noise reduction,
- sharpening to emphasize character edges.

## Files updated

- `configs/config.py`
  - added `USE_SR` and `SR_SCALE`
- `src/data/transforms.py`
  - added `apply_lightweight_sr()`
- `src/data/dataset.py`
  - applies SR after resize, before albumentations normalize
- `src/models/components.py`
  - added `LearnableDeblurBlock` (optional, before STN)
- `src/models/restran.py`
  - `use_learnable_sr` flag wires deblur into forward pass
- `train.py` / `predict.py`
  - added `--no-sr`, `--no-learnable-sr`
  - passes SR settings into the dataset and model

## Why this approach is suitable for ResTran + STN

`STN` is strongest when it receives cleaner, more legible plate crops. The SR preprocessing helps the STN by enhancing edges before alignment, while `ResTran` receives frames with improved character structure.

A separate deep SR model such as EDSR/Real-ESRGAN would be heavier and usually not worth the overhead for this OCR pipeline unless you want to train and benchmark an extra stage. For this project, a lightweight SR step is the better default.

## How to run training

### What `python train.py` uses by default

If you run:

```bash
python train.py
```

then the training will use the default values from `configs/config.py`, unless you override them with CLI flags.

Current defaults are:
- `experiment_name`: `restran34_with_stn`
- `epochs`: `1`
- `batch_size`: `64`
- `learning_rate`: `5e-4`
- `data_root`: `dataset/data/train`
- `seed`: `42`
- `num_workers`: `10`
- `transformer_heads`: `8`
- `transformer_layers`: `3`
- `aug_level`: `full`
- `output_dir`: `results`
- `use_stn`: `True`
- `use_sr`: `True`
- `use_learnable_sr`: `True`
- `sr_scale`: `2`

So `python train.py` is equivalent to training with:

```bash
python train.py \
  --output-dir results
```

plus all the default config values above.

### Customizable training command

You can change the important training settings directly from the command line:

```bash
python train.py \
  --experiment-name my_experiment \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.0003 \
  --data-root dataset/data/train \
  --seed 42 \
  --num-workers 8 \
  --transformer-heads 8 \
  --transformer-layers 3 \
  --aug-level full \
  --output-dir results \
  --submission-mode
```

### Useful flags

- `--no-stn`: disable Spatial Transformer Network
- `--no-sr`: disable OpenCV super-resolution preprocessing
- `--no-learnable-sr`: disable learnable deblur block in the model
- `--submission-mode`: train on full data and generate predictions for test set after training
- `--aug-level light`: use lighter augmentation instead of the full pipeline

### Examples

Default run with SR enabled:

```bash
python train.py
```

Disable SR if you want a baseline comparison:

```bash
python train.py --no-sr
```

If you also want to disable STN:

```bash
python train.py --no-stn
```

Custom run with more epochs and a different learning rate:

```bash
python train.py --epochs 50 --batch-size 32 --lr 0.0001 --experiment-name restran_sr_exp
```

Submission mode example:

```bash
python train.py --submission-mode
```

## How to run inference / test

Run test inference with a checkpoint:

```bash
python predict.py --checkpoint results/restran34_with_stn_best.pth --mode test
```

Validation mode:

```bash
python predict.py --checkpoint results/restran34_with_stn_best.pth --mode val --data-root dataset/data/train
```

Disable SR for inference:

```bash
python predict.py --checkpoint results/restran34_with_stn_best.pth --mode test --no-sr
```

## Notes

- Keep SR enabled during both training and testing for the best consistency.
- If you train with `--no-sr`, then test with `--no-sr` as well.
- If you use a different checkpoint trained before this change, try matching the same SR/STN settings used during training.

## Suggested experiment plan

1. Baseline: `--no-sr`
2. SR only on both train/test: default
3. SR + STN together: default with `USE_STN=True`
4. SR without STN: `--no-stn`

This will help measure whether SR actually improves plate OCR accuracy in your data.
