# Vietnamese Music Classification

Classification of five Vietnamese traditional music genres using a convolutional neural network and 30-second audio spectrograms. The project compares **Log-STFT** and **Log-Mel** inputs with the same CNN2D architecture, fixed dataset split, and three training seeds.

The five classes are **Cải lương**, **Ca trù**, **Chầu văn**, **Chèo**, and **Hát xẩm**.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [CheckDTS.ipynb](CheckDTS.ipynb) | Audit audio metadata, duration, silence, decoding errors, and exact duplicates. |
| [Split.ipynb](Split.ipynb) | Remove exact duplicate copies, create the fixed split, and extract Log-STFT arrays. |
| [CNN2D_STFT30.ipynb](CNN2D_STFT30.ipynb) | Initial single-seed STFT experiment. |
| [STFT_CNN2D_MultiSeed.ipynb](STFT_CNN2D_MultiSeed.ipynb) | Optimized STFT training with seeds 42, 123, and 2026. |
| [LogMel_CNN2D_MultiSeed.ipynb](LogMel_CNN2D_MultiSeed.ipynb) | Log-Mel extraction and training with the same three seeds. |
| [Demo.ipynb](Demo.ipynb) | Google Colab inference using the supplied Log-Mel checkpoint. |

Notebooks contain executable cells and their saved outputs. Instructions are collected here.

## Run the demo in Google Colab

1. Open `Demo.ipynb` in Google Colab using **File → Open notebook → GitHub**, and enter this repository URL.
2. Download [best_cnn2d_mel.pt](best_cnn2d_mel.pt) from this repository using the file's download button.
3. In Colab, open the **Files** sidebar and upload the checkpoint. Its exact location must be `/content/best_cnn2d_mel.pt`.
4. Run the cells from top to bottom. CPU inference is supported; a GPU is optional.
5. When prompted, upload one audio file. WAV, MP3, FLAC, OGG, and M4A are accepted when the runtime has the required decoder. Use WAV if a compressed format fails to decode.
6. View the predicted genre, softmax scores, and Log-Mel visualization.

The demo uses the first 30 seconds of longer recordings and zero-pads shorter files. Colab temporary files disappear when the runtime is reset, so upload the checkpoint again in a new runtime.

This is a **closed-set classifier**: it always selects one of the five classes. It cannot reliably reject other music genres, speech, or noise. A softmax score is not a calibrated probability that the prediction is correct.

## Local training setup

The training notebooks were run using Python 3.12, PyTorch, CUDA, and an NVIDIA RTX 4070 Ti SUPER with 16 GB VRAM, under Ubuntu WSL. The two multi-seed notebooks require CUDA.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas librosa soundfile scikit-learn matplotlib tqdm ipykernel
```

Install a CUDA-compatible PyTorch build for your driver, then verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Select the `.venv` kernel in VS Code or Jupyter. Raw audio is not included in this repository. Place your copy of the VNTM3 audio collection in five class folders named `cailuong`, `catru`, `chauvan`, `cheo`, and `hatxam`.

The default data location is `D:\Data\VNTM3` on Windows, exposed as `/mnt/d/Data/VNTM3` in WSL. If you use another location, update the path variables in each notebook. Generated manifests contain absolute paths; regenerate them if you move the dataset.

Run in this order:

1. **CheckDTS.ipynb** writes the audit CSV files to `_audit/`.
2. **Split.ipynb** writes the fixed split manifests and spectrograms to `_stft30_v1/`.
3. Optionally run **CNN2D_STFT30.ipynb** for the initial single-seed baseline.
4. **STFT_CNN2D_MultiSeed.ipynb** writes results to `_cnn2d_stft_multiseed_v1/`.
5. **LogMel_CNN2D_MultiSeed.ipynb** reuses the STFT manifest and writes results to `_cnn2d_mel30_multiseed_v1/`.

Batch size is 96 and the number of DataLoader workers is 8 in the multi-seed notebooks. Lower batch size if GPU memory is insufficient, or lower worker count for a smaller CPU. The single-seed notebook uses batch size 64 and 12 workers.

Feature caches are reused when present. Use a new output directory when changing extraction parameters or input data: cache checks do not track every parameter. Training cells start new runs; they do not resume optimizer state after an interruption.

## Dataset preparation

The supplied audit outputs contain 2,499 audio files. Four duplicate copies are excluded using file-byte SHA-256, leaving 2,495 files. Raw files are not deleted.

| Genre | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Cải lương | 350 | 75 | 75 |
| Ca trù | 347 | 74 | 75 |
| Chầu văn | 349 | 75 | 75 |
| Chèo | 350 | 75 | 75 |
| Hát xẩm | 350 | 75 | 75 |
| **Total** | **1,746** | **374** | **375** |

The split is stratified by genre, approximately 70%/15%/15%, with random state 42. Audio is loaded as mono, resampled to 22,050 Hz, and cropped or zero-padded to 661,500 samples (30 seconds). Short recordings are retained and padded; they do not all contain 30 seconds of actual music. The audit flags whole-file RMS below 0.001; no files met that condition in the saved run. The pipeline does not trim silent intervals.

## Spectrograms

| Parameter | Log-STFT | Log-Mel |
| --- | --- | --- |
| FFT size | 2,048 | 2,048 |
| Hop length | 512 samples | 512 samples |
| Frequency representation | 1,025 linear-frequency bins | 128 Mel bands |
| Original matrix | 1,025 × 1,292 | 128 × 1,292 |
| Power and dB conversion | Squared magnitude; reference = file maximum | Mel power; reference = file maximum |
| Dynamic range | 80 dB | 80 dB |
| Normalization | [-80, 0] to [0, 1] | [-80, 0] to [0, 1] |
| CNN input | 1 × 256 × 320 | 1 × 256 × 320 |

Bilinear interpolation resizes both representations to the same CNN input size. The multi-seed notebooks save resized arrays as float16 caches. Increasing Mel height from 128 to 256 by interpolation does not add frequency information.

## CNN and training

The CNN has 865,957 parameters. Four blocks use 32, 64, 128, and 192 channels. Each block contains two 3×3 convolutions, BatchNorm, ReLU, 2×2 max pooling, and spatial dropout (0.05, 0.10, 0.15, 0.20). Global average pooling is followed by Linear 192→128, ReLU, dropout 0.40, and Linear 128→5.

- Loss: cross-entropy.
- Optimizer: AdamW, learning rate 0.0003, weight decay 0.0001.
- Training: up to 50 epochs; early stopping patience 10; minimum improvement 0.0001.
- Checkpoint selection: validation macro-F1.
- Scheduler: ReduceLROnPlateau, factor 0.5, patience 3, minimum LR 0.000001.
- Training-only augmentation: circular time shift, frequency masking, time masking, and small spectrogram intensity scaling.
- Performance settings: AMP FP16, channels-last, TF32, pinned memory, persistent workers, and prefetching.

Each seed initializes and trains a separate model on the **same split**. Test evaluation occurs after selecting that run's checkpoint using validation. The runs are not cross-validation folds or an ensemble. cuDNN speed settings mean identical seeds do not guarantee bitwise reproducibility.

## Recorded results

These results come from the outputs saved in the submitted notebooks; they are not a new training run performed while preparing this repository.

| Experiment | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Initial STFT, seed 42 | 97.87% | 97.85% |
| STFT, three seeds | 97.24% ± 1.37 percentage points | 97.22% ± 1.40 percentage points |
| Log-Mel, three seeds | 97.16% ± 0.31 percentage points | 97.12% ± 0.31 percentage points |

Mean and sample standard deviation use seeds 42, 123, and 2026. STFT has a slightly higher mean; Log-Mel varies less across these runs. These three runs do not establish statistically significant superiority. The initial STFT experiment uses a different batch size and execution configuration from the multi-seed experiments.

Each training notebook exports checkpoint files, training history, classification reports, confusion matrices, and per-file predictions. Multi-seed notebooks additionally export per-seed and mean/std summaries. The STFT reference values in the Log-Mel comparison cell are hard-coded to the recorded STFT run and need updating if that experiment changes.

## Evaluation limits

This is a **fixed stratified file-level split**. Reliable original-recording IDs are not available in the provided pipeline, so separation of recording sources is not guaranteed. SHA-256 removes byte-identical copies, but does not detect re-encoded duplicates, overlapping excerpts, or different excerpts from the same recording. Therefore these scores should not be presented as guaranteed unseen-recording performance or directly compared with studies using a different split or dataset.

Training curves use augmented inputs with dropout active, whereas validation uses unaugmented inputs in evaluation mode. Their scores are not measured under identical conditions.

The repository contains the experiment notebooks and a demo checkpoint, not a dataset download pipeline. Dataset provenance and redistribution terms should be obtained from the original dataset distributor; no dataset ownership or redistribution license is asserted here.
