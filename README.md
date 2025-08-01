# GazeLT: Visual attention–guided long-tailed disease classification in chest radiographs
[Moinak Bhattacharya](https://sites.google.com/stonybrook.edu/moinakbhattacharya), [Gagandeep Singh](https://www.columbiaradiology.org/profile/gagandeep-singh-mbbs), [Shubham Jain](https://www3.cs.stonybrook.edu/~jain/), [Prateek Prasanna](https://you.stonybrook.edu/imaginelab/)

**Status**: Under Review

---
![Architecture Diagram](https://github.com/user-attachments/files/21555092/figure2.pdf)
*Figure 1: Overview of the GazeLT architecture showing visual attention guidance for long-tailed disease classification.*

---

## 🔗 Resources

- **TWI & TWD Weights**: [Download here](https://drive.google.com/drive/folders/1nrfZ2rBj9If-yy-O3lCRqUe9bO93OaHj?usp=sharing)
- **GazeLT Weights & Results**: [Download here](https://drive.google.com/drive/folders/10wA9KePZ6Yux2G_jiI9Y2urgYDJP_S4b?usp=share_link)
- **Datasets**:
  - [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
  - [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

  After downloading, organize as follows:
  ```bash
  mkdir -p data/nih
  mkdir -p data/mimic
  ```

  Place NIH images (and metadata if needed) into: `data/nih/`
  
  Place MIMIC-CXR-JPG files into: `data/mimic/`
- **Label Files**:
  - Download and place inside a `labels/` directory:
    ```bash
    mkdir -p labels
    # Download from browser or manually place contents of:
    # https://drive.google.com/drive/folders/1tlaoqIRBdJWcjIDXVYRE1BOmUtwc_qW-?usp=sharing
    ```

---

## 🚀 Getting Started

Clone this repository:

```bash
git clone https://github.com/lordmoinak1/gazelt.git
cd gazelt
```

## ⚙️ Environment Setup

Create and activate a Conda environment:

```bash
conda create -n gazelt_env python=3.8 -y
conda activate gazelt_env
```

## 🔧 Install Dependencies

```pip3 install -r requirements.txt```

---

## 👁️ Generate TWI and TWD Features

```bash
  mkdir -p tw/weights
  mkdir -p tw/features/nih
  mkdir -p tw/features/mimic
```
```bash
  python3 src/gaze_models.py \
      --global \
      --nih \
      --model_path tw/weights/global_epoch_99_attentionloss_0.002267777990709874.pt \
      --data_path data/nih \
      --labels_path labels \
      --save_features_path tw/features/nih

  python3 src/gaze_models.py \
      --focal \
      --nih \
      --model_path tw/weights/focal_epoch_99_attentionloss_0.0024850438985595247.pt \
      --data_path data/nih \
      --labels_path labels \
      --save_features_path tw/features/nih

  python3 src/gaze_models.py \
      --global \
      --mimic \
      --model_path tw/weights/global_epoch_99_attentionloss_0.002267777990709874.pt \
      --data_path data/mimic \
      --labels_path labels \
      --save_features_path tw/features/nih

  python3 src/gaze_models.py \
      --focal \
      --mimic \
      --model_path tw/weights/focal_epoch_99_attentionloss_0.0024850438985595247.pt \
      --data_path data/mimic \
      --labels_path labels \
      --save_features_path tw/features/nih
```

---

## 🚀 Training Instructions
### NIH-CXR-LT

```bash
CUDA_VISIBLE_DEVICES=X python3 src/main_gazelt.py \
                    --data_dir /path/to/nih/images \
                    --label_dir /path/to/LongTailCXR/labels \
                    --out_dir nih_results_gazelt \
                    --dataset nih-cxr-lt \
                    --loss ldam \
                    --rw_method sklearn \
                    --drw \
                    --max_epochs 100 \
                    --patience 15 \
                    --batch_size 256 \
                    --lr 1e-4 \
```
### MIMIC-CXR-LT
```bash
CUDA_VISIBLE_DEVICES=X python src/main_gazelt.py \
                    --data_dir /path/to/physionet.org/files/mimic-cxr-jpg/2.0.0 \
                    --label_dir /path/to/LongTailCXR/labels \
                    --out_dir mimic_results_gazelt \
                    --dataset mimic-cxr-lt \
                    --loss ldam \
                    --rw_method sklearn \
                    --drw \
                    --max_epochs 100 \
                    --patience 15 \
                    --batch_size 256 \
                    --lr 1e-4 \
```
---

## 📊 Long-Tailed Classification Results on NIH-CXR-LT

| Method             | Head  | Medium | Tail  | Avg    | bAcc   |
|--------------------|-------|--------|-------|--------|--------|
| Softmax            | 0.419 | 0.056  | 0.017 | 0.164  | 0.115  |
| CB Softmax         | 0.295 | 0.415  | 0.217 | 0.309  | 0.269  |
| RW Softmax         | 0.248 | 0.359  | 0.258 | 0.288  | 0.260  |
| Focal Loss         | 0.362 | 0.056  | 0.042 | 0.153  | 0.122  |
| CB Focal Loss      | 0.371 | 0.333  | 0.117 | 0.274  | 0.232  |
| RW Focal Loss      | 0.286 | 0.293  | 0.117 | 0.232  | 0.197  |
| LDAM               | 0.410 | 0.133  | 0.142 | 0.228  | 0.178  |
| CB LDAM            | 0.357 | 0.285  | 0.208 | 0.284  | 0.235  |
| CB LDAM-DRW        | 0.476 | 0.356  | 0.250 | 0.361  | 0.281  |
| RW LDAM            | 0.305 | 0.419  | 0.292 | 0.338  | 0.279  |
| RW LDAM-DRW        | 0.410 | 0.367  | 0.308 | 0.362  | 0.289  |
| MixUp              | 0.419 | 0.044  | 0.017 | 0.160  | 0.118  |
| Balanced-MixUp     | 0.443 | 0.081  | 0.108 | 0.211  | 0.155  |
| Decoupling-cRT     | 0.433 | 0.374  | 0.300 | 0.369  | 0.294  |
| Decoupling-τ-norm  | 0.457 | 0.230  | 0.083 | 0.257  | 0.214  |
| GazeRadar          | 0.390 | 0.074  | 0.075 | 0.187  | 0.140  |
| RadioTransformer   | 0.386 | 0.093  | 0.083 | 0.193  | 0.131  |
| **GazeLT (Ours)**  | 0.404 | 0.411  | 0.417 | 0.410  | 0.315  |

## 📊 Long-Tailed Classification Results on MIMIC-CXR-LT

| Method             | Head  | Medium | Tail  | Avg    | bAcc   |
|--------------------|-------|--------|-------|--------|--------|
| Softmax            | 0.503 | 0.039  | 0.022 | 0.188  | 0.169  |
| CB Softmax         | 0.493 | 0.167  | 0.222 | 0.294  | 0.227  |
| RW Softmax         | 0.473 | 0.139  | 0.133 | 0.249  | 0.211  |
| Focal Loss         | 0.477 | 0.044  | 0.022 | 0.181  | 0.172  |
| CB Focal Loss      | 0.373 | 0.117  | 0.344 | 0.278  | 0.191  |
| RW Focal Loss      | 0.403 | 0.283  | 0.211 | 0.299  | 0.239  |
| LDAM               | 0.497 | 0.000  | 0.000 | 0.166  | 0.165  |
| CB LDAM            | 0.467 | 0.161  | 0.211 | 0.280  | 0.225  |
| CB LDAM-DRW        | 0.520 | 0.156  | 0.356 | 0.344  | 0.267  |
| RW LDAM            | 0.437 | 0.250  | 0.167 | 0.284  | 0.243  |
| RW LDAM-DRW        | 0.447 | 0.256  | 0.311 | 0.338  | 0.275  |
| MixUp              | 0.543 | 0.011  | 0.011 | 0.189  | 0.176  |
| Balanced-MixUp     | 0.480 | 0.039  | 0.011 | 0.177  | 0.168  |
| Decoupling-cRT     | 0.490 | 0.306  | 0.367 | 0.387  | 0.296  |
| Decoupling-τ-norm  | 0.520 | 0.167  | 0.067 | 0.251  | 0.230  |
| GazeRadar          | 0.527 | 0.006  | 0.000 | 0.279  | 0.174  |
| RadioTransformer   | 0.493 | 0.000  | 0.000 | 0.260  | 0.164  |
| **GazeLT (Ours)**  | 0.480 | 0.278  | 0.489 | 0.418  | 0.292  |

---

## Citation
If you find this repository useful, please consider giving a star :star: and cite the following
```
to be updated
```

📬 Contact
For questions or collaborations, please reach out to the corresponding authors via their personal webpages linked above.



