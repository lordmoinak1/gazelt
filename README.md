# GazeLT: Visual attention–guided long-tailed disease classification in chest radiographs
[Moinak Bhattacharya](https://sites.google.com/stonybrook.edu/moinakbhattacharya), [Gagandeep Singh](https://www.columbiaradiology.org/profile/gagandeep-singh-mbbs), [Shubham Jain](https://www3.cs.stonybrook.edu/~jain/), [Prateek Prasanna](https://you.stonybrook.edu/imaginelab/)

**Status**: Under Review

---

## 🔗 Resources

- **Pre-trained Weights**: [Download here](https://drive.google.com/drive/folders/10wA9KePZ6Yux2G_jiI9Y2urgYDJP_S4b?usp=share_link)
- **Datasets**:
  - [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
  - [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

  After downloading, organize as follows:
  ```bash
  mkdir -p data/nih
  mkdir -p data/mimic
  ```

  Place NIH images (and metadata if needed) into: data/nih/ \\
  Place MIMIC-CXR-JPG files into: data/mimic/
- **Label Files**:
  - Download and place inside a `labels/` directory:
    ```bash
    mkdir -p labels
    # Download from browser or manually place contents of:
    # https://drive.google.com/drive/folders/1tlaoqIRBdJWcjIDXVYRE1BOmUtwc_qW-?usp=sharing
    ```

---

## ⚙️ Environment Setup

Create and activate a Conda environment:

```bash
conda create -n gazelt_env python=3.8 -y
conda activate gazelt_env
```

# Install dependencies
```pip install -r requirements.txt```


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
  
## Citation
If you find this repository useful, please consider giving a star :star: and cite the following
```
to be updated
```

📬 Contact
For questions or collaborations, please reach out to the corresponding authors via their personal webpages linked above.



