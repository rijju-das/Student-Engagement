# Optimizing Student Engagement Detection using Facial and Behavioral Features (accepted in NCAA 2025)

With the spirit of reproducible research, this repository includes a complete collection of codes required to generate the results and diagrams presented in the paper:

> R. Das, and S. Dev, Optimizing Student Engagement Detection using Facial and Behavioral Features, *Neural Computing and Applications*, 2025.




## Setup

1. **Clone repo**  
   ```bash
   git clone https://github.com/rijju-das/Student-Engagement.git
   cd Student-Engagement
   ```

2. **Install dependencies**

   ```bash
   pip install numpy pandas scikit-learn xgboost tensorflow keras matplotlib mediapipe
   ```

## 2. Feature Extraction

The `Feature_extract/` folder contains two pipelines:

1. **OpenFace features**  
2. **MediaPipe FaceMesh landmarks**

---

### 2.1. OpenFace Features

Generates per-frame Action Units, head pose, gaze, and landmarks using OpenFace v2.2.

1. **Install/OpenFace setup**  
   - Download and build OpenFace; ensure the `Feature_extract/OpenFace` binary is on your `PATH`.

2. **Prepare your frame folders**  
   Place your frame sequences under:
   WACV data/
      ├── dataset/1/
      ├── dataset/2/
      └── dataset/3/

4. **Run the notebook**  
This will:
- Process each folder (`1`, `2`, `3`)
- Produce `processedData0.csv`, `processedData1.csv`, `processedData2.csv`
- Concatenate into `processedDataOF.csv`  
and save all CSVs under `WACV data/`.

```bash
cd Feature_extract
jupyter nbconvert --to notebook --execute Extract_OpenFace_features.ipynb



## 1. AU Mappings

Convert raw OpenFace AU outputs into aggregate engagement features.

```bash
cd AU_mappings
python map_aus.py \
  --input_dir ../Feature_extract/openface_output \
  --output_dir ./mapped_features
```

* **map\_aus.py**
  Reads per‐frame AUs (`.csv`) and computes time‐windowed stats (mean, std, etc.) for each AU.

Mapped features land in `AU_mappings/mapped_features/`.





## 3. Classical ML Models

Train and evaluate XGBoost, SVM, and Random Forest baselines.

```bash
cd ML_models
```
# Train XGBoost
```bash
python train_xgb.py \
  --features_dir ../AU_mappings/mapped_features \
  --out_model models/xgb_model.pkl \
  --report_dir results/xgb
```
# Train SVM
```bash
python train_svm.py \
  --features_dir ../AU_mappings/mapped_features \
  --out_model models/svm_model.pkl \
  --report_dir results/svm
```
# Evaluate all
```bash
python evaluate_ml.py \
  --models_dir models \
  --test_data_dir "../WACV data" \
  --out_csv results/ml_summary.csv
```

* **train\_xgb.py**, **train\_svm.py**, **evaluate\_ml.py** each support `--help` for more flags.

---

## 4. Deep-Learning Models

Implement a CNN on WACV, then fine-tune on DAiSEE via transfer learning.

```bash
cd DL_models

# 1) Pretrain on WACV
python train_cnn.py \
  --data_dir "../WACV data" \
  --epochs 50 \
  --save_model models/cnn_wacv.h5

# 2) Transfer to DAiSEE
python transfer_learning.py \
  --base_model models/cnn_wacv.h5 \
  --target_data_dir "../DAiSEE data" \
  --epochs 30 \
  --save_model models/cnn_daisee.h5

# 3) Evaluate
python evaluate_dl.py \
  --model models/cnn_daisee.h5 \
  --test_data "../DAiSEE data/test" \
  --out_csv ../Results/dl_summary.csv
```

* **train\_cnn.py**, **transfer\_learning.py**, **evaluate\_dl.py** all include usage details.

---

## 5. Reproducing Figures & Tables

Once you have `ml_summary.csv` and `dl_summary.csv`:

```bash
cd Results
python plot_results.py \
  --ml_csv ../ML_models/results/ml_summary.csv \
  --dl_csv ../DL_models/results/dl_summary.csv \
  --out_dir figures
```

You’ll find all paper‐figures (ROC curves, bar‐charts, confusion matrices) in `Results/figures/`.


## Data

* **WACV data/** – Pre‐split source dataset (features & labels) used for pretraining.
* **DAiSEE data/** – Download from [DAiSEE](https://sites.google.com/view/daisee/) and place in root as shown.

---

## 📖 Citation

```bibtex
@article{das2025optimizing,
  title={Optimizing Student Engagement Detection using Facial and Behavioral Features},
  author={Das, R. and Dev, S.},
  journal={Neural Computing and Applications},
  year={2025},
}
```

---

## 👤 Contact

Riju Das ([riju.das@ucd.ie](mailto:riju.das@ucd.ie))
Ph.D. Scholar – University College Dublin

Feel free to raise an issue for any question or data‐path tweak.
