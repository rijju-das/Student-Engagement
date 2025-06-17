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

### 2.1. OpenFace Feature Extraction

This step uses the [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) toolkit by Tadas Baltrušaitis et al. to extract per‐frame Action Units, head pose, gaze, and facial landmarks.

- **How to Run**  
  Open and execute the `Extract_OpenFace_features.ipynb` notebook (either in Google Colab or on your local machine within the project’s environment).  

- **Outputs**  
  After completion, the notebook will write the following CSVs into your `WACV data/` folder:  
  - `processedData0.csv`  
  - `processedData1.csv`  
  - `processedData2.csv`  
  - `processedDataOF.csv`  

These files contain your extracted OpenFace features and labels, ready for the next mapping and modeling steps.  


### 2.2. MediaPipe Feature Extraction

This step uses Google’s [MediaPipe FaceMesh](https://github.com/google/mediapipe) to extract 468-point facial landmarks and then merges them with your OpenFace features.

- **How to Run**  
  1. Ensure both `Extract_MediaPipe_features.py` and `mediaPipeFeatureExtractor.py` live in `Feature_extract/`.  
  2. From that directory, run:
     ```bash
     cd Feature_extract
     python Extract_MediaPipe_features.py
     ```

- **Outputs**  
  The script will read your OpenFace CSVs, extract MediaPipe landmarks for each frame, merge on `ImageID`, and write out three merged files to `WACV data/`:
  - `merged_data0.csv`
  - `merged_data1.csv`
  - `merged_data2.csv`

These merged CSVs contain both the 468 2D landmark coordinates and your previously extracted OpenFace features, ready for the subsequent mapping and modeling steps.  


## 3. Classical ML Model Training

All scripts in `ML_models/` assume you’ve already generated and merged your feature CSVs under `WACV data/`.

- **Scripts & folders**  
  - `ML_classification.py`  
    Defines the classification routines for Decision Tree, SVM, Random Forest & XGBoost.  
  - `train_model_ML.py`  
    Orchestrates the end-to-end ML pipeline: loads your merged CSVs, trains each classifier, and saves results.  
  - `trained_models/`  
    Where all trained model files are written.

- **How to run**  
  ```bash
  cd ML_models
  python train_model_ML.py \
    --data_dir "../WACV data" \
    --output_dir "../Results"


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
