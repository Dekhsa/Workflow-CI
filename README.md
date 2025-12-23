# Workflow-CI: MLOps CI/CD Pipeline

![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-blue)
![Python](https://img.shields.io/badge/Python-3.12.7-green)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)

Automated CI/CD pipeline untuk training, tracking, dan deployment model fraud detection menggunakan MLflow, DagsHub, dan Docker Hub.

## 📋 Daftar Isi

- [Overview](#overview)
- [Struktur Project](#struktur-project)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Details](#model-details)
- [Usage](#usage)

## 🎯 Overview

Project ini mengimplementasikan **complete MLOps pipeline** untuk fraud detection model dengan:
- **Automated Training**: Training otomatis ketika push ke main branch
- **Model Tracking**: Tracking experiments menggunakan MLflow & DagsHub
- **Containerization**: Docker image untuk deployment
- **Artifact Management**: Upload artifacts ke GitHub Actions
- **Version Control**: Model versioning dengan Git SHA

## 📁 Struktur Project

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── MLProject/
│   ├── MLProject               # MLflow project metadata
│   ├── conda.yaml              # Conda environment specification
│   ├── modelling.py            # Main training script
│   ├── creditcardfraud_preprocessing.csv  # Dataset
│   └── docker_hub_link.txt     # Docker Hub URLs (auto-generated)
└── README.md
```

## ✨ Features

### 🤖 Machine Learning
- **Model**: XGBoost Classifier dengan hyperparameter tuning (GridSearchCV)
- **Preprocessing**: StandardScaler pipeline
- **Cross-Validation**: RepeatedStratifiedKFold (5 folds, 3 repeats)
- **Metrics**: F1 Score, Precision, Recall, AUPRC
- **Visualizations**: Confusion Matrix, PR Curve, Feature Importance

### 🔄 CI/CD Pipeline
1. **Setup Environment**: Python 3.12.7, MLflow, Conda
2. **Train Model**: MLflow project execution
3. **Track Experiments**: DagsHub integration
4. **Build Docker Image**: Containerize trained model
5. **Push to Docker Hub**: Automated deployment
6. **Upload Artifacts**: Store model artifacts di GitHub

### 📊 Model Tracking
- MLflow tracking dengan DagsHub backend
- Automatic experiment logging
- Dataset profiling & versioning
- Artifact storage (plots, metrics, model)

## 🛠 Requirements

### Python Dependencies
```yaml
- python=3.12.7
- mlflow==2.19.0
- scikit-learn
- pandas
- dagshub
- xgboost
- matplotlib
- seaborn
- numpy
```

### GitHub Secrets
Setup secrets di repository **Settings → Secrets and variables → Actions**:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `DAGSHUB_USERNAME` | DagsHub username | `Dekhsa` |
| `DAGSHUB_TOKEN` | DagsHub access token | `xxxxx...` |
| `DOCKER_USERNAME` | Docker Hub username | `dekhsa` |
| `DOCKER_PASSWORD` | Docker Hub password/token | `xxxxx...` |

## 🚀 Setup

### 1. Clone Repository
```bash
git clone https://github.com/Dekhsa/Workflow-CI.git
cd Workflow-CI
```

### 2. Setup DagsHub
1. Buat akun di [DagsHub](https://dagshub.com)
2. Buat repository: `SMSML_Muhamad-Dekhsa-Afnan`
3. Generate access token dari **Settings → Access Tokens**

### 3. Setup Docker Hub
1. Buat akun di [Docker Hub](https://hub.docker.com)
2. (Optional) Buat access token dari **Account Settings → Security**

### 4. Configure GitHub Secrets
Tambahkan semua secrets yang diperlukan ke GitHub repository.

### 5. Prepare Dataset
Pastikan dataset `creditcardfraud_preprocessing.csv` ada di folder `MLProject/` dengan kolom:
- Features: semua kolom kecuali target
- Target: `is_fraud` (binary: 0/1)

### 6. Push to Main Branch
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

Pipeline akan otomatis berjalan! 🎉

## 🔄 CI/CD Pipeline

### Pipeline Steps

```
1. Checkout repository
2. Set up Python 3.12.7
3. Check Environment
4. Install dependencies
5. Run mlflow project          ← Training
6. Get latest MLflow run_id    ← Tracking
7. Install Python dependencies
8. Upload to GitHub            ← Artifact storage
9. Build Docker Model          ← Containerization
10. Log in to Docker Hub
11. Tag Docker image
12. Push Docker Image          ← Deployment
13. Post Log in to Docker Hub
```

### Trigger
Pipeline akan berjalan otomatis ketika:
- Push ke branch `main`
- Manual trigger dari Actions tab

### Output
Setelah pipeline selesai:
- ✅ Model tracked di DagsHub
- ✅ Docker image di Docker Hub: `<username>/fraud-detection-model:latest`
- ✅ Artifacts tersimpan di GitHub Actions (90 hari)
- ✅ Docker URLs tersimpan di `MLProject/docker_hub_link.txt`

## 🧠 Model Details

### Algorithm
**XGBoost Classifier** dengan hyperparameter tuning

### Hyperparameter Grid
```python
{
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "scale_pos_weight": [1.0, 5.0]
}
```

### Training Process
1. Load & preprocess data
2. Train-test split (80-20, stratified)
3. StandardScaler normalization
4. GridSearchCV with 3-fold CV
5. Best model evaluation
6. Cross-validation (5 folds × 3 repeats)
7. Log metrics & artifacts to MLflow

### Metrics Tracked
- **F1 Score**: Balance between precision & recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **AUPRC**: Area Under Precision-Recall Curve
- **CV Statistics**: Mean & std for all metrics

## 📖 Usage

### Local Training
```bash
# Setup environment
conda env create -f MLProject/conda.yaml
conda activate fraud_env

# Set DagsHub credentials
export MLFLOW_TRACKING_URI="https://dagshub.com/Dekhsa/SMSML_Muhamad-Dekhsa-Afnan.mlflow"
export MLFLOW_TRACKING_USERNAME="your_username"
export MLFLOW_TRACKING_PASSWORD="your_token"

# Run MLflow project
mlflow run MLProject/ --env-manager=conda
```

### Run Docker Container
```bash
# Pull image from Docker Hub
docker pull <username>/fraud-detection-model:latest

# Run model server
docker run -p 8080:8080 <username>/fraud-detection-model:latest

# Make prediction
curl -X POST http://localhost:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

### View Experiments
1. **DagsHub**: https://dagshub.com/Dekhsa/SMSML_Muhamad-Dekhsa-Afnan
2. **GitHub Actions**: Repository → Actions tab
3. **Docker Hub**: https://hub.docker.com/r/<username>/fraud-detection-model

### Download Artifacts
1. Pergi ke **Actions** → Pilih workflow run
2. Scroll ke bawah ke **Artifacts** section
3. Download `mlflow-model-artifacts.zip`

## 🔧 Troubleshooting

### Error: "Column 'is_fraud' not found"
- Pastikan dataset memiliki kolom `is_fraud` sebagai target
- Atau edit `modelling.py` untuk menggunakan nama kolom yang sesuai

### Error: "DAGSHUB_USER_TOKEN not found"
- Pastikan semua GitHub Secrets sudah di-setup dengan benar
- Check typo pada nama secrets

### Docker Build Failed
- Pastikan MLflow run berhasil dan model ter-log
- Check MLflow tracking URI dan credentials

### Pipeline Stuck
- Check GitHub Actions logs untuk error detail
- Verify conda environment setup berhasil

## 📝 License

MIT License - Free to use and modify

## 👤 Author

**Muhamad Dekhsa Afnan**
- GitHub: [@Dekhsa](https://github.com/Dekhsa)
- DagsHub: [Dekhsa](https://dagshub.com/Dekhsa)

## 🙏 Acknowledgments

- MLflow untuk experiment tracking
- DagsHub untuk MLflow hosting
- GitHub Actions untuk CI/CD
- Docker Hub untuk container registry

---

⭐ **Star this repository** jika bermanfaat!
