# 🏠 Housing Price Prediction Model

A machine learning project that predicts California housing prices using Random Forest Regression with scikit-learn. This project demonstrates end-to-end ML workflow including data preprocessing, model training, cross-validation, and inference.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Project Overview

This project demonstrates:
- ✅ Data preprocessing with scikit-learn pipelines
- ✅ Stratified train-test splitting to maintain data distribution
- ✅ 10-fold cross-validation for robust model evaluation
- ✅ Model persistence using joblib
- ✅ Separate training and inference workflows
- ✅ Best practices in machine learning development

### Key Features
- **Automated Pipeline**: Handles missing values, scaling, and encoding automatically
- **Stratified Splitting**: Ensures representative train/test sets based on income categories
- **Cross-Validation**: 10-fold CV provides reliable performance estimates
- **Production Ready**: Serialized model and pipeline for easy deployment

---

## 🚀 Quick Start

### Prerequisites

Make sure you have the following installed:
- **Python 3.8 or higher**
- **pip** (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download the dataset** (if not included):
```bash
python download_data.py
```
Or manually download from: https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

---

## 🎯 Usage

### Training the Model

Run the script for the first time to train the model:

```bash
python3 main.py
```

**What happens during training:**
1. Loads the California housing dataset
2. Creates stratified train/test split (80/20)
3. Preprocesses data (imputation, scaling, encoding)
4. Trains Random Forest Regressor
5. Performs 10-fold cross-validation
6. Saves trained model and pipeline as `.pkl` files
7. Creates `input.csv` with test data for inference

**Expected Output:**
Cross-validation RMSE scores: [51039.08053738 48741.94041426 45940.42771745 50501.41453432
 47387.7896427  49595.25845731 51625.68567717 48865.70709952
 47322.87631489 53301.08748462]
Average RMSE: 49432.12678796127
Model and pipeline trained and saved successfully.

### Making Predictions (Inference)

Run the script again to perform inference on test data:

```bash
python3 main.py
```

**What happens during inference:**
1. Loads the saved model and pipeline
2. Reads `input.csv` (test data)
3. Applies the same preprocessing transformations
4. Generates predictions
5. Saves results to `output.csv`

**Expected Output:**
RMSE for the predictions: 47197.66824186381
MAE for the predictions: 30929.476097383722
R² Score: 0.8290804707970139
Error percentage: 22.882853095835635
Inference completed and output saved to output.csv.

## 📈 Model Performance

### Cross-Validation Results
- **Algorithm:** Random Forest Regressor
- **Cross-Validation:** 10-fold
- **Average RMSE:** ~$49,500
- **Standard Deviation:** ~$1,200

### Model Details
- **Number of Features:** 9 (8 numerical + 1 categorical)
- **Training Set Size:** ~16,512 samples (80%)
- **Test Set Size:** ~4,128 samples (20%)
- **Random State:** 42 (for reproducibility)

## 🛠️ Technologies & Libraries Used

- **Python 3.8+** - Programming language
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **joblib** - Model serialization

---

## 🔍 Technical Details

### Preprocessing Pipeline

**Numerical Features:**
1. **Imputation:** Missing values filled with median
2. **Scaling:** StandardScaler for normalization

**Categorical Features:**
1. **Encoding:** OneHotEncoder with `handle_unknown='ignore'`

### Model Architecture

```python
RandomForestRegressor(
    random_state=42,
    # Default parameters:
    # n_estimators=100
    # max_depth=None
    # min_samples_split=2
)
```
## Author

**NomanTariq**  
GitHub: [@NomanTariq7216](https://github.com/NomanTariq7216)  
Email: nomans7216@gmail.com

## License

MIT License
