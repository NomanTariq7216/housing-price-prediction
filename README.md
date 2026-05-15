# HOUSING PRICE PREDICTION

A machine learning project that predicts California housing prices using Random Forest Regression with scikit-learn. This project demonstrates end-to-end ML workflow including data preprocessing, model training, cross-validation, and inference.


## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Usage

**First run (Training):**
```bash
python main.py
```
This will train the model, perform 10-fold cross-validation, and save `model.pkl` and `pipeline.pkl`.

**Second run (Inference):**
```bash
python main.py
```
This will load the trained model and make predictions on `input.csv`, saving results to `output.csv`.

## Dataset

Download the California Housing dataset:
```bash
wget https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv
```

Or place `housing.csv` in the project root.

## Project Structure

├── main.py           # Main script (training & inference)
├── housing.csv        # Dataset
├── requirements.txt   # Dependencies
├── model.pkl         # Trained model (generated)
├── pipeline.pkl      # Preprocessing pipeline (generated)
├── input.csv         # Test data (generated)
└── output.csv        # Predictions (generated)

## Technical Details

- **Algorithm:** Random Forest Regressor
- **Preprocessing:** Median imputation, standard scaling, one-hot encoding
- **Validation:** 10-fold cross-validation
- **Train/Test Split:** 80/20 (stratified by income)

## Dependencies

scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.1

## Features

✅ Automated preprocessing pipeline  
✅ Stratified train-test splitting  
✅ Cross-validation for model evaluation  
✅ Model persistence with joblib  

## Author

**Noman Tariq**  
GitHub: [@NomanTariq7216](https://github.com/NomanTariq7216)  
Email: nomans7216@gmail.com
## License

MIT License
