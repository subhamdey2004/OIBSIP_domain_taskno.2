# Wine Quality Prediction

A machine learning project that predicts wine quality based on chemical characteristics using three different classifier models.

## 📋 Project Overview

This project demonstrates the application of machine learning classification techniques to predict wine quality. The dataset contains physicochemical properties of wines (density, acidity, alcohol content, etc.) and their quality ratings.

**Objective:** Build and compare three classifier models to accurately predict wine quality scores.

### Models Implemented

1. **Random Forest Classifier** - Ensemble learning method using multiple decision trees
2. **Support Vector Classifier (SVC)** - Kernel-based method for binary/multi-class classification
3. **Stochastic Gradient Descent (SGD)** - Online learning algorithm for classification

## 📂 Project Structure

```
wine-quality-simple/
├── data/
│   └── wine_quality.csv          # Dataset (download from Kaggle)
├── main.py                        # Main project script
├── requirements.txt               # Python dependencies
├── results.png                    # Generated visualizations
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Create project directory:**
   ```bash
   mkdir wine-quality-simple
   cd wine-quality-simple
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate        # macOS/Linux
   # or
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset:**
   - Go to: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
   - Download the CSV file
   - Create a `data/` folder in your project
   - Place `wine_quality.csv` inside the `data/` folder

5. **Run the project:**
   ```bash
   python main.py
   ```

## 📊 Dataset Information

**Source:** Kaggle - Wine Quality Dataset

**Dataset Size:** ~6,500 wine samples

**Features (11):**
- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol Content

**Target Variable:** Quality (score between 0-10)

## 🔧 How It Works

### Step 1: Data Loading
- Loads the wine quality dataset from CSV
- Displays dataset shape and basic information

### Step 2: Data Preprocessing
- Separates features (X) from target variable (quality)
- Splits data: 80% training, 20% testing
- Scales features using StandardScaler for better model performance

### Step 3: Model Training
Trains three classifier models:
- Random Forest with 100 trees
- SVC with RBF kernel
- SGD with log loss

### Step 4: Model Evaluation
- Calculates accuracy for each model
- Generates classification reports with precision, recall, and F1-score
- Compares model performance

### Step 5: Visualization
Generates a 4-panel visualization showing:
1. **Quality Distribution** - Histogram of wine quality scores
2. **Feature Correlation** - Heatmap showing relationships between features
3. **Model Comparison** - Bar chart comparing accuracy of all three models
4. **Feature Importance** - Top 5 most important features (from Random Forest)

## 📈 Output

### Console Output
```
Loading data...
Dataset shape: (6497, 12)

==================================================
MODEL RESULTS
==================================================

Random Forest:
Accuracy: 0.8234

SVC:
Accuracy: 0.7891

SGD:
Accuracy: 0.7456

==================================================
Best Model: Random Forest
==================================================
```

### File Output
- **results.png** - High-resolution (300 DPI) visualization with all 4 plots

## 📦 Requirements

```
pandas        - Data manipulation and analysis
numpy         - Numerical computing
scikit-learn  - Machine learning library
matplotlib    - Data visualization
seaborn       - Statistical data visualization
```

Install with: `pip install -r requirements.txt`

## 🎯 Key Concepts & Challenges

### Machine Learning Concepts
- **Classification:** Predicting discrete labels (quality scores)
- **Feature Scaling:** StandardScaler normalizes features for better model performance
- **Train-Test Split:** Ensures unbiased model evaluation
- **Model Comparison:** Evaluates multiple algorithms to find the best performer

### Challenges Addressed
- **Multi-class Classification:** Handling multiple quality scores (not just binary)
- **Feature Engineering:** Using chemical properties as predictors
- **Imbalanced Data:** Different quality scores have different frequencies
- **Model Selection:** Comparing ensemble, kernel-based, and gradient-based approaches

## 💡 Usage Examples

### Using the Trained Model (Advanced)

```python
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Make predictions on new data
new_wine = [[7.0, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3.0, 0.45, 8.8]]
scaler = StandardScaler()
prediction = model.predict(scaler.transform(new_wine))
print(f"Predicted Quality: {prediction}")
```

## 📊 Expected Results

Typical accuracy scores:
- **Random Forest:** 80-85%
- **SVC:** 75-80%
- **SGD:** 70-75%

Results may vary based on random seed and data sampling.

## 🔍 Code Structure

**main.py** consists of:
1. Data loading section
2. Data preprocessing section
3. Model training loop
4. Model evaluation section
5. Visualization section

All contained in a single, easy-to-read file (~80 lines).

## 🛠️ Customization

### Change Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Adjust Random Forest Parameters
```python
RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
```

### Modify SVC Kernel
```python
SVC(kernel='linear')  # or 'poly', 'sigmoid'
```

## 🎓 Learning Outcomes

After completing this project, you'll understand:
- How to load and preprocess real-world datasets
- Implementing multiple classification algorithms
- Model evaluation and comparison techniques
- Data visualization with matplotlib and seaborn
- Feature importance analysis
- Classification metrics (accuracy, precision, recall, F1-score)

## 📝 Common Issues & Solutions

**Issue:** `FileNotFoundError: data/wine_quality.csv`
- **Solution:** Make sure you created the `data/` folder and placed the CSV file inside

**Issue:** `ModuleNotFoundError: No module named 'pandas'`
- **Solution:** Run `pip install -r requirements.txt`

**Issue:** Virtual environment not activating
- **Solution:** Make sure you're in the project directory and use the correct activation command for your OS

## 🔗 Resources

- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)



---

**Happy Learning! 🎉**
