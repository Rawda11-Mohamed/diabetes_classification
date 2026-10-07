# 🩺 Diabetes Classification

A machine learning project that uses a **Decision Tree Classifier** to predict diabetes based on clinical and health-related features.

The project focuses on exploring the diabetes dataset, analyzing its features, training a Decision Tree model, tuning its hyperparameters using **GridSearchCV**, and evaluating the model using classification metrics and visualizations.

---

## 📌 Project Overview

Diabetes classification is a binary classification problem where the goal is to predict whether a patient is likely to have diabetes based on medical and demographic measurements.

In this project, a **Decision Tree Classifier** is trained using patient health data. The model is then optimized using **GridSearchCV** to find better hyperparameters and improve its generalization performance.

The complete workflow includes:

```text
Dataset
   ↓
Data Exploration
   ↓
Feature / Target Selection
   ↓
Train-Test Split
   ↓
Decision Tree
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Feature Importance
   ↓
Decision Tree Visualization
```

---

## 📊 Dataset

The project uses the **Diabetes Dataset** downloaded through KaggleHub from the `mathchi/diabetes-data-set` dataset.

The dataset contains **768 records** and the following features:

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `Pregnancies`              | Number of pregnancies                          |
| `Glucose`                  | Plasma glucose concentration                   |
| `BloodPressure`            | Diastolic blood pressure                       |
| `SkinThickness`            | Triceps skin fold thickness                    |
| `Insulin`                  | 2-Hour serum insulin                           |
| `BMI`                      | Body Mass Index                                |
| `DiabetesPedigreeFunction` | Diabetes pedigree function                     |
| `Age`                      | Patient age                                    |
| `Outcome`                  | Target variable: 0 = No Diabetes, 1 = Diabetes |

The dataset contains 8 input features and `Outcome` as the target variable.

---

## 🔍 Exploratory Data Analysis

The notebook starts by loading and inspecting the dataset using **Pandas**.

The analysis includes:

* Displaying the first rows of the dataset
* Descriptive statistics
* Examining feature distributions
* Understanding the target variable
* Visualizing model results
* Analyzing feature importance

The dataset contains 768 observations, with `Outcome` representing the binary target class.

---

## 🤖 Machine Learning Model

### Decision Tree Classifier

The main machine learning algorithm used in this project is:

```text
DecisionTreeClassifier
```

The dataset is divided into training and testing sets using an **80/20 split**.

```text
80% → Training Data
20% → Testing Data
```

The initial Decision Tree model is trained using:

```python
DecisionTreeClassifier(random_state=42)
```

---

## ⚙️ Hyperparameter Tuning

To improve the Decision Tree model, **GridSearchCV** is used to search through different combinations of hyperparameters.

The parameters explored include:

* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

The grid search uses:

```text
5-Fold Cross Validation
```

A total of **192 parameter combinations** were evaluated, resulting in **960 model fits**.

### Best Parameters

The best configuration found by GridSearchCV was:

```text
criterion = entropy
max_depth = 5
min_samples_leaf = 10
min_samples_split = 2
```

---

## 📈 Model Performance

### Before Hyperparameter Tuning

The initial Decision Tree achieved:

| Metric            |   Score |
| ----------------- | ------: |
| Training Accuracy | 100.00% |
| Testing Accuracy  |  74.68% |

The difference between training and testing performance indicates that the initial tree was overfitting the training data.

### After Hyperparameter Tuning

After applying GridSearchCV:

| Metric            |  Score |
| ----------------- | -----: |
| Training Accuracy | 79.97% |
| Testing Accuracy  | 79.87% |

The tuned model provides a better balance between training and testing performance.

### Classification Report

For the tuned model:

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| 0                    |      0.86 |   0.83 |     0.84 |
| 1                    |      0.70 |   0.74 |     0.72 |
| **Overall Accuracy** |           |        | **0.80** |

The model achieved approximately **79.87% test accuracy**.

---

## 📊 Model Evaluation

Several evaluation techniques are included in the notebook.

### Confusion Matrix

A confusion matrix is generated to visualize:

* True Negatives
* False Positives
* False Negatives
* True Positives

```python
ConfusionMatrixDisplay.from_estimator(
    best_dt,
    X_test,
    y_test
)
```

### Feature Importance

The Decision Tree feature importance is also visualized to identify which health indicators contribute most to the model's predictions.

```python
best_dt.feature_importances_
```

### Decision Tree Visualization

The final tuned Decision Tree is visualized using:

```python
plot_tree()
```

This provides a visual representation of the decision rules learned by the model.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Decision Tree Classifier
* GridSearchCV

### Visualization

* Matplotlib

### Dataset

* Kaggle
* KaggleHub

### Development Environment

* Google Colab
* Jupyter Notebook

---

## 📁 Project Structure

```text
diabetes_classification/
│
├── Diabetes.ipynb
│
└── README.md
```

The main notebook contains the complete workflow, including:

```text
Dataset Download
      ↓
Data Loading
      ↓
Data Exploration
      ↓
Decision Tree Training
      ↓
Model Evaluation
      ↓
Grid Search
      ↓
Best Model Selection
      ↓
Feature Importance
      ↓
Decision Tree Visualization
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed.

The project can also be run directly using **Google Colab**.

### Install Dependencies

The notebook uses the following main Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn kagglehub
```

### Run the Project

Open:

```text
Diabetes.ipynb
```

Then run the notebook cells sequentially.

The dataset is downloaded automatically using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "mathchi/diabetes-data-set"
)
```

---

## 🧠 Machine Learning Workflow

### 1. Load Dataset

The diabetes dataset is downloaded and loaded using Pandas.

### 2. Explore the Data

Basic dataset inspection and descriptive statistics are performed.

### 3. Select Features and Target

```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

### 4. Split the Dataset

The data is divided into training and testing sets.

```text
Training → 80%
Testing  → 20%
```

### 5. Train Decision Tree

An initial Decision Tree model is trained and evaluated.

### 6. Tune Hyperparameters

GridSearchCV evaluates multiple parameter combinations using 5-fold cross-validation.

### 7. Evaluate the Best Model

The optimized model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 8. Analyze the Model

Feature importance and the final Decision Tree structure are visualized.

---

## 📌 Key Results

The project demonstrates the effect of hyperparameter tuning on a Decision Tree model.

```text
Initial Model
Test Accuracy: 74.68%

        ↓

GridSearchCV

        ↓

Optimized Model
Test Accuracy: 79.87%
```

The optimized Decision Tree achieved a test accuracy of approximately **79.87%**, while also reducing the large gap between training and testing performance seen in the initial model.

---

## ⚠️ Disclaimer

This project is an **educational machine learning project** and should not be used as a medical diagnostic system.

The predictions are based on a specific dataset and machine learning model and should not replace professional medical advice, clinical testing, or diagnosis.

---

## 👩‍💻 Author

**Rawda Mohamed**

Computer Science Graduate | Machine Learning & Flutter Enthusiast
