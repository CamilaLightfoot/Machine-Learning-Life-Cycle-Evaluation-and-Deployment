# Machine-Learning-Life-Cycle-Evaluation-and-Deployment
## Overview
This lab covers the **fifth step** of the Machine Learning Life Cycle — **Model Evaluation and Deployment**.  
The goal is to perform **model selection** to identify the optimal **Logistic Regression model** for a predictive task using the Airbnb NYC dataset.  
We evaluate models using multiple performance metrics, conduct **feature selection**, and save the best-performing model for deployment.


## Objectives
In this lab, we:
1. **Build** a DataFrame and define the ML problem.
2. **Create** labeled examples and split the data into **training** and **test** sets.
3. **Train, test, and evaluate** a Logistic Regression model with the default hyperparameter values.
4. **Find the optimal** model using `GridSearchCV`.
5. **Plot** Precision-Recall and ROC curves and compute **AUC** for both default and optimal models.
6. **Perform feature selection** using `SelectKBest`.
7. **Save the best-performing model** as a `.pkl` file for deployment.

## Files

### 1. [ModelSelectionForLogisticRegression.ipynb](https://github.com/CamilaLightfoot/Machine-Learning-Life-Cycle-Evaluation-and-Deployment/blob/main/ModelSelectionForLogisticRegression%20(3).ipynb)
A Jupyter Notebook containing the full Lab 5 workflow:
- Data preparation and splitting.
- Model training and evaluation.
- Hyperparameter tuning using `GridSearchCV`.
- Plotting evaluation metrics.
- Feature selection using `SelectKBest`.
- Saving the final model to a `.pkl` file.

### 2. **best_model.pkl**
Serialized Logistic Regression model (best-performing version) saved using `pickle`:
- Can be loaded in any Python environment to make predictions without retraining.

## Database used: [airbnbData_train.csv]
Prepared dataset containing Airbnb NYC listing data:
- Includes numerical and one-hot encoded categorical features.
- Target variable prepared for classification with Logistic Regression.

## Technologies Used
- **Python** (Pandas, NumPy)
- **Scikit-learn**: LogisticRegression, GridSearchCV, SelectKBest
- **Matplotlib** & **Seaborn** for visualizations
- **Pickle** for model persistence
- **Jupyter Notebook** for development

## 🚀 How to Run

### Option 1: Jupyter Notebook
git clone https://github.com/yourusername/Lab5-ML-Evaluation-Deployment.git
cd Lab5-ML-Evaluation-Deployment
pip install -r requirements.txt
jupyter notebook ModelSelectionForLogisticRegression.ipynb
