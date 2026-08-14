# Titanic Analytics & ML Pipeline

## Overview

This module performs exploratory data analysis, preprocessing,
classification, class-imbalance comparison, hyperparameter tuning,
and regression using the Titanic dataset.

## Dataset

The Titanic dataset is stored locally as `titanic.csv`.

## Analysis

The pipeline includes:

- Dataset profiling and missing-value analysis
- Missing-value handling
- IQR outlier detection for Age and Fare
- Mean, median, mode and skewness
- Survival analysis by sex and passenger class
- Correlation analysis
- Multivariate visualizations
- Age z-score exploration

## Classification

Three models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Class imbalance was compared using:

- Baseline
- Class-weight balanced model
- SMOTE

Random Forest hyperparameters were tuned using GridSearchCV.

## Regression

Linear Regression was used to predict Fare.

Metrics:

- MAE
- MSE
- RMSE
- R2
- Adjusted R2

A residual plot was generated to inspect heteroscedasticity.

## Final Result

Random Forest performed best among the evaluated classifiers based
on F1 score.

The complete tuned Random Forest preprocessing and model pipeline
was saved as:

`titanic_survival_pipeline.joblib`

The saved pipeline was reloaded and tested using raw, unprocessed
input data.