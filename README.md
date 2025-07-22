# Drivers License Test Scores Analysis and Qualification Prediction

## Project Overview

This project involves exploring a dataset containing driver's license test scores and applicant information, performing data cleaning and preprocessing, conducting exploratory data analysis (EDA), training and evaluating classification models to predict driver qualification, and interpreting the results. The goal is to identify key factors influencing qualification and build a predictive model.

## Dataset

The dataset, `Drivers License Data.csv`, contains information about applicants, including demographics, training details, various driving test scores, and their qualification status.

## Exploratory Data Analysis (EDA) Findings

Based on the initial analysis and visualizations:

*   **Numerical Features**: Applicants who qualified generally scored higher across most driving test components, particularly in `'Theory Test'`, `'Signals'`, `'Road Signs'`, and `'Steer Control'`.
*   **Categorical Features**:
    *   Applicants with `'Advanced'` training had a higher qualification rate.
    *   Applicants with `'Fast'` reactions had a significantly higher qualification rate.
    *   `'Gender'` and `'Race'` distributions were relatively similar in terms of qualification.
*   **Key Predictors**: `'Theory Test'` score, `'Training'` level, and `'Reactions'` appear to be strong indicators of qualification.

## Methodology

The project followed these steps:

1.  **Data Loading and Initial Exploration**: Loaded the data and examined its structure and content.
2.  **Data Cleaning**: Handled missing values in the `'Training'` column by imputing with the `mode`.
3.  **Exploratory Data Analysis (EDA)**: Generated descriptive statistics, frequency counts, histograms, and box plots to understand data distributions and relationships.
4.  **Data Preprocessing**: Dropped the `'Applicant ID'` column and applied `one-hot encoding` to categorical features.
5.  **Model Selection**: Chose `Logistic Regression`, `Random Forest`, and `LightGBM` for binary classification.
6.  **Model Training**: Trained the selected models on the training data.
7.  **Model Evaluation**: Evaluated models using `accuracy`, `precision`, `recall`, `F1-score`, and `ROC AUC score`.
8.  **Model Interpretation**: Analyzed `feature importances` from the best-performing model.
9.  **Model Saving**: Saved the trained `LightGBM` model using `joblib`.

## Model Performance

| Model              | Accuracy | Precision | Recall | F1-score | ROC AUC Score |
| :----------------- | :------- | :-------- | :----- | :------- | :------------ |
| Logistic Regression| 0.7900   | 0.7818    | 0.8269 | 0.8037   | 0.8726        |
| Random Forest      | 0.8000   | 0.7857    | 0.8462 | 0.8148   | 0.8804        |
| LightGBM           | **0.8100** | **0.8000** | **0.8462** | **0.8224** | **0.8866**    |

`LightGBM` performed best across the evaluation metrics.

## Feature Importance (LightGBM)

The most important features for predicting qualification according to the LightGBM model were:

*   `'Signals'`
*   `'Steer Control'`
*   `'Parking'`
*   `'Speed Control'`
*   `'Confidence'`
*   `'Night Drive'`
*   `'Road Signs'`
*   `'Mirror Usage'`

## Conclusion and Next Steps

The analysis successfully identified key factors influencing driver qualification and built a predictive model. The `LightGBM` model provides a good baseline for predicting qualification.

Further steps could include:

*   Hyperparameter tuning of the `LightGBM` model for potentially improved performance.
*   Exploring other advanced classification models.
*   Investigating potential interactions between features.
*   Gathering more data to improve model generalization.
