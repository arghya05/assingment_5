# Machine Learning Models Repository

This repository contains two main components:
1. A comprehensive CTR (Click-Through Rate) Prediction Pipeline
2. A Lightweight MNIST Classification Model

## 1. CTR Prediction Pipeline

### Overview
A production-ready pipeline for CTR prediction that automatically:
- Performs intelligent feature engineering
- Compares multiple state-of-the-art models
- Optimizes hyperparameters
- Provides model explanations using SHAP and LIME

### Features

#### Automated Feature Engineering
- Time-based features (hour, day of week, weekend flags)
- Location clustering
- Historical CTR patterns
- Distance calculations
- Interaction features

#### Model Selection
Automatically compares and selects the best model from:
- XGBoost
- LightGBM
- CatBoost

#### Performance Metrics
Evaluates models using:
- RMSE (Root Mean Square Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

#### Model Explainability
- SHAP (SHapley Additive exPlanations) values
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance rankings

### Usage
