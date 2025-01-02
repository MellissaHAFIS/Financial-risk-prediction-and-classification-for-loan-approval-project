# Financial risk prediction and classification for loan approval
![image.png](attachment:3763ecd7-482f-4d84-be17-38f693e67101.png)

## Overview
In the modern financial landscape, effective risk assessment and accurate loan approval predictions are crucial for minimizing losses and ensuring the sustainability of financial institutions. The ability to evaluate the financial health and reliability of loan applicants can streamline decision-making processes, reduce defaults, and optimize resource allocation.

This project leverages a synthetic dataset of personal and financial data to address two key challenges:

 **1. Risk Assessment:** Estimating **a continuous Risk Score** that reflects the likelihood of financial instability or default for each applicant.
 
 **2. Loan Approval Prediction:** Developing a binary classification model to determine whether a loan application should be **approved or denied** based on the applicant’s profile.
 
By building robust machine learning models, this project aims to provide insights into key factors influencing risk and loan approval decisions, offering a foundation for improved lending strategies.

Then, this project demonstrates a complete machine learning pipeline for both regression and classification tasks, including data preprocessing, feature engineering, model selection, hyperparameter optimization, and performance evaluation. The work aims to build robust predictive models and follows best practices in machine learning workflows.

## Problem satement: 
This project addresses two specific tasks:

 1. Risk Score Prediction: Predict a continuous variable, the RiskScore, which quantifies the risk associated with an individual.
    - Example: Predicting a RiskScore of 0.75 for an applicant suggests a higher likelihood of default than a score of 0.25.
     
 2. Loan Approval Prediction: Perform binary classification to determine the loan approval outcome (LoanApproved = 1 for approval, 0 for denial).
    - Example: Based on the applicant's data, predicting LoanApproved = 0 means the applicant is not eligible for the loan.

## Objectives of the project

1. Develop **a regression model** to accurately predict the RiskScore for each applicant (Supervised learning -Offline).
2. Build **a classification model** to predict whether a loan application will be approved (Supervised learning -Offline).
3. Analyze and interpret key features influencing risk and approval decisions.
4. Evaluate the performance of the models using appropriate metrics.
6. Present actionable insights for improving risk assessment and decision-making processes.
7. Give and developpe the solution for this problem in 3 weeks
   
## Dataset
It's a synthetic sataset for Risk Assessment and Loan Approval Modeling that we toke from Kaggle https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval/data.

This synthetic dataset comprises 20,000 records and 36 features of personal and financial data, designed to facilitate the development of predictive models for risk assessment. 

The dataset includes diverse features such as demographic information, credit history, employment status, income levels, existing debt, and other relevant financial metrics, providing a comprehensive foundation for sophisticated data-driven analysis and decision-making.

The dataset consists of:
- **Features**: Numeric and categorical features, processed through feature engineering.
- **Target Variables**:
  - Regression: A continuous target variable.
  - Classification: A binary target variable.

## Pipeline Workflow
1. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Scaled numeric features and encoded categorical features.
   - Combined and engineered features for enhanced model input.

2. **Model Selection**:
   - Regression Models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - XGBoost Regressor
   - Classification Models:
     - Logistic Regression
     - k-Nearest Neighbors
     - Decision Tree
     - Support Vector Classifier
     - Naive Bayes
     - Random Forest
     - Gradient Boosting
     - AdaBoost
     - XGBoost
     - CatBoost

3. **Model Evaluation**:
   - Cross-validation used to assess model performance.
   - Metrics:
     - Regression: R² score.
     - Classification: Accuracy, precision, recall, and F1 score.

4. **Hyperparameter Optimization**:
   - Tuned hyperparameters for XGBoost and Logistic Regression using GridSearchCV.
   - Enhanced model performance with optimized configurations.

5. **Results**:
   - **Best Regression Model**: XGBoost Regressor
     - Mean R² Score: 0.8921
   - **Best Classification Model**: Logistic Regression
     - Mean Accuracy: 0.9643

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - scikit-learn
  - XGBoost
  - CatBoost
  - NumPy
  - pandas

## Results and Insights
- Identified optimal models for both regression and classification tasks.
- Demonstrated the importance of cross-validation and hyperparameter tuning.
- Achieved high performance with selected models.
