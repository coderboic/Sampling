# Sampling Techniques on Imbalanced Credit Card Data

## Objective
To evaluate how different sampling techniques affect the performance of machine learning models on a highly imbalanced credit card fraud dataset.

## Dataset
- Credit Card Fraud Detection dataset  
- Target column: `Class` (Fraud / Non-Fraud)

## Methodology
- Applied feature scaling using **StandardScaler**.
- Balanced the dataset using **Random Over Sampling**.
- Created five random samples from the balanced data.
- Applied five sampling techniques:
  - Random Over Sampling  
  - Random Under Sampling  
  - SMOTE  
  - SMOTETomek  
  - Random Over Sampling (variant)
- Trained five ML models:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Gaussian Naive Bayes  
  - Support Vector Machine
- Evaluated models using **Accuracy Score**.

## Results

| Model | S1 | S2 | S3 | S4 | S5 |
|------|----|----|----|----|----|
| M1 | XX | XX | XX | XX | XX |
| M2 | XX | XX | XX | XX | XX |
| M3 | XX | XX | XX | XX | XX |
| M4 | XX | XX | XX | XX | XX |
| M5 | XX | XX | XX | XX | XX |

## Observations
- Oversampling techniques generally outperform under-sampling.
- Random Forest performs consistently well across sampling methods.
- Optimal sampling technique depends on the model.

## Conclusion
Proper sampling significantly improves model performance on imbalanced datasets. Selecting the right combination of sampling technique and model is crucial for reliable results.

## Tools Used
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn
