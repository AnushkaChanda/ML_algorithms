# Epileptic Seizure Recognition – Data Preprocessing & Balancing

This project focuses on preprocessing and balancing the **Epileptic Seizure Recognition dataset** using key data science techniques. The goal is to prepare the dataset for accurate seizure classification in future modeling tasks.

## 📊 Key Features:
- Loaded dataset and performed basic exploration (shape, types, missing values)
- Analyzed class imbalance in the target variable `y`
- Applied **SMOTE** to balance seizure vs. non-seizure classes
- Standardized feature values using `StandardScaler`
- Split the data into training and testing sets (80/20 stratified split)
- Visualized class distribution before balancing

## 📁 Dataset:
The dataset used is: `Epileptic Seizure Recognition.csv`  
(Make sure to place the dataset in the same directory or adjust the path.)

## 📌 Technologies Used:
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn (`SMOTE`)

## 🚀 Next Steps (Optional Enhancements):
- Apply classification algorithms (e.g., Logistic Regression, Random Forest)
- Evaluate model performance with metrics and visualizations
- Deploy or convert into a notebook for further analysis

## ✅ Output:
- Balanced dataset with equal distribution of seizure classes
- Scaled features ready for ML models
- Barplot visualizing original class imbalance