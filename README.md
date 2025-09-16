# Employee-Turnover-Prediction-and-Retention-Strategy

## Overview

The **Employee Turnover Analytics** project is designed to predict employee turnover based on various factors such as satisfaction level, evaluation scores, average monthly hours, project count, and other employee-related metrics. This project uses Machine Learning techniques to perform data analysis, handle class imbalance, train classification models, and generate retention strategies for employees at risk of leaving.

The goal of this project is to help companies proactively retain their workforce by identifying high-risk employees and suggesting personalized retention strategies.

## Project Structure

- **Data Quality Checks:** Identify and handle missing values in the dataset.
- **Exploratory Data Analysis (EDA):** Investigate which factors contribute most to employee turnover.
- **Clustering:** Cluster employees who left based on satisfaction and evaluation scores.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Handle class imbalance in the dataset.
- **Model Training:** Perform 5-fold cross-validation on multiple models and evaluate their performance.
- **Model Evaluation:** Identify the best model based on performance metrics such as ROC/AUC and Confusion Matrix.
- **Retention Strategies:** Suggest retention strategies for targeted employees based on predicted turnover probabilities.

## Data

The dataset used for this project has been sourced from [Kaggle: HR Employee Attrition](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv). It contains the following columns:

| Column Name            | Description |
|------------------------|-------------|
| `satisfaction_level`    | Satisfaction level of the employee (numeric) |
| `last_evaluation`       | Rating between 0 and 1, received by an employee at their last evaluation |
| `number_project`        | Number of projects the employee has been involved in |
| `average_montly_hours`  | Average number of hours an employee worked per month |
| `time_spend_company`    | Number of years the employee has been with the company |
| `Work_accident`         | Whether the employee had a work accident (binary: 0 or 1) |
| `left`                  | Whether the employee left the company (binary: 0 = Stayed, 1 = Left) |
| `promotion_last_5years` | Number of promotions in the last 5 years |
| `Department`            | Department to which the employee belongs |
| `salary`                | Salary of the employee in USD |

## Steps Involved

### 1. Data Quality Checks
- Check for missing values in the dataset.
- Handle missing values appropriately.

### 2. Exploratory Data Analysis (EDA)
- **Correlation Matrix Heatmap:** Visualize correlations between numerical features.
- **Distribution Plots:** 
  - Employee Satisfaction (`satisfaction_level`)
  - Employee Evaluation (`last_evaluation`)
  - Employee Average Monthly Hours (`average_montly_hours`)
- **Bar Plot:** Analyze the number of projects worked on by employees who left vs. those who stayed.

### 3. Clustering of Employees Who Left
- Perform **K-means clustering** of employees who left based on their `satisfaction_level` and `last_evaluation` columns.

### 4. Handle Class Imbalance Using SMOTE
- Pre-process the data by converting categorical columns into numerical columns.
- Use **Stratified Train-Test Split** (80:20).
- **Apply SMOTE** to balance the class distribution in the training set.

### 5. Model Training & Evaluation
- Train three different models using **5-fold cross-validation**:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Gradient Boosting Classifier
- Evaluate models based on performance metrics:
  - **Classification Report**
  - **ROC/AUC Curve**
  - **Confusion Matrix**

### 6. Best Model Selection
- Compare models using **ROC/AUC** and **Confusion Matrix**.
- Discuss which metric (Recall or Precision) should be prioritized.

### 7. Employee Retention Strategies
- Use the best-performing model to predict the probability of employee turnover in the test data.
- Categorize employees based on predicted probability scores into four zones:
  - **Safe Zone (Green)**: Probability < 20%
  - **Low-Risk Zone (Yellow)**: 20% < Probability < 60%
  - **Medium-Risk Zone (Orange)**: 60% < Probability < 90%
  - **High-Risk Zone (Red)**: Probability > 90%
- Suggest retention strategies based on the probability score.

## Technologies Used

- **Python**: Programming language used for data processing and model building.
- **Pandas**: Data manipulation and analysis library.
- **NumPy**: Library for numerical computing.
- **Matplotlib** and **Seaborn**: Data visualization libraries.
- **Scikit-learn**: Library for machine learning and model evaluation.
- **imblearn**: SMOTE technique for handling class imbalance.
- **KMeans**: Clustering algorithm.
- **Logistic Regression, Random Forest, Gradient Boosting**: Machine learning models for classification.

## How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Utkarsh-152/Employee-Turnover-Analytics.git

2. **Open Jupyter Notebook**

3. **Execute all cells**
   
This will generate relevant outputs including visualizations, model reports, and retention strategies.

## Results & Inferences

- The EDA phase highlighted key factors contributing to employee turnover.
- Clustering revealed different employee groups based on satisfaction and evaluation scores.
- After addressing the class imbalance using SMOTE, all models showed improved performance.
- The Random Forest Classifier emerged as the best-performing model, with high ROC/AUC scores and strong classification metrics.
- Based on predicted probabilities, retention strategies were designed for employees in different risk zones.

## Conclusion
This project demonstrates the power of machine learning to predict and understand employee turnover. By analyzing various factors and implementing a robust machine learning pipeline, companies can proactively identify employees at risk of leaving and implement targeted retention strategies.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Kaggle for providing the dataset.
- Scikit-learn for the machine learning tools.
- Imbalanced-learn for the SMOTE technique.
- Matplotlib and Seaborn for data visualization.
