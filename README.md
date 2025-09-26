# **Loan Approval Prediction using Machine Learning**

This project applies Logistic Regression and a Decision Tree Classifier to predict loan approval status using customer financial and personal information. The workflow includes data preprocessing, class balancing with SMOTE, model trxaining, and evaluation with metrics and visualizations.

## **📂 Project Workflow**
**1. Data Preprocessing**

Loaded the dataset and checked column types.

Encoded categorical features (education, self_employed, loan_status) into numeric values.

Scaled features for Logistic Regression to improve performance.

**2. Handling Class Imbalance (SMOTE)**

Original data was imbalanced (more approvals than rejections).

Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

Visualized the distribution of loan status before and after SMOTE using heatmaps.

**3. Model Training**

Logistic Regression trained on scaled features.

Decision Tree trained without scaling (trees don’t require scaling).

**4. Model Evaluation**

We used several metrics to evaluate model performance:

Accuracy → Overall correctness of predictions.

Precision → Of all predicted approvals, how many were actually correct?

Recall (Sensitivity) → Of all actual approvals, how many were correctly predicted?

F1-Score → Harmonic mean of Precision & Recall (balances both).

Macro Avg → Average of metrics across both classes (treats each class equally).

Weighted Avg → Average weighted by support (accounts for imbalance).

**5. Visualizations**



a) Correlation Heatmap

The correlation heatmap shows how different features in the dataset are related to each other.

Helps detect redundant features (high correlation between them).

Highlights important predictors that might affect the target (loan_status).

Provides insights for feature selection and model improvement.

<img width="1327" height="689" alt="download" src="https://github.com/user-attachments/assets/73cc6261-d781-4fe2-add6-7f0deab9ff7f" />


b) Class Distribution Before and After SMOTE

Shows how SMOTE balances the dataset.

<img width="530" height="386" alt="download" src="https://github.com/user-attachments/assets/95608aa0-c8b7-4954-aea2-0cb923748fe7" />
<img width="530" height="386" alt="download" src="https://github.com/user-attachments/assets/9fc34f68-f454-45cc-a45f-d9e73cafc1bb" />


c) Confusion Matrix

Each confusion matrix shows True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).

Helps understand where models make mistakes (rejecting approved loans or approving bad loans).

<img width="454" height="391" alt="download" src="https://github.com/user-attachments/assets/c9770c59-27ef-44a3-bc57-dd619f8c745f" />
<img width="450" height="391" alt="download" src="https://github.com/user-attachments/assets/3c61ae3f-ee48-4a40-a646-534d08f6a7f4" />


d) ROC Curve

Plots True Positive Rate vs False Positive Rate at different thresholds.

The Area Under the Curve (AUC) indicates model performance (closer to 1 = better).

<img width="613" height="468" alt="download" src="https://github.com/user-attachments/assets/f3383f73-3a2b-4dbf-9a8e-b4070686ab0a" />


## **📊 Results**

Logistic Regression performed consistently with good balance between precision and recall.

Decision Tree performed well but tended to overfit in some cases.

SMOTE improved Recall significantly, meaning fewer missed approvals.

<img width="691" height="372" alt="download" src="https://github.com/user-attachments/assets/f7030ea7-eefb-4c3b-b4ba-bf97ed784948" />



## **🚀 Next Steps**

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

Try ensemble models (Random Forest, XGBoost) for improved accuracy.

Deploy model as a simple Flask/Django API or Streamlit web app for real-time predictions.

## **🛠️ Tech Stack**

Python

Pandas, NumPy (Data handling)

Scikit-learn (ML models, SMOTE, evaluation metrics)

Matplotlib, Seaborn (Visualization)

## 📌 **Acknowledgement**
This project was developed as part of the Elevvo Pathways Internship (Machine Learning Track), focusing on applied ML problem-solving and model interpretability.

The dataset (Loan-Approval-Prediction-Dataset by KAI) link: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
