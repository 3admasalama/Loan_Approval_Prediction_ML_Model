Loan Approval Prediction using Machine Learning

This project applies Logistic Regression and a Decision Tree Classifier to predict loan approval status using customer financial and personal information. The workflow includes data preprocessing, class balancing with SMOTE, model trxaining, and evaluation with metrics and visualizations.

📂 Project Workflow
1. Data Preprocessing

Loaded the dataset and checked column types.

Encoded categorical features (education, self_employed, loan_status) into numeric values.

Scaled features for Logistic Regression to improve performance.

2. Handling Class Imbalance (SMOTE)

Original data was imbalanced (more approvals than rejections).

Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

Visualized the distribution of loan status before and after SMOTE using heatmaps.

3. Model Training

Logistic Regression trained on scaled features.

Decision Tree trained without scaling (trees don’t require scaling).

4. Model Evaluation

We used several metrics to evaluate model performance:

Accuracy → Overall correctness of predictions.

Precision → Of all predicted approvals, how many were actually correct?

Recall (Sensitivity) → Of all actual approvals, how many were correctly predicted?

F1-Score → Harmonic mean of Precision & Recall (balances both).

Macro Avg → Average of metrics across both classes (treats each class equally).

Weighted Avg → Average weighted by support (accounts for imbalance).

5. Visualizations
a) Class Distribution Before and After SMOTE

Shows how SMOTE balances the dataset.

b) Confusion Matrix

Each confusion matrix shows True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP).

Helps understand where models make mistakes (rejecting approved loans or approving bad loans).

c) ROC Curve

Plots True Positive Rate vs False Positive Rate at different thresholds.

The Area Under the Curve (AUC) indicates model performance (closer to 1 = better).

📊 Results

Logistic Regression performed consistently with good balance between precision and recall.

Decision Tree performed well but tended to overfit in some cases.

SMOTE improved Recall significantly, meaning fewer missed approvals.

🚀 Next Steps

Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).

Try ensemble models (Random Forest, XGBoost) for improved accuracy.

Deploy model as a simple Flask/Django API or Streamlit web app for real-time predictions.

🛠️ Tech Stack

Python

Pandas, NumPy (Data handling)

Scikit-learn (ML models, SMOTE, evaluation metrics)

Matplotlib, Seaborn (Visualization)

📌 This project was developed as part of the Elevvo Pathways Internship (Machine Learning Track), focusing on applied ML problem-solving and model interpretability.

The dataset (Loan-Approval-Prediction-Dataset by KAI) link: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset/data
