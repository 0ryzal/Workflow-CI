import pandas as pd
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def execute_model_training():
    dataset_df = pd.read_csv("breast_cancer_preprocessing.csv")
    predictors = dataset_df.drop('target', axis=1)
    target_vals = dataset_df['target']
    
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(predictors, target_vals, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("MSML_Basic_Breast_Cancer")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        clf_rf = RandomForestClassifier(random_state=42)
        clf_rf.fit(X_train_data, y_train_data)
        
        preds = clf_rf.predict(X_test_data)
        probs = clf_rf.predict_proba(X_test_data)[:, 1]
        
        acc = accuracy_score(y_test_data, preds)
        prec = precision_score(y_test_data, preds)
        rec = recall_score(y_test_data, preds)
        f1_metric = f1_score(y_test_data, preds)
        roc_metric = roc_auc_score(y_test_data, probs)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1_metric:.4f}")
        print(f"ROC-AUC: {roc_metric:.4f}")

if __name__ == "__main__":
    execute_model_training()
