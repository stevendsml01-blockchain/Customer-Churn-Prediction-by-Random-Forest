import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

# ====================== SMOTE ======================
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ====================== LOAD DATA ======================
df = pd.read_csv('steva_ecommerce.csv')

FEATURES = [
    'Age',
    'Membership_Years',
    'Lifetime_Value',
    'Total_Purchases',
    'Days_Since_Last_Purchase',
    'Average_Order_Value',
    'Returns_Rate',
    'Cart_Abandonment_Rate'
]

X = df[FEATURES]
y = df['Churned']

# ====================== TRAIN TEST SPLIT ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("="*70)
print("DATA SPLIT")
print("="*70)

print(f"Train Shape : {X_train.shape}")
print(f"Test Shape  : {X_test.shape}")

print("\nClass Distribution:")
print(y.value_counts(normalize=True))

# ============================================================
# 1. BASELINE MODEL (BALANCED)
# ============================================================

print("\n" + "="*70)
print("1. BASELINE RANDOM FOREST")
print("="*70)

rf_baseline = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)
y_proba_baseline = rf_baseline.predict_proba(X_test)[:, 1]

print("\nBaseline Classification Report:")
print(classification_report(y_test, y_pred_baseline))

# ROC-AUC
roc_auc_baseline = roc_auc_score(y_test, y_proba_baseline)

# PR-AUC
pr_auc_baseline = average_precision_score(y_test, y_proba_baseline)

print(f"Baseline ROC-AUC : {roc_auc_baseline:.4f}")
print(f"Baseline PR-AUC  : {pr_auc_baseline:.4f}")

# ============================================================
# 2. SMOTE + RANDOM FOREST PIPELINE
# ============================================================

print("\n" + "="*70)
print("2. GRIDSEARCHCV + SMOTE")
print("="*70)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {

    'rf__n_estimators': [100],

    'rf__max_depth': [
        10,
        15
    ],

    'rf__min_samples_split': [
        2,
        5
    ],

    'rf__min_samples_leaf': [
        1,
        2
    ],

    'rf__max_features': [
        'sqrt',
        'log2',
        0.3
    ],

    'rf__class_weight': [
        'balanced'
    ]
}

# Stratified KFold
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='recall',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest CV Recall: {grid_search.best_score_:.4f}")

# ============================================================
# 3. BEST MODEL
# ============================================================

best_model = grid_search.best_estimator_

# Probabilities
y_proba_tuned = best_model.predict_proba(X_test)[:, 1]

# ============================================================
# 4. THRESHOLD TUNING
# ============================================================

print("\n" + "="*70)
print("3. THRESHOLD TUNING")
print("="*70)

# Default threshold 0.5
y_pred_default = (y_proba_tuned >= 0.5).astype(int)

# Lower threshold
threshold = 0.30

y_pred_adjusted = (y_proba_tuned >= threshold).astype(int)

print(f"\nThreshold Used : {threshold}")

print("\nAdjusted Classification Report:")
print(classification_report(y_test, y_pred_adjusted))

# ============================================================
# 5. METRICS
# ============================================================

roc_auc_tuned = roc_auc_score(
    y_test,
    y_proba_tuned
)

pr_auc_tuned = average_precision_score(
    y_test,
    y_proba_tuned
)

print(f"Tuned ROC-AUC : {roc_auc_tuned:.4f}")
print(f"Tuned PR-AUC  : {pr_auc_tuned:.4f}")

# ============================================================
# 6. COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    'Metric': [
        'Accuracy',
        'Precision (Churn)',
        'Recall (Churn)',
        'F1-Score (Churn)',
        'ROC-AUC',
        'PR-AUC'
    ],

    'Baseline': [

        accuracy_score(
            y_test,
            y_pred_baseline
        ),

        precision_score(
            y_test,
            y_pred_baseline
        ),

        recall_score(
            y_test,
            y_pred_baseline
        ),

        f1_score(
            y_test,
            y_pred_baseline
        ),

        roc_auc_baseline,
        pr_auc_baseline
    ],

    'Tuned': [

        accuracy_score(
            y_test,
            y_pred_adjusted
        ),

        precision_score(
            y_test,
            y_pred_adjusted
        ),

        recall_score(
            y_test,
            y_pred_adjusted
        ),

        f1_score(
            y_test,
            y_pred_adjusted
        ),

        roc_auc_tuned,
        pr_auc_tuned
    ]
})

print("\nMODEL COMPARISON")
print(comparison.round(4))

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

def plot_cm(cm, title):

    group_names = [
        'True Neg',
        'False Pos',
        'False Neg',
        'True Pos'
    ]

    group_counts = [
        f"{value:0.0f}"
        for value in cm.flatten()
    ]

    labels = [
        f"{v1}\n{v2}"
        for v1, v2 in zip(group_names, group_counts)
    ]

    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt='',
        cmap='Blues'
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

# Baseline CM
plt.figure(figsize=(6,5))

plot_cm(
    confusion_matrix(y_test, y_pred_baseline),
    "Baseline Model"
)

plt.tight_layout()
plt.show()

# Tuned CM
plt.figure(figsize=(6,5))

plot_cm(
    confusion_matrix(y_test, y_pred_adjusted),
    "Tuned Model"
)

plt.tight_layout()
plt.show()

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

rf_model = best_model.named_steps['rf']

importances = pd.Series(
    rf_model.feature_importances_,
    index=FEATURES
)

plt.figure(figsize=(10,6))

importances.sort_values().plot(
    kind='barh'
)

plt.title("Feature Importance")

plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(importances.sort_values(ascending=False))

# ============================================================
# 9. SHAP
# ============================================================

try:

    import shap

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(rf_model)

    shap_values = explainer.shap_values(X_test)

    # Kelas 1 = Churn
    shap.summary_plot(
        shap_values[1],
        X_test
    )

    shap.summary_plot(
        shap_values[1],
        X_test,
        plot_type='bar'
    )

except ImportError:

    print("\nSHAP belum terinstall.")
    print("Install dengan:")
    print("pip install shap")

print("\n✅ Pipeline selesai!")