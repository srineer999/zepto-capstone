import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os

# ==========================================
# PART A - PROFILING, CLEANING AND DATA STORY
# ==========================================

# Load local Titanic dataset
df = pd.read_csv("analytics/titanic.csv")

print("\n========== DATASET PROFILE ==========")
print("Shape:", df.shape)

print("\nINFO:")
df.info()

print("\nDESCRIBE:")
print(df.describe(include="all"))

# ------------------------------------------
# 1. MISSING VALUES
# ------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_report = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage
})

print(missing_report)

# ------------------------------------------
# 2. MISSING VALUE HANDLING
# Rule:
# < 5%   -> drop rows
# 5-30%  -> impute
# > 30%  -> drop column
# ------------------------------------------

print("\n========== MISSING VALUE STRATEGY ==========")

for column in df.columns:

    pct = missing_percentage[column]

    if pct == 0:
        continue

    elif pct < 5:
        print(column, ": <5% missing -> dropping missing rows")
        df = df.dropna(subset=[column])

    elif pct <= 30:
        print(column, ": 5-30% missing -> imputing")

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    else:
        print(column, ": >30% missing -> dropping column")
        df = df.drop(columns=[column])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ------------------------------------------
# 3. UNIVARIATE ANALYSIS - AGE AND FARE
# ------------------------------------------

print("\n========== UNIVARIATE ANALYSIS ==========")

for column in ["age", "fare"]:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    print("\nCOLUMN:", column)
    print("Mean:", df[column].mean())
    print("Median:", df[column].median())
    print("Mode:", df[column].mode()[0])
    print("Skewness:", df[column].skew())
    print("IQR Outlier Count:", len(outliers))

    plt.figure()
    sns.histplot(df[column], kde=True)
    plt.title(f"{column.title()} Distribution")
    plt.savefig(f"analytics/{column}_histogram.png")
    plt.close()

    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f"{column.title()} Boxplot")
    plt.savefig(f"analytics/{column}_boxplot.png")
    plt.close()

# ------------------------------------------
# 4. BIVARIATE ANALYSIS
# ------------------------------------------

print("\n========== SURVIVAL ANALYSIS ==========")

print("\nSurvival by Sex:")
print(df.groupby("sex")["survived"].mean())

print("\nSurvival by Class:")
print(df.groupby("class")["survived"].mean())

print("\nSurvival by Sex and Class:")
print(df.groupby(["sex", "class"], observed=True)["survived"].mean())

# Required correlation columns
corr_columns = [
    "survived",
    "age",
    "sibsp",
    "parch",
    "fare",
    "pclass"
]

correlation = df[corr_columns].corr()

print("\nCORRELATION MATRIX:")
print(correlation)

# Find strongest off-diagonal correlations
pairs = []

for i in range(len(correlation.columns)):
    for j in range(i + 1, len(correlation.columns)):

        pairs.append((
            correlation.columns[i],
            correlation.columns[j],
            correlation.iloc[i, j]
        ))

pairs = sorted(
    pairs,
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\nTWO STRONGEST CORRELATIONS:")

for pair in pairs[:2]:
    print(pair)

# ------------------------------------------
# 5. FOUR MULTIVARIATE CHARTS
# ------------------------------------------

print("\n========== CREATING CHARTS ==========")

# Chart 1
plt.figure()
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Sex")
plt.savefig("analytics/survival_by_sex.png")
plt.close()

print(
    "Chart 1 Interpretation: Survival rate differs substantially "
    "between male and female passengers."
)

# Chart 2
plt.figure()
sns.barplot(data=df, x="class", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.savefig("analytics/survival_by_class.png")
plt.close()

print(
    "Chart 2 Interpretation: Passenger class is associated "
    "with different survival rates."
)

# Chart 3
plt.figure()
sns.barplot(data=df, x="class", y="survived", hue="sex")
plt.title("Survival by Class and Sex")
plt.savefig("analytics/survival_class_sex.png")
plt.close()

print(
    "Chart 3 Interpretation: Survival patterns vary jointly "
    "according to passenger sex and class."
)

# Chart 4
plt.figure()
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("analytics/correlation_heatmap.png")
plt.close()

print(
    "Chart 4 Interpretation: The heatmap shows the strength "
    "and direction of relationships among numeric variables."
)

# ------------------------------------------
# 6. EXPLORATORY AGE Z-SCORE
# ------------------------------------------

print("\n========== AGE Z-SCORE EXPLORATION ==========")

age_before = df["age"].copy()
plt.figure()
sns.histplot(age_before, kde=True)
plt.title("Age Before Z-score Transformation")
plt.savefig("analytics/age_before_zscore.png")
plt.close()

age_zscore = zscore(df["age"])

print("Age before transformation:")
print(age_before.describe())

print("\nAge Z-score after transformation:")
print(pd.Series(age_zscore).describe())

plt.figure()
sns.histplot(age_zscore, kde=True)
plt.title("Age Z-score Distribution")
plt.savefig("analytics/age_zscore.png")
plt.close()

print(
    "\nZ-score interpretation: Standardization changes age to "
    "approximately mean 0 and standard deviation 1 while "
    "preserving the distribution shape."
)

print("\nPART A COMPLETED SUCCESSFULLY!")




# ==========================================
# PART B - PREDICTIVE MODELING
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


print("\n========== PART B: PREDICTIVE MODELING ==========")

# ------------------------------------------
# 7. TRAIN / TEST SPLIT
# ------------------------------------------

# Features and target
features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------
# 8. PREPROCESSING
# ------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ------------------------------------------
# 9. THREE CLASSIFIERS
# ------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n================================")
    print(name)
    print("================================")

    print("\nConfusion Matrix:")
    print(cm)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    # Decision tree visualization
    if name == "Decision Tree":

        transformed_X = pipeline.named_steps[
            "preprocessing"
        ].transform(X_train)

        plt.figure(figsize=(20, 10))

        plot_tree(
            model,
            filled=True,
            max_depth=3
        )

        plt.title("Decision Tree Visualization")
        plt.savefig(
            "analytics/decision_tree.png",
            bbox_inches="tight"
        )
        plt.close()

# ------------------------------------------
# MODEL COMPARISON TABLE
# ------------------------------------------

comparison = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
)

print("\n========== CLASSIFIER COMPARISON ==========")
print(comparison)

comparison.to_csv(
    "analytics/classifier_comparison.csv",
    index=False
)

# ------------------------------------------
# 11. CLASS IMBALANCE COMPARISON
# Baseline vs Balanced vs SMOTE
# ------------------------------------------

print("\n========== CLASS IMBALANCE COMPARISON ==========")

# Baseline Logistic Regression
baseline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_f1 = f1_score(y_test, baseline_pred)

print("\nBaseline F1:", baseline_f1)


# Balanced Logistic Regression
balanced = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

balanced.fit(X_train, y_train)

balanced_pred = balanced.predict(X_test)

balanced_f1 = f1_score(y_test, balanced_pred)

print("Balanced F1:", balanced_f1)


# SMOTE - training data only
smote_model = ImbPipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

smote_model.fit(X_train, y_train)

smote_pred = smote_model.predict(X_test)

smote_f1 = f1_score(y_test, smote_pred)

print("SMOTE F1:", smote_f1)

print("\nImbalance conclusion:")

if balanced_f1 >= baseline_f1 and balanced_f1 >= smote_f1:
    print(
        "The class-weight balanced approach produced the "
        "best F1 score among the three approaches."
    )

elif smote_f1 >= baseline_f1:
    print(
        "SMOTE produced the best F1 score among the "
        "three approaches."
    )

else:
    print(
        "The baseline model produced the best F1 score, "
        "so additional imbalance handling did not improve it."
    )

print("\nPART B CLASSIFICATION COMPLETED!")



# ==========================================
# 12. RANDOM FOREST HYPERPARAMETER TUNING
# ==========================================

from sklearn.model_selection import GridSearchCV

print("\n========== RANDOM FOREST TUNING ==========")

rf_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        oob_score=True
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(grid_search.best_score_)

best_rf = grid_search.best_estimator_

print("\nRandom Forest OOB Score:")
print(best_rf.named_steps["model"].oob_score_)

# Evaluate tuned model
tuned_pred = best_rf.predict(X_test)
tuned_prob = best_rf.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Test Results:")
print("Accuracy:", accuracy_score(y_test, tuned_pred))
print("Precision:", precision_score(y_test, tuned_pred))
print("Recall:", recall_score(y_test, tuned_pred))
print("F1:", f1_score(y_test, tuned_pred))
print("ROC-AUC:", roc_auc_score(y_test, tuned_prob))

print("\nHYPERPARAMETER TUNING COMPLETED!")



# ==========================================
# 13. REGRESSION - PREDICT FARE
# ==========================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n========== REGRESSION ==========")

# Use cleaned data
reg_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

X_reg = df[reg_features]
y_reg = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical = [
    "sex",
    "embarked"
]

reg_preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        reg_numeric
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        reg_categorical
    )
])

regression_pipeline = Pipeline([
    ("preprocessing", reg_preprocessor),
    ("model", LinearRegression())
])

regression_pipeline.fit(Xr_train, yr_train)

yr_pred = regression_pipeline.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
mse = mean_squared_error(yr_test, yr_pred)
rmse = np.sqrt(mse)
r2 = r2_score(yr_test, yr_pred)

# Adjusted R2
n = len(yr_test)
p = len(
    regression_pipeline
    .named_steps["preprocessing"]
    .get_feature_names_out()
)

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nRegression Metrics:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

# Residuals
residuals = yr_test - yr_pred

plt.figure()
plt.scatter(yr_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Fare Regression Residual Plot")
plt.savefig(
    "analytics/regression_residuals.png",
    bbox_inches="tight"
)
plt.close()

print("\nHeteroscedasticity conclusion:")
print(
    "The residual plot should be inspected for a funnel-shaped "
    "pattern. A visible changing spread of residuals suggests "
    "heteroscedasticity; a roughly constant spread suggests "
    "approximately constant variance."
)

print("\nREGRESSION COMPLETED!")


# ==========================================
# 14. FINAL MODEL COMPARISON
# ==========================================

print("\n========== FINAL MODEL COMPARISON ==========")

# Add tuned Random Forest to comparison
tuned_row = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy_score(y_test, tuned_pred),
    "Precision": precision_score(y_test, tuned_pred),
    "Recall": recall_score(y_test, tuned_pred),
    "F1": f1_score(y_test, tuned_pred),
    "ROC-AUC": roc_auc_score(y_test, tuned_prob)
}])

final_classification = pd.concat(
    [comparison, tuned_row],
    ignore_index=True
)

print("\nClassification Models:")
print(final_classification)

final_classification.to_csv(
    "analytics/final_classification_comparison.csv",
    index=False
)

# Regression comparison
regression_comparison = pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}])

print("\nRegression Model:")
print(regression_comparison)

regression_comparison.to_csv(
    "analytics/regression_comparison.csv",
    index=False
)

# Recommendation
best_model_row = final_classification.loc[
    final_classification["F1"].idxmax()
]

print("\n========== FINAL RECOMMENDATION ==========")

print(
    f"Based on the F1 score, {best_model_row['Model']} "
    f"performed best among the evaluated classifiers."
)

print(
    "The model provides a useful balance between precision "
    "and recall for survival prediction."
)

print(
    "The classification models are suitable for predicting "
    "the Titanic survival target."
)

print(
    "For regression, Linear Regression achieved an R2 of "
    f"{r2:.3f} for predicting fare."
)


# ==========================================
# 15. SAVE COMPLETE PIPELINE
# ==========================================

import joblib

# Save the tuned Random Forest pipeline
joblib.dump(
    best_rf,
    "analytics/titanic_survival_pipeline.joblib"
)

print("\nSaved complete pipeline:")
print("analytics/titanic_survival_pipeline.joblib")

# Load it again
loaded_pipeline = joblib.load(
    "analytics/titanic_survival_pipeline.joblib"
)

# Test with raw, unprocessed data
raw_sample = X_test.iloc[:1]

loaded_prediction = loaded_pipeline.predict(raw_sample)

print("\nRaw sample prediction:")
print(loaded_prediction)

print("\nFINAL ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")