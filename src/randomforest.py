import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


def treat_data_for_Random_Forest():
    df = pd.read_csv("data/raw/cyberbullying_tweets.csv")
    df["label"] = (df["cyberbullying_type"] != "not_cyberbullying").astype(int)
    y_harassement = df["label"]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, temp_idx in splitter.split(df, y_harassement):
        train_split = df.iloc[train_idx]
        temp_split = df.iloc[temp_idx]

    # 50% val, 50% test dans les 20% restants (10% val / 10% test globalement)
    y_temp = temp_split["label"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in splitter.split(temp_split, y_temp):
        val_split = temp_split.iloc[val_idx]
        test_split = temp_split.iloc[test_idx]

    # Vectorisation après le split
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(train_split["tweet_text"])
    y_train = train_split["label"]

    X_val = vectorizer.transform(val_split["tweet_text"])
    y_val = val_split["label"]

    X_test = vectorizer.transform(test_split["tweet_text"])
    y_test = test_split["label"]
    return X_train, y_train, X_val, y_val, X_test, y_test, vectorizer


def train_model_precision(X_train, y_train, param_grid, scorer):
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring=scorer,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    return grid


def get_params(grid, X_test, y_test):
    y_pred = grid.predict(X_test)
    print("Best params:", grid.best_params_)
    report = classification_report(y_test, y_pred)
    print(report)
    matrix = confusion_matrix(y_test, y_pred)
    print(
        "🧪 Prédit 0    🧪 Prédit 1\n",
        f"🎯 Réel 0 (négatif)    ✅ TN = {matrix[0][0]}    ❌ FP = {matrix[0][1]}\n",
        f"🎯 Réel 1 (positif)    ❌ FN = {matrix[1][0]}    ✅ TP = {matrix[1][1]}",
    )
    result = {"metrics": report, "confusion_matrix": matrix.tolist()}
    return result


def train_rainforest():
    param_grid = {
        "n_estimators": [100, 200, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5],
    }
    scorer_1 = make_scorer(precision_score)
    scorer_2 = make_scorer(f1_score, pos_label=0)
    X_train, y_train, X_val, y_val, _, _, vectorizer = treat_data_for_Random_Forest()
    grid_1 = train_model_precision(X_train, y_train, param_grid, scorer_1)
    result_1 = get_params(grid_1, X_val, y_val)
    with open("figures/results_precision.json", "w") as file:
        json.dump(result_1, file, indent=2)
    grid_2 = train_model_precision(X_train, y_train, param_grid, scorer_2)
    joblib.dump(grid_2, "Train/random_forest_model_for_classifying.joblib")
    joblib.dump(vectorizer, "Train/vectorizer_model_for_classifying.joblib")
    result_2 = get_params(grid_2, X_val, y_val)
    with open("figures/results_f1_score.json", "w") as file:
        json.dump(result_2, file, indent=2)


if __name__ == "__main__":
    train_rainforest()
