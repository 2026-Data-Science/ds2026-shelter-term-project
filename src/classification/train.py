import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from src.classification.preprocessing import build_preprocessing_pipeline
from src.classification.features import (
    TARGET_COLUMN,
    FORBIDDEN_MODEL_INPUT_COLUMNS,
)

df = pd.read_csv("data/train.csv")

# Drop columns that are not allowed as model input
X = df.drop(
    columns=[col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
)

# Target variable
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,  # class 비율 유지
    random_state=42  # 랜덤하게 나누되, 매번 같은 결과가 나오도록 시드 고정
)

models = {
    "DummyClassifier": DummyClassifier(
        strategy="most_frequent" # train 데이터에서 가장 많은 클래스만 예측한다.
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, # 수렴을 위해 최대 반복 횟수 증가
        random_state=42,
        class_weight="balanced",
        solver="lbfgs", # 대규모 데이터셋에 적합한 최적화 알고리즘
        n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, # 트리 개수
        random_state=42,
        class_weight="balanced", # 클래스 불균형 문제 보정
        n_jobs=-1 # 병렬 처리
    )
}

param_grids = {
    "DummyClassifier": {},

    "LogisticRegression": {
        "classifier__C": [0.1, 1, 10],
        "classifier__class_weight": [None, "balanced"],
    },

    "RandomForest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
        "classifier__class_weight": [None, "balanced"],
    }
}

results = []
best_models = {}

for model_name, classifier in models.items():
    print("=" * 60)
    print(f"Tuning {model_name}...")

    model = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", classifier)
    ])

    # DummyClassifier는 튜닝할 파라미터가 없으므로 그냥 fit
    if not param_grids[model_name]:
        model.fit(X_train, y_train)
        best_model = model
        best_params = "No tuning"
    else:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[model_name],
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best params: {best_params}")
        print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")

    best_models[model_name] = best_model

    print(f"Predicting with best {model_name}...")
    pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    results.append({
        "model": model_name,
        "best_params": best_params,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
    })

    print(classification_report(y_test, pred))

print("=" * 60)
print("Tuned Model Comparison")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

print(pd.DataFrame(results).to_string())