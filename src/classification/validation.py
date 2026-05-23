import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score

from src.classification.preprocessing import build_preprocessing_pipeline
from src.classification.features import (
    TARGET_COLUMN,
    FORBIDDEN_MODEL_INPUT_COLUMNS,
)

# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv("data/train.csv")

# 모델 입력에 사용하면 안 되는 컬럼 제거
X = df.drop(
    columns=[col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
)

# 정답 라벨
y = df[TARGET_COLUMN]

# =========================
# 2. 모델 정의
# =========================
models = {
    "DummyClassifier": DummyClassifier(
        strategy="most_frequent"
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# =========================
# 3. K-Fold 설정
# =========================
# StratifiedKFold:
# 각 fold마다 Adoption, Transfer, Died 등의 클래스 비율이 비슷하게 유지되도록 나눔
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 평가 지표
scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

# =========================
# 4. K-Fold 검증 실행
# =========================
results = []

for model_name, classifier in models.items():
    print("=" * 60)
    print(f"Evaluating {model_name} with 5-Fold Cross Validation...")

    model = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", classifier)
    ])

    cv_result = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    result = {
        "model": model_name,
        "accuracy_mean": cv_result["test_accuracy"].mean(),
        "accuracy_std": cv_result["test_accuracy"].std(),
        "macro_f1_mean": cv_result["test_macro_f1"].mean(),
        "macro_f1_std": cv_result["test_macro_f1"].std(),
        "weighted_f1_mean": cv_result["test_weighted_f1"].mean(),
        "weighted_f1_std": cv_result["test_weighted_f1"].std(),
    }

    results.append(result)

    print(f"Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    print(f"Macro F1: {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}")
    print(f"Weighted F1: {result['weighted_f1_mean']:.4f} ± {result['weighted_f1_std']:.4f}")

print("=" * 60)
print("Model Comparison")
print(pd.DataFrame(results))