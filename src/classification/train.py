import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier

from src.classification.preprocessing import build_preprocessing_pipeline
from src.classification.features import (
    TARGET_COLUMN,
    FORBIDDEN_MODEL_INPUT_COLUMNS,
)

# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv("data/train.csv")

X = df.drop(
    columns=[col for col in FORBIDDEN_MODEL_INPUT_COLUMNS if col in df.columns]
)

y = df[TARGET_COLUMN]

# =========================
# 2. Train / Test 분리
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 3. 최적 모델 정의
# =========================
models = {
    "DummyClassifier": DummyClassifier(
        strategy="most_frequent" # 최소 기준선(baseline) 모델인 Dummy는 따로 하이퍼파라미터 튜닝이 필요 없다.
    ),

    "LogisticRegression": LogisticRegression(
        C=1,
        class_weight=None,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# =========================
# 4. 학습 및 평가
# =========================
results = []

for model_name, classifier in models.items():

    print("=" * 60)
    print(f"Training {model_name}...")

    model = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", classifier)
    ])

    # 학습
    model.fit(X_train, y_train)

    # 예측
    print(f"Predicting with {model_name}...")
    pred = model.predict(X_test)

    # 성능 계산
    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    # 상세 리포트 출력
    print(classification_report(y_test, pred))

# =========================
# 5. 최종 비교
# =========================
print("=" * 60)
print("Final Model Comparison")

result_df = pd.DataFrame(results)

pd.set_option("display.max_columns", None)
print(result_df.to_string(index=False))