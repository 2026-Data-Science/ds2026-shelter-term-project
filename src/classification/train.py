import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

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

results = []

for model_name, classifier in models.items():
    print("=" * 60)
    print(f"Training {model_name}...")

    model = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    print(f"Predicting with {model_name}...")
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    })

    print(classification_report(y_test, pred))

print("=" * 60)
print("Model Comparison")
print(pd.DataFrame(results))