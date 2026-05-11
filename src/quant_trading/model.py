from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from .features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelResult:
    model: XGBClassifier
    feature_columns: list[str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    y_test: pd.Series
    predictions: pd.Series
    probabilities: pd.Series
    metrics: dict[str, float]


def _safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def train_xgboost_model(frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> ModelResult:
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5 for a walk-forward split.")

    usable = frame.dropna().copy().sort_index()
    feature_columns = [column for column in FEATURE_COLUMNS if column in usable.columns]
    if not feature_columns:
        raise ValueError("No feature columns are available for training.")

    split_index = max(1, int(len(usable) * (1 - test_size)))
    if split_index >= len(usable):
        raise ValueError("Not enough data to create a test split.")

    train_frame = usable.iloc[:split_index].copy()
    test_frame = usable.iloc[split_index:].copy()

    x_train = train_frame[feature_columns]
    y_train = train_frame["target"].astype(int)
    x_test = test_frame[feature_columns]
    y_test = test_frame["target"].astype(int)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=1,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    predicted_probabilities = pd.Series(model.predict_proba(x_test)[:, 1], index=test_frame.index, name="probability_up")
    predictions = pd.Series((predicted_probabilities >= 0.5).astype(int), index=test_frame.index, name="prediction")

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_test, predicted_probabilities),
        "train_samples": float(len(train_frame)),
        "test_samples": float(len(test_frame)),
    }

    return ModelResult(
        model=model,
        feature_columns=feature_columns,
        train_frame=train_frame,
        test_frame=test_frame,
        y_test=y_test,
        predictions=predictions,
        probabilities=predicted_probabilities,
        metrics=metrics,
    )
