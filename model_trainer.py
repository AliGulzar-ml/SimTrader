import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import MODEL_PATH


def train_model(df: pd.DataFrame, model_path: str = MODEL_PATH):
    """
    Train a fast gradient-boosted tree model on engineered features.
    Uses HistGradientBoostingClassifier which is optimized in C and
    runs very efficiently on Apple Silicon CPUs.
    """
    df = df.copy()
    y = df["target"].values
    feature_cols = [c for c in df.columns if c not in ("future_ret", "target")]
    X = df[feature_cols].values.astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        max_iter=400,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}, validation samples: {len(X_val)}")
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    print(f"Validation accuracy: {acc:.4f}")

    joblib.dump((model, feature_cols), model_path)
    print(f"Model saved to {model_path}")
