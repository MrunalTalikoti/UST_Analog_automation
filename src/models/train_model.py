import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib


# --------------------------------------------------
# Paths (project-root safe)
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]

INTERIM_DATA = PROJECT_DIR / "data" / "interim"
MODELS_DIR = PROJECT_DIR / "models"


def main():
    # --------------------------------------------------
    # 1. Load preprocessed data
    # --------------------------------------------------
    X_train = np.load(INTERIM_DATA / "X_train.npy")
    X_test  = np.load(INTERIM_DATA / "X_test.npy")
    y_train = np.load(INTERIM_DATA / "y_train.npy")
    y_test  = np.load(INTERIM_DATA / "y_test.npy")

    # --------------------------------------------------
    # 2. Define model (baseline)
    # --------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------
    # 3. Train model
    # --------------------------------------------------
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # 4. Evaluate model
    # --------------------------------------------------
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ Model training complete")
    print(f"RMSE : {rmse:.4f} dB")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------
    # 5. Save trained model
    # --------------------------------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "gain_model.pkl")

    print("📦 Model saved to models/gain_model.pkl")


if __name__ == "__main__":
    main()
