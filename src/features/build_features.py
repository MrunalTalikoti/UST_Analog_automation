import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


PROJECT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA = PROJECT_DIR / "data" / "processed" / "csamp_long.csv"
INTERIM_DATA = PROJECT_DIR / "data" / "interim"
MODELS_DIR = PROJECT_DIR / "models"


def main():
    # --------------------------------------------------
    # 1. Load transformed dataset
    # --------------------------------------------------
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(
            f"Transformed dataset not found at {PROCESSED_DATA}"
        )

    df = pd.read_csv(PROCESSED_DATA)

    # --------------------------------------------------
    # 2. Feature / target split
    # --------------------------------------------------
    X = df[["PMOS_W", "NMOS_W"]].values
    y = df["Gain"].values

    # --------------------------------------------------
    # 3. Train-test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # --------------------------------------------------
    # 4. Feature scaling (StandardScaler)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # 5. Save preprocessing artifacts
    # --------------------------------------------------
    INTERIM_DATA.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(INTERIM_DATA / "X_train.npy", X_train_scaled)
    np.save(INTERIM_DATA / "X_test.npy", X_test_scaled)
    np.save(INTERIM_DATA / "y_train.npy", y_train)
    np.save(INTERIM_DATA / "y_test.npy", y_test)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # --------------------------------------------------
    # 6. Sanity checks (prints only)
    # --------------------------------------------------
    print("✅ Feature preprocessing completed successfully")
    print(f"X_train shape : {X_train_scaled.shape}")
    print(f"X_test shape  : {X_test_scaled.shape}")
    print(f"Mean (train)  : {X_train_scaled.mean(axis=0)}")
    print(f"Std (train)   : {X_train_scaled.std(axis=0)}")


if __name__ == "__main__":
    main()
