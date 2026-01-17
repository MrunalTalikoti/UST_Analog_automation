import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_DIR / "data" / "raw" / "CSAmp.csv"
PROCESSED_DATA = PROJECT_DIR / "data" / "processed" / "csamp_long.csv"


def main():
    # Load
    df = pd.read_csv(RAW_DATA)

    # Transform
    df = df.rename(columns={df.columns[0]: "PMOS_W"})
    df_long = df.melt(
        id_vars="PMOS_W",
        var_name="NMOS_W",
        value_name="Gain"
    )

    # Type safety
    df_long["PMOS_W"] = pd.to_numeric(df_long["PMOS_W"])
    df_long["NMOS_W"] = pd.to_numeric(df_long["NMOS_W"])
    df_long["Gain"] = pd.to_numeric(df_long["Gain"], errors="coerce")

    # Drop invalid points
    df_long = df_long.dropna()

    # Save
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(PROCESSED_DATA, index=False)

    print(" Dataset Transformation complete.")


if __name__ == "__main__":
    main()
