"""Rebuild data/processed/*.csv from raw UCI downloads.

Produces the numeric-only, headerless, variables-as-columns CSVs that
scripts/realworld_benchmark.py expects, matching the paper's setup:
  - uci_air_quality.csv      13 numeric vars, 5000 steps
  - uci_appliances_energy.csv 28 numeric vars, 5000 steps
  - uci_metro_traffic.csv     5 numeric vars, 5000 steps
"""
import gzip
import os
import shutil

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)


def air_quality():
    df = pd.read_csv(os.path.join(RAW, "AirQualityUCI.csv"), sep=";", decimal=",")
    # Drop date/time and trailing empty columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    # -200 is the dataset's missing-value sentinel; forward-fill like the paper's pipeline
    df = df.replace(-200, np.nan).ffill().bfill()
    df = df.iloc[:5000]
    assert df.shape[1] == 13, df.shape
    df.to_csv(os.path.join(OUT, "uci_air_quality.csv"), index=False, header=False)
    print("air_quality:", df.shape)


def appliances():
    df = pd.read_csv(os.path.join(RAW, "energydata_complete.csv"))
    df = df.drop(columns=["date"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    df = df.iloc[:5000]
    assert df.shape[1] == 28, df.shape
    df.to_csv(os.path.join(OUT, "uci_appliances_energy.csv"), index=False, header=False)
    print("appliances:", df.shape)


def metro():
    gz = os.path.join(RAW, "Metro_Interstate_Traffic_Volume.csv.gz")
    csv = os.path.join(RAW, "Metro_Interstate_Traffic_Volume.csv")
    if not os.path.exists(csv):
        with gzip.open(gz, "rb") as f_in, open(csv, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    df = pd.read_csv(csv)
    df = df[["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]]
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    df = df.iloc[:5000]
    assert df.shape[1] == 5, df.shape
    df.to_csv(os.path.join(OUT, "uci_metro_traffic.csv"), index=False, header=False)
    print("metro:", df.shape)


if __name__ == "__main__":
    air_quality()
    appliances()
    metro()
