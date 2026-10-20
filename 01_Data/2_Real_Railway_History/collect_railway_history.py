
"""Download the historical arrival/departure records of your trains from Hugging Face into this folder.

IMPORTANT: do NOT open the generated CSV files in Excel and save them. Excel keeps only 1,048,575 rows and silently cuts the rest.
Run check_history_completeness.py afterwards to confirm the file is complete.
"""
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

BASE_DIR = Path(__file__).resolve().parent   # this folder: 01_Data/2_Real_Railway_History
TRAIN_LIST = BASE_DIR / "railway_real_eta.txt"   # one train number per line

REPO_ID = "yadavsaurabh/indian-railways-runs"

ALL_OUTPUT = BASE_DIR / "historical_all_rows.csv"
OBS_OUTPUT = BASE_DIR / "historical_reported_observations.csv"
SUMMARY_OUTPUT = BASE_DIR / "train_coverage_summary.csv"
MISSING_OUTPUT = BASE_DIR / "missing_train_files.txt"

BASE_DIR.mkdir(parents=True, exist_ok=True)

if not TRAIN_LIST.exists():
    raise SystemExit(
        f"Create {TRAIN_LIST} and paste one train number per line."
    )

# Preserve train numbers as strings and remove duplicates.
train_numbers = []
for line in TRAIN_LIST.read_text(encoding="utf-8-sig").splitlines():
    value = line.strip()
    if value:
        number = value.split(".")[0]
        if number.isdigit():
            train_numbers.append(number.zfill(5))

train_numbers = list(dict.fromkeys(train_numbers))

# Start fresh so repeated runs don't duplicate old records.
for path in (ALL_OUTPUT, OBS_OUTPUT, SUMMARY_OUTPUT, MISSING_OUTPUT):
    if path.exists():
        path.unlink()

summaries = []
missing = []
all_header_written = False
obs_header_written = False

for index, train_no in enumerate(train_numbers, start=1):
    # Dataset folders are split by the first digit.
    folder = "trains_1" if train_no[0] in "56789" else "trains"
    filename = f"{folder}/{train_no}.csv"

    print(f"[{index}/{len(train_numbers)}] Checking train {train_no}...")

    try:
        csv_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename
        )
    except EntryNotFoundError:
        print("  File not present in dataset.")
        missing.append(train_no)
        continue
    except Exception as exc:
        print(f"  Download error: {exc}")
        missing.append(f"{train_no} - ERROR: {exc}")
        continue

    try:
        df = pd.read_csv(
            csv_path,
            dtype={
                "train_no": str,
                "station_code": str,
                "platform": str
            }
        )
    except Exception as exc:
        print(f"  Could not read CSV: {exc}")
        missing.append(f"{train_no} - READ ERROR: {exc}")
        continue

    # Normalize identifiers and flags.
    df["train_no"] = df["train_no"].astype("string").str.zfill(5)
    reported = pd.to_numeric(df["reported"], errors="coerce").eq(1)

    has_actual_arrival = (
        df["act_arrival"].notna()
        & df["act_arrival"].astype("string").str.strip().ne("")
    )
    has_actual_departure = (
        df["act_departure"].notna()
        & df["act_departure"].astype("string").str.strip().ne("")
    )

    # Keep the original rows, including unreported timetable stops.
    df.to_csv(
        ALL_OUTPUT,
        mode="a",
        index=False,
        header=not all_header_written
    )
    all_header_written = True

    # Reported observations with at least one actual time.
    observed = df.loc[
        reported & (has_actual_arrival | has_actual_departure)
    ].copy()

    if not observed.empty:
        observed.to_csv(
            OBS_OUTPUT,
            mode="a",
            index=False,
            header=not obs_header_written
        )
        obs_header_written = True

    summaries.append({
        "train_no": train_no,
        "train_name": (
            df["train_name"].dropna().iloc[0]
            if df["train_name"].notna().any() else ""
        ),
        "total_station_rows": len(df),
        "reported_station_rows": int(reported.sum()),
        "rows_with_actual_arrival": int(
            (reported & has_actual_arrival).sum()
        ),
        "rows_with_actual_departure": int(
            (reported & has_actual_departure).sum()
        ),
        "rows_with_any_actual_time": len(observed),
        "journey_dates": df["date"].nunique(),
        "first_journey_date": df["date"].min(),
        "last_journey_date": df["date"].max()
    })

    print(
        f"  Rows: {len(df)}, reported: {int(reported.sum())}, "
        f"with actual times: {len(observed)}"
    )

pd.DataFrame(summaries).to_csv(SUMMARY_OUTPUT, index=False)

MISSING_OUTPUT.write_text(
    "\n".join(missing),
    encoding="utf-8"
)

print("\nCollection finished.")
print("All rows:              ", ALL_OUTPUT)
print("Reported observations: ", OBS_OUTPUT)
print("Coverage summary:      ", SUMMARY_OUTPUT)
print("Missing/error files:   ", MISSING_OUTPUT)
print("Train files found:     ", len(summaries))
print("Train files missing:   ", len(missing))
