#!/usr/bin/env python3
"""
train_model.py

Trains a Random Forest classifier on the CSV produced by extract_features.py.

Split strategy (read this before trusting the results):
  For any label with 2+ independent captures (source_file), this script
  holds out whole file(s) for testing -- a proper capture-level split,
  so the model is tested on traffic it has never seen at all.

  For any label with only ONE capture (currently RECON, ARP_DOS), a true
  capture-level split isn't possible yet. As a fallback, that label's
  windows are split CHRONOLOGICALLY within the single file (first
  --train-frac of time-ordered windows -> train, rest -> test). This is
  weaker evidence of generalisation -- capture 2-3 more independent
  sessions for these classes to remove this fallback.

Class imbalance:
  Uses class_weight='balanced' in the Random Forest so classes with far
  fewer rows (e.g. RECON/ARP_DOS) aren't drowned out by classes with many
  more rows (e.g. MITM). This does NOT replace collecting more data for
  under-represented classes -- it just stops the model ignoring them
  during training.

Usage:
    python3 train_model.py --csv dataset.csv --train-frac 0.7 --test-file-frac 0.3 --model rf_model.pkl
"""

import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Columns that describe WHERE a row came from, not network behaviour.
# These must NOT be used as model features or the model can "cheat" by
# memorizing file identity / window position instead of learning behaviour.
NON_FEATURE_COLS = ["source_file", "label", "window_start", "window_index"]


def chronological_split_single_file(group, train_frac):
    """Fallback for a label with only one capture: split its windows
    chronologically."""
    group_sorted = group.sort_values("window_index")
    cutoff = int(len(group_sorted) * train_frac)
    return group_sorted.iloc[:cutoff], group_sorted.iloc[cutoff:]


def split_dataset(df, train_frac, test_file_frac):
    """
    Per label:
      - If the label has 2+ distinct source_file values, hold out whole
        file(s) as the test set (capture-level split).
      - If the label has only 1 file, fall back to a chronological
        within-file split.
    """
    train_parts = []
    test_parts = []

    for label, label_df in df.groupby("label"):
        files = sorted(label_df["source_file"].unique())

        if len(files) >= 2:
            n_test_files = max(1, round(len(files) * test_file_frac))
            n_test_files = min(n_test_files, len(files) - 1)  # keep >=1 train file
            test_files = files[-n_test_files:]
            train_files = files[:-n_test_files]

            train_parts.append(label_df[label_df["source_file"].isin(train_files)])
            test_parts.append(label_df[label_df["source_file"].isin(test_files)])

            print(f"  [{label}] capture-level split -- "
                  f"train files: {train_files} | test files: {test_files}")
        else:
            train_g, test_g = chronological_split_single_file(label_df, train_frac)
            train_parts.append(train_g)
            test_parts.append(test_g)
            print(f"  [{label}] only one capture ({files[0]}) -- "
                  f"FALLBACK chronological split: "
                  f"{len(train_g)} train windows, {len(test_g)} test windows "
                  f"(capture 2-3 more sessions for this class)")

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="Path to dataset.csv from extract_features.py")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="[Single-capture labels only] fraction of windows used for training (default 0.7)")
    ap.add_argument("--test-file-frac", type=float, default=0.3,
                    help="[Multi-capture labels] fraction of FILES (not rows) held out for testing (default 0.3)")
    ap.add_argument("--model", default="rf_model.pkl", help="Output path for the trained model")
    ap.add_argument("--n-estimators", type=int, default=100, help="Number of trees in the forest")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print("Rows per label:")
    print(df["label"].value_counts().to_string())
    print()
    print("Captures (source_file) per label:")
    print(df.groupby("label")["source_file"].unique().to_string())
    print()

    print("Splitting dataset:")
    train_df, test_df = split_dataset(df, args.train_frac, args.test_file_frac)
    print()
    print(f"Total train rows: {len(train_df)} | Total test rows: {len(test_df)}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}\n")

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        class_weight="balanced",  # reweight loss so minority classes aren't ignored
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=== Classification report (per-class precision/recall/F1) ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Confusion matrix ===")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Rows = actual, Columns = predicted")
    print("Labels order:", labels)
    print(cm)

    print("\n=== Feature importances ===")
    importances = sorted(
        zip(feature_cols, clf.feature_importances_),
        key=lambda x: -x[1]
    )
    for name, score in importances[:10]:
        print(f"  {name}: {score:.4f}")

    joblib.dump({"model": clf, "feature_cols": feature_cols}, args.model)
    print(f"\nSaved trained model to {args.model}")


if __name__ == "__main__":
    main()
