"""Train and serialize a baseline SWaT-derived Cyber-Physical Anomaly Detection Model.

Dataset Context:
Secure Water Treatment (SWaT) testbed physical sensor suite:
- LIT101: Level Indicator Transmitter (Raw water tank level, cm)
- FIT101: Flow Indicator Transmitter (Inlet flow rate, m3/h)
- AIT201: Chemical Conductivity / pH Analyzer (uS/cm)
- P101: Pump 101 Actuator State (0=OFF, 1=ON)
- DPIT301: Differential Pressure Indicator Transmitter (Filter pressure, kPa)

Target:
- 0: Normal plant operations
- 1: Cyber-physical attack / operational anomaly
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def generate_swat_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic physical process sensor readings with simulated cyber-attacks."""
    rng = np.random.RandomState(random_state)

    # 1. Normal operating conditions (85% of telemetry)
    n_normal = int(n_samples * 0.85)
    lit101_norm = rng.normal(loc=650.0, scale=35.0, size=n_normal)   # Tank level ~650 cm
    fit101_norm = rng.normal(loc=2.45, scale=0.15, size=n_normal)    # Flow ~2.45 m3/h
    ait201_norm = rng.normal(loc=255.0, scale=8.0, size=n_normal)    # Conductivity ~255
    p101_norm = rng.choice([0, 1], size=n_normal, p=[0.2, 0.8])      # Pump active
    dpit301_norm = rng.normal(loc=19.5, scale=2.0, size=n_normal)    # Differential pressure

    X_normal = np.column_stack([lit101_norm, fit101_norm, ait201_norm, p101_norm, dpit301_norm])
    y_normal = np.zeros(n_normal, dtype=int)

    # 2. Attack / Anomaly conditions (15% of telemetry: overflow attack, pump tampering, filter clogging)
    n_attack = n_samples - n_normal
    lit101_att = rng.uniform(low=850.0, high=1050.0, size=n_attack)  # Tank overflow attack
    fit101_att = rng.uniform(low=4.5, high=7.0, size=n_attack)       # High inlet surge
    ait201_att = rng.uniform(low=300.0, high=450.0, size=n_attack)   # Chemical tampering
    p101_att = rng.choice([0, 1], size=n_attack, p=[0.7, 0.3])
    dpit301_att = rng.uniform(low=35.0, high=65.0, size=n_attack)    # Dangerous pressure surge

    X_attack = np.column_stack([lit101_att, fit101_att, ait201_att, p101_att, dpit301_att])
    y_attack = np.ones(n_attack, dtype=int)

    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])

    # Shuffle dataset
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def train_and_export_model(output_path: str = "artifacts/baseline_swat_model.joblib") -> str:
    """Train Random Forest classifier on SWaT dataset and serialize via joblib."""
    print("Generating synthetic SWaT telemetry dataset...")
    X, y = generate_swat_synthetic_data(n_samples=3000, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier (n_estimators=30, max_depth=6)...")
    clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    print("\nModel Evaluation on Test Split:")
    print(classification_report(y_test, preds, target_names=["Normal (0)", "Attack (1)"]))

    # Ensure target directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save artifact
    joblib.dump(clf, out_file)
    print(f"Artifact successfully serialized to: {out_file.resolve()}")
    return str(out_file.resolve())


if __name__ == "__main__":
    train_and_export_model()
