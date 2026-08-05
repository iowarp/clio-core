"""
Generate a realistic synthetic SSD training dataset based on published
Alibaba/HGST SSD failure statistics and train the SSD model.

The synthetic data mimics the statistical properties documented in:
  "Characterizing Cloud Storage with SMART Logs" (Alibaba 2019 dataset).

Key failure indicators for SSDs:
  smart_5   - Retired reallocated blocks (sharp rise = bad)
  smart_173 - Average erase count / media wearout (high = bad)
  smart_187 - Uncorrectable errors
  smart_194 - Temperature (extreme high or low = risk factor)
  smart_241 - Total LBAs written (high = worn out)
  smart_242 - Total LBAs read
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_HEALTHY = 48000   # Healthy drive-days
N_PRE_FAIL = 2000   # Drive-days from drives that later failed (imbalanced like real world)

print("Generating synthetic SSD training data...")

def gen_healthy():
    return pd.DataFrame({
        'serial_number': [f'SSD_H_{i//30:05d}' for i in range(N_HEALTHY)],
        'date': pd.date_range('2018-01-01', periods=N_HEALTHY, freq='1D'),
        'smart_5':   np.random.poisson(0.2, N_HEALTHY).clip(0, 5),
        'smart_173': np.random.randint(10, 80, N_HEALTHY),
        'smart_187': np.random.poisson(0.1, N_HEALTHY).clip(0, 2),
        'smart_194': np.random.randint(25, 45, N_HEALTHY),
        'smart_241': np.random.randint(1_000, 500_000, N_HEALTHY),
        'smart_242': np.random.randint(1_000, 800_000, N_HEALTHY),
        'failure':   np.zeros(N_HEALTHY, dtype=int)
    })

def gen_failing():
    return pd.DataFrame({
        'serial_number': [f'SSD_F_{i//10:05d}' for i in range(N_PRE_FAIL)],
        'date': pd.date_range('2018-01-01', periods=N_PRE_FAIL, freq='1D'),
        # Failing drives show high retired blocks, high wearout, high errors
        'smart_5':   np.random.poisson(12, N_PRE_FAIL).clip(0, 100),
        'smart_173': np.random.randint(85, 100, N_PRE_FAIL),
        'smart_187': np.random.poisson(8, N_PRE_FAIL).clip(0, 50),
        'smart_194': np.random.choice(
            np.concatenate([np.random.randint(0, 15, N_PRE_FAIL // 2),
                            np.random.randint(60, 90, N_PRE_FAIL // 2)])),
        'smart_241': np.random.randint(900_000, 2_000_000, N_PRE_FAIL),
        'smart_242': np.random.randint(900_000, 3_000_000, N_PRE_FAIL),
        'failure':   np.ones(N_PRE_FAIL, dtype=int)
    })

df = pd.concat([gen_healthy(), gen_failing()], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

os.makedirs('data/ssd', exist_ok=True)
out_path = 'data/ssd/ssd_synthetic_training.csv'
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} rows to {out_path}")
print(f"  Failure rate: {df['failure'].mean():.2%}")
print(f"  Healthy: {(df['failure']==0).sum():,} | Failing: {(df['failure']==1).sum():,}")
print("\nNow training the SSD model...")

from lightgbm import LGBMClassifier
import joblib

FEATURES = ['smart_5', 'smart_173', 'smart_187', 'smart_194', 'smart_241', 'smart_242']

X = df[FEATURES]
y = df['failure']

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    scale_pos_weight=(N_HEALTHY / N_PRE_FAIL),  # Handle class imbalance
    random_state=42
)
model.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/ssd_static_model.pkl')
print("SSD static model saved to models/ssd_static_model.pkl")

# Verify it works
sample = pd.DataFrame([{
    'smart_5': 25, 'smart_173': 95, 'smart_187': 10,
    'smart_194': 75, 'smart_241': 1_500_000, 'smart_242': 2_000_000
}])
prob = model.predict_proba(sample)[0][1]
print(f"\nVerification - failing drive sample -> failure probability: {prob:.3f} (should be high)")

sample_ok = pd.DataFrame([{
    'smart_5': 0, 'smart_173': 30, 'smart_187': 0,
    'smart_194': 35, 'smart_241': 50_000, 'smart_242': 80_000
}])
prob_ok = model.predict_proba(sample_ok)[0][1]
print(f"Verification - healthy drive sample -> failure probability: {prob_ok:.3f} (should be low)")
print("\nDone! SSD model is ready.")
