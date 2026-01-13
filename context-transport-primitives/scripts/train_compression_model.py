#!/usr/bin/env python3
"""
Compression Performance Prediction Model Training

Trains XGBoost regression models to predict:
1. Compression Ratio given data statistics and compressor choice
2. PSNR (for lossy compressors) given data statistics and compressor choice

Features:
- Compression Library (one-hot encoded)
- Data Type (float, char/int)
- Shannon Entropy (bits/byte)
- MAD (Mean Absolute Deviation)
- Second Derivative Mean (curvature)

Target Variables:
- Compression Ratio (higher = better)
- PSNR (dB) for lossy compressors only (higher = better)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'model_output'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_prepare_data(lossless_csv, lossy_csv):
    """
    Load and prepare data from CSV files.

    Returns:
        df_combined: Combined DataFrame with all data
        df_lossless: Lossless data only (for compression ratio prediction)
        df_lossy: Lossy data only (for PSNR prediction)
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load lossless data
    df_lossless = pd.read_csv(lossless_csv)
    df_lossless['Data Type'] = 'char'
    df_lossless['PSNR (dB)'] = np.nan  # No PSNR for lossless
    print(f"✓ Loaded {len(df_lossless)} lossless records")

    # Load lossy data
    df_lossy = pd.read_csv(lossy_csv)
    df_lossy['Data Type'] = 'float'
    print(f"✓ Loaded {len(df_lossy)} lossy records")

    # Combine datasets
    df_combined = pd.concat([df_lossless, df_lossy], ignore_index=True)
    print(f"✓ Total records: {len(df_combined)}")
    print()

    # Use ALL CPU utilization data to increase training set size
    # This gives us 5x more data (0%, 25%, 50%, 75%, 100%)
    print(f"✓ Using all CPU utilization levels: {len(df_combined)} records")
    print(f"  (5x increase from using all CPU levels instead of just 0%)")
    print()

    return df_combined, df_lossless, df_lossy

def prepare_features(df):
    """
    Prepare feature matrix and target variables.

    Features:
    - Library (one-hot encoded)
    - Data Type (one-hot encoded)
    - Shannon Entropy
    - MAD
    - Second Derivative Mean

    Returns:
        X: Feature matrix (pandas DataFrame with column names)
        y_ratio: Compression ratio target
        y_psnr: PSNR target (for lossy only)
        feature_names: List of feature names
        label_encoders: Dictionary of label encoders for categorical features
    """
    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    # Select relevant columns (include CPU utilization as feature)
    feature_cols = [
        'Library',
        'Data Type',
        'Target CPU Util (%)',  # NEW: CPU load affects compression time
        'Shannon Entropy (bits/byte)',
        'MAD',
        'Second Derivative Mean'
    ]

    df_features = df[feature_cols + ['Compression Ratio', 'PSNR (dB)']].copy()

    # Handle missing values in PSNR (lossless compressors don't have PSNR)
    df_features['PSNR (dB)'] = df_features['PSNR (dB)'].fillna(-1)  # Use -1 as sentinel for lossless

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_features, columns=['Library', 'Data Type'], drop_first=False)

    # Separate features and targets
    feature_names = [col for col in df_encoded.columns if col not in ['Compression Ratio', 'PSNR (dB)']]
    X = df_encoded[feature_names]
    y_ratio = df_encoded['Compression Ratio']
    y_psnr = df_encoded['PSNR (dB)']

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Number of features: {len(feature_names)}")
    print(f"✓ Features:")
    for i, name in enumerate(feature_names):
        if i < 10 or 'Library' not in name:  # Show first 10 and all non-library features
            print(f"    {i+1}. {name}")
        elif i == 10:
            print(f"    ... ({len([f for f in feature_names if 'Library' in f])} library one-hot features)")
    print()

    return X, y_ratio, y_psnr, feature_names, df_features

def train_compression_ratio_model(X, y, feature_names):
    """
    Train XGBoost model to predict compression ratio.
    Uses GridSearchCV to find best hyperparameters.
    """
    print("=" * 80)
    print("TRAINING COMPRESSION RATIO MODEL")
    print("=" * 80)

    # Split data: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print()

    # Define hyperparameter grid with stronger regularization to prevent overfitting
    param_grid = {
        'max_depth': [2, 3, 4],  # Shallower trees to reduce overfitting
        'learning_rate': [0.05, 0.1],  # Lower learning rates
        'n_estimators': [100, 200],  # Moderate number of trees
        'subsample': [0.6, 0.8],  # More subsampling
        'colsample_bytree': [0.6, 0.8],  # More feature subsampling
        'reg_lambda': [1.0, 10.0],  # L2 regularization (Ridge)
        'min_child_weight': [5, 10],  # Minimum samples per leaf
    }

    print("Hyperparameter search space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()

    # Create base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Grid search with cross-validation
    print("Running grid search (this may take a few minutes)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print()
    print("Best hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print()

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print("Model Performance:")
    print(f"  Training RMSE:   {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")
    print(f"  Training MAE:    {train_mae:.4f}")
    print(f"  Validation MAE:  {val_mae:.4f}")
    print(f"  Training R²:     {train_r2:.4f}")
    print(f"  Validation R²:   {val_r2:.4f}")
    print()

    # Check for overfitting
    if train_r2 - val_r2 > 0.1:
        print("⚠ Warning: Potential overfitting detected (train R² >> validation R²)")
    else:
        print("✓ No significant overfitting detected")
    print()

    return best_model, (X_train, X_val, y_train, y_val, y_pred_train, y_pred_val)

def train_psnr_model(X, y, feature_names):
    """
    Train XGBoost model to predict PSNR (lossy compressors only).
    """
    print("=" * 80)
    print("TRAINING PSNR MODEL (Lossy Compressors Only)")
    print("=" * 80)

    # Filter out lossless data (PSNR = -1)
    mask = y != -1
    X_lossy = X[mask]
    y_lossy = y[mask]

    if len(X_lossy) == 0:
        print("✗ No lossy data available for PSNR prediction")
        return None, None

    print(f"✓ Using {len(X_lossy)} lossy compression records")
    print()

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_lossy, y_lossy, test_size=0.2, random_state=42
    )

    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print()

    # Define hyperparameter grid (smaller for PSNR)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }

    print("Hyperparameter search space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()

    # Create base model
    base_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Grid search
    print("Running grid search...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print()
    print("Best hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print()

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)

    print("Model Performance:")
    print(f"  Training RMSE:   {train_rmse:.4f} dB")
    print(f"  Validation RMSE: {val_rmse:.4f} dB")
    print(f"  Training MAE:    {train_mae:.4f} dB")
    print(f"  Validation MAE:  {val_mae:.4f} dB")
    print(f"  Training R²:     {train_r2:.4f}")
    print(f"  Validation R²:   {val_r2:.4f}")
    print()

    if train_r2 - val_r2 > 0.1:
        print("⚠ Warning: Potential overfitting detected")
    else:
        print("✓ No significant overfitting detected")
    print()

    return best_model, (X_train, X_val, y_train, y_val, y_pred_train, y_pred_val)

def plot_predictions(model_name, y_train, y_pred_train, y_val, y_pred_val, unit=""):
    """
    Plot predicted vs actual values for train and validation sets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set
    axes[0].scatter(y_train, y_pred_train, alpha=0.5, s=20)
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel(f'Actual {model_name} {unit}', fontweight='bold')
    axes[0].set_ylabel(f'Predicted {model_name} {unit}', fontweight='bold')
    axes[0].set_title(f'Training Set: {model_name}', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation set
    axes[1].scatter(y_val, y_pred_val, alpha=0.5, s=20, color='orange')
    axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_xlabel(f'Actual {model_name} {unit}', fontweight='bold')
    axes[1].set_ylabel(f'Predicted {model_name} {unit}', fontweight='bold')
    axes[1].set_title(f'Validation Set: {model_name}', fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance from trained model.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:20]  # Top 20 features

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(range(len(indices)), importance[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontweight='bold')
    ax.set_title(f'Top 20 Features: {model_name}', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    return fig

def save_models_and_metadata(ratio_model, psnr_model, feature_names, best_params):
    """
    Save trained models and metadata for deployment.
    """
    print("=" * 80)
    print("SAVING MODELS")
    print("=" * 80)

    # Save compression ratio model using joblib
    ratio_model_path = OUTPUT_DIR / 'compression_ratio_model.pkl'
    joblib.dump(ratio_model, ratio_model_path)
    print(f"✓ Saved compression ratio model: {ratio_model_path.name}")

    # Save PSNR model
    if psnr_model is not None:
        psnr_model_path = OUTPUT_DIR / 'psnr_model.pkl'
        joblib.dump(psnr_model, psnr_model_path)
        print(f"✓ Saved PSNR model: {psnr_model_path.name}")

    # Save feature names and metadata
    metadata = {
        'feature_names': feature_names,
        'compression_ratio_params': best_params['ratio'],
        'psnr_params': best_params.get('psnr', None),
        'model_version': '1.0',
        'description': 'XGBoost models for compression performance prediction'
    }

    metadata_path = OUTPUT_DIR / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path.name}")
    print()

def main():
    """Main training pipeline."""
    print()
    print("=" * 80)
    print("COMPRESSION PERFORMANCE PREDICTION - MODEL TRAINING")
    print("=" * 80)
    print()

    # Load data
    build_dir = Path(__file__).parent.parent.parent / 'build'
    lossless_csv = build_dir / 'compression_parameter_study_results.csv'
    lossy_csv = build_dir / 'compression_lossy_parameter_study_results.csv'

    if not lossless_csv.exists() or not lossy_csv.exists():
        print("✗ Error: CSV files not found. Run benchmarks first.")
        return

    df_combined, df_lossless, df_lossy = load_and_prepare_data(lossless_csv, lossy_csv)

    # Prepare features
    X, y_ratio, y_psnr, feature_names, df_features = prepare_features(df_combined)

    # Train compression ratio model
    ratio_model, ratio_data = train_compression_ratio_model(X, y_ratio, feature_names)
    X_train_r, X_val_r, y_train_r, y_val_r, y_pred_train_r, y_pred_val_r = ratio_data

    # Train PSNR model
    psnr_model, psnr_data = train_psnr_model(X, y_psnr, feature_names)

    # Generate visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Compression ratio plots
    fig1 = plot_predictions('Compression Ratio', y_train_r, y_pred_train_r,
                           y_val_r, y_pred_val_r, unit="(x)")
    fig1.savefig(OUTPUT_DIR / 'compression_ratio_predictions.pdf', bbox_inches='tight')
    print("✓ Saved: compression_ratio_predictions.pdf")

    fig2 = plot_feature_importance(ratio_model, feature_names, 'Compression Ratio Model')
    fig2.savefig(OUTPUT_DIR / 'compression_ratio_feature_importance.pdf', bbox_inches='tight')
    print("✓ Saved: compression_ratio_feature_importance.pdf")

    # PSNR plots
    if psnr_model is not None:
        X_train_p, X_val_p, y_train_p, y_val_p, y_pred_train_p, y_pred_val_p = psnr_data
        fig3 = plot_predictions('PSNR', y_train_p, y_pred_train_p,
                               y_val_p, y_pred_val_p, unit="(dB)")
        fig3.savefig(OUTPUT_DIR / 'psnr_predictions.pdf', bbox_inches='tight')
        print("✓ Saved: psnr_predictions.pdf")

        fig4 = plot_feature_importance(psnr_model, feature_names, 'PSNR Model')
        fig4.savefig(OUTPUT_DIR / 'psnr_feature_importance.pdf', bbox_inches='tight')
        print("✓ Saved: psnr_feature_importance.pdf")

    print()

    # Save models
    best_params = {
        'ratio': ratio_model.get_params(),
        'psnr': psnr_model.get_params() if psnr_model is not None else None
    }
    save_models_and_metadata(ratio_model, psnr_model, feature_names if isinstance(feature_names, list) else feature_names.tolist(), best_params)

    # Summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  • compression_ratio_model.pkl - XGBoost model for compression ratio")
    if psnr_model is not None:
        print(f"  • psnr_model.pkl - XGBoost model for PSNR (lossy only)")
    print(f"  • model_metadata.json - Feature names and hyperparameters")
    print(f"  • compression_ratio_predictions.pdf - Model predictions visualization")
    print(f"  • compression_ratio_feature_importance.pdf - Feature importance")
    if psnr_model is not None:
        print(f"  • psnr_predictions.pdf - PSNR predictions visualization")
        print(f"  • psnr_feature_importance.pdf - PSNR feature importance")
    print()
    print("Models can be loaded for inference using:")
    print("  import joblib")
    print("  ratio_model = joblib.load('compression_ratio_model.pkl')")
    print("  psnr_model = joblib.load('psnr_model.pkl')")
    print()

if __name__ == '__main__':
    main()
