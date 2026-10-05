"""
EYE ORDER COME AI — Train Cursor Model (Regression)
เทรน Regression model สำหรับทำนายพิกัดเมาส์จาก 7 features

เปรียบเทียบ 5 อัลกอริทึม:
  1. Ridge Regression
  2. SVR (RBF kernel)  
  3. Random Forest Regressor
  4. Gradient Boosting Regressor
  5. MLPRegressor (Neural Network)

Metrics: MAE, RMSE, R² Score
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # ใช้ backend ที่ไม่ต้องแสดง GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CALIBRATION_DATA_PATH, CURSOR_MODEL_PATH, MODELS_DIR, DATA_DIR
)
from core.feature_extractor import FeatureExtractor
from utils.math_utils import fold_head_angle

# เวอร์ชันของนิยามฟีเจอร์ — v2 = pitch/roll พับเข้าช่วง [-90, 90] (v1 วนข้าม ±180°)
FEATURE_VERSION = 2
# ตัดเฟรมแรกของแต่ละจุดทิ้ง: ศีรษะยังเคลื่อนมาไม่ถึงจุด ทั้งที่ label เป็นจุดนั้นแล้ว
SETTLE_FRAMES = 15
# โมเดลที่ให้ผลลัพธ์ "ต่อเนื่อง" เหมาะกับการคุมเคอร์เซอร์ (ต้นไม้ให้ค่าเป็นขั้นบันได คุมยาก)
SMOOTH_MODELS = {"Ridge Regression", "SVR (RBF)", "MLP (Neural Network)"}


def _assign_groups(df):
    """แบ่งกลุ่มตามการเยี่ยมจุดเป้าหมายแต่ละครั้ง (label เปลี่ยน = กลุ่มใหม่)
    
    เฟรมติดกันของจุดเดียวกันแทบเหมือนกัน ถ้าสุ่มแบ่ง train/test ทีละแถวจะ "รั่ว"
    และ MAE ดูดีเกินจริง จึงต้องแบ่งตามกลุ่มทั้งก้อน
    """
    key = df["screen_x"].astype(str) + "_" + df["screen_y"].astype(str)
    return (key != key.shift()).cumsum().values


def _clean(df, groups):
    """ตัดเฟรมช่วงศีรษะกำลังเคลื่อนที่ + ค่าผิดปกติ (MAD) ในแต่ละจุด"""
    df = df.copy()
    df["g"] = groups
    df["_i"] = df.groupby("g").cumcount()
    df = df[df["_i"] >= SETTLE_FRAMES]
    
    def keep(g):
        ok = np.ones(len(g), dtype=bool)
        for c in ("nose_x", "nose_y", "yaw", "pitch", "roll"):
            med = g[c].median()
            mad = (g[c] - med).abs().median() * 1.4826 + 1e-6
            ok &= ((g[c] - med).abs() <= 3.0 * mad).values
        return pd.Series(ok, index=g.index)
    
    mask = df.groupby("g", group_keys=False).apply(keep)
    return df[mask.values]


def load_data(data_path=None):
    """โหลดข้อมูล calibration (รองรับไฟล์เก่าที่ pitch/roll วนข้าม ±180°)
    
    Returns:
        tuple: (X, y, df, groups) หรือ None ถ้าไม่มีข้อมูล
    """
    if data_path is None:
        data_path = CALIBRATION_DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"  ❌ ไม่พบข้อมูล: {data_path}")
        print("  กรุณารัน calibration ก่อน:")
        print("    python calibration/nine_point_calibration.py")
        return None
    
    df = pd.read_csv(data_path)
    print(f"  📂 โหลดข้อมูล: {data_path}")
    print(f"     ขนาด: {df.shape[0]} ตัวอย่าง × {df.shape[1]} columns")
    
    # ไฟล์จากเวอร์ชันเก่าเก็บ pitch/roll ใกล้ ±180° → พับกลับ (ค่าที่พับแล้วไม่เปลี่ยน)
    df["pitch"] = fold_head_angle(df["pitch"].values)
    df["roll"] = fold_head_angle(df["roll"].values)
    
    groups = _assign_groups(df)
    n_raw = len(df)
    df = _clean(df, groups)
    print(f"     หลังตัดเฟรมช่วงเคลื่อนที่/ค่าผิดปกติ: {len(df)}/{n_raw} ตัวอย่าง "
          f"({df['g'].nunique()} การเยี่ยมจุด)")
    
    feature_names = FeatureExtractor.get_cursor_feature_names()
    X = df[feature_names].values
    y = df[["screen_x", "screen_y"]].values
    
    return X, y, df, df["g"].values


def define_models():
    """กำหนดโมเดลและ hyperparameter grid สำหรับ GridSearchCV (ปรับจูนความเร็ว)
    
    Returns:
        dict: {name: {"model": model, "params": param_grid}}
    """
    models = {
        "Ridge Regression": {
            "model": MultiOutputRegressor(Ridge()),
            "params": {
                "estimator__alpha": [1.0, 10.0, 100.0],
            },
        },
        "SVR (RBF)": {
            "model": MultiOutputRegressor(SVR(kernel="rbf"), n_jobs=-1),
            "params": {
                "estimator__C": [10, 100],
                "estimator__gamma": ["scale", 0.01],
            },
        },
        "Random Forest": {
            "model": MultiOutputRegressor(
                RandomForestRegressor(random_state=42, n_jobs=-1), n_jobs=-1
            ),
            "params": {
                "estimator__n_estimators": [100],
                "estimator__max_depth": [10, 20],
            },
        },
        "Gradient Boosting": {
            "model": MultiOutputRegressor(
                GradientBoostingRegressor(random_state=42, n_estimators=100), n_jobs=-1
            ),
            "params": {
                "estimator__learning_rate": [0.1],
                "estimator__max_depth": [3, 5],
            },
        },
        "MLP (Neural Network)": {
            "model": MLPRegressor(
                random_state=42, max_iter=300, early_stopping=True, n_iter_no_change=5
            ),
            "params": {
                "hidden_layer_sizes": [(128, 64), (64, 32)],
                "activation": ["relu"],
            },
        },
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler, groups_train):
    """เทรนทุกโมเดล + ประเมินผล
    
    Returns:
        dict: {name: {"model": best_model, "metrics": {...}, "best_params": {...}}}
    """
    models = define_models()
    results = {}
    
    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION")
    print("=" * 70)
    
    for name, config in models.items():
        print(f"\n  ── {name} ──")
        start_time = time.time()
        
        model = config["model"]
        params = config["params"]
        
        # GridSearchCV
        print(f"    GridSearchCV ({len(params)} param sets)...")
        grid_search = GridSearchCV(
            model, params,
            cv=list(GroupKFold(n_splits=5).split(X_train, y_train, groups_train)),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # ทำนาย
        y_pred = best_model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # MAE แยกแกน
        mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
        
        elapsed = time.time() - start_time
        
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "metrics": {
                "MAE": mae,
                "MAE_X": mae_x,
                "MAE_Y": mae_y,
                "RMSE": rmse,
                "R2": r2,
            },
            "y_pred": y_pred,
            "train_time": elapsed,
            "smooth": name in SMOOTH_MODELS,
        }
        
        print(f"    Best params: {best_params}")
        print(f"    MAE: {mae:.2f} px (X: {mae_x:.2f}, Y: {mae_y:.2f})")
        print(f"    RMSE: {rmse:.2f} px")
        print(f"    R²: {r2:.4f}")
        print(f"    Time: {elapsed:.1f}s")
    
    return results


def select_best_model(results):
    """เลือกโมเดลที่ MAE ต่ำสุด "ในกลุ่มที่ต่อเนื่อง" (Ridge/SVR/MLP)
    
    Random Forest / Gradient Boosting ให้ผลเป็นขั้นบันได (ค่าคงที่เป็นช่วงๆ)
    เคอร์เซอร์จึงกระโดดและคุมยากแม้ MAE จะต่ำ จึงไม่ใช้ควบคุมเคอร์เซอร์สด
    
    Returns:
        tuple: (best_name, best_result)
    """
    candidates = {k: v for k, v in results.items() if v.get("smooth")} or results
    best_name = min(candidates, key=lambda k: candidates[k]["metrics"]["MAE"])
    return best_name, results[best_name]


def save_model(model, scaler, name, new_mae=None, X_test_scaled=None, y_test=None, model_path=None):
    """บันทึกโมเดล (ตรวจสอบก่อนว่าใหม่กว่าหรือดีกว่าของเดิมไหม)"""
    if model_path is None:
        model_path = CURSOR_MODEL_PATH
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    should_save = True
    if os.path.exists(model_path) and new_mae is not None and X_test_scaled is not None and y_test is not None:
        try:
            old_data = joblib.load(model_path)
            if old_data.get("feature_version", 1) != FEATURE_VERSION:
                raise ValueError("โมเดลเดิมใช้นิยามฟีเจอร์เวอร์ชันเก่า — เขียนทับ")
            old_model = old_data.get("model")
            old_scaler = old_data.get("scaler")
            old_name = old_data.get("name", "Existing Model")
            
            if old_model is not None:
                X_raw = scaler.inverse_transform(X_test_scaled)
                X_old = old_scaler.transform(X_raw) if old_scaler else X_raw
                old_pred = old_model.predict(X_old)
                old_mae = mean_absolute_error(y_test, old_pred)
                
                if old_mae < new_mae - 0.01:
                    should_save = False
                    print(f"\n  🛡️ โมเดลเดิมดีกว่า! (โมเดลเดิม {old_name} MAE: {old_mae:.2f} px vs โมเดลใหม่ {name} MAE: {new_mae:.2f} px)")
                    print("     ระบบจึงเปิดเซฟโหมดและรักษาไฟล์โมเดลเดิมไว้ ไม่ให้คะแนนดรอปครับ 👍")
        except Exception:
            pass
    
    if should_save:
        data = {
            "model": model,
            "scaler": scaler,
            "name": name,
            "feature_version": FEATURE_VERSION,
        }
        joblib.dump(data, model_path)
        print(f"\n  💾 บันทึกโมเดลที่: {model_path}")


def plot_results(results, y_test, save_dir=None):
    """สร้างกราฟเปรียบเทียบ
    
    Args:
        results: dict จาก train_and_evaluate
        y_test: ค่าจริง
        save_dir: โฟลเดอร์สำหรับบันทึกกราฟ
    """
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # ── Plot 1: Model Comparison Bar Chart ──
    names = list(results.keys())
    maes = [results[n]["metrics"]["MAE"] for n in names]
    rmses = [results[n]["metrics"]["RMSE"] for n in names]
    r2s = [results[n]["metrics"]["R2"] for n in names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = sns.color_palette("viridis", len(names))
    
    axes[0].barh(names, maes, color=colors)
    axes[0].set_xlabel("MAE (pixels)")
    axes[0].set_title("Mean Absolute Error (↓ lower is better)")
    for i, v in enumerate(maes):
        axes[0].text(v + 1, i, f"{v:.1f}", va='center')
    
    axes[1].barh(names, rmses, color=colors)
    axes[1].set_xlabel("RMSE (pixels)")
    axes[1].set_title("Root Mean Squared Error (↓ lower is better)")
    for i, v in enumerate(rmses):
        axes[1].text(v + 1, i, f"{v:.1f}", va='center')
    
    axes[2].barh(names, r2s, color=colors)
    axes[2].set_xlabel("R² Score")
    axes[2].set_title("R² Score (↑ higher is better)")
    for i, v in enumerate(r2s):
        axes[2].text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cursor_model_comparison.png"), dpi=150)
    plt.close()
    
    # ── Plot 2: Prediction vs Actual (best model) ──
    best_name, _ = select_best_model(results)
    y_pred = results[best_name]["y_pred"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5, s=10,
                    color='steelblue')
    axes[0].plot([y_test[:, 0].min(), y_test[:, 0].max()],
                [y_test[:, 0].min(), y_test[:, 0].max()],
                'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel("Actual Screen X")
    axes[0].set_ylabel("Predicted Screen X")
    axes[0].set_title(f"X Coordinate — {best_name}")
    axes[0].legend()
    
    axes[1].scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5, s=10,
                    color='coral')
    axes[1].plot([y_test[:, 1].min(), y_test[:, 1].max()],
                [y_test[:, 1].min(), y_test[:, 1].max()],
                'r--', linewidth=2, label='Perfect prediction')
    axes[1].set_xlabel("Actual Screen Y")
    axes[1].set_ylabel("Predicted Screen Y")
    axes[1].set_title(f"Y Coordinate — {best_name}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cursor_prediction_scatter.png"), dpi=150)
    plt.close()
    
    # ── Plot 3: Error Distribution ──
    errors_x = y_test[:, 0] - y_pred[:, 0]
    errors_y = y_test[:, 1] - y_pred[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(errors_x, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Error X (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"X Error Distribution — {best_name}")
    
    axes[1].hist(errors_y, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel("Error Y (pixels)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Y Error Distribution — {best_name}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cursor_error_distribution.png"), dpi=150)
    plt.close()
    
    print(f"\n  📊 กราฟบันทึกที่: {save_dir}")


def print_summary_table(results):
    """แสดงตารางสรุปผล"""
    print("\n" + "=" * 80)
    print("  SUMMARY — CURSOR REGRESSION MODEL COMPARISON")
    print("=" * 80)
    print(f"  {'Model':<25} {'MAE':>8} {'MAE_X':>8} {'MAE_Y':>8} "
          f"{'RMSE':>8} {'R²':>8} {'Time':>8}")
    print("  " + "-" * 75)
    
    best_name, _ = select_best_model(results)
    
    for name, res in results.items():
        m = res["metrics"]
        marker = " ★" if name == best_name else (" (ขั้นบันได)" if not res.get("smooth") else "")
        print(f"  {name:<25} {m['MAE']:>7.2f} {m['MAE_X']:>7.2f} "
              f"{m['MAE_Y']:>7.2f} {m['RMSE']:>7.2f} {m['R2']:>7.4f} "
              f"{res['train_time']:>6.1f}s{marker}")
    
    print("  " + "-" * 75)
    print(f"  ★ Best model: {best_name}")
    print("=" * 80)


def main():
    """Main training pipeline"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  EYE ORDER COME AI — CURSOR MODEL TRAINING              ║")
    print("╚" + "═" * 58 + "╝")
    
    # 1. โหลดข้อมูล
    data = load_data()
    if data is None:
        return
    X, y, df, groups = data
    
    print(f"\n  Features: {X.shape[1]} ตัวแปร")
    print(f"  Targets: Screen_X, Screen_Y")
    print(f"  ตัวอย่าง: {X.shape[0]}")
    
    # 2. Train/Test Split — แบ่งตามกลุ่ม (การเยี่ยมจุด) ไม่ใช่ทีละแถว
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} (แบ่งตามกลุ่ม ไม่รั่ว)")
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. เทรนและประเมิน
    results = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, groups_train
    )
    
    # 5. แสดงสรุป
    print_summary_table(results)
    
    # 6. เลือกโมเดลที่ดีที่สุด
    best_name, best_result = select_best_model(results)
    
    # 7. บันทึกโมเดล (มีระบบป้องกันถ้าของเดิมดีกว่า)
    save_model(
        best_result["model"], scaler, best_name,
        new_mae=best_result["metrics"]["MAE"],
        X_test_scaled=X_test_scaled, y_test=y_test
    )
    
    # 8. สร้างกราฟ
    plot_results(results, y_test)
    
    print("\n  ✅ การเทรนเสร็จสมบูรณ์!")
    return results


if __name__ == "__main__":
    main()
