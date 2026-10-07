"""
EYE ORDER COME AI — Train Cursor Model (Regression)
เทรน Regression model สำหรับทำนายพิกัดเมาส์จาก 7 features

เปรียบเทียบ 5 อัลกอริทึม:
  1. Ridge Regression
  2. SVR (RBF kernel)  
  3. Random Forest Regressor
  4. Gradient Boosting Regressor
  5. MLPRegressor (Neural Network)

ขั้นตอน:
  1. ทำความสะอาด: ตัดเฟรมช่วงศีรษะกำลังเคลื่อนที่ + ค่าผิดปกติ
  2. Feature selection: เทียบชุดฟีเจอร์ด้วย GroupKFold CV แล้วเลือกชุดที่ MAE ต่ำสุด
     (nose_x/nose_y เป็น "ตำแหน่งหน้า" เปลี่ยนตามท่านั่ง ส่วน nose_offset เป็น "การหมุนหัว" ล้วน)
  3. เปรียบเทียบ 5 อัลกอริทึมบนชุดฟีเจอร์ที่เลือก (GridSearchCV + GroupKFold แยกตามการเยี่ยมจุด)
  4. Metrics จาก out-of-fold prediction ของทุกการเยี่ยมจุด: MAE, RMSE, R², jitter

ทุกโมเดลถูกห่อด้วย EdgeSafeRegressor (core/cursor_models.py): นอกช่วงที่ calibrate จะต่อเนื่อง
แบบเชิงเส้น → หันหัวเลยขอบจอแล้วเคอร์เซอร์ "ติดขอบ" ไม่วิ่งกลับเข้ากลางจอ

Metrics: MAE, RMSE, R² Score, Jitter (ค่าสั่นของเคอร์เซอร์ตอนหัวนิ่ง)
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

from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
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
    CALIBRATION_DATA_PATH, CURSOR_MODEL_PATH, DATA_DIR
)
from core.cursor_models import EdgeSafeRegressor
from core.feature_extractor import CURSOR_FEATURE_NAMES, CURSOR_FEATURE_VERSION

FEATURE_VERSION = CURSOR_FEATURE_VERSION
ALL_FEATURES = list(CURSOR_FEATURE_NAMES)
# ชุดฟีเจอร์ที่นำมาเทียบ (feature selection)
FEATURE_SETS = {
    "Head pose matrix (yaw/pitch)": ["head_yaw", "head_pitch"],
    "Head pose + roll": ["head_yaw", "head_pitch", "head_roll"],
    "Nose offset (x/y)": ["nose_offset_x", "nose_offset_y"],
    "Head pose + nose offset": ["head_yaw", "head_pitch", "nose_offset_x", "nose_offset_y"],
    "All 7 features": ALL_FEATURES,
}
CV_FOLDS = 6
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
        for c in ("head_yaw", "head_pitch", "nose_offset_x", "nose_offset_y"):
            med = g[c].median()
            mad = (g[c] - med).abs().median() * 1.4826 + 1e-6
            ok &= ((g[c] - med).abs() <= 3.0 * mad).values
        return pd.Series(ok, index=g.index)
    
    mask = df.groupby("g", group_keys=False).apply(keep)
    return df[mask.values]


def load_data(data_path=None):
    """โหลดข้อมูล calibration
    
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
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        print(f"  ❌ ข้อมูล calibration เป็นรุ่นเก่า (ไม่มี {missing}) — กด Calibrate ใหม่")
        return None
    
    groups = _assign_groups(df)
    n_raw = len(df)
    df = _clean(df, groups)
    print(f"     หลังตัดเฟรมช่วงเคลื่อนที่/ค่าผิดปกติ: {len(df)}/{n_raw} ตัวอย่าง "
          f"({df['g'].nunique()} การเยี่ยมจุด)")
    
    X = df[ALL_FEATURES].values
    y = df[["screen_x", "screen_y"]].values
    
    return X, y, df, df["g"].values


def _jitter_xy(df, pred):
    """ค่าสั่นของเคอร์เซอร์ตอนหัวนิ่ง แยกแกน = median ของ std ภายในการเยี่ยมจุดเดียวกัน (px)"""
    d = pd.DataFrame({"g": df["g"].values, "px": pred[:, 0], "py": pred[:, 1]})
    s = d.groupby("g")[["px", "py"]].std().median()
    return float(s["px"]), float(s["py"])


def _jitter(df, pred):
    return float(np.mean(_jitter_xy(df, pred)))


def _cv(groups):
    return GroupKFold(n_splits=min(CV_FOLDS, len(np.unique(groups))))


def select_feature_set(df, y, groups):
    """Feature selection: เทียบชุดฟีเจอร์ด้วยโมเดลอ้างอิง (SVR) + GroupKFold CV"""
    print("\n" + "=" * 70)
    print("  FEATURE SELECTION (GroupKFold CV, โมเดลอ้างอิง SVR RBF)")
    print("=" * 70)
    ref = make_pipeline(StandardScaler(), EdgeSafeRegressor(
        MultiOutputRegressor(SVR(kernel="rbf", C=100, epsilon=5))))
    rows = []
    for name, cols in FEATURE_SETS.items():
        pred = cross_val_predict(ref, df[cols].values, y, groups=groups, cv=_cv(groups))
        mae = mean_absolute_error(y, pred)
        rows.append((name, cols, mae, _jitter(df, pred)))
        print(f"  {name:<30} MAE {mae:7.1f} px   jitter {rows[-1][3]:5.1f} px   ({len(cols)} ฟีเจอร์)")
    best = min(rows, key=lambda r: r[2])
    print(f"  ★ เลือก: {best[0]} → {best[1]}")
    return best[1], rows


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


def train_and_evaluate(X, y, groups, df):
    """เทรนทุกโมเดล (GridSearchCV) + ประเมินด้วย out-of-fold prediction ของทุกการเยี่ยมจุด

    Returns:
        dict: {name: {"model": pipeline(scaler+model), "metrics": {...}, "y_pred": oof, ...}}
    """
    models = define_models()
    results = {}
    cv = _cv(groups)
    
    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION (out-of-fold)")
    print("=" * 70)
    
    for name, config in models.items():
        print(f"\n  ── {name} ──")
        start_time = time.time()
        pipe = make_pipeline(StandardScaler(), EdgeSafeRegressor(config["model"]))
        grid = {f"{pipe.steps[-1][0]}__estimator__{k}": v for k, v in config["params"].items()}
        grid_search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        grid_search.fit(X, y, groups=groups)
        best_model = grid_search.best_estimator_
        best_params = {k.split("__estimator__", 1)[1]: v for k, v in grid_search.best_params_.items()}
        
        y_pred = cross_val_predict(best_model, X, y, groups=groups, cv=cv)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mae_x = mean_absolute_error(y[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y[:, 1], y_pred[:, 1])
        jitter_x, jitter_y = _jitter_xy(df, y_pred)
        jitter = (jitter_x + jitter_y) / 2
        elapsed = time.time() - start_time
        
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "metrics": {"MAE": mae, "MAE_X": mae_x, "MAE_Y": mae_y, "RMSE": rmse, "R2": r2,
                        "Jitter": jitter, "Jitter_X": jitter_x, "Jitter_Y": jitter_y},
            "y_pred": y_pred,
            "train_time": elapsed,
            "smooth": name in SMOOTH_MODELS,
        }
        
        print(f"    Best params: {best_params}")
        print(f"    MAE: {mae:.2f} px (X: {mae_x:.2f}, Y: {mae_y:.2f})  RMSE: {rmse:.2f}  "
              f"R²: {r2:.4f}  Jitter: {jitter:.1f} px  ({elapsed:.1f}s)")
    
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


def save_model(pipeline, name, feature_cols, metrics, model_path=None):
    """บันทึกโมเดล (เทรนบนข้อมูลทั้งหมดแล้ว) + ชุดฟีเจอร์ที่ใช้"""
    model_path = model_path or CURSOR_MODEL_PATH
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    scaler, model = pipeline.steps[0][1], pipeline.steps[-1][1]
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "name": name,
        "feature_version": FEATURE_VERSION,
        "feature_cols": list(feature_cols),
        "metrics": metrics,
        # noise ของโมเดลเองแยกแกน (px) → CursorPredictor ใช้ปรับ deadzone ให้พอดีกับ noise แต่ละแกน
        "noise_px": [metrics["Jitter_X"], metrics["Jitter_Y"]],
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, model_path)
    print(f"\n  💾 บันทึกโมเดลที่: {model_path}")


def plot_results(results, y_test, save_dir=None, feature_rows=None):
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
    
    # ── Plot 4: Feature selection ──
    if feature_rows:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = [r[0] for r in feature_rows]
        maes = [r[2] for r in feature_rows]
        best = int(np.argmin(maes))
        bars = ax.barh(labels, maes, color=["#4CAF50" if i == best else "#90A4AE"
                                            for i in range(len(labels))])
        for bar, v in zip(bars, maes):
            ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center")
        ax.set_xlabel("MAE (pixels, GroupKFold CV)")
        ax.set_title("Cursor Feature Selection (SVR RBF)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cursor_feature_selection.png"), dpi=150)
        plt.close()

    print(f"\n  📊 กราฟบันทึกที่: {save_dir}")


def print_summary_table(results):
    """แสดงตารางสรุปผล"""
    print("\n" + "=" * 80)
    print("  SUMMARY — CURSOR REGRESSION MODEL COMPARISON")
    print("=" * 80)
    print(f"  {'Model':<25} {'MAE':>8} {'MAE_X':>8} {'MAE_Y':>8} "
          f"{'RMSE':>8} {'R²':>8} {'Jitter':>7} {'Time':>8}")
    print("  " + "-" * 75)
    
    best_name, _ = select_best_model(results)
    
    for name, res in results.items():
        m = res["metrics"]
        marker = " ★" if name == best_name else (" (ขั้นบันได)" if not res.get("smooth") else "")
        print(f"  {name:<25} {m['MAE']:>7.2f} {m['MAE_X']:>7.2f} "
              f"{m['MAE_Y']:>7.2f} {m['RMSE']:>7.2f} {m['R2']:>7.4f} {m['Jitter']:>7.1f} "
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
    _, y, df, groups = data
    print(f"\n  Targets: Screen_X, Screen_Y   ตัวอย่าง: {len(y)}   "
          f"การเยี่ยมจุด: {len(np.unique(groups))} (แบ่ง fold ตามนี้ ไม่รั่ว)")
    
    # 2. Feature selection
    feature_cols, feature_rows = select_feature_set(df, y, groups)
    X = df[feature_cols].values
    
    # 3. เทรนและประเมิน 5 โมเดล
    results = train_and_evaluate(X, y, groups, df)
    
    # 4. สรุป + เลือกโมเดล
    print_summary_table(results)
    best_name, best_result = select_best_model(results)
    
    # 5. เทรนตัวที่เลือกบนข้อมูลทั้งหมด แล้วบันทึก
    final = best_result["model"].fit(X, y)
    save_model(final, best_name, feature_cols, best_result["metrics"])
    
    # 6. กราฟ (ใช้ out-of-fold prediction)
    plot_results(results, y, feature_rows=feature_rows)
    
    print("\n  ✅ การเทรนเสร็จสมบูรณ์!")
    return results


if __name__ == "__main__":
    main()
