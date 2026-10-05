"""
EYE ORDER COME AI — Model Evaluation & Comparison Report
ประเมินและเปรียบเทียบผลโมเดลทั้ง cursor (regression) และ click (classification)
สร้างรายงานสรุปครบถ้วน
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CALIBRATION_DATA_PATH, GESTURE_DATA_PATH,
    CURSOR_MODEL_PATH, CLICK_MODEL_PATH,
    DATA_DIR, CLASS_NAMES
)
from core.feature_extractor import FeatureExtractor


def evaluate_cursor_model():
    """ประเมิน Cursor Regression Model"""
    print("\n" + "=" * 60)
    print("  CURSOR MODEL EVALUATION")
    print("=" * 60)
    
    # โหลดโมเดล
    if not os.path.exists(CURSOR_MODEL_PATH):
        print("  ❌ ไม่พบโมเดล cursor")
        return None
    
    model_data = joblib.load(CURSOR_MODEL_PATH)
    model = model_data["model"]
    scaler = model_data.get("scaler")
    name = model_data.get("name", "Unknown")
    
    print(f"  โมเดล: {name}")
    
    # โหลดข้อมูล
    if not os.path.exists(CALIBRATION_DATA_PATH):
        print("  ❌ ไม่พบข้อมูล calibration")
        return None
    
    # ใช้ขั้นตอนโหลด/ทำความสะอาด/แบ่งกลุ่มเดียวกับตอนเทรน (พับมุม + แบ่งตามการเยี่ยมจุด)
    from sklearn.model_selection import GroupShuffleSplit
    from training.train_cursor_model import load_data as load_cursor_data
    loaded = load_cursor_data()
    if loaded is None:
        return None
    X, y, _, groups = loaded
    _, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]
    
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    print(f"\n  Results:")
    print(f"    MAE:    {mae:.2f} px (X: {mae_x:.2f}, Y: {mae_y:.2f})")
    print(f"    RMSE:   {rmse:.2f} px")
    print(f"    R²:     {r2:.4f}")
    
    # Learning Curve
    save_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_scaled, y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="neg_mean_absolute_error"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        train_mean = -train_scores.mean(axis=1)
        test_mean = -test_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_std = test_scores.std(axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='steelblue',
                label='Training MAE')
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='steelblue')
        ax.plot(train_sizes, test_mean, 'o-', color='coral',
                label='Validation MAE')
        ax.fill_between(train_sizes, test_mean - test_std,
                       test_mean + test_std, alpha=0.1, color='coral')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('MAE (pixels)')
        ax.set_title(f'Learning Curve — {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cursor_learning_curve.png"), dpi=150)
        plt.close()
        print(f"\n  📊 Learning curve: {save_dir}/cursor_learning_curve.png")
    except Exception as e:
        print(f"  ⚠️ ไม่สามารถสร้าง learning curve: {e}")
    
    return {
        "name": name,
        "MAE": mae, "MAE_X": mae_x, "MAE_Y": mae_y,
        "RMSE": rmse, "R2": r2,
    }


def evaluate_click_model():
    """ประเมิน Click Classification Model"""
    print("\n" + "=" * 60)
    print("  CLICK MODEL EVALUATION")
    print("=" * 60)
    
    # โหลดโมเดล
    if not os.path.exists(CLICK_MODEL_PATH):
        print("  ❌ ไม่พบโมเดล click")
        return None
    
    model_data = joblib.load(CLICK_MODEL_PATH)
    model = model_data["model"]
    scaler = model_data.get("scaler")
    name = model_data.get("name", "Unknown")
    
    print(f"  โมเดล: {name}")
    
    # โหลดข้อมูล
    if not os.path.exists(GESTURE_DATA_PATH):
        print("  ❌ ไม่พบข้อมูล gesture")
        return None
    
    df = pd.read_csv(GESTURE_DATA_PATH)
    X = df.drop(columns=["class"]).values
    y = df["class"].values.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    unique_classes = sorted(np.unique(y_test))
    class_labels = [CLASS_NAMES.get(i, f"Class {i}") for i in unique_classes]
    
    report = classification_report(
        y_test, y_pred,
        target_names=class_labels,
        zero_division=0
    )
    
    print(f"\n  Results:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"\n  Classification Report:")
    print(report)
    
    # Learning Curve
    save_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_scaled, y,
            cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy"
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_std = test_scores.std(axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='steelblue',
                label='Training Accuracy')
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='steelblue')
        ax.plot(train_sizes, test_mean, 'o-', color='coral',
                label='Validation Accuracy')
        ax.fill_between(train_sizes, test_mean - test_std,
                       test_mean + test_std, alpha=0.1, color='coral')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Learning Curve — {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "click_learning_curve.png"), dpi=150)
        plt.close()
        print(f"\n  📊 Learning curve: {save_dir}/click_learning_curve.png")
    except Exception as e:
        print(f"  ⚠️ ไม่สามารถสร้าง learning curve: {e}")
    
    return {
        "name": name,
        "Accuracy": acc, "F1": f1,
        "confusion_matrix": cm,
    }


def generate_full_report():
    """สร้างรายงานสรุปทั้งหมด"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  EYE ORDER COME AI — FULL EVALUATION REPORT             ║")
    print("╚" + "═" * 58 + "╝")
    
    cursor_result = evaluate_cursor_model()
    click_result = evaluate_click_model()
    
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    
    if cursor_result:
        print(f"\n  🎯 Cursor Model: {cursor_result['name']}")
        print(f"     MAE: {cursor_result['MAE']:.2f} px")
        print(f"     R²:  {cursor_result['R2']:.4f}")
    else:
        print("\n  ❌ Cursor Model: ยังไม่ได้เทรน")
    
    if click_result:
        print(f"\n  🖱️ Click Model: {click_result['name']}")
        print(f"     Accuracy: {click_result['Accuracy']:.4f}")
        print(f"     F1-Score: {click_result['F1']:.4f}")
    else:
        print("\n  ❌ Click Model: ยังไม่ได้เทรน")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    generate_full_report()
