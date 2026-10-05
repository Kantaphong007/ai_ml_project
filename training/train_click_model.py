"""
EYE ORDER COME AI — Train Click Model (Classification)
เทรน Classification model สำหรับจำแนกคำสั่งคลิก 6 คลาส
จาก sliding window 30 เฟรม (90 features)

เปรียบเทียบ 5 อัลกอริทึม:
  1. Random Forest Classifier
  2. Gradient Boosting Classifier
  3. SVM (RBF kernel)
  4. MLPClassifier (Neural Network)
  5. KNN Classifier

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    GESTURE_DATA_PATH, CLICK_MODEL_PATH, DATA_DIR, CLASS_NAMES
)
from core.feature_extractor import FeatureExtractor


def load_data(data_path=None):
    """โหลดข้อมูล gesture
    
    Returns:
        tuple: (X, y, df) หรือ None ถ้าไม่มีข้อมูล
    """
    if data_path is None:
        data_path = GESTURE_DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"  ❌ ไม่พบข้อมูล: {data_path}")
        print("  กรุณารัน gesture collection ก่อน:")
        print("    python calibration/gesture_collection.py")
        return None
    
    df = pd.read_csv(data_path)
    print(f"  📂 โหลดข้อมูล: {data_path}")
    print(f"     ขนาด: {df.shape[0]} ตัวอย่าง × {df.shape[1]} columns")
    
    X = df.drop(columns=["class"]).values
    y = df["class"].values.astype(int)
    
    # แสดงจำนวนต่อ class
    print(f"\n  จำนวนต่อคลาส:")
    for cls in sorted(np.unique(y)):
        count = np.sum(y == cls)
        name = CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"    Class {cls} ({name}): {count}")
    
    return X, y, df


def define_models():
    """กำหนดโมเดลและ hyperparameter grid (ปรับจูนความเร็ว)
    
    Returns:
        dict: {name: {"model": model, "params": param_grid}}
    """
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100],
                "max_depth": [10, 20],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42, n_estimators=100),
            "params": {
                "learning_rate": [0.1],
                "max_depth": [3, 5],
            },
        },
        "SVM (RBF)": {
            "model": SVC(kernel="rbf", probability=True, random_state=42),
            "params": {
                "C": [1, 10],
                "gamma": ["scale"],
            },
        },
        "MLP (Neural Network)": {
            "model": MLPClassifier(
                random_state=42, max_iter=300, early_stopping=True, n_iter_no_change=5
            ),
            "params": {
                "hidden_layer_sizes": [(128, 64), (64, 32)],
                "activation": ["relu"],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(n_jobs=-1),
            "params": {
                "n_neighbors": [3, 5],
                "weights": ["distance"],
            },
        },
    }
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler):
    """เทรนทุกโมเดล + ประเมินผล
    
    Returns:
        dict: {name: {"model": best_model, "metrics": {...}, ...}}
    """
    models = define_models()
    results = {}
    
    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models.items():
        print(f"\n  ── {name} ──")
        start_time = time.time()
        
        model = config["model"]
        params = config["params"]
        
        # GridSearchCV
        print(f"    GridSearchCV ({len(params)} param sets)...")
        grid_search = GridSearchCV(
            model, params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_
        
        # ทำนาย
        y_pred = best_model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=[CLASS_NAMES.get(i, f"Class {i}")
                         for i in sorted(np.unique(y_test))],
            zero_division=0
        )
        
        elapsed = time.time() - start_time
        
        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "CV_Score": cv_score,
            },
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred,
            "train_time": elapsed,
        }
        
        print(f"    Best params: {best_params}")
        print(f"    CV Score:  {cv_score:.4f}")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Time:      {elapsed:.1f}s")
    
    return results


def select_best_model(results):
    """เลือกโมเดลที่ดีที่สุดจาก F1 Score"""
    best_name = max(results, key=lambda k: results[k]["metrics"]["F1"])
    return best_name, results[best_name]


def save_model(model, scaler, name, new_f1=None, X_test_scaled=None, y_test=None, model_path=None):
    """บันทึกโมเดล (ตรวจสอบก่อนว่าดีกว่าของเดิมไหม)"""
    if model_path is None:
        model_path = CLICK_MODEL_PATH
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    should_save = True
    if os.path.exists(model_path) and new_f1 is not None and X_test_scaled is not None and y_test is not None:
        try:
            old_data = joblib.load(model_path)
            old_model = old_data.get("model")
            old_scaler = old_data.get("scaler")
            old_name = old_data.get("name", "Existing Model")
            
            if old_model is not None:
                X_raw = scaler.inverse_transform(X_test_scaled)
                X_old = old_scaler.transform(X_raw) if old_scaler else X_raw
                old_pred = old_model.predict(X_old)
                old_f1 = f1_score(y_test, old_pred, average="weighted", zero_division=0)
                
                if old_f1 > new_f1 + 0.001:
                    should_save = False
                    print(f"\n  🛡️ โมเดลเดิมดีกว่า! (โมเดลเดิม {old_name} F1: {old_f1:.4f} vs โมเดลใหม่ {name} F1: {new_f1:.4f})")
                    print("     ระบบจึงเปิดเซฟโหมดและรักษาไฟล์โมเดลเดิมไว้ ไม่ให้คะแนนดรอปครับ 👍")
        except Exception:
            pass
    
    if should_save:
        data = {
            "model": model,
            "scaler": scaler,
            "name": name,
        }
        joblib.dump(data, model_path)
        print(f"\n  💾 บันทึกโมเดลที่: {model_path}")


def plot_results(results, y_test, save_dir=None):
    """สร้างกราฟเปรียบเทียบ"""
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # ── Plot 1: Model Comparison ──
    names = list(results.keys())
    accs = [results[n]["metrics"]["Accuracy"] for n in names]
    f1s = [results[n]["metrics"]["F1"] for n in names]
    precs = [results[n]["metrics"]["Precision"] for n in names]
    recs = [results[n]["metrics"]["Recall"] for n in names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(names))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, accs, width, label='Accuracy',
                   color='#2196F3')
    bars2 = ax.bar(x - 0.5*width, precs, width, label='Precision',
                   color='#4CAF50')
    bars3 = ax.bar(x + 0.5*width, recs, width, label='Recall',
                   color='#FF9800')
    bars4 = ax.bar(x + 1.5*width, f1s, width, label='F1-Score',
                   color='#F44336')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Click Classification — Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.15)
    
    # ใส่ค่าบน bar
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_model_comparison.png"), dpi=150)
    plt.close()
    
    # ── Plot 2: Confusion Matrix (best model) ──
    best_name = max(results, key=lambda k: results[k]["metrics"]["F1"])
    cm = results[best_name]["confusion_matrix"]
    
    unique_classes = sorted(np.unique(y_test))
    class_labels = [CLASS_NAMES.get(i, f"Class {i}") for i in unique_classes]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {best_name}')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # ── Plot 3: Per-class F1 for best model ──
    best_model = results[best_name]["model"]
    y_pred = results[best_name]["y_pred"]
    
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("husl", len(unique_classes))
    bars = ax.bar(class_labels, per_class_f1, color=colors)
    ax.set_xlabel('Class')
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Per-Class F1-Score — {best_name}')
    ax.set_ylim(0, 1.15)
    plt.xticks(rotation=30, ha='right')
    
    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_per_class_f1.png"), dpi=150)
    plt.close()
    
    print(f"\n  📊 กราฟบันทึกที่: {save_dir}")


def print_summary_table(results):
    """แสดงตารางสรุปผล"""
    print("\n" + "=" * 85)
    print("  SUMMARY — CLICK CLASSIFICATION MODEL COMPARISON")
    print("=" * 85)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10} {'CV':>10} {'Time':>8}")
    print("  " + "-" * 80)
    
    best_name = max(results, key=lambda k: results[k]["metrics"]["F1"])
    
    for name, res in results.items():
        m = res["metrics"]
        marker = " ★" if name == best_name else ""
        print(f"  {name:<25} {m['Accuracy']:>9.4f} {m['Precision']:>9.4f} "
              f"{m['Recall']:>9.4f} {m['F1']:>9.4f} {m['CV_Score']:>9.4f} "
              f"{res['train_time']:>6.1f}s{marker}")
    
    print("  " + "-" * 80)
    print(f"  ★ Best model: {best_name}")
    
    # แสดง classification report ของ best model
    print(f"\n  Classification Report ({best_name}):")
    print(results[best_name]["classification_report"])
    print("=" * 85)


def main():
    """Main training pipeline"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  EYE ORDER COME AI — CLICK MODEL TRAINING               ║")
    print("╚" + "═" * 58 + "╝")
    
    # 1. โหลดข้อมูล
    data = load_data()
    if data is None:
        return
    X, y, df = data
    
    print(f"\n  Features: {X.shape[1]} ตัวแปร (30 frames × 3 features)")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  ตัวอย่าง: {X.shape[0]}")
    
    # 2. Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. เทรนและประเมิน
    results = train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    )
    
    # 5. แสดงสรุป
    print_summary_table(results)
    
    # 6. เลือกโมเดลที่ดีที่สุด
    best_name, best_result = select_best_model(results)
    
    # 7. บันทึกโมเดล (มีระบบป้องกันถ้าของเดิมดีกว่า)
    save_model(
        best_result["model"], scaler, best_name,
        new_f1=best_result["metrics"]["F1"],
        X_test_scaled=X_test_scaled, y_test=y_test
    )
    
    # 8. สร้างกราฟ
    plot_results(results, y_test)
    
    print("\n  ✅ การเทรนเสร็จสมบูรณ์!")
    return results


if __name__ == "__main__":
    main()
