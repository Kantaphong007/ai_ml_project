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
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    CALIBRATION_DATA_PATH, CURSOR_MODEL_PATH, CLICK_MODEL_PATH, DATA_DIR,
)


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
    
    # ใช้ขั้นตอนโหลด/ทำความสะอาดเดียวกับตอนเทรน + ชุดฟีเจอร์ที่โมเดลเลือกไว้
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from training.train_cursor_model import load_data as load_cursor_data, ALL_FEATURES
    loaded = load_cursor_data()
    if loaded is None:
        return None
    _, y, df, groups = loaded
    cols = model_data.get("feature_cols") or ALL_FEATURES
    if any(c not in df.columns for c in cols):
        print("  ❌ โมเดล/ข้อมูล cursor คนละรุ่นกัน — Calibrate + Train Cursor ใหม่")
        return None
    X = df[cols].values
    print(f"  ฟีเจอร์: {cols}")

    # โมเดลที่บันทึกเทรนบนข้อมูลทั้งหมดแล้ว → ประเมินด้วย GroupKFold (เทรนใหม่ทุก fold) ไม่ให้ดีเกินจริง
    pipe = make_pipeline(clone(scaler), clone(model)) if scaler else clone(model)
    cv = GroupKFold(n_splits=min(6, len(np.unique(groups))))
    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae_x = mean_absolute_error(y[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y[:, 1], y_pred[:, 1])

    print("\n  Results (GroupKFold out-of-fold):")
    print(f"    MAE:    {mae:.2f} px (X: {mae_x:.2f}, Y: {mae_y:.2f})")
    print(f"    RMSE:   {rmse:.2f} px")
    print(f"    R²:     {r2:.4f}")

    # Learning Curve
    save_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)

    try:
        train_sizes, train_scores, test_scores = learning_curve(
            pipe, X, y, groups=groups,
            cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.2, 1.0, 8),
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
    """ประเมิน Click Classification Model (hybrid episode classifier)

    ผลหลักมาจากตอนเทรน (out-of-fold + event-level เทียบ rule-based) ที่บันทึกในไฟล์โมเดล
    และสร้าง learning curve ใหม่จาก data/gesture_signals.csv
    """
    from core.episode_features import FEATURE_VERSION
    from config.tuning import load_tuning
    from training.signal_data import SIGNALS_PATH, load_sessions
    from training.train_click_model import build_dataset, make_cv

    print("\n" + "=" * 60)
    print("  CLICK MODEL EVALUATION")
    print("=" * 60)

    if not os.path.exists(CLICK_MODEL_PATH):
        print("  ❌ ไม่พบโมเดล click — Record Signals แล้วกด Train Click")
        return None
    data = joblib.load(CLICK_MODEL_PATH)
    if data.get("feature_version") != FEATURE_VERSION:
        print("  ❌ โมเดล click เป็นรูปแบบเก่า — Record Signals แล้วกด Train Click ใหม่")
        return None

    name, m, ev = data["name"], data["metrics"], data["event_eval"]
    print(f"  โมเดล: {name}  (เทรน {data.get('trained', '?')}, {data.get('n_samples', '?')} episode)")
    print(f"  ใช้สั่งคลิกจริง: {'ใช่' if data.get('use_live') else 'ไม่ (ใช้ rule-based)'}")
    print("\n  Episode level (out-of-fold):")
    print(f"    Accuracy {m['Accuracy']:.4f} | Precision {m['Precision']:.4f} | "
          f"Recall {m['Recall']:.4f} | F1(macro) {m['F1']:.4f}")
    print(f"\n  Event level ({ev['method']}):")
    print(f"    {'':<14}{'Rule-based':>12}{'ML':>10}")
    for label, acc in ev["rule"]["acc"].items():
        print(f"    {label:<14}{acc:>12.1%}{ev['ml']['acc'].get(label, 0):>10.1%}")
    print(f"    {'balanced acc':<14}{ev['rule']['balanced_acc']:>12.1%}{ev['ml']['balanced_acc']:>10.1%}")
    print(f"    {'FP/min':<14}{ev['rule']['fp_per_min']:>12.2f}{ev['ml']['fp_per_min']:>10.2f}")

    # Learning curve (ข้อมูลมากขึ้น → ดีขึ้นไหม = ควรอัดเพิ่มหรือยัง)
    sessions = load_sessions(SIGNALS_PATH, smile_comp=load_tuning()[0].smile_pitch_comp)
    if sessions:
        try:
            gp, _ = load_tuning()
            X, y, groups, sids = build_dataset(sessions, gp)
            cv, cv_groups, _ = make_cv(y, groups, sids)
            train_sizes, train_scores, test_scores = learning_curve(
                data["model"], X, y, groups=cv_groups, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 8), scoring="f1_macro")
            save_dir = os.path.join(DATA_DIR, "plots")
            os.makedirs(save_dir, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            for scores, color, label in ((train_scores, "steelblue", "Training F1 (macro)"),
                                         (test_scores, "coral", "Validation F1 (macro)")):
                mean, std = scores.mean(axis=1), scores.std(axis=1)
                ax.plot(train_sizes, mean, "o-", color=color, label=label)
                ax.fill_between(train_sizes, mean - std, mean + std, alpha=0.1, color=color)
            ax.set_xlabel("Training Set Size (episodes)")
            ax.set_ylabel("F1 (macro)")
            ax.set_title(f"Learning Curve — {name}")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "click_learning_curve.png"), dpi=150)
            plt.close()
            print(f"\n  📊 Learning curve: {save_dir}/click_learning_curve.png")
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถสร้าง learning curve: {e}")

    return {"name": name, "Accuracy": m["Accuracy"], "F1": m["F1"],
            "event_ml": ev["ml"], "event_rule": ev["rule"]}


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
        print(f"     Accuracy: {click_result['Accuracy']:.4f}  F1 (macro): {click_result['F1']:.4f}")
        print(f"     ใช้งานจริง: ถูกต้อง {click_result['event_ml']['balanced_acc']:.1%} "
              f"(rule-based {click_result['event_rule']['balanced_acc']:.1%}), "
              f"คลิกผิด {click_result['event_ml']['fp_per_min']:.2f}/นาที")
    else:
        print("\n  ❌ Click Model: ยังไม่ได้เทรน")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    generate_full_report()
