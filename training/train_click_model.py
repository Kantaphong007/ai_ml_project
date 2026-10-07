"""
EYE ORDER COME AI — Train Click Model (Classification)
เทรนโมเดลจำแนกท่าทางตา 5 คลาส จาก "episode" ของสัญญาณจริง (Record Signals)

แนวทาง Hybrid:
  GestureDetector หา episode (ช่วงหลับตา) → สกัดฟีเจอร์ 19 ตัว → โมเดล ML บอกว่าเป็นท่าอะไร
  ใช้ทั้งตอนเทรนและตอนใช้งานจริงด้วยโค้ดเดียวกัน (core/episode_features.py)

ขั้นตอน:
  1. Dataset: replay data/gesture_signals.csv ผ่าน GestureDetector → episode + label จากท่าที่สั่ง
     + Data augmentation: สลับตาซ้าย↔ขวา (ขยิบซ้าย ↔ ขยิบขวา) ได้ข้อมูล ×2
  2. เปรียบเทียบ 5 อัลกอริทึม (GridSearchCV + Grouped CV แยกตาม session กันข้อมูลรั่ว)
       Random Forest · Gradient Boosting · SVM (RBF) · MLP · KNN
  3. Metrics ระดับ episode: Accuracy, Precision, Recall, F1 (macro), Confusion Matrix
  4. Metrics ระดับ event (ใช้งานจริง): replay สัญญาณผ่านระบบทั้งระบบ เทียบกับ rule-based
     (leave-one-session-out ถ้ามี ≥2 session) → ความแม่นต่อท่า + คลิกผิด/นาที
  5. บันทึก models/click_model.pkl (+ use_live = ML ไม่แพ้ rule-based) และกราฟใน data/plots/

Usage:
    python training/train_click_model.py
    python training/train_click_model.py --force-live     # ใช้ ML สั่งคลิกแม้ผลแพ้ rule-based
"""
import argparse
import os
import sys
import time
import warnings
from collections import Counter

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, StratifiedGroupKFold, cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import CLICK_MODEL_PATH, DATA_DIR
from config.tuning import load_tuning
from core.click_classifier import ClickClassifier
from core.episode_features import EYE_CLASS_NAMES, FEATURE_NAMES, FEATURE_VERSION
from training.signal_data import (
    EXPECTED, GRACE_S, SIGNALS_PATH, Session, evaluate, load_sessions, replay,
)

CLASSES = sorted(EYE_CLASS_NAMES)
# พารามิเตอร์ที่กำหนดว่า episode ถูกตัดตรงไหน — เปลี่ยนแล้วต้องเทรนใหม่
SEGMENTATION_KEYS = ["close_ratio", "open_ratio", "open_confirm_s", "half_open_abort_s",
                     "drag_hold_s", "episode_timeout_s", "ref_window_s", "ref_percentile"]
LIVE_MARGIN = 0.02      # ML ต้องได้คะแนน event ≥ rule-based − ค่านี้ ถึงใช้สั่งคลิกจริง
_I_DUR = FEATURE_NAMES.index("duration")
_I_GAP = FEATURE_NAMES.index("prev_gap")


# ──────────────────────────────────────────────
# 1. Dataset
# ──────────────────────────────────────────────
def mirror_session(sess):
    """สลับตาซ้าย↔ขวา + สลับ label ขยิบซ้าย↔ขวา (data augmentation)"""
    g = sess.df.copy()
    g["ear_l"], g["ear_r"] = sess.df["ear_r"].values, sess.df["ear_l"].values
    g["label"] = g["label"].replace({"wink_left": "wink_right", "wink_right": "wink_left"})
    return Session(sess.sid, g)


def extract_episodes(sess, gp):
    """replay session แล้วเก็บฟีเจอร์ทุกจุดตัดสิน (ตรงกับที่โมเดลจะเห็นตอนใช้งานจริง)"""
    recs = []
    replay(sess, gp, hook=lambda kind, start, t, feats: recs.append(
        {"kind": kind, "start": start, "t": t, "feats": feats}))
    return recs


def label_episodes(sess, recs):
    """ใส่ label จากท่าที่สั่งในแต่ละ trial

    - ขยิบ: episode ที่ยาวที่สุดในหน้าต่าง = ขยิบ, ที่เหลือ = กะพริบธรรมชาติ (0)
    - drag: จุดตัดสิน hold = 4 (ถ้าไม่ได้ค้างถึง drag_hold_s = ทำท่าไม่สำเร็จ ตัดทิ้ง)
    - ดับเบิลคลิก: กะพริบครั้งที่สอง (มีครั้งแรกอยู่ในหน้าต่างเดียวกัน) = 3, ครั้งแรก = 0
    - นอกนั้น (กะพริบปกติ, ก้ม/เงย, ยิ้ม, ช่วงพัก, idle) = 0
    คืน (labels, keep) — keep=False = ตัดออกเพราะ label ไม่ชัด
    """
    labels = [0] * len(recs)
    keep = [True] * len(recs)
    for _, label, t0, t1 in sess.trials:
        idx = [i for i, r in enumerate(recs) if t0 - 0.3 <= r["start"] <= t1 + GRACE_S]
        if not idx:
            continue
        if label in ("wink_left", "wink_right"):
            main = max(idx, key=lambda i: recs[i]["feats"][_I_DUR])
            if recs[main]["kind"] == "end" and recs[main]["feats"][_I_DUR] >= 0.15:
                labels[main] = 1 if label == "wink_left" else 2
            else:
                keep[main] = False          # ค้างนานจนเป็น drag / สั้นจนไม่ใช่ขยิบ
        elif label == "drag":
            holds = [i for i in idx if recs[i]["kind"] == "hold"]
            if holds:
                labels[holds[0]] = 4
            else:
                keep[max(idx, key=lambda i: recs[i]["feats"][_I_DUR])] = False
        elif label == "double_blink":
            pairs = [i for i in idx if recs[i]["kind"] == "end" and i - 1 in idx
                     and recs[i]["feats"][_I_GAP] <= 1.2]
            if pairs:
                labels[pairs[-1]] = 3
    return labels, keep


def build_dataset(sessions, gp, augment=True):
    """→ X (n, 19), y (n,), groups (n,), sids (n,)"""
    X, y, groups, sids = [], [], [], []
    for s in sessions:
        variants = [s, mirror_session(s)] if augment else [s]
        for v in variants:
            recs = extract_episodes(v, gp)
            labels, keep = label_episodes(v, recs)
            for k, (r, lb, ok) in enumerate(zip(recs, labels, keep)):
                if ok:
                    X.append(r["feats"])
                    y.append(lb)
                    groups.append(f"{s.sid}#{k}")    # ต้นฉบับ+ภาพสะท้อนอยู่กลุ่มเดียวกัน
                    sids.append(s.sid)
    return np.array(X, dtype=np.float64), np.array(y, dtype=int), np.array(groups), np.array(sids)


# ──────────────────────────────────────────────
# 2. Models
# ──────────────────────────────────────────────
def define_models():
    """5 อัลกอริทึม + hyperparameter grid (ทุกตัวมี StandardScaler นำหน้า)"""
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return {
        "Random Forest": (pipe(RandomForestClassifier(random_state=42, class_weight="balanced")),
                          {"clf__n_estimators": [200], "clf__max_depth": [None, 8],
                           "clf__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=42)),
                              {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                               "clf__learning_rate": [0.1]}),
        "SVM (RBF)": (pipe(SVC(kernel="rbf", probability=True, class_weight="balanced",
                               random_state=42)),
                      {"clf__C": [1, 10, 30], "clf__gamma": ["scale"]}),
        "MLP (Neural Network)": (pipe(MLPClassifier(random_state=42, max_iter=1500)),
                                 {"clf__hidden_layer_sizes": [(32,), (64, 32)],
                                  "clf__alpha": [1e-3, 1e-2]}),
        "KNN": (pipe(KNeighborsClassifier()),
                {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["distance"]}),
    }


def make_cv(y, groups, sids):
    """แยก fold ตาม session (ถ้ามี ≥2) ไม่งั้นแยกตาม episode — ภาพสะท้อนอยู่ fold เดียวกันเสมอ"""
    n_sessions = len(set(sids))
    if n_sessions >= 2:
        return GroupKFold(n_splits=min(5, n_sessions)), sids, f"GroupKFold ตาม session ({min(5, n_sessions)} fold)"
    min_count = min(Counter(y).values())
    k = max(2, min(5, min_count))
    return (StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42), groups,
            f"StratifiedGroupKFold ตาม episode ({k} fold, มี session เดียว)")


def compare_models(X, y, cv, cv_groups):
    results = {}
    print("\n" + "=" * 70)
    print("  TRAINING & EVALUATION (out-of-fold)")
    print("=" * 70)
    for name, (model, grid) in define_models().items():
        t0 = time.time()
        gs = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X, y, groups=cv_groups)
        y_pred = cross_val_predict(clone(gs.best_estimator_), X, y, groups=cv_groups, cv=cv)
        m = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y, y_pred, average="macro", zero_division=0),
            "CV_Score": gs.best_score_,
        }
        results[name] = {
            "estimator": gs.best_estimator_, "best_params": gs.best_params_, "metrics": m,
            "y_pred": y_pred, "confusion_matrix": confusion_matrix(y, y_pred, labels=CLASSES),
            "report": classification_report(y, y_pred, labels=CLASSES,
                                            target_names=[EYE_CLASS_NAMES[c] for c in CLASSES],
                                            zero_division=0),
            "train_time": time.time() - t0,
        }
        print(f"\n  ── {name} ──  {gs.best_params_}")
        print(f"    Accuracy {m['Accuracy']:.4f} | Precision {m['Precision']:.4f} | "
              f"Recall {m['Recall']:.4f} | F1(macro) {m['F1']:.4f} | {results[name]['train_time']:.1f}s")
    return results


# ──────────────────────────────────────────────
# 3. Event-level evaluation (ใช้งานจริง)
# ──────────────────────────────────────────────
def _wrap(model):
    c = ClickClassifier()
    c.model = model
    return c


def event_level_eval(sessions, gp, estimator, X, y, sids):
    """replay ทั้งระบบด้วย ML เทียบ rule-based — แยก session ที่ใช้ทดสอบออกจากการเทรน"""
    rule = evaluate(sessions, gp)
    if len(sessions) >= 2:
        clfs = {}
        for s in sessions:
            m = sids != s.sid
            clfs[s.sid] = _wrap(clone(estimator).fit(X[m], y[m]))
        ml = evaluate(sessions, gp, classifier=clfs)
        how = "leave-one-session-out"
    else:
        ml = evaluate(sessions, gp, classifier=_wrap(clone(estimator).fit(X, y)))
        how = "in-sample (มี session เดียว — ผลจะดีเกินจริง ควรอัดเพิ่ม)"
    return rule, ml, how


# ──────────────────────────────────────────────
# 4. Report / plots / save
# ──────────────────────────────────────────────
def print_summary_table(results, best_name):
    print("\n" + "=" * 85)
    print("  SUMMARY — CLICK CLASSIFICATION MODEL COMPARISON (episode level, out-of-fold)")
    print("=" * 85)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1(macro)':>10} {'CV':>8} {'Time':>8}")
    print("  " + "-" * 83)
    for name, res in results.items():
        m = res["metrics"]
        print(f"  {name:<25} {m['Accuracy']:>10.4f} {m['Precision']:>10.4f} {m['Recall']:>10.4f} "
              f"{m['F1']:>10.4f} {m['CV_Score']:>8.4f} {res['train_time']:>7.1f}s"
              f"{' ★' if name == best_name else ''}")
    print("  " + "-" * 83)
    print(f"  ★ Best model: {best_name}\n\n  Classification Report ({best_name}):")
    print(results[best_name]["report"])


def print_event_eval(rule, ml, how):
    print("\n" + "=" * 70)
    print(f"  EVENT-LEVEL (ใช้งานจริง) — {how}")
    print("=" * 70)
    print(f"  {'ท่า':<14} {'Rule-based':>12} {'ML':>10}")
    for label in EXPECTED:
        if label in rule["acc"]:
            print(f"  {label:<14} {rule['acc'][label]:>12.1%} {ml['acc'].get(label, 0):>10.1%}")
    print(f"  {'balanced acc':<14} {rule['balanced_acc']:>12.1%} {ml['balanced_acc']:>10.1%}")
    print(f"  {'FP/นาที':<14} {rule['fp_per_min']:>12.2f} {ml['fp_per_min']:>10.2f}")
    print(f"  {'score':<14} {rule['score']:>12.3f} {ml['score']:>10.3f}")


def plot_results(results, y, best_name, rule, ml, final_model, save_dir=None):
    save_dir = save_dir or os.path.join(DATA_DIR, "plots")
    os.makedirs(save_dir, exist_ok=True)
    names = list(results)
    class_labels = [EYE_CLASS_NAMES[c] for c in CLASSES]

    # 1) เปรียบเทียบโมเดล
    fig, ax = plt.subplots(figsize=(12, 6))
    x, w = np.arange(len(names)), 0.2
    for j, (metric, color) in enumerate([("Accuracy", "#2196F3"), ("Precision", "#4CAF50"),
                                         ("Recall", "#FF9800"), ("F1", "#F44336")]):
        vals = [results[n]["metrics"][metric] for n in names]
        bars = ax.bar(x + (j - 1.5) * w, vals, w, label=metric if metric != "F1" else "F1 (macro)",
                      color=color)
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (out-of-fold)")
    ax.set_title("Click Classification — Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_model_comparison.png"), dpi=150)
    plt.close()

    # 2) Confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(results[best_name]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name} (out-of-fold)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_confusion_matrix.png"), dpi=150)
    plt.close()

    # 3) F1 ต่อคลาส
    f1s = f1_score(y, results[best_name]["y_pred"], labels=CLASSES, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_labels, f1s, color=sns.color_palette("husl", len(CLASSES)))
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-Score")
    ax.set_title(f"Per-Class F1-Score — {best_name}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_per_class_f1.png"), dpi=150)
    plt.close()

    # 4) ระดับ event: ML vs rule-based
    labels = [k for k in EXPECTED if k in rule["acc"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [rule["acc"][k] for k in labels], 0.4, label="Rule-based", color="#9E9E9E")
    ax.bar(x + 0.2, [ml["acc"].get(k, 0) for k in labels], 0.4, label="ML (hybrid)", color="#3F51B5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy per gesture")
    ax.set_title(f"Event-level: Rule-based (FP {rule['fp_per_min']:.2f}/min) vs "
                 f"ML (FP {ml['fp_per_min']:.2f}/min)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "click_event_level.png"), dpi=150)
    plt.close()

    # 5) Feature importance (ถ้าโมเดลมี)
    clf = final_model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        order = np.argsort(clf.feature_importances_)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(np.array(FEATURE_NAMES)[order], clf.feature_importances_[order], color="#009688")
        ax.set_title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "click_feature_importance.png"), dpi=150)
        plt.close()
    print(f"\n  📊 กราฟบันทึกที่: {save_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="เทรนโมเดลจำแนกท่าทางตา (hybrid episode classifier)")
    ap.add_argument("--data", default=SIGNALS_PATH)
    ap.add_argument("--out", default=CLICK_MODEL_PATH)
    ap.add_argument("--force-live", action="store_true",
                    help="ใช้ ML สั่งคลิกจริงแม้ผลระดับ event แพ้ rule-based")
    ap.add_argument("--no-augment", action="store_true", help="ไม่ทำ mirror augmentation")
    ap.add_argument("--plots", default=os.path.join(DATA_DIR, "plots"), help="โฟลเดอร์เก็บกราฟ")
    args = ap.parse_args(argv)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  EYE ORDER COME AI — CLICK MODEL TRAINING (Hybrid)      ║")
    print("╚" + "═" * 58 + "╝")

    gp, _ = load_tuning()
    sessions = load_sessions(args.data, smile_comp=gp.smile_pitch_comp)
    if not sessions:
        return None

    X, y, groups, sids = build_dataset(sessions, gp, augment=not args.no_augment)
    counts = Counter(y.tolist())
    print(f"\n  📂 {len(sessions)} session → {len(y)} episode "
          f"({'รวม mirror augmentation ×2' if not args.no_augment else 'ไม่ augment'})")
    for c in CLASSES:
        print(f"    Class {c} ({EYE_CLASS_NAMES[c]}): {counts.get(c, 0)}")
    missing = [EYE_CLASS_NAMES[c] for c in CLASSES if counts.get(c, 0) < 4]
    if missing:
        print(f"\n  ❌ ข้อมูลไม่พอ (<4 ตัวอย่าง): {missing}\n     Record Signals เพิ่มก่อน")
        return None

    cv, cv_groups, cv_desc = make_cv(y, groups, sids)
    print(f"\n  Cross-validation: {cv_desc}")
    results = compare_models(X, y, cv, cv_groups)
    best_name = max(results, key=lambda n: results[n]["metrics"]["F1"])
    print_summary_table(results, best_name)

    best_est = results[best_name]["estimator"]
    rule, ml, how = event_level_eval(sessions, gp, best_est, X, y, sids)
    print_event_eval(rule, ml, how)

    use_live = args.force_live or ml["score"] >= rule["score"] - LIVE_MARGIN
    final_model = clone(best_est).fit(X, y)
    data = {
        "model": final_model,
        "name": best_name,
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "classes": CLASSES,
        "use_live": bool(use_live),
        "segmentation": {k: getattr(gp, k) for k in SEGMENTATION_KEYS},
        "metrics": results[best_name]["metrics"],
        "best_params": results[best_name]["best_params"],
        "event_eval": {"method": how,
                       "rule": {k: rule[k] for k in ("score", "balanced_acc", "fp_per_min", "acc")},
                       "ml": {k: ml[k] for k in ("score", "balanced_acc", "fp_per_min", "acc")}},
        "n_samples": int(len(y)),
        "sessions": [s.sid for s in sessions],
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(data, args.out)
    print(f"\n  💾 บันทึกโมเดลที่: {args.out}")
    if use_live:
        print(f"  ✅ ระบบจะใช้ {best_name} ตัดสินท่าทางตาตอนใช้งานจริง")
    else:
        print(f"  ⚠️ ML ได้คะแนนระดับ event ต่ำกว่า rule-based ({ml['score']:.3f} < {rule['score']:.3f})"
              " → ระบบยังใช้ rule-based\n     อัดข้อมูลเพิ่มแล้วเทรนใหม่ หรือใช้ --force-live")

    plot_results(results, y, best_name, rule, ml, final_model, args.plots)
    print("\n  ✅ การเทรนเสร็จสมบูรณ์!")
    return data


if __name__ == "__main__":
    main()
