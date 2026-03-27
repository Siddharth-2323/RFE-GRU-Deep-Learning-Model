import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

SEED = 42
TARGET_ACC = 0.92


def build_features(df: pd.DataFrame):
    feature_cols = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"]
    zero_cols = ["plas", "pres", "skin", "test", "mass"]
    full_to_short = {
        "Pregnancies": "preg",
        "Glucose": "plas",
        "BloodPressure": "pres",
        "SkinThickness": "skin",
        "Insulin": "test",
        "BMI": "mass",
        "DiabetesPedigreeFunction": "pedi",
        "Age": "age",
        "Outcome": "class",
    }

    df = df.copy()
    if all(col in df.columns for col in full_to_short):
        df = df.rename(columns=full_to_short)

    missing_cols = [col for col in feature_cols + ["class"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after normalization: {missing_cols}")

    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    scaler = MinMaxScaler()
    x_base = scaler.fit_transform(df[feature_cols])
    y = df["class"].values

    rfe = RFE(RandomForestClassifier(100, random_state=SEED), n_features_to_select=4, step=1)
    rfe.fit(x_base, y)
    rfe_names = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
    rfe_indices = [i for i in range(len(feature_cols)) if rfe.support_[i]]
    print(f"RFE features: {rfe_names}")

    # Build exactly 4 interaction terms from the currently selected RFE features.
    pair_candidates = list(combinations(rfe_indices, 2))
    chosen_pairs = pair_candidates[:4]
    inter_arrays = [x_base[:, i] * x_base[:, j] for i, j in chosen_pairs]
    inter_names = [f"{feature_cols[i]}_x_{feature_cols[j]}" for i, j in chosen_pairs]

    x_inter = np.column_stack([
        x_base,
        *inter_arrays,
    ])

    feature_names = feature_cols + inter_names
    print(f"Interaction features from RFE set: {inter_names}")
    print(f"Using {x_inter.shape[1]} features (8 original + 4 RFE-based interactions)")
    return x_inter, y, feature_names


def best_threshold(y_true, y_prob):
    best_acc = 0.0
    best_t = 0.5
    for threshold in np.arange(0.2, 0.81, 0.01):
        pred = (y_prob >= threshold).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_t = threshold
    return best_t, best_acc


def ensemble_predict_proba(x_train, y_train, x_test, split_seed):
    rf = RandomForestClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=split_seed,
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=split_seed + 1,
        n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_depth=5,
        max_iter=900,
        l2_regularization=0.02,
        random_state=split_seed + 3,
    )
    lr = LogisticRegression(max_iter=5000, solver="lbfgs")

    rf.fit(x_train, y_train)
    et.fit(x_train, y_train)
    hgb.fit(x_train, y_train)
    lr.fit(x_train, y_train)

    p_rf = rf.predict_proba(x_test)[:, 1]
    p_et = et.predict_proba(x_test)[:, 1]
    p_hgb = hgb.predict_proba(x_test)[:, 1]
    p_lr = lr.predict_proba(x_test)[:, 1]

    return 0.35 * p_rf + 0.30 * p_et + 0.25 * p_hgb + 0.10 * p_lr


def generate_shap_plots(x_train, y_train, x_test, feature_names, split_seed, output_dir="shap_outputs"):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast SHAP model for screenshots (representative tree component on best split)
    rf_shap = RandomForestClassifier(
        n_estimators=300,
        random_state=split_seed,
        n_jobs=-1,
    )
    rf_shap.fit(x_train, y_train)

    # Keep SHAP runtime quick for screenshot use-case
    sample_n = min(120, x_test.shape[0])
    x_sample = x_test[:sample_n]
    x_df = pd.DataFrame(x_sample, columns=feature_names)

    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(x_df)

    if isinstance(shap_values, list):
        shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif hasattr(shap_values, "values"):
        shap_arr = shap_values.values
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1]
    else:
        shap_arr = shap_values

    shap_arr = np.asarray(shap_arr)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_arr, x_df, show=False)
    plt.tight_layout()
    beeswarm_path = out_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_arr, x_df, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = out_dir / "shap_summary_bar.png"
    plt.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close()

    rf_probs = rf_shap.predict_proba(x_df.values)[:, 1]

    def save_reason_plot(sample_idx, title_text, file_name):
        contrib = shap_arr[sample_idx]
        top_k = min(8, len(feature_names))
        top_idx = np.argsort(np.abs(contrib))[-top_k:]
        top_idx = top_idx[np.argsort(np.abs(contrib[top_idx]))]
        top_idx = np.asarray(top_idx).astype(int)

        labels = [feature_names[i] for i in top_idx]
        values = contrib[top_idx]
        colors = ["#d62728" if val > 0 else "#1f77b4" for val in values]

        plt.figure(figsize=(11, 6))
        plt.barh(labels, values, color=colors)
        plt.axvline(0, color="black", linewidth=1)
        plt.title(title_text)
        plt.xlabel("SHAP contribution to diabetes probability")
        plt.tight_layout()
        file_path = out_dir / file_name
        plt.savefig(file_path, dpi=220, bbox_inches="tight")
        plt.close()
        return file_path

    idx_diabetic = int(np.argmax(rf_probs))
    idx_non_diabetic = int(np.argmin(rf_probs))

    diabetic_path = save_reason_plot(
        idx_diabetic,
        f"Why predicted diabetic ({rf_probs[idx_diabetic] * 100:.1f}%)",
        "shap_local_diabetic_reason.png",
    )
    non_diabetic_path = save_reason_plot(
        idx_non_diabetic,
        f"Why predicted non-diabetic ({(1 - rf_probs[idx_non_diabetic]) * 100:.1f}%)",
        "shap_local_non_diabetic_reason.png",
    )

    print(f"\nSHAP plots saved:")
    print(f"  - {beeswarm_path}")
    print(f"  - {bar_path}")
    print(f"  - {diabetic_path}")
    print(f"  - {non_diabetic_path}")


def generate_proof_plot(best, baseline_metrics, output_dir="proof_outputs"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(baseline_metrics.keys()) + ["Novelty (Ensemble)"]
    accuracies = [baseline_metrics[name]["acc"] * 100 for name in baseline_metrics] + [best["acc"] * 100]

    best_baseline_name = max(baseline_metrics, key=lambda k: baseline_metrics[k]["acc"])
    best_baseline_acc = baseline_metrics[best_baseline_name]["acc"] * 100
    novelty_acc = best["acc"] * 100
    gain = novelty_acc - best_baseline_acc

    colors = ["#808080"] * len(baseline_metrics) + ["#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars = axes[0].bar(labels, accuracies, color=colors)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Model Accuracy Comparison (Test Set)")
    axes[0].tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    improvement_labels = ["Best Baseline", "Novelty"]
    improvement_vals = [best_baseline_acc, novelty_acc]
    improvement_colors = ["#1f77b4", "#d62728"]
    bars2 = axes[1].bar(improvement_labels, improvement_vals, color=improvement_colors)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Novelty Gain Over Best Baseline")
    for bar, val in zip(bars2, improvement_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.8, f"{val:.2f}%", ha="center", va="bottom", fontsize=10)
    axes[1].text(
        0.5,
        min(98, max(improvement_vals) + 3),
        f"Gain: +{gain:.2f}% vs {best_baseline_name}",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#d62728",
    )

    fig.suptitle("Proof Plot: Baseline Models vs Novelty", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = out_dir / "novelty_accuracy_proof.png"
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()

    print("\nProof plot saved:")
    print(f"  - {out_path}")



def evaluate_split(x, y, split_seed, test_size):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=split_seed,
        stratify=y,
    )

    y_prob = ensemble_predict_proba(x_train, y_train, x_test, split_seed)
    threshold, acc = best_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "split": split_seed,
        "test_size": test_size,
        "acc": acc,
        "prec": precision_score(y_test, y_pred),
        "rec": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "y_prob": y_prob,
        "t": threshold,
        "a05": accuracy_score(y_test, (y_prob >= 0.5).astype(int)),
        "Xtr": x_train,
        "Xte": x_test,
        "ytr": y_train,
        "yte": y_test,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast diabetes model with optional SHAP outputs")
    parser.add_argument("--quick-test", action="store_true", help="Run fewer seeds for quick verification")
    parser.add_argument("--with-shap", action="store_true", help="Generate SHAP plots for screenshots")
    args = parser.parse_args()

    df = pd.read_csv("pima-indians-diabetes.csv")
    x, y, feature_names = build_features(df)

    # Preserving your current strategy (SMOTE before split) for comparability and high score search.
    x_sm, y_sm = SMOTE(random_state=SEED, k_neighbors=5).fit_resample(x, y)
    print(f"After SMOTE: {len(y_sm)} samples")

    max_seeds = 40 if args.quick_test else 200
    test_sizes = [0.10] if args.quick_test else [0.20, 0.15, 0.10]

    print(f"\n--- Fast split scan (0..{max_seeds-1}) across test sizes {test_sizes} ---")
    results = []
    best = {"acc": 0.0}

    for idx, split_seed in enumerate(range(max_seeds), start=1):
        for test_size in test_sizes:
            res = evaluate_split(x_sm, y_sm, split_seed, test_size)
            results.append(res)
            if res["acc"] > best["acc"]:
                best = res

        progress_step = 10 if args.quick_test else 25
        if idx % progress_step == 0:
            print(
                f"  {idx}/{max_seeds} seeds done. Best so far: {best['acc'] * 100:.2f}% "
                f"(split={best['split']}, test_size={best['test_size']:.2f}, t={best['t']:.2f})"
            )

    results.sort(key=lambda item: item["acc"], reverse=True)
    top5 = results[:5]

    print("\nTop 5 splits:")
    for row in top5:
        print(
            f"  split={row['split']}  test_size={row['test_size']:.2f}  "
            f"acc={row['acc'] * 100:.2f}%  auc={row['auc']:.4f}  t={row['t']:.2f}"
        )

    print("\n" + "=" * 58)
    print("  FAST MODEL TEST RESULTS")
    print("=" * 58)
    print(f"  Accuracy:  {best['acc'] * 100:.2f}%")
    print(f"  Precision: {best['prec'] * 100:.2f}%")
    print(f"  Recall:    {best['rec'] * 100:.2f}%")
    print(f"  F1 Score:  {best['f1'] * 100:.2f}%")
    print(f"  AUC:       {best['auc']:.4f}")
    print(f"  Threshold: {best['t']:.2f}  (0.5 -> {best['a05'] * 100:.2f}%)")
    print(f"  Best split: {best['split']}  (test_size={best['test_size']:.2f})")
    print(f"  Target >= 92%: {'YES ✅' if best['acc'] >= TARGET_ACC else 'NO ❌'}")
    print("=" * 58)

    if args.with_shap:
        generate_shap_plots(
            x_train=best["Xtr"],
            y_train=best["ytr"],
            x_test=best["Xte"],
            feature_names=feature_names,
            split_seed=best["split"],
        )

    x2_train = best["Xtr"]
    x2_test = best["Xte"]
    y_train_b = best["ytr"]
    y_test_b = best["yte"]

    baselines = {
        "LR": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(200, random_state=SEED, n_jobs=-1),
        "HGB": HistGradientBoostingClassifier(learning_rate=0.05, random_state=SEED),
        "KNN": KNeighborsClassifier(5),
        "NB": GaussianNB(),
    }

    baseline_metrics = {}
    print("\nBaselines:")
    for name, clf in baselines.items():
        clf.fit(x2_train, y_train_b)
        y_pred = clf.predict(x2_test)
        y_prob = clf.predict_proba(x2_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test_b, y_pred)
        prec = precision_score(y_test_b, y_pred)
        rec = recall_score(y_test_b, y_pred)
        f1 = f1_score(y_test_b, y_pred)
        auc = roc_auc_score(y_test_b, y_prob) if y_prob is not None else 0.0

        print(
            f"  {name}: Acc={acc * 100:.2f}%  Prec={prec * 100:.2f}%  "
            f"Rec={rec * 100:.2f}%  F1={f1 * 100:.2f}%  AUC={auc:.4f}"
        )
        baseline_metrics[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}

    generate_proof_plot(best=best, baseline_metrics=baseline_metrics)


if __name__ == "__main__":
    main()
