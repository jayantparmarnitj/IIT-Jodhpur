
import math
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None, depth=0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # for leaf: class label
        self.depth = depth
        self.impurity = None  # node impurity (Gini)
        self.n_samples = 0
        self.feature_importance_contrib = defaultdict(float)  # track impurity decrease for features at this node

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):

        self.max_depth = max_depth if (max_depth is None or max_depth > 0) else None
        self.min_samples_split = max(2, min_samples_split)
        self.n_features = n_features  # for feature randomness
        self.root = None
        self.n_features_in_ = None
        self._feature_importances_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        ps = counts / counts.sum()
        return 1.0 - np.sum(ps ** 2)

    def _best_split(self, X, y, features_idx):
        best_feature = None
        best_thresh = None
        best_impurity_decrease = 0
        best_left_idx = None
        best_right_idx = None

        parent_impurity = self._gini(y)
        n_parent = y.size

        if parent_impurity == 0:  # pure
            return None, None, 0, None, None

        for fi in features_idx:
            X_col = X[:, fi]
            sorted_idx = np.argsort(X_col)
            X_sorted = X_col[sorted_idx]
            y_sorted = y[sorted_idx]
            for i in range(1, len(X_sorted)):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue
                if y_sorted[i] == y_sorted[i - 1]:
                    # still possible good split; we will evaluate anyway
                    pass
                thresh = (X_sorted[i] + X_sorted[i - 1]) / 2.0
                left_mask = X_col <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                left_imp = self._gini(y[left_mask])
                right_imp = self._gini(y[right_mask])
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                weighted_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
                impurity_decrease = parent_impurity - weighted_imp
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature = fi
                    best_thresh = thresh
                    best_left_idx = left_mask
                    best_right_idx = right_mask

        return best_feature, best_thresh, best_impurity_decrease, best_left_idx, best_right_idx

    def _build_tree(self, X, y, depth=0):
        node = TreeNode(depth=depth)
        node.n_samples = y.size
        node.impurity = self._gini(y)

        # stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or node.n_samples < self.min_samples_split or node.impurity == 0:
            # leaf
            counts = np.bincount(y)
            node.value = np.argmax(counts)
            return node

        # features to consider
        if self.n_features is None or self.n_features >= X.shape[1]:
            feat_idxs = list(range(X.shape[1]))
        else:
            feat_idxs = random.sample(range(X.shape[1]), self.n_features)

        best_feature, best_thresh, best_imp_decrease, left_idx, right_idx = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            counts = np.bincount(y)
            node.value = np.argmax(counts)
            return node

        node.feature_index = best_feature
        node.threshold = best_thresh

        # recursion
        left_node = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        node.left = left_node
        node.right = right_node

        # track feature importance contribution: impurity decrease
        node.feature_importance_contrib[best_feature] += best_imp_decrease

        return node

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.n_features_in_ = X.shape[1]
        if self.n_features is None:
            self.n_features = self.n_features_in_
        self.root = self._build_tree(X, y, depth=0)
        # aggregate feature importance by summing contributions across nodes
        importances = np.zeros(self.n_features_in_)
        def traverse(node):
            if node is None:
                return
            for fi, contrib in node.feature_importance_contrib.items():
                importances[fi] += contrib
            traverse(node.left)
            traverse(node.right)
        traverse(self.root)
        # normalize
        if importances.sum() > 0:
            self._feature_importances_ = importances / importances.sum()
        else:
            self._feature_importances_ = importances
        return self

    def _predict_one(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    @property
    def feature_importances_(self):
        # returns array length = n_features_in_
        if self._feature_importances_ is None:
            return np.zeros(self.n_features_in_)
        return self._feature_importances_

# -------------------------
# Random Forest Implementation
# -------------------------
class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, n_features=None, bootstrap=True, random_state=None, oob_score=False):
        """
        n_features: number of features to consider at each split for each tree.
                    If None, defaults to sqrt(total_features) (rounded down).
        oob_score: if True, compute OOB estimate
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.trees = []
        self.oob_score = oob_score
        self.oob_predictions = None
        self.feature_importances_ = None
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        return indices, oob_mask

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        n_samples, n_total_features = X.shape
        if self.n_features is None:
            # typical default: sqrt(#features)
            self.n_features = max(1, int(math.sqrt(n_total_features)))

        self.trees = []
        # For OOB predictions
        if self.oob_score:
            oob_votes = [defaultdict(int) for _ in range(n_samples)]
            oob_counts = np.zeros(n_samples, dtype=int)

        for i in tqdm(range(self.n_trees), desc="Training trees"):
            # bootstrap sample
            sample_idx, oob_mask = self._bootstrap_sample(X, y) if self.bootstrap else (np.arange(n_samples), np.zeros(n_samples, dtype=bool))
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=self.n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append({
                "tree": tree,
                "sample_idx": sample_idx,
                "oob_mask": oob_mask
            })

            if self.oob_score:
                # For samples that are OOB for this tree, get prediction and add vote
                oob_indices = np.where(oob_mask)[0]
                if oob_indices.size > 0:
                    preds = tree.predict(X[oob_indices])
                    for idx, pred in zip(oob_indices, preds):
                        oob_votes[idx][pred] += 1
                        oob_counts[idx] += 1

        # aggregate feature importances (average of tree importances)
        importances = np.zeros(n_total_features)
        for rec in self.trees:
            importances += rec["tree"].feature_importances_
        importances /= max(1, len(self.trees))
        self.feature_importances_ = importances / (importances.sum() + 1e-12)

        if self.oob_score:
            # compute OOB predictions where we have at least one OOB vote
            oob_pred = np.full(n_samples, -1)
            for i in range(n_samples):
                if oob_counts[i] > 0:
                    # choose highest vote
                    votes = oob_votes[i]
                    oob_pred[i] = max(votes.items(), key=lambda kv: kv[1])[0]
            # compute OOB error on samples with oob_pred != -1
            valid = oob_pred != -1
            if valid.sum() > 0:
                oob_acc = (oob_pred[valid] == y[valid]).mean()
                self.oob_score_ = oob_acc
            else:
                self.oob_score_ = None

        return self

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        # collect votes
        votes = np.zeros((n_samples, ), dtype=int)
        # We'll gather predictions from all trees and majority vote
        all_preds = np.zeros((len(self.trees), n_samples), dtype=int)
        for ti, rec in enumerate(self.trees):
            preds = rec["tree"].predict(X)
            all_preds[ti] = preds
        # majority vote
        final_preds = []
        for j in range(n_samples):
            col = all_preds[:, j]
            counts = np.bincount(col)
            final_preds.append(np.argmax(counts))
        return np.array(final_preds)

    def predict_proba(self, X):
        # proportion of trees voting for class 1
        X = np.asarray(X)
        n_samples = X.shape[0]
        all_preds = np.zeros((len(self.trees), n_samples), dtype=int)
        for ti, rec in enumerate(self.trees):
            all_preds[ti] = rec["tree"].predict(X)
        probs = all_preds.mean(axis=0)  # fraction of trees predicting 1
        # return 2-col probs [P(0), P(1)]
        return np.vstack([1 - probs, probs]).T


def load_wine_quality(which="red"):

    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
    if which == "red":
        url = base + "/winequality-red.csv"
    else:
        url = base + "/winequality-white.csv"
    df = pd.read_csv(url, sep=";")
    return df

def prepare_binary(df, threshold=5):
    # Convert quality to binary: > threshold -> 1, else 0
    df2 = df.copy()
    df2['target'] = (df2['quality'] > threshold).astype(int)
    X = df2.drop(['quality', 'target'], axis=1).values
    y = df2['target'].values
    feature_names = [c for c in df2.columns if c not in ('quality', 'target')]
    return X, y, feature_names

def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_experiment(
    which_wine='red',
    n_trees_list=[1,5,10,25,50,100],
    max_depth=None,
    min_samples_split=2,
    test_size=0.2,
    random_state=42,
    use_oob=True
):
    # 1. Load
    print(f"Loading {which_wine} wine dataset...")
    df = load_wine_quality(which_wine)
    X, y, feature_names = prepare_binary(df, threshold=5)
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # set n_features for trees (sqrt rule)
    total_features = X.shape[1]
    n_features_at_split = max(1, int(math.sqrt(total_features)))

    # For each n_trees, train and evaluate
    accuracy_vs_trees = []
    results_for_table = {}

    # Train a RandomForest with chosen n_trees_list values and keep the best or last for feature importance
    rf_models = {}
    for n_trees in n_trees_list:
        print(f"\nTraining Random Forest with {n_trees} trees...")
        rf = RandomForestClassifier(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_features=n_features_at_split,
            bootstrap=True,
            random_state=random_state,
            oob_score=use_oob
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        metrics = evaluate_model(y_test, preds)
        print(f"Test Accuracy (n_trees={n_trees}): {metrics['accuracy']:.4f}")
        accuracy_vs_trees.append(metrics['accuracy'])
        results_for_table[f"RF_{n_trees}"] = metrics
        rf_models[n_trees] = rf

    # Train a single (deep) Decision Tree to overfit
    print("\nTraining a single deep Decision Tree (to illustrate overfitting)...")
    single_tree = DecisionTreeClassifier(max_depth=None, min_samples_split=2, n_features=None)
    single_tree.fit(X_train, y_train)
    preds_tree = single_tree.predict(X_test)
    metrics_tree = evaluate_model(y_test, preds_tree)
    results_for_table["SingleTree_deep"] = metrics_tree
    print("Single tree test accuracy:", metrics_tree['accuracy'])

    # Choose RF with max n_trees for feature importance plots
    chosen_n = n_trees_list[-1]
    rf_chosen = rf_models[chosen_n]
    importances = rf_chosen.feature_importances_
    # sort for plotting
    idxs = np.argsort(importances)[::-1]
    sorted_feats = [feature_names[i] for i in idxs]
    sorted_importances = importances[idxs]

    # Print comparison table for single tree vs RF with chosen_n
    print("\n=== Evaluation Metrics (Test Set) ===")
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    print("{:<20s} {:>8s} {:>10s} {:>8s} {:>8s}".format(*headers))
    rf_metrics_chosen = results_for_table[f"RF_{chosen_n}"]
    print("{:<20s} {:8.4f} {:10.4f} {:8.4f} {:8.4f}".format(f"RandomForest_{chosen_n}", rf_metrics_chosen['accuracy'], rf_metrics_chosen['precision'], rf_metrics_chosen['recall'], rf_metrics_chosen['f1']))
    print("{:<20s} {:8.4f} {:10.4f} {:8.4f} {:8.4f}".format("SingleTree_deep", metrics_tree['accuracy'], metrics_tree['precision'], metrics_tree['recall'], metrics_tree['f1']))

    # Plot: Accuracy vs Number of Trees
    plt.figure(figsize=(8,5))
    plt.plot(n_trees_list, accuracy_vs_trees, marker='o')
    plt.title("Test Accuracy vs Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_trees.png")
    print("\nSaved plot: accuracy_vs_trees.png")

    # Plot: Feature Importances
    plt.figure(figsize=(10,6))
    plt.bar(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feats)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Feature Importances (RandomForest, n_trees={chosen_n})")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    print("Saved plot: feature_importances.png")

    # OOB score (if available)
    if use_oob and hasattr(rf_chosen, "oob_score_"):
        print(f"\nOOB Score (for RF with {chosen_n} trees): {rf_chosen.oob_score_}")
    elif use_oob:
        print("\nOOB Score not available.")

    return {
        "results_table": results_for_table,
        "accuracy_vs_trees": (n_trees_list, accuracy_vs_trees),
        "feature_importances": (feature_names, importances),
        "single_tree_metrics": metrics_tree,
        "rf_chosen": rf_chosen,
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    # Parameters (you can tune)
    which_wine_version = "red"   # or "white"
    n_trees_list = [1, 5, 10, 25, 50, 100]
    out = run_experiment(which_wine=which_wine_version, n_trees_list=n_trees_list, max_depth=None, min_samples_split=2, test_size=0.2, random_state=42, use_oob=True)

    # Save a small CSV summary of metrics for reporting
    rows = []
    for k, metrics in out['results_table'].items():
        rows.append({"Model": k, **metrics})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv("rf_vs_single_tree_metrics.csv", index=False)
    print("\nSaved metrics CSV: rf_vs_single_tree_metrics.csv")
    print("Done.")
