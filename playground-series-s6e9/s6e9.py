"""Playground Series S6E9 - Will_Buy_EV (binary classification, ROC AUC).

Shared logic for the submission notebook. Three sections: read, train, test.
Runs unchanged locally and on Kaggle.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# --- params
TARGET = 'Will_Buy_EV'
ID = 'id'
SEED = 42
FOLDS = 5

KAGGLE_DIR = Path('/kaggle/input/playground-series-s6e9')
DATA_DIR = KAGGLE_DIR if KAGGLE_DIR.exists() else Path(__file__).parent


# --- read
def read(name):
    """Load a competition CSV. Drops the stray index column loadkaggle.py adds."""
    df = pd.read_csv(DATA_DIR / f'{name}.csv')
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]


def read_all():
    train = read('train')
    test = read('test')
    train[TARGET] = (train[TARGET] == 'Yes').astype(int) # convert string to binary
    return train, test


def prep(df, cats=None):
    """
    Feature frame: object columns -> category with a shared category set.
    Don't touch test; pass `cats` from the train frame so test encodes identically.
    """
    X = df.drop(columns=[ID, TARGET], errors='ignore').copy()
    obj = X.select_dtypes('object').columns
    if cats is None:
        cats = {c: sorted(X[c].unique()) for c in obj}
    for c in obj:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X, cats


def plot_features(df, cols=None, ncols=4, sample=50_000, path=None):
    """
    One panel per feature, distribution split by target: histograms for numeric,
    count bars for categorical / low-cardinality. Blue = target 1, orange = 0;
    matching shapes mean the feature does not separate the classes.
    Pass `path` to also save the figure (png).
    """
    d = df.sample(min(sample, len(df)), random_state=SEED)
    cols = cols or [c for c in d.columns if c not in (ID, TARGET)]
    colors = {0: '#eb6834', 1: '#2a78d6'}
    fig, axes = plt.subplots(-(-len(cols) // ncols), ncols, figsize=(4 * ncols, 3 * -(-len(cols) // ncols)))
    for ax, c in zip(axes.flat, cols):
        if d[c].dtype == object or d[c].nunique() <= 10:
            d.groupby(c)[TARGET].value_counts().unstack().plot.bar(ax=ax, color=colors, width=0.8, legend=False)
        else:
            for t, col in colors.items():
                ax.hist(d.loc[d[TARGET] == t, c], bins=40, alpha=0.6, color=col)
        ax.set_title(c, fontsize=10); ax.set_xlabel(''); ax.tick_params(labelsize=8)
    for ax in axes.flat[len(cols):]:
        ax.axis('off')
    fig.legend(handles=[Patch(color=colors[1], label='Yes'), Patch(color=colors[0], label='No')], title=TARGET, loc='lower right')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches='tight')
        print(f'wrote {path}')
    plt.show()


# engineering
def engineer(df):
    """Optimizing for increased predictive capacity."""

    # standardize all

    # quartile 


# --- train
def make_model():
    # HGB: boosted trees suit mixed tabular data, fast on 668k rows (histogram binning),
    # reads pandas category dtype directly, tolerates NaN, early stopping picks iters.
    # Swap for LightGBM / XGBoost / CatBoost (and ensemble) once the baseline is stable.
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        early_stopping=True,
        categorical_features='from_dtype',
        random_state=SEED,
    )


def train(X, y):
    """Stratified K-fold CV. Returns fitted fold models and out-of-fold probabilities."""
    oof = np.zeros(len(y))
    models = []
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f'fold {k}  auc {roc_auc_score(y.iloc[va], oof[va]):.5f}  iters {m.n_iter_}')
        models.append(m)
    print(f'oof auc {roc_auc_score(y, oof):.5f}')
    return models, oof

def pi(s):
    samp = train.sample(100_000, random_state=s.SEED)
    Xs, _ = s.prep(samp)
    ys = samp[s.TARGET]
    cut = int(len(Xs) * 0.8)
    m = s.make_model().fit(Xs.iloc[:cut], ys.iloc[:cut])
    pi = permutation_importance(m, Xs.iloc[cut:], ys.iloc[cut:], scoring='roc_auc', n_repeats=5, random_state=s.SEED, n_jobs=-1)
    return pd.Series(pi.importances_mean, index=Xs.columns).sort_values(ascending=False).round(4)

def oof_report(X, y, oof):
    """Out-of-fold analysis: ROC curve, calibration by decile, weakest slices."""
    RocCurveDisplay.from_predictions(y, oof)
    plt.title(f'OOF ROC  auc {roc_auc_score(y, oof):.4f}')
    plt.show()

    # calibration: predicted vs actual rate per probability decile
    dec = pd.qcut(oof, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'pred': oof, 'actual': y.values}).groupby(dec).mean()
    cal['n'] = pd.Series(oof).groupby(dec).size()
    print('--- calibration by decile')
    print(cal.round(4).to_string())

    # per-category AUC: where does the model rank worst?
    print('--- auc by category level')
    for c in X.select_dtypes('category').columns:
        for lvl, idx in X.groupby(c, observed=True).indices.items():
            if y.iloc[idx].nunique() == 2:
                print(f'{c:28s} {str(lvl):10s} n={len(idx):7d}  auc {roc_auc_score(y.iloc[idx], oof[idx]):.4f}')


# --- test
def test(models, X_test, ids, path='submission.csv'):
    """Average fold probabilities and write the submission."""
    pred = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    sub = pd.DataFrame({ID: ids, TARGET: pred})
    sub.to_csv(path, index=False)
    print(f'wrote {path}  rows {len(sub)}  mean {pred.mean():.4f}')
    return sub
