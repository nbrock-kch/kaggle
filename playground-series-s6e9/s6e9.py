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
from sklearn.inspection import permutation_importance
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
    count bars for categorical / low-cardinality; integer columns get one bin per
    value. Blue = target 1, orange = 0;
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
            # integer-valued columns get one bin per value so no bin ever straddles two values
            v = d[c].dropna()
            whole = (v % 1 == 0).all() and v.nunique() <= 100
            bins = np.arange(v.min() - 0.5, v.max() + 1.5) if whole else 40
            for t, col in colors.items():
                ax.hist(d.loc[d[TARGET] == t, c], bins=bins, alpha=0.6, color=col)
        ax.set_title(c, fontsize=10); ax.set_xlabel(''); ax.tick_params(labelsize=8)
    for ax in axes.flat[len(cols):]:
        ax.axis('off')
    fig.legend(handles=[Patch(color=colors[1], label='Yes'), Patch(color=colors[0], label='No')], title=TARGET, loc='lower right')
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches='tight')
        print(f'wrote {path}')
    plt.show()


# --- engineering
# Each step is df -> df on the raw frame (target column optional), so the same call
# works on train and on test. Try one step at a time in the notebook; keep PASSes in engineer().
def clip_outliers(df, cols, lo=0.01, hi=0.99):
    """Winsorize: clip each column to its [lo, hi] quantiles, pulling outliers in."""
    df = df.copy()
    for c in cols:
        df[c] = df[c].clip(df[c].quantile(lo), df[c].quantile(hi))
    return df


def log_transform(df, cols):
    """log1p on right-skewed non-negative columns."""
    df = df.copy()
    for c in cols:
        df[c] = np.log1p(df[c])
    return df


def smooth(df, cols, width=3):
    """Bin a noisy numeric column into buckets of `width` (floor), damping high variance."""
    df = df.copy()
    for c in cols:
        df[c] = (df[c] // width) * width
    return df


def merge_levels(df, col, mapping):
    """Fold rare category levels into others, e.g. {'Other': 'Male'}."""
    df = df.copy()
    df[col] = df[col].replace(mapping)
    return df


def add_age_is_decade(df):
    """Flag ages on a round decade (30, 40, ...)."""
    df = df.copy()
    df['age_is_decade'] = (df['Age'] % 10 == 0).astype(int)
    return df


def drop_features(df, cols):
    """Remove low-importance columns."""
    return df.drop(columns=cols)


def engineer(df):
    """Kept steps (PASS in the notebook), in order. Applied to train and test alike."""
    # df = clip_outliers(df, ['Annual_Income_USD', 'Daily_Commute_km', 'Charging_Stations_Near_Work'])
    # df = log_transform(df, ['Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work'])
    # df = smooth(df, ['Charging_Stations_Near_Work'])
    # df = merge_levels(df, 'Gender', {'Other': 'Male'})
    # df = add_age_is_decade(df)
    # df = drop_features(df, ['Gender', 'Current_Car_Type'])
    return df


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


def run(df):
    """prep -> train on a (possibly engineered) frame. Prints the gain over the saved baseline."""
    X, cats = prep(df)
    y = df[TARGET]
    models, oof = train(X, y)
    base = Path('baseline_results.txt')
    if base.exists():
        b = float(base.read_text().split('oof auc ')[1].split()[0])
        print(f'baseline {b:.5f}  delta {roc_auc_score(y, oof) - b:+.5f}')
    return X, y, cats, models, oof


def pi(df, sample=100_000):
    """
    Permutation importance: one quick fit on a sample, AUC drop when each feature is shuffled.
    Takes a frame that still has the target column (raw or engineered) and preps it itself,
    so the same call works before and after engineering.
    """
    d = df.sample(min(sample, len(df)), random_state=SEED)
    X, _ = prep(d)
    y = d[TARGET]
    cut = int(len(X) * 0.8)
    m = make_model().fit(X.iloc[:cut], y.iloc[:cut])
    r = permutation_importance(m, X.iloc[cut:], y.iloc[cut:], scoring='roc_auc', n_repeats=5, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).round(4)


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
