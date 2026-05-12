import pandas as pd
import time
import zipfile
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# params
NTREES = 200
NJOBS = 3
PERMUTATIONS = 10
TARGET = 'SalePrice'
REFRESH_OFFLINE_FILE = False


def train(model, X, y, calc_importances=False):
    start = time.time()
    print('training...')
    model.fit(X, y)
    # results
    # print(f'processing time {(time.time() - start):.2f}s')
    print(f'training score {model.score(X, y):.4f}')
    print(f'OOB score {model.oob_score_:.4f}')
    if calc_importances:
        print('--- PERMUTATION IMPORTANCE')
        start = time.time()
        importance = permutation_importance(model, X, y,
                                            scoring='r2', n_repeats=PERMUTATIONS, n_jobs=NJOBS)
        order = importance.importances_mean.argsort()
        importancesdf = pd.DataFrame({
            'feature': X.columns.values[order],
            'importance': importance.importances_mean[order]
        })
        importancesdf = importancesdf.sort_values('importance', ascending=False)
        importancesdf.to_csv('importances.csv')
        print(importancesdf.to_string())
        print(f'processingtime {(time.time() - start) / 60:.2f}m')


def dropnulls(data, cols):
    # use if unable to fill NaN
    before = len(data.index)
    data.dropna(subset=[cols], inplace=True)
    diff = before - len(data.index)
    if diff > 0:
        print('drop NA', diff, 'rows dropped')


def get_data(preview=False, refresh=False):
    # LOAD DATA - run loadkaggle 
    filename = 'prepped_data.csv'
    if refresh:
        print('running data prep again...')
        # --- DATA PREP
        prep = pd.read_csv('train.csv')

        # handle nulls
        print('NaN columns count', len(prep[prep.isna().any(axis=1)].index))
        # fill NA where applicable
        # is it ok to assume "No" if the quality record is blank/NA? Is it possible that this should be "other" in some cases?
        nans = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Electrical', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                'Fence', 'MiscFeature']
        for n in nans:
            prep[n] = prep[n].fillna('No')
        # features have no min in data - assume zero if not entered
        nomin = ['LotFrontage', 'MasVnrArea']
        for no in nomin:
            prep[no] = prep[no].fillna(0)
        # print('NaN columns count', len(prep[prep.isna().any(axis=1)].index))

        # check categorical unique values
        # obj = prep.select_dtypes(include=['object'])
        # uniques = obj.nunique()
        # print(uniques.to_string())
        # for c in obj.columns:
        #     uniques[c] = prep[c].unique()
        # print(uniques.to_string())
        # print(prep.loc[:, obj.columns[obj.nunique() >= 5]].head(10).to_string())

        prep.to_csv(filename)
        print('created new offline file:', filename)
    else:
        print('reading data from offline file:', filename)
        prep = pd.read_csv(filename, index_col='Id')
        
    # data types - retain "object" for categorical numeric columns
    cats = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    for c in cats:
        prep[c] = prep[c].astype(object)

    if preview:
        print(prep.sort_values(TARGET, ascending=True).head(10).to_string())
        print(prep.describe().to_string())

    return prep


# --- CHOOSE MODEL / CREATE MODEL / TUNE
homeprice_rf = RandomForestRegressor(n_estimators=NTREES, n_jobs=NJOBS, oob_score=True)
# confusion matrix - evaluate other models

# --- BASELINE
print('--- BASELINE')
baseline_data = get_data(refresh=REFRESH_OFFLINE_FILE)
baseline_data = pd.get_dummies(baseline_data)
baseline_X, baseline_y = baseline_data.drop(columns=TARGET), baseline_data[TARGET]
# train(homeprice_rf, baseline_X, baseline_y)

# --- FEATURE ENGINEERING
train_data = get_data(refresh=False)
trainX, trainy = train_data.drop(columns=TARGET), train_data[TARGET]

# variability / outliers
# options with high variability - possibly custom homes
options_max_var = ['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
print(train_data[options_max_var].describe())
outliers = train_data[
    (train_data['BsmtFinSF2'] > 1200)
    # | (train_data['PoolArea'] > 0)
    # | (train_data['MiscVal'] > 4000)
    ]
print(outliers[['SalePrice', 'LotArea', 'BedroomAbvGr', 'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']])
# pool is rare
# 1-2 huge outliers - what are these homes? basement only?
# for o in options_max_var:
#     plt.title(o)
#     plt.scatter(train_data.index, train_data[o])
#     plt.show()


for c in train_data.columns:
    if 'bed' in c.lower():
        print(c)
exit()












options_75_var = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF',
               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea',
               'WoodDeckSF', 'OpenPorchSF']

# yearbuilt high variability - possibly old homes
# plt.scatter(train_data.index, train_data['YearBuilt'])
# plt.show()
# visual cutoffs at 1940 1980 2000
train_data['YearBuiltPre1940'] = train_data[train_data['YearBuilt'] <= 1940]
train_data['YearBuiltPre1980'] = train_data[(train_data['YearBuilt'] > 1940) & (train_data['YearBuilt'] <= 1980)]
train_data['YearBuiltPre2000'] = train_data[(train_data['YearBuilt'] > 1980) & (train_data['YearBuilt'] <= 2000)]
train_data.drop('YearBuilt', axis=1, inplace=True)

# upper/lower price outliers
sorted = train_data.sort_values('SalePrice', ascending=False)
upper_outliers, lower_outliers = sorted.head().index.tolist(), sorted.tail().index.tolist()
train_data['upper_outliers'] = train_data.index.isin(upper_outliers)
train_data['lower_outliers'] = train_data.index.isin(lower_outliers)
# print(train_data[train_data['upper_outliers'] | train_data['lower_outliers']])

# combining features for total sqft
train_data['TotalPorchSF'] = train_data['OpenPorchSF'] + train_data['EnclosedPorch'] + train_data['3SsnPorch'] + train_data['ScreenPorch']
train_data['TotalBsmtFinSF'] = train_data['BsmtFinSF1'] + train_data['BsmtFinSF2']

# booleans - does the presence/absence of the option (e.g. porch, basement) matter?
# porch - insignificant
# porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
# train_data['porch'] = (train_data[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']] > 0).any(axis=1)
# basement - insignificant
# train_data['basement'] = (train_data[['BsmtFinType1', 'BsmtFinType2']] != 'No').any(axis=1)
# basement = train_data[train_data['basement'] == True]
# deck - insignificant
# train_data['deck'] = (train_data['WoodDeckSF'] > 0)
# pool - insignificant
# train_data['pool'] = (train_data['PoolArea'] > 0)
# miscellaneous - insignificant
# train_data['misc'] = (train_data['MiscVal'] > 0)
# masonry - insignificant
# train_data['masonry'] = (train_data['MasVnrArea'] > 0)

train_data = pd.get_dummies(train_data)
train(homeprice_rf, trainX, trainy)

# importances
# simplify - focus in on effective features (higher importances)
importances = pd.read_csv('importances.csv')
effective_index = importances[importances['importance'] >= 0.0001]
train(homeprice_rf, trainX, trainy)


# test
# homeprice_test = pd.read_csv(r'C:\Users\nbrock.HAYDEN-HOMES\OneDrive - Hayden Homes\coding\PycharmProjects\houseprices\house-prices-advanced-regression-techniques.zip\test.csv')
