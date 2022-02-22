# utf-8
import pandas as pd
import subprocess

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def project_start1():
    data = pd.read_csv("house.csv")
    data = data.drop('Id', axis=1)
    # print(data.head(1))
    # corrMatrix = data.corr()
    # sns.heatmap(corrMatrix, annot=True)
    # plt.title('Correlation matrix')
    # plt.show()
    # plt.clf()
    y = data['SalePrice']

    MSSubClassDummies = pd.get_dummies(data['MSSubClass'], prefix='MSSubClass')
    MSZoningDummies = pd.get_dummies(data['MSZoning'], prefix='MSZoning')
    StreetDummies = pd.get_dummies(data['Street'], prefix='Street')
    AlleyDummies = pd.get_dummies(data['Alley'], prefix='Alley')
    LotShapeDummies = pd.get_dummies(data['LotShape'], prefix='LotShape')
    LandContourDummies = pd.get_dummies(data['LandContour'], prefix='LandContour')
    UtilitiesDummies = pd.get_dummies(data['Utilities'], prefix='Utilities')
    LotConfigDummies = pd.get_dummies(data['LotConfig'], prefix='LotConfig')
    LandSlopeDummies = pd.get_dummies(data['LandSlope'], prefix='LandSlope')
    Condition1Dummies = pd.get_dummies(data['Condition1'], prefix='Condition1')
    Condition2Dummies = pd.get_dummies(data['Condition2'], prefix='Condition2')
    BldgTypeDummies = pd.get_dummies(data['BldgType'], prefix='BldgType')
    HouseStyleDummies = pd.get_dummies(data['HouseStyle'], prefix='HouseStyle')
    RoofStyleDummies = pd.get_dummies(data['RoofStyle'], prefix='RoofStyle')
    RoofMatlDummies = pd.get_dummies(data['RoofMatl'], prefix='RoofMatl')
    Exterior1stDummies = pd.get_dummies(data['Exterior1st'], prefix='Exterior1st')
    Exterior2ndDummies = pd.get_dummies(data['Exterior2nd'], prefix='Exterior2nd')
    MasVnrTypeDummies = pd.get_dummies(data['MasVnrType'], prefix='MasVnrType')
    FoundationDummies = pd.get_dummies(data['Foundation'], prefix='Foundation')
    BsmtExposureDummies = pd.get_dummies(data['BsmtExposure'], prefix='BsmtExposure')
    RoofMatlDummies = pd.get_dummies(data['RoofMatl'], prefix='RoofMatl')
    HeatingDummies = pd.get_dummies(data['Heating'], prefix='Heating')
    CentralAirDummies = pd.get_dummies(data['CentralAir'], prefix='CentralAir')
    ElectricalDummies = pd.get_dummies(data['Electrical'], prefix='Electrical')
    FunctionalDummies = pd.get_dummies(data['Functional'], prefix='Functional')
    GarageTypeDummies = pd.get_dummies(data['GarageType'], prefix='GarageType')
    GarageFinishDummies = pd.get_dummies(data['GarageFinish'], prefix='GarageFinish')
    PavedDriveDummies = pd.get_dummies(data['PavedDrive'], prefix='PavedDrive')
    FenceDummies = pd.get_dummies(data['Fence'], prefix='Fence')
    MiscFeatureDummies = pd.get_dummies(data['MiscFeature'], prefix='MiscFeature')
    SaleTypeDummies = pd.get_dummies(data['SaleType'], prefix='SaleType')
    SaleConditionDummies = pd.get_dummies(data['SaleCondition'], prefix='SaleCondition')

    mapping1 = {k: v for v, k in enumerate(['Po', 'Fa', 'TA', 'Gd', 'Ex'])}
    data['ExterQual'] = data['ExterQual'].map(mapping1)
    data['ExterCond'] = data['ExterCond'].map(mapping1)
    data['KitchenQual'] = data['KitchenQual'].map(mapping1)
    data['HeatingQC'] = data['HeatingQC'].map(mapping1)

    mapping2 = {k: v for v, k in enumerate(['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'])}
    data['BsmtQual'] = data['BsmtQual'].map(mapping2)
    data['BsmtCond'] = data['BsmtCond'].map(mapping2)
    data['FireplaceQu'] = data['FireplaceQu'].map(mapping2)
    data['GarageQual'] = data['GarageQual'].map(mapping2)
    data['GarageCond'] = data['GarageCond'].map(mapping2)

    mapping3 = {k: v for v, k in enumerate(['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])}
    data['BsmtFinType1'] = data['BsmtFinType1'].map(mapping3)
    data['BsmtFinType2'] = data['BsmtFinType2'].map(mapping3)

    numericalPredictors = data[
        ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
         'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
         'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
         'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
         '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'ExterQual', 'ExterCond', 'KitchenQual',
         'HeatingQC', 'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'SalePrice']]
    X = pd.concat(
        [numericalPredictors, MSSubClassDummies, MSZoningDummies, StreetDummies, AlleyDummies, LotShapeDummies,
         LandContourDummies, UtilitiesDummies, LotConfigDummies, LandSlopeDummies, BldgTypeDummies, Condition1Dummies,
         Condition2Dummies, FunctionalDummies, GarageTypeDummies, HouseStyleDummies, RoofStyleDummies, RoofMatlDummies,
         Exterior1stDummies, Exterior2ndDummies, MasVnrTypeDummies, BsmtExposureDummies, FoundationDummies,
         RoofMatlDummies, HeatingDummies, CentralAirDummies, ElectricalDummies, GarageFinishDummies, PavedDriveDummies,
         FenceDummies, MiscFeatureDummies, SaleTypeDummies, SaleConditionDummies], axis=1).dropna()

    y = X['SalePrice']
    X = X.drop(['SalePrice'], axis=1)

    return X, y


def visualize_tree(tree, feature_names, fn_name):
    with open("{0}.dot".format(fn_name), 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        print("Could not run dot, ie graphviz, to "
              "produce visualization")


def unpruned_tree(X, y):
    dregressor = DecisionTreeRegressor(random_state=0, min_samples_split=20)
    dregressor.fit(X, Y)

    return dregressor


def pruned_tree(X, y):
    parameters = {'max_depth': range(1, 30)}
    r2_scorer = make_scorer(r2_score, greater_is_better=True)
    clf = GridSearchCV(DecisionTreeRegressor(random_state=1), parameters, n_jobs=4, cv=10,
                       scoring=r2_scorer, return_train_score=True)
    clf.fit(X=X, y=Y)
    tree_model = clf.best_estimator_

    return clf, tree_model


def boosted_tree(X, y):
    parameters = {'learning_rate': np.linspace(0.001, 0.5, 20), 'n_estimators': np.arange(1, 40, 2)}
    r2_scorer = make_scorer(r2_score, greater_is_better=True)
    clf = GridSearchCV(GradientBoostingRegressor(random_state=0), parameters, n_jobs=4, cv=10, scoring=r2_scorer, return_train_score=True)
    clf.fit(X=X, y=y)
    tree_model = clf.best_estimator_

    return clf, tree_model


X, Y = project_start1()

unpruned_model = unpruned_tree(X, Y)
visualize_tree(unpruned_tree(X, Y), X.columns.tolist(), "unpruned")

print("Unpruned R^2 Score: {0}".format(np.mean(cross_val_score(unpruned_model, X, Y))))

"---"

clf, pruned_model = pruned_tree(X, Y)

test_R2 = {}
train_R2 = {}

for idx, pm in enumerate(clf.cv_results_['param_max_depth'].data):
    test_R2[pm] = abs(clf.cv_results_['mean_test_score'][idx]) # Taking absolute value as returned value of MSE is negative
    train_R2[pm] = abs(clf.cv_results_['mean_train_score'][idx])


test_R2 = {}
train_R2 = {}
for idx, pm in enumerate(clf.cv_results_['param_max_depth'].data):
    test_R2[pm] = abs(clf.cv_results_['mean_test_score'][idx]) # Taking absolute value as returned value of MSE is negative
    train_R2[pm] = abs(clf.cv_results_['mean_train_score'][idx])

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(121)

lists = sorted(test_R2.items())
x, y = zip(*lists)


plt.plot(x, y, color='b', label='Test Score')
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('Test R^2')
ax.set_title('Test R^2 vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()

ax = fig.add_subplot(122)
lists = sorted(train_R2.items())
x, y = zip(*lists)

plt.plot(x, y, color='r', label='Test Score')
ax.set_xlabel('Depth of the Tree')
ax.set_ylabel('Train R^2')
ax.set_title('Train R^2 vs Depth of the Tree')
ax.set_xlim([1, 29])
ax.grid()

plt.show()

visualize_tree(pruned_model, X.columns.tolist(), "pruned")

print("Pruned R^2 Score: {0}".format(clf.best_score_))

"---"

clf2, boosted_model = boosted_tree(X, Y)

print("Boosted R^2 Score: {0}".format(clf2.best_score_))
print(boosted_model)