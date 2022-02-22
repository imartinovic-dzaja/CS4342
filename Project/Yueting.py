from numpy.core.numeric import cross
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import decomposition
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import pprint
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mpl_toolkits import mplot3d
from sklearn import tree
from enum import auto
import math
import re
from numpy.random.mtrand import RandomState
from scipy.sparse import data
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from sklearn.datasets import make_blobs
from scipy import stats
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.utils import resample
import itertools
import sklearn
from itertools import combinations
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#Project
def project_start1():
    data = pd.read_csv("house.csv")
    data = data.drop('Id',axis=1)
    #print(data.head(1))
    #corrMatrix = data.corr()
    #sns.heatmap(corrMatrix, annot=True)
    #plt.title('Correlation matrix')
    #plt.show()
    #plt.clf()
    y = data['SalePrice']
    
    MSSubClassDummies = pd.get_dummies(data['MSSubClass'], prefix='MSSubClass')
    MSZoningDummies = pd.get_dummies(data['MSZoning'], prefix='MSZoning')
    StreetDummies = pd.get_dummies(data['Street'], prefix='Street')
    AlleyDummies = pd.get_dummies(data['Alley'], prefix='Alley')
    LotShapeDummies = pd.get_dummies(data['LotShape'], prefix='LotShape')
    LandContourDummies = pd.get_dummies(data['LandContour'], prefix='LandContour')
    UtilitiesDummies =  pd.get_dummies(data['Utilities'], prefix='Utilities')
    LotConfigDummies  = pd.get_dummies(data['LotConfig'], prefix='LotConfig')
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
    
    numericalPredictors = data[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'ExterQual', 'ExterCond', 'KitchenQual', 'HeatingQC', 'BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageQual', 'GarageCond', 'SalePrice']]
    X = pd.concat([numericalPredictors, MSSubClassDummies, MSZoningDummies, StreetDummies, AlleyDummies, LotShapeDummies, LandContourDummies, UtilitiesDummies, LotConfigDummies, LandSlopeDummies, BldgTypeDummies,Condition1Dummies, Condition2Dummies, FunctionalDummies, GarageTypeDummies, HouseStyleDummies, RoofStyleDummies, RoofMatlDummies, Exterior1stDummies, Exterior2ndDummies,MasVnrTypeDummies, BsmtExposureDummies, FoundationDummies, RoofMatlDummies,  HeatingDummies, CentralAirDummies, ElectricalDummies, GarageFinishDummies, PavedDriveDummies, FenceDummies, MiscFeatureDummies, SaleTypeDummies, SaleConditionDummies], axis=1).dropna()
    
    y = X['SalePrice']
    X = X.drop(['SalePrice'], axis=1)

    return X, y

def project_start():
    house = pd.read_csv('house.csv')
    
    #Correlation matrix
    corrMatrix = house.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.title('Correlation matrix')
    plt.show()
    plt.clf()

    #NaN data filtering
    house[house.columns[house.isna().sum() > 0]].isna().sum().sort_values().plot.bar()
    plt.title('Null values count')
    plt.ylabel('counts')
    plt.xlabel('Column')
    #plt.show()
    
    #Drop all columns with more than 1000 NaNs
    #house = house.drop('Fence',axis=1)
    #house = house.drop('Alley',axis=1)
    #house = house.drop('MiscFeature',axis=1)
    #house = house.drop('PoolQC',axis=1)

    #Convert all categorical variables to dummy variables
    #house = pd.get_dummies(house)

    house['MSZoning'].fillna("RL", inplace = True)
    house.Utilities.fillna('AllPub',inplace = True)
    house.Exterior1st.fillna("VinylSd", inplace = True)
    house.Exterior2nd.fillna("VinylSd", inplace = True)
    house.MasVnrArea.fillna(0., inplace=True)
    house.BsmtCond.fillna("No", inplace=True)
    house.BsmtExposure.fillna("NB", inplace=True)
    house.BsmtFinType1.fillna("NB", inplace=True)
    house.BsmtFinType2.fillna("NB", inplace=True)
    house.BsmtFinSF1.fillna(0., inplace=True)
    house.BsmtFinSF2.fillna(0., inplace=True)
    house.BsmtUnfSF.fillna(0., inplace=True)
    house.TotalBsmtSF.fillna(0., inplace=True)
    house.Electrical.fillna("SBrkr", inplace = True)
    house.BsmtFullBath.fillna(0., inplace=True)
    house.BsmtHalfBath.fillna(0., inplace=True)
    house.KitchenQual.fillna("TA", inplace = True)
    house.Functional.fillna('Typ', inplace = True)
    house.FireplaceQu.fillna("No", inplace = True)
    house.GarageType.fillna("No", inplace = True)
    house.GarageYrBlt.fillna(0, inplace = True)
    house.GarageFinish.fillna("No", inplace = True)
    house.GarageCars.fillna(0, inplace = True)
    house.GarageArea.fillna(0, inplace = True)
    house.GarageQual.fillna("No", inplace = True)
    house.GarageCond.fillna("No", inplace = True)
    house.SaleType.fillna("Con", inplace = True)
    house.SaleCondition.fillna("Normal", inplace = True)

    house['LotFrontage'] = house.groupby(['Neighborhood', 'Street'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    house.BsmtFullBath.replace(3.0, 2.0, inplace=True)
    house.BsmtFullBath = house.BsmtFullBath.astype('int')
    house.BsmtHalfBath = house.BsmtHalfBath.astype('int')
    house.KitchenAbvGr = pd.cut(house.KitchenAbvGr,2)
    house.KitchenAbvGr = house.KitchenAbvGr.astype('category').cat.rename_categories([0, 1])
    house.TotRmsAbvGrd = house.TotRmsAbvGrd.apply(lambda row: 4 if row < 5 else 10)
    house.Fireplaces = house.Fireplaces.apply(lambda row: 2 if row >= 2 else row)
    house.Fireplaces = house.Fireplaces.astype('int')
    house['GarageAgeCat'] = house.GarageYrBlt.apply(lambda row: 'recent' if row >= 2000 else 'old')
    house.GarageCars = house.GarageCars.astype('int')
    house['LotArea_log'] = np.log(house['LotArea'])

    house = pd.get_dummies(house)
    return house

def Linear_project():
    #house = project_start()
    X,Y = project_start1()
    XY = X
    XY.sort_values(by = ['OverallQual'])
    #Train Test split
    #cols = sorted(house)
    #cols.remove('SalePrice')
    #X = house[cols]
    #Y = house['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,shuffle=True)



    #Linear regression
    model = LinearRegression()
    model.fit(X_train,y_train)
    neg_mean_absolute_error = cross_val_score(model, X, Y, cv=5,scoring='neg_mean_absolute_error')
    max_error = cross_val_score(model, X, Y, cv=5,scoring='max_error')
    explained_variance = cross_val_score(model, X, Y, cv=5,scoring='explained_variance')
    r2 = cross_val_score(model, X, Y, cv=5,scoring='r2')
    print('Negative mean absolute error:',np.mean(neg_mean_absolute_error))
    print('max error:',np.mean(max_error))
    print('explained variance:',np.mean(explained_variance))
    print('R2:',np.mean(r2))

    #plot
    prediction = model.predict(XY)
    plt.scatter(X['OverallQual'],Y,c = 'b',marker='.')
    plt.scatter(XY['OverallQual'],prediction,c = 'r',marker='x')
    plt.xlabel("Overall Quality")
    plt.ylabel('Price')
    plt.title('Linear regression')
    plt.show()

def Lasso_project():
    X,Y = project_start1()
    R2 = []
    Highest = 0
    Alpha = -1
    alpL = np.arange(300,600,step=2)
    for alp in alpL:
        Lass = linear_model.Lasso(alpha=alp).fit(X,Y)
        U = np.mean(cross_val_score(Lass,X,Y,cv = 5,scoring='r2'))
        print(U)
        if U > Highest:
            Highest = U
            Alpha = alp
        R2.append(U)
    print(R2)
    plt.plot(alpL,R2)
    plt.xlabel('Alpha')
    plt.ylabel('R2')
    plt.show()
    print('Best alpha is', Alpha)
    print('Highest R2 with this alpha is', Highest)

def Ridge_project():
    X,Y = project_start1()
    Las = []
    Highest = 0
    Alpha = -1
    alpL = np.arange(1,50,step=1)
    for alp in alpL:
        Lass = linear_model.Ridge(alpha=alp).fit(X,Y)
        U = np.mean(cross_val_score(Lass,X,Y,cv = 5,scoring='r2'))
        if U > Highest:
            Highest = U
            Alpha = alp
        Las.append(U)
    plt.plot(alpL,Las)
    plt.xlabel('Alpha')
    plt.ylabel('R2')
    plt.show()
    print('Best alpha is', Alpha)
    print('Highest R2 with this alpha is', Highest)

def Ploy_feature():
    X,Y = project_start1()
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,shuffle=True)
    degrees = [1,2,3]
    y_train_pred = np.zeros((len(X_train), len(degrees)))
    y_test_pred = np.zeros((len(X_test), len(degrees)))
    for i in [0,1,2]:
        model = make_pipeline(PolynomialFeatures(i+1), LinearRegression())
        model.fit(X_train, y_train)
        y_train_pred[:, i] = model.predict(X_train)
        y_test_pred[:, i] = model.predict(X_test)
        print(y_train_pred)
        # make pipeline: create features, then feed them to linear_reg model
        # predict on test and train data
        # store the predictions of each degree in the corresponding column

    # compare r2 for train and test sets (for all polynomial fits)
    print("MSE values: \n")
    testerror = []
    trainerror = []
    for i in [0,1,2]:
        train_r2 = round(sklearn.metrics.r2_score(y_train, y_train_pred[:, i]), 2)
        test_r2 = round(sklearn.metrics.r2_score(y_test, y_test_pred[:, i]), 2)
        print('Tr R2:',train_r2,'Te R2:',test_r2)
        trainerror.append(train_r2)
        testerror.append(test_r2)
        print("Polynomial degree {0}: train score={1}, test score={2}".format(i+1, 
                                                                            train_r2, 
                                                                            test_r2))
    plt.figure(figsize=(16, 8))
    plt.plot(degrees, testerror,c = 'red')
    plt.ylabel('R2')
    plt.xlabel('Degrees')
    plt.show()                                                                     
    plt.plot(degrees, testerror,c = 'blue')
    plt.ylabel('R2')
    plt.xlabel('Degrees')
    plt.show()

def PCA_project():
    house = project_start()

    #Train Test split
    cols = sorted(house)
    cols.remove('SalePrice')
    X = house[cols]
    Y = house['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,shuffle=True)

    #PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    X_train_PC, X_test_PC, y_train_PC, y_test_PC = train_test_split(X,Y, test_size=0.2,shuffle=True)
    model = LinearRegression()
    model.fit(X_train_PC,y_train_PC)
    scores = cross_val_score(model,X_train_PC,y_train_PC,cv=5,scoring='r2')
    print('Mean R2 scores when trying to fit PC1 and PC2:', np.mean(scores))

    #Plotting PC1 and PC2
    pca = PCA()
    PCA_fitted = pca.fit(X)
    plt.clf()
    plt.plot(np.cumsum(PCA_fitted.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    plt.clf()
    plt.scatter(principalDf['PC1'],principalDf['PC2'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    plt.clf()

if __name__ == '__main__':
    pd.set_option('display.max.columns', None)
    Linear_project()

