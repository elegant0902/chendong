import pandas as pd
import numpy as np 

"""Step 1: 检视源数据集"""

train_csv=pd.read_csv('./train.csv',index_col=0)
test_csv=pd.read_csv('./test.csv',index_col=0)
price=pd.DataFrame({'price':train_csv['SalePrice'],'log(price+1)':np.log1p(train_csv['SalePrice'])})
y_train=np.log1p(train_csv.pop('SalePrice'))
#print(y_train)

"""Step 2: 合并数据"""

all_data=pd.concat((train_csv,test_csv),axis=0)
#print(all_data.shape)  #all_df的行列数

"""Step 3: 变量转化"""

all_data['MSSubClass']=all_data['MSSubClass'].astype(str)#数据类型转化 int64 -> string
all_dummy_data=pd.get_dummies(all_data)#所有的category数据进行编码
#print(all_dummy_data.head())#编码的前五行
mean_cols=all_dummy_data.mean()#缺失数据的平均值
#print(mean_cols.head(10))
all_dummy_data=all_dummy_data.fillna(mean_cols)#将平均值补全缺失数据
#print(all_dummy_data.isnull().sum().sum())#检查是否还存在缺失数据
numeric_cols=all_data.columns[all_data.dtypes!='object']#挑选出不是numerical类型的数据
#print(numeric_cols)

"""计算标准分布：(X-X')/s 对数据进行平滑处理"""
numeric_col_means = all_dummy_data.loc[:,numeric_cols].mean()#均值
numeric_col_std = all_dummy_data.loc[:,numeric_cols].std()#方差
all_dummy_data.loc[:,numeric_cols] = (all_dummy_data.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std#标准化
#print(all_dummy_data.loc[:,numeric_cols])

"""step4:建立模型"""

"""把数据集分回训练集和测试集"""

dummy_train_data=all_dummy_data.loc[train_csv.index]
dummy_test_data=all_dummy_data.loc[test_csv.index]
print(dummy_train_data.shape,dummy_test_data.shape)#检验训练集和测试集

"""岭回归Ridge Regression与k折交叉验证法cross_val_score"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
"""将DF转化成Numpy Array"""
X_train = dummy_train_data.values
X_test = dummy_test_data.values
#print(X_train.shape,X_test.shape)

"""Ridge Regression验证参数"""
alphas = np.logspace(-3, 2, 50)
#print(alphas)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

"""Random Forest"""
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feat in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
"""验证参数值的选取"""
#import matplotlib.pyplot as plt
#plt.plot(max_features, test_scores)
#plt.title("Max Features vs CV Error")
#plt.show()

"""Step 5:Ensemble"""
"""根据调整好的参数对数据进行预测"""
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)

y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
y_final = (y_ridge + y_rf) / 2

"""Step 6:提交结果"""
submission_df = pd.DataFrame({'Id' : test_csv.index, 'SalePrice': y_final})
print(submission_df.head(10))