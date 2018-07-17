import numpy as np
import  pandas as pd
"""Step 1: 检视源数据集"""
train_df = pd.read_csv('./train.csv',index_col=0)#将训练集读入数据
test_df = pd.read_csv('./test.csv',index_col=0)#将测试集读入数据

#prices = pd.DataFrame({"price":train_df["SalePrice"],"log(price + 1)":np.log1p(train_df["SalePrice"])})
#将prices平滑化处理
y_train = np.log1p(train_df.pop('SalePrice'))#将训练集saleprice作为标签，并且进行平滑处理
#print(y_train.head())#y_train的前五行
"""Step 2: 合并数据"""
all_df = pd.concat((train_df,test_df),axis=0)#训练集与测试集合起来作为整个集合
#print(all_df.shape)  #all_df的行列数
"""Step 3: 变量转化"""
#print(all_df['MSSubClass'].dtypes) #MSSubClass的数据类型dtype(int64)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)#数据类型转化 int64 -> string
#print(all_df['MSSubClass'].value_counts())#MSSubClass下的数据类别进行统计

#print(pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass').head())
#对MSSubClass下的category进行One-Hot编码

all_dummy_df = pd.get_dummies(all_df)#所有的category数据进行编码
#print(all_dummy_df.head())#编码的前五行

"""处理numerical变量，将缺失数据用平均值补全"""
#print(all_dummy_df.isnull().sum().sort_values(ascending=False).head(10))
#将缺失数据按照类别进行求和降序排序
mean_cols = all_dummy_df.mean()#缺失数据的平均值
#print(mean_cols.head(10))
all_dummy_df = all_dummy_df.fillna(mean_cols)#将平均值补全缺失数据
#print(all_dummy_df.isnull().sum().sum())#检查是否还存在缺失数据

"""标准化numerical数据"""
numeric_cols = all_df.columns[all_df.dtypes != 'object']#挑选出不是numerical类型的数据
#print(numeric_cols)#将numerical数据打印出来

"""计算标准分布：(X-X')/s 对数据进行平滑处理"""
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()#均值
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()#方差
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std#标准化

"""step4:建立模型"""

"""把数据集分回训练集和测试集"""

dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
#print(dummy_train_df.shape,dummy_test_df.shape)#检验训练集和测试集

"""岭回归Ridge Regression与k折交叉验证法cross_val_score"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
"""将DF转化成Numpy Array"""
X_train = dummy_train_df.values
X_test = dummy_test_df.values
"""Ridge Regression验证参数"""
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

"""验证参数值的选取"""
#import matplotlib.pyplot as plt
#plt.plot(alphas, test_scores)
#plt.title("Alpha vs CV Error")
#plt.show()

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
submission_df = pd.DataFrame(data= {'Id' : test_df.index, 'SalePrice': y_final})
print(submission_df.head(10))