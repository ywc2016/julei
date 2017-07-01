# -*- coding: utf-8 -*-

import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 分出变量和标签
dataset_indicator = pd.read_excel(u'file/indicators_of_customer_value_and_loss.xlsx',
                                  sheetname='Sheet1')
dataset_basic = pd.read_excel(u'file/member_basic_info.xlsx',
                              sheetname='Sheet1')

# dataset = pd.merge(dataset_basic,dataset_indicator,on='Member')
# dataset.to_excel(u'file/merge.xlsx', sheet_name='Sheet1')

dataset = pd.read_excel(u'file/merge.xlsx',
                        sheetname='Sheet2')

Y = dataset[['label']]
X = dataset.drop(['label'], axis=1)

# 将数据分为训练集和测试集，测试集用来预测，训练集用来学习模型
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# xgboost 有封装好的分类器和回归器，可以直接用 XGBClassifier 建立模型
model = XGBClassifier()
model.fit(X_train, y_train)

# xgboost 的结果是每个样本属于第一类的概率，需要用 round 将其转换为 0 1 值

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# 得到 Accuracy

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# xgboost可以在模型训练时，评价模型在测试集上的表现，也可以输出每一步的分数，只需要将
# model = XGBClassifier()
# model.fit(X_train, y_train)
# 变为：
# 那么它会在每加入一颗树后打印出 logloss,并打印出 Early Stopping 的点：

# model = XGBClassifier()
# eval_set = [(X_test, y_test)]
# model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

# gradient boosting还有一个优点是可以给出训练好的模型的特征重要性，
# 这样就可以知道哪些变量需要被保留，哪些可以舍弃。
#
# 需要引入下面两个类：

from xgboost import plot_importance
from matplotlib import pyplot

# 和前面的代码相比，就是在 fit 后面加入两行画出特征的重要性

model.fit(X, Y)

plot_importance(model)
pyplot.show()
