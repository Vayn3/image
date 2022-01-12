import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import GridSearchCV

"""
随机森林进行乘客生存预测
"""
# 获取数据
titan = pd.read_csv("数据集/泰坦尼克号/train.csv")

# 数据的处理
x = titan[['Pclass', 'Age', 'Sex']]
y = titan['Survived']

# 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
x['Age'].fillna(x['Age'].mean(), inplace=True)

# 对于x转换成字典数据x.to_dict(orient="records")
# 字典特征抽取
dict = DictVectorizer(sparse=False)

x = dict.fit_transform(x.to_dict(orient="records"))

print(dict.get_feature_names())
print(x)


# 分割训练集合测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

estimator = RandomForestClassifier()

# 加入网格搜索与交叉验证
# 参数准备
param_dict = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5, 8, 15, 25, 30]}
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
estimator.fit(x_train, y_train)

# 模型评估
# 方法1：直接比对真实值和预测值
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接比对真实值和预测值:\n", y_test == y_predict)

# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)

# 最佳参数：best_params_
print("最佳参数：\n", estimator.best_params_)
# 最佳结果：best_score_
print("最佳结果：\n", estimator.best_score_)
# 最佳估计器：best_estimator_
print("最佳估计器:\n", estimator.best_estimator_)
# 交叉验证结果：cv_results_
print("交叉验证结果:\n", estimator.cv_results_)