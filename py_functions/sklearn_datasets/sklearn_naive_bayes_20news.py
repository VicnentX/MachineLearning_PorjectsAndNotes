from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


"""
sklearn.naive_bayes.MultinomialNB(alpha = 1.0)
alpha:拉普拉斯平滑系数

20个希望嫩足数据集包含20个新闻类别
"""

news = fetch_20newsgroups(subset="all")

# 进行数据分割
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

# tf - idf = 词频(TF) * 逆文档频率(IDF)
# 对数据进行特性抽取
tf = TfidfVectorizer()

# 以训练集合中的词的列表进行每篇文章重要性统计「"aa","bb","cc","dd"」\
x_train = tf.fit_transform(x_train)
print(tf.get_feature_names())
x_test = tf.transform(x_test)

# 进行朴素贝叶斯算法预测
mlt = MultinomialNB(alpha=1.0)
print(x_train.toarray())
mlt.fit(x_train, y_train)
y_predict = mlt.predict(x_test)
print("预测的文章类别为 ： ", y_predict)
# 得出准确率
print("预测的准确率为 : ", mlt.score(x_test, y_test))




