from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


texts = ["dog cat fish", "dog cat cat", "fish bird", "bird"]
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)

print(cv.get_feature_names())
print("cv_fit is like: ")
print(cv_fit)
print("cv_fit.toarray() is like ; ")
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))
y_train = [1, 1, 1, 0]

classifier = MultinomialNB()
classifier.fit(cv.transform(texts), y_train)


x_test = ["dog cat ", "dog cat", "fish dog elephant", "bird elephant"]
y_test = [1, 1, 1, 0]
result = classifier.score(cv.transform(x_test), y_test)
print(result)

