import numpy as np
import utils
import utils.mnist_reader as mnist_reader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from time import time

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

clf = LinearSVC()

print ("Training Initiated")
t0 = time()
clf.fit(X_train[:10000], y_train[:10000])
print "Training Time:", round(time()-t0, 3), "s"

print("")

print ("Making Predictions")
t1 = time()
pred = clf.predict(X_test)
print "Prediction Time:", round(time()-t1, 3), "s"
print("")
# print(pred.shape, y_test.shape) 

# acc=np.sum(pred.reshape(-1).astype(int)==y_test.reshape(-1))
acc = accuracy_score(pred, y_test)
# print("Accuracy ", round(float(acc)/pred.shape[0], 3))
print "Accuracy:", acc, "%"