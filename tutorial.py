from time import gmtime

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split


digits = datasets.load_digits()

print digits.images[0]

# Our digits are stored in a 2 dimensional array lets flatten each before we can train the model
# Get number of samples
samples = digits.images.shape[0]
# Reshape array
X = digits.images.reshape([samples,-1])

# Split set into a training set and a test set
X_train, X_test, y_train, y_test  = train_test_split(X, digits.target, test_size=0.2, random_state=gmtime().tm_sec)

# Initialize a SVM with gamme=0.001
clf = svm.SVC(gamma=0.001)

# Fit with training data
clf.fit(X_train, y_train)

# Score our fit using the test data
classification_score = clf.score(X_test,y_test)
print "Prediction score for our SVM", (classification_score*100).round(2), "%"

# Do a prediction for one sample
print clf.predict([X_test[0]])
# Reshape back to 2 dimmensions and print
print X_test[0].reshape(8,8)
