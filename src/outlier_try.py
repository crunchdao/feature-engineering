
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

from pyod.models.vae import VAE
import pdb


contamination = 0.1  # percentage of outliers
n_train = 20000  # number of training points
n_test = 2000  # number of testing points
n_features = 300  # number of features

# Generate sample data
X_train, X_test, y_train, y_test = \
    generate_data(n_train=n_train,
                    n_test=n_test,
                    n_features=n_features,
                    contamination=contamination,
                    random_state=42)

#x_train: can be like our f_matrix
# y_train: array([0., 0., 0., ..., 1., 1., 1.])

#X_train.shape--> (20000, 300), y_train.shape --> (20000,)
#X_test.shape--> (2000, 300), y_test.shape --> (2000,)

# train VAE detector (Beta-VAE)
clf_name = 'VAE'
clf = VAE(epochs=30, contamination=contamination, gamma=0.8, capacity=0.2)
clf.fit(X_train)


# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers), (20000,)
y_train_scores = clf.decision_scores_  # raw outlier scores, (20000,)

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores
pdb.set_trace()
# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)