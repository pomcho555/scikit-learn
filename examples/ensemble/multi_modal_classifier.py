# ensamble learning

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# kears  package import
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

# local import
from cnn import cnn_model

def make_test_ensemble(X, y):
    """handle multi-modal input like image and text
    Author pomcho555
    
    Prameters
    ---------
            X: tupled data like (image_vector, text_vector)
            y: well knwon array label, [0, 0, 1]
            
    Returns
    -------
    None
    """

    # load each child-classifier

    # Naive Baise Classifier
    clf1 = GaussianNB()

    # CNN
    clf2 = KerasClassifier(build_fn=cnn_model, verbose=0)

    # RF
    clf3 = RandomForestClassifier(n_estimators=2, random_state=1)

    eclf =  MultiModalClassifier(estimators=[('nb', clf1), ('cnn', clf2), ('rf', clf3)], voting='hard')
    
    eclf.fit(X, y)
    


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    """Private function used to fit an estimator within a job."""

    X = _multi_modal_valdation(estimator, X)

    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator

def _multi_modal_valdation(clf, X):
    if len(X[0]) == 2:
        if clf.__class__.__name__ == 'KerasClassifier':
            X,  _ = unzip(X)
        else:
            _, X = unzip(X)
        X = np.stack(X)
    return X


from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import Bunch

# local import
from data_helper import unzip

class MultiModalClassifier(VotingClassifier):

        def fit(self, X, y, sample_weight=None):
            """ Fit the estimators.
            Parameters
            ----------
            X : {array-like, sparse matrix}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y : array-like, shape = [n_samples]
                Target values.
            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted.
                Note that this is supported only if all underlying estimators
                support sample weights.
            Returns
            -------
            self : object
            """
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError('Multilabel and multi-output'
                                          ' classification is not supported.')

            if self.voting not in ('soft', 'hard'):
                raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                                 % self.voting)

            if self.estimators is None or len(self.estimators) == 0:
                raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                     ' should be a list of (string, estimator)'
                                     ' tuples')

            if (self.weights is not None and
                    len(self.weights) != len(self.estimators)):
                raise ValueError('Number of classifiers and weights must be equal'
                                 '; got %d weights, %d estimators'
                                 % (len(self.weights), len(self.estimators)))

            if sample_weight is not None:
                for name, step in self.estimators:
                    if not has_fit_parameter(step, 'sample_weight'):
                        raise ValueError('Underlying estimator \'%s\' does not'
                                         ' support sample weights.' % name)
            names, clfs = zip(*self.estimators)
            self._validate_names(names)

            n_isnone = np.sum([clf is None for _, clf in self.estimators])
            if n_isnone == len(self.estimators):
                raise ValueError('All estimators are None. At least one is '
                                 'required to be a classifier!')

            self.le_ = LabelEncoder().fit(y)
            self.classes_ = self.le_.classes_
            self.estimators_ = []

            transformed_y = self.le_.transform(y)

            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                                                     sample_weight=sample_weight)
                    for clf in clfs if clf is not None)

            self.named_estimators_ = Bunch(**dict())
            for k, e in zip(self.estimators, self.estimators_):
                self.named_estimators_[k[0]] = e
            return self

        def _collect_probas(self, X):
            """Collect results from clf.predict calls. """
            predicts = []
            for clf in self.estimators_:
                clf_X = _multi_modal_valdation(clf, X)
                tmp_pre = clf.predict_proba(clf_X)
#                 if clf.__class__.__name__ == 'KerasClassifier':
#                     tmp_pre = np.reshape(tmp_pre, -1)
                predicts.append(tmp_pre)
                del clf_X


            return np.asarray(predicts).T

            #return np.asarray([clf.predict_proba(X) for clf in self.estimators_])


        def _predict(self, X):
            """Collect results from clf.predict calls. """
            predicts = []
            for clf in self.estimators_:
                clf_X = _multi_modal_valdation(clf, X)
            #                 print(clf.predict(clf_X))
                tmp_pre = clf.predict(clf_X)
                if clf.__class__.__name__ == 'KerasClassifier':
                    tmp_pre = np.reshape(tmp_pre, -1)
                predicts.append(tmp_pre)
                del clf_X

            return np.asarray(predicts).T

            return np.asarray([clf.predict(_multi_modal_valdation(clf, X)) for clf in self.estimators_]).T
