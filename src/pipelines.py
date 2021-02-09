import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
import keras


class EmbedCalibrate:
    """
    A two step transformer and classifier learner

    Parameters
    ----------

    validation_frac : float in (0, 1)
        Fraction of samples to use for honest posteriors or postprocessing


    Attributes
    ----------
    network_ : object
        Keras model. Copied before fitting from self.network

    transformer_ : object
        Keras model, subset of self.network_, to transformer inputs
    """

    def __init__(
        self,
        network,
        compile_kwargs={},
        fit_kwargs={},
        transform_layer_idx=-2,
        classifier=None,
        validation_frac=None,
    ):
        self.network = keras.models.clone_model(network)
        self.compile_kwargs = compile_kwargs
        self.fit_kwargs = fit_kwargs
        self.transform_layer_idx = transform_layer_idx
        self.classifier = classifier
        self.validation_frac = validation_frac

    def fit(self, X, y):
        """
        Fits the transformer and classifier

        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response data matrix).

        Returns
        -------
        self
            The object itself.
        """
        check_X_y(X, y)

        self.network_ = keras.models.clone_model(self.network)
        self.transformer_ = keras.models.Model(
            inputs=self.network_.inputs,
            outputs=self.network_.layers[self.transform_layer_idx].output,
        )

        _, y = np.unique(y, return_inverse=True)

        # Create validation (i.e. honest) split to fit/tune classifiers
        if self.validation_frac:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_frac)

            self._fit_transformer(X_train, y_train)
            if self.classifier is not None:
                self._fit_classifier(self.transformer_.predict(X_val), y_val)
        # Regular (softmax) output
        else:
            self._fit_transformer(X, y)
            if self.classifier is not None:
                self._fit_classifier(self.transformer_.predict(X), y)

        return self

    def _fit_transformer(self, X, y):
        self.network_.compile(**self.compile_kwargs)

        self.network_.fit(
            X, keras.utils.to_categorical(y), **self.fit_kwargs)

    def _fit_classifier(self, X, y):
        assert self.classifier is not None
        self.classifier = self.classifier.fit(X, y)

    def predict(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            the transformed input data

        Returns
        -------
        y_hat : ndarray of shape (n_samples,)
            predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """
        Returns the posterior probabilities of each class for data X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            the transformed input data

        Returns
        -------
        y_proba_hat : ndarray of shape (n_samples, n_classes)
            Predicted class posteriors
        """
        check_is_fitted(self)
        check_array(X)
        if self.classifier is None:
            proba = self.network_.predict(X)
            return proba
        else:
            transformed = self.transformer_.predict(X)
            return self.classifier.predict_proba(transformed)
