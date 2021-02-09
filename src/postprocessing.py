# Code taken from https://github.com/markus93/NN_calibration
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters
    ----------
        x : numpy.ndarray
            array containing m samples with n-dimensions (m, n)
    Returns
    -------
        x_softmax : numpy.ndarray
            softmaxed values for initial (m, n) array
    """
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=1)


class TemperatureScaling:
    """
    Tunes a scaling parameter on the logits to better calibrate output.
    Note, predicted classes remain unchanged under temperature scaling.

    Parameters
    ----------
        temp : float (default 1)
            starting temperature
        maxiter : int
            maximum iterations done by optimizer
    """
    def __init__(self, maxiter=50, solver="BFGS", temp=None):
        self.maxiter = maxiter
        self.solver = solver
        self.temp = temp

    def _loss_fun(self, temp, logits, y_true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self._predict_proba(logits, temp)
        loss = log_loss(y_true=y_true, y_pred=scaled_probs)
        return loss

    def fit(self, logits, y_true):
        """
        Trains the model and finds optimal temperature

        Parameters
        ----------
            logits : np.ndarray, shape (samples, classes)
                the output from neural network for each class

            y_true : tensor
                one-hot-encoding of true labels.

        Returns
        -------
        self : object
            The instance of self
        """
        if self.temp:
            self.temp_ = None
            return self

        y_true = y_true.flatten()  # Flatten y_val
        opt = minimize(self._loss_fun, x0=1, args=(logits, y_true), options={
                       'maxiter': self.maxiter}, method=self.solver)
        self.temp_ = opt.x[0]
        self.opt_ = opt

        return self

    def _predict_proba(self, logits, temp):
        """
        Internal prob prediction for optimization
        """
        return softmax(logits / temp)

    def predict_proba(self, logits):
        """
        Scales logits based on the temperature and returns calibrated
        probabilities

        Parameters
        ----------
            logits : numpy.ndarray, shape (samples, classes)
                logits values of data for each class

            temp: float, optional
                used for optimizing during fit

        Returns
        -------
            probs : numpy.ndarray, shape (samples, classes)
                calibrated probabilities
        """

        if self.temp:
            return self._predict_proba(logits, self.temp)
        else:
            return self._predict_proba(logits, self.temp_)
