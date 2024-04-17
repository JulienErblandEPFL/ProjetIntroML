import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind ="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):  
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!

        if self.lmda == 0:
            self.weight = np.linalg.pinv(training_data)@training_labels # without ridge (with pinv for optimisation)
        else:
            regulator = self.lmda*np.identity(training_data.shape[1])   #for ridge
            self.weight = np.linalg.inv(training_data.T@training_data + regulator)@training_data.T@training_labels
        
        ###
        ##
        
        return self.predict(training_data)


    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        pred_regression_targets = test_data@self.weight
        ###
        ##

        return pred_regression_targets
