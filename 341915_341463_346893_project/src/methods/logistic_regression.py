import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind ="classification" ):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
    

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!

        #initialisation
        labels_onehot = label_to_onehot(training_labels)

        D = training_data.shape[1]  # number of features
        C = get_n_classes(training_labels)  # number of classes
        self.weights = np.random.normal(0, 0.1, (D, C))
        
        #gradient descent
        for i in range(self.max_iters):
            #find the gradient and do a gradient step
            self.gradient = training_data.T@(self.softmax(training_data@self.weights) - labels_onehot)  #cross entropy calculation
            self.weights = self.weights - self.lr*self.gradient


        
        ###
        ##
        return self.predict(training_data)

    def softmax(self,data):
        #version 1 de softmax
        up = np.exp(data)
        prob_matrix = up/np.sum(up, axis = 1, keepdims = True)
        """
        #version 2 de softmax
        prob_matrix = np.empty((len(data), self.weights.shape[1]))  # Shape: (N, C)
        for i in range(len(data)):
            # Calculate scores for each class for the ith data sample
            scores = np.exp(data[i] @ self.weights)  # Shape: (C,)
            
            # Calculate softmax probabilities
            probabilities = scores / np.sum(scores)
            
            # Store probabilities for the ith data sample
            prob_matrix[i] = probabilities
        """
        

        return prob_matrix

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        probabilities = self.softmax(test_data @ self.weights)
        pred_labels = onehot_to_label(probabilities)
        ###
        ##
        return pred_labels
