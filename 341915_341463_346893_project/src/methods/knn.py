import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification",distance = "euclidian",predict = "average"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.distance = distance
        self.predict_method = predict

    def euclidean_dist(self, one_element, training_data):
        """Compute the Euclidean distance between a single
        vector and all training_data.

        Inputs:
            example: shape (D,)
            training_data: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sqrt(np.sum((training_data - one_element) ** 2, axis=1))
    
    def chi_square_dist(self,one_element,training_data):
        """Compute the Chi-square distance between a single
        vector and all training_data.

        Inputs:
            example: shape (D,)
            training_data: shape (NxD) 
        Outputs:
            chi-square distances: shape (N,)
        """
        epsilon = 1e-6 #to avoid division by zero
        return np.sqrt(np.sum((one_element - training_data)**2 / (one_element + training_data + epsilon), axis=1))

    
    def find_k_nearest_neighbors(self, k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        sorted_indices = np.argsort(distances)
        nn_indices = sorted_indices[:k]

        if self.task_kind == "regression" and self.predict_method == "weighted_average":   #Compute the weights only if we will need them later
            epsilon = 1e-6
            distances[distances == 0] = epsilon #avoid division by zeros, a little epsilon will increase the weights of neighbors with distance == 0     
            self.weights = 1/distances[nn_indices]
        return nn_indices
    
    def predict_label(self, neighbours_labels):
        """
        Predict the label using the k-nearest neighbors in the classification task
        """
        return np.argmax(np.bincount(neighbours_labels))
    

    def predict_average(self, neighbors):
        """
        Predict the value using the average of the k-nearest neighbors in the regression task
        """
        return np.sum(neighbors,axis = 0)/self.k
    
    def predict_weighted_average(self,neigbors):
        """
        Predict the value using the weighted average of the k-nearest neighbors in the regression task
        """
        num_vectors = len(self.weights)
        multiplied_vectors = np.empty_like(neigbors)
        
        for i in range(num_vectors):
            multiplied_vectors[i] = neigbors[i] * self.weights[i]

        return np.sum(multiplied_vectors)/np.sum(self.weights)   #sum of the weights can't be equal to 0 because of the precautions taken before


    def kNN_one_example(self, unlabeled, training_features, training_labels, k):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """     
        # Compute distances
        if self.distance == "euclidian":
            distances = self.euclidean_dist(unlabeled,training_features)
        elif self.distance == "chi-square":
            distances = self.chi_square_dist(unlabeled,training_features)    
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k,distances)

        neighbors = training_labels[nn_indices]

        if self.task_kind == "classification":
            # Pick the most common
            best_label = self.predict_label(neighbors)
            return best_label
        
        elif self.task_kind == "regression":
            if self.predict_method == "average":
                return self.predict_average(neighbors)
            elif self.predict_method == "weighted_average":
                return self.predict_weighted_average(neighbors)

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        self.labels = training_labels
        self.training_data = training_data
        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        return np.apply_along_axis(func1d=self.kNN_one_example, axis=1, arr=test_data, 
                                training_features=self.training_data, 
                               training_labels=self.labels, k=self.k)
        
        """
        test_labels = []

        # For each example in the test data
        for example in test_data:
            # Predict label using kNN for the current example
            pred_label = self.kNN_one_example(example, self.training_data, self.labels, self.k)
            # Append predicted label to the list of test labels
            test_labels.append(pred_label)

        # Convert the list of test labels to numpy array
        test_labels = np.array(test_labels)

        return test_labels
        """