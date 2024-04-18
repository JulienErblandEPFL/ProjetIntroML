import numpy as np
import torch

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def euclidean_dist(self, example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        return np.sqrt(np.sum((training_examples - example) ** 2, axis=1))
    
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
        return sorted_indices[:k]
    
    def predict_label(self, neighbours_labels):
        """
        # Convert arrays to tuples to make them hashable
        array_tuples = [tuple(subarray) for subarray in arrays]
        
        # Count occurrences of each array
        counts = {}
        for array_tuple in array_tuples:
            if array_tuple in counts:
                counts[array_tuple] += 1
            else:
                counts[array_tuple] = 1
        
        # Find the array with maximum count
        max_count = 0
        most_common_array = None
        for array_tuple, count in counts.items():
            if count > max_count:
                max_count = count
                most_common_array = array_tuple
        
        # Convert tuple back to list
        most_common_array = list(most_common_array)
        """

        
        return np.argmax(np.bincount(neighbours_labels))
    



    def kNN_one_example(self, unlabeled_example, training_features, training_labels, k):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """
        # WRITE YOUR CODE HERE
        
        # Compute distances
        distances = self.euclidean_dist(unlabeled_example,training_features)
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k,distances)
        
        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]
        # Pick the most common
        best_label = self.predict_label(neighbor_labels)

        return best_label

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