import numpy as np
from sklearn.neighbors import KDTree

class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        self.tree = KDTree(self.train_X, metric='manhattan')


    def predict(self, X, use_kdtree, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use
        use_kdtree, bool value, shows if we want to use KDtree structure
        
        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        if use_kdtree:
            return self.predict_labels_with_kdtree(X)
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distance_matrix = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i, x in enumerate(X):
            for j, train_x in enumerate(self.train_X):
                distance_matrix[i][j] = np.sum(np.abs(x-train_x))
        return distance_matrix

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distance_matrix = np.array([np.sum(np.abs(x-self.train_X), axis=1) for x in X])
      
        return distance_matrix


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        distance_matrix = np.sum(np.abs(X[:, np.newaxis]-self.train_X), axis=2)
        
        return distance_matrix


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            k_neighbors = self.train_y[np.argpartition(distances[i, :], kth=self.k)[:self.k]]
            predictions[i] += np.bincount(k_neighbors).argmax()
        return predictions


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        predictions = np.zeros(n_test)
        for i in range(n_test):
            k_neighbors = self.train_y[np.argpartition(distances[i, :], kth=self.k)[:self.k]]
            predictions[i] += np.bincount(k_neighbors).argmax()
        return predictions
      
        
    def predict_labels_with_kdtree(self, X):
        """
        Returns model predictions for classification
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """
        
        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dist, ind = self.tree.query(X[i].reshape(1, -1), k=self.k) 
            k_neighbors = self.train_y[ind[0]]
            predictions[i] += np.bincount(k_neighbors).argmax()
        return predictions
        