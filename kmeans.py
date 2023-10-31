
'''
File: kmeans.py
Project: Downloads
File Created: Oct 2023
Author: Abhinav Vishnuvajhala
'''

import numpy as np

class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        N = self.points.shape[0]
        rand = np.random.choice(N, self.K, replace=False)
        self.centers = self.points[rand]
        
        return self.centers
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """

        raise NotImplementedError

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """

        raise NotImplementedError

    def update_assignment(self):  # [5 pts]
        self.assignments = np.argmin(pairwise_dist(self.points, self.centers), axis = 1)

        return self.assignments

        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison. 
        """        

        raise NotImplementedError

    def update_centers(self):  # [5 pts]
        # c_points = self.points[self.assignments == self.centers]
        # self.centers = np.mean(c_points, axis = 0)
        for i in range(self.K):
            cluster_mask = self.assignments == i
            center = np.mean(self.points[cluster_mask], axis=0)
            self.centers[i] = center

        return self.centers
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """

        raise NotImplementedError

    def get_loss(self):  # [5 pts]
        self.loss = 0
        for i in range(self.K):
            points = self.points[self.assignments == i]
            self.loss += np.sum(pairwise_dist(points, self.centers[None,i])**2)
            
        return self.loss
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """

        raise NotImplementedError

    def train(self):    # [10 pts]
        x = 1
        loss_prev = 0
        while (x <= self.max_iters):
            self.update_assignment()
            self.update_centers()
            newClusters = np.nonzero(np.isnan(self.centers).any(axis = 1))[0]
            if len(newClusters) > 0:
                newCenters = self.init_centers()
                self.centers[newClusters] = newCenters[newClusters]
                self.update_assignment()
            self.update_centers()
            self.loss = self.get_loss()
            if (np.abs(self.loss - loss_prev)) / (self.loss + 1e-10) < self.rel_tol: 
                break

            loss_prev = self.loss
            x += 1

        return self.centers, self.assignments, self.loss    

        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.   
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """

        raise NotImplementedError


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)

        #x_2 = np.sum(x * x, axis = 1)[:, np.newaxis]
        #y_2 = np.sum(y * y, axis = 1)
        #return np.sqrt(np.maximum(x_2 + y_2 - 2 * (x @ y.T), 0))

        dist = -2 * x @ y.T + np.sum(y**2, axis = 1) + np.sum(x**2, axis = 1)[:, np.newaxis]
        return np.sqrt(np.maximum(dist, 0))
        
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """

        raise NotImplementedError

def rand_statistic(xGroundTruth, xPredicted): # [5 pts]
    fp, fn, tp, tn = 0, 0, 0, 0

    for i in range (len(xGroundTruth)):
        for j in range (i + 1, len(xGroundTruth)):
            if (xGroundTruth[i] == xGroundTruth[j]):
                if (xPredicted[i] == xPredicted[j]): tp += 1
                else: fn += 1

            else:
                if (xPredicted[i] == xPredicted[j]): fp += 1
                else: tn += 1

    rand = (tp + tn)/(tp + tn + fp + fn)
    return rand
    
    """
    Args:
        xPredicted : N x 1 numpy array, N = no. of test samples
        xGroundTruth: N x 1 numpy array, N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float
    """
    
    raise NotImplementedError