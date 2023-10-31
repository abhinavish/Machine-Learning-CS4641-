import numpy as np
from tqdm import tqdm
from kmeans import KMeans


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

FULL_MATRIX = True # Set False if the covariance matrix is a diagonal matrix

class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """

        max_vals = np.max(logit, axis=1, keepdims=True)
        exp_logit = logit - max_vals
        exp_logit = np.exp(exp_logit)
        denominator = np.sum(exp_logit, axis=1, keepdims=True)
        prob = exp_logit / denominator
        return prob

        raise NotImplementedError

    def logsumexp(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """

        max = np.max(logit, axis = 1, keepdims = True)

        exp = np.exp(logit - max)
        exp2 = np.sum(exp, axis = 1, keepdims = True)
        return np.log(exp2) + max

        raise NotImplementedError
    
    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):  # [5pts]

        sigma_i += 1e-32
        cov = np.diag(sigma_i)

        a = np.power(2 * np.pi, mu_i.shape[0] / 2.0)

        b = np.prod(np.sqrt(cov))

        c = 1.0 / (a * b) #first

        e = np.exp(-0.5 * np.sum((points - mu_i) ** 2 / cov, axis = 1))

        return c * e

    
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """

        raise NotImplementedError

    def create_pi(self):
        """
        Initialize the prior probabilities 
        Args:
        Return:
        pi: numpy array of length K, prior

        """ 
        pi = np.full(self.K, 1/self.K)
        return pi

        return NotImplementedError

    def create_mu(self):
        """
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        
        random_integers = np.random.choice(self.N, size=self.K, replace=False)
        
        mu = self.points[random_integers, :]
        return mu                    

        return NotImplementedError
    
    def create_sigma(self):
        """
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the 
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """

        sigma = np.array([np.eye(self.D)+ 1e-32] * self.K)
        return sigma

        return NotImplementedError
    
    def _init_components(self, **kwargs):  # [5pts]

        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case

            Hint: np.random.seed(5) may be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5) #Do Not Remove Seed

        pi = self.create_pi()
        mu = self.create_mu()
        sigma = self.create_sigma()

        return pi, mu, sigma

        raise NotImplementedError

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        ll = np.zeros((self.N, self.K))
        
        if (full_matrix is False):

            for k in range(self.K):
                log_likelihood_k = np.log(pi[k] + 1e-32) + np.log(self.normalPDF(self.points, mu[k], sigma[k]) + 1e-32)
        
                ll[:, k] = log_likelihood_k 

        return ll        


        # === graduate implementation
        #if full_matrix is True:
            #...

        # === undergraduate implementation
        #if full_matrix is False:
            # ...
        raise NotImplementedError

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX, **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation
        if (full_matrix is False):
            gamma = self.softmax(self._ll_joint(pi, mu, sigma, full_matrix = False))

        return gamma


        raise NotImplementedError

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        # === graduate implementation
        #if full_matrix is True:
            # ...

        # === undergraduate implementation

        if full_matrix is False:
            pi = np.zeros(self.K)
            mu = np.zeros((self.K, self.D))
            sigma = np.zeros((self.K, self.D, self.D))
            N_k = np.zeros(self.K)

            for k in range(self.K): 
                N_k[k] = np.sum(gamma[:, k])
                mu[k] = np.dot(gamma.T[k], self.points) / N_k[k]

                c = self.points - mu[k]
                a = np.multiply(gamma.T[k], c.T)
                sigma[k] = np.diag(np.diag(np.dot(a, c) / N_k[k]))
        
            pi = N_k / self.N
            
            return pi, mu, sigma

        raise NotImplementedError

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)

