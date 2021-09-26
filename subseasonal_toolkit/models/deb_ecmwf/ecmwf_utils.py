from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    """Computes the geometric median of the columns of X, up to a tolerance epsilon.
    The geometric median is the vector that minimizes the mean Euclidean norm to
    each column of X.
    """
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def ssm(X, alpha=1):
    """Computes stabilized sample mean (Orenstein, 2019) of each column of X
    
    Args:
        alpha: if infinity, recovers the mean; if 0 approximates median
    """
    # Compute first, second, and third uncentered moments
    mu = np.mean(X,0)
    mu2 = np.mean(np.square(X),0)
    mu3 = np.mean(np.power(X,3),0)
    # Return mean - (third central moment)/(3*(2+numrows(X))*variance)
    return mu - (mu3 - 3*mu*mu2+2*np.power(mu,3)).div(3*(2+alpha*X.shape[0])*(mu2 - np.square(mu)))
