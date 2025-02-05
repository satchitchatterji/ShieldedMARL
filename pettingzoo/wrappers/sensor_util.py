import numpy as np
from scipy.special import erf

class OnlineStats:
    """Calculate mean and standard deviation of a stream of numbers using Welford's method."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        return self.M2 / (self.n - 1) if self.n > 1 else float('nan')

    def stddev(self):
        return np.sqrt(self.variance())


# Define the CDF of the standard normal distribution using the error function
def normal_cdf(z):
    return 0.5 * (1 + erf(z / np.sqrt(2)))

# Define the normalized distance metric function
def normalized_pgg_distance(f, mu, sigma):
    if sigma is None or sigma == 0:
        return 1
    z = (f - mu) / sigma
    cdf = normal_cdf(z)
    return 1 - 2 * np.abs(cdf - 0.5)