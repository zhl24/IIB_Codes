import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt
from scipy.special import logsumexp



def calculate_mse(time_series_1, time_series_2):
    """
    Calculate the Mean Squared Error (MSE) between two time series and round it to 5 significant figures.

    :param time_series_1: Array representing the first time series
    :param time_series_2: Array representing the second time series
    :return: Mean Squared Error between the two time series, rounded to 5 significant figures
    """
    if len(time_series_1) != len(time_series_2):
        raise ValueError("Time series must have the same length")

    mse = np.mean((np.array(time_series_1) - np.array(time_series_2)) ** 2)
    return round(mse, 5)  # Round the result to 5 significant figures




#This function takes in an array of log likelihoods, then perform normalisation in the log domain and  finally return the normalised probabilities in the usual domain
def log_probs_to_normalised_probs(log_likelihoods):
    # Step 1: Compute the normalization factor using logsumexp
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = np.squeeze(log_likelihoods)
    normalization_factor = logsumexp(log_likelihoods)

    # Step 2: Normalize each log likelihood by subtracting the normalization factor
    normalized_log_likelihoods = log_likelihoods - normalization_factor

    # Step 3: Convert back to probability domain
    probabilities = np.exp(normalized_log_likelihoods)
    
    probabilities = probabilities/np.sum(probabilities)
    return np.squeeze(probabilities)


def weighted_sum(particles, weights):
    

    # Initialize the sum as a zero array of the same shape as the first particle
    weighted_sum = np.zeros_like(particles[0])

    # Iterate over each particle and its weight
    for particle, weight in zip(particles, weights):
        weighted_sum += np.array(particle) * weight

    return weighted_sum


def inverted_gamma_to_mean_variance(alpha, beta):
    """
    Convert Inverted Gamma distribution parameters to mean and variance.
    
    Parameters:
    - alpha: shape parameter of the Inverted Gamma distribution (> 0).
    - beta: scale parameter of the Inverted Gamma distribution (> 0).
    
    Returns:
    - A tuple containing the mean and variance of the Inverted Gamma distribution.
      Returns (None, None) if the mean or variance does not exist.
    """
    mean = max(0,beta / (alpha - 1))
    
    
    variance = max(0,beta**2 / ((alpha - 1)**2 * (alpha - 2)))
    
    return mean, variance