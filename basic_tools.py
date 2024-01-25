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