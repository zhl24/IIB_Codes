import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt




def Kalman_Predict(x_mean,x_variance,x_transition,x_noise_variance,x_noise_mean = 0): #Hidden state x predict step. 
#Note that the noise is generally not zero mean in our case.
#This function takes the mean and variance of the current hidden state and the transition architecture to predict the next state mean and variance
    return x_mean*x_transition+x_noise_mean,x_variance*x_transition**2 + x_noise_variance

def Kalman_Correct(x_mean,x_variance,observation,emission,emission_noise_variance):
    I = observation - emission * x_mean
    return x_mean + (emission * x_variance)/(emission**2 * x_variance + emission_noise_variance)*I, x_variance * (1-emission**2 * x_variance/(emission**2*x_variance+emission_noise_variance))
#evaluation points just form the time axis
#We then simply need the subordinator jumps and jump times to compute the x_noise mean and variances
#emission is simply 1 in our model. emission_noise_variance is just the simga_n term in the model

def Noisy_Normal_Gamma_SDE_Filter(noisy_sequence,observation_noise_var,SDE,subordinator_jumps,jump_times,evaluation_points,Kalman_Predict,Kalman_Correct,x_mean_prior=0,x_variance_prior=1):
    A = SDE.A
    h = SDE.h
    T = SDE.T
    X0 = x_mean_prior
    var0 = x_variance_prior
    NVM_generator = SDE.NVM
    muw = NVM_generator.muw
    sigmaw = NVM_generator.sigmaw 
    s = 0
    #The priors would be removed after inference
    inferred_x = []
    inferred_var = [] 
    for index,evaluation_point in enumerate(evaluation_points):
        x_transition = np.exp(A*(evaluation_point-s))
        x_noise_mean = 0
        x_noise_variance = 0
        for i in range(len(jump_times)):
             jump = subordinator_jumps[i]
             jump_time = jump_times[i]
             if s<=jump_time<=evaluation_point:
                x_noise_mean += jump*np.exp(A*(evaluation_point-jump_time))*h
                x_noise_variance += jump*np.exp(2*A*(evaluation_point-jump_time))*h**2
        x_noise_mean = x_noise_mean * muw
        x_noise_variance = x_noise_variance * sigmaw**2
        if index != 0:
             mean_predict,var_predict = Kalman_Predict(inferred_x[-1],inferred_var[-1],x_transition,x_noise_variance,x_noise_mean)
        else:
            mean_predict,var_predict = Kalman_Predict(X0,var0,x_transition,x_noise_variance,x_noise_mean)
        mean_correct,var_correct = Kalman_Correct(mean_predict,var_predict,noisy_sequence[index],1,observation_noise_var)
        inferred_x.append(mean_correct)
        inferred_var.append(var_correct)
        s = evaluation_point
    return inferred_x,inferred_var
        




#The most general multi-dimensional Kalman filtering functions for the NVM SDE model:
#Here we define a single step in Kalman filter such that it can 
# We just need to define f, g, W (mw,Q) and V (mv,R)

#The first two inputs are the current state. f is the transition matrix, mw is the state noise mean and Q is the state noise covariance

def Kalman_transit(X,P,f,Q,mw = 0,B=0,u=0):
    if B == 0 :
        return f@X+mw, f@P@f.transpose()+Q
    else:
        return f@X+mw + B@u, f@P@f.transpose()+Q
#We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
#the observation mean and noises which determine most of the Kalman filtering difficulties
def Kalman_correct(X,P,Y,g,R,mv = 0):
    Ino = Y - g @ X -mv#The innovation term. Note that for the biased noise case there would be a subtraction here. All other terms are not affected
    S = g @ P @ g.transpose() + R #The innovation covariance
    K = P @ g.transpose() @ np.linalg.inv(S)
    n = np.shape(P)[0]
    I = np.identity(n)
    return X + K@Ino,(I-K@g)@P






# Bootstrap Particle Filtering in General NVM State Space:




