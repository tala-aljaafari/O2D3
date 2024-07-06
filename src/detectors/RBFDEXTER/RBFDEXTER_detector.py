import os
import argparse
import numpy as np
import time 
import hashlib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.feature_extraction import settings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import pandas as pd
import pickle
import concurrent.futures
import random
from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

def make_kernel_matrix(x1, sigma=1, width=10):

    '''
    function to make a kernel matrix
    Parameters:
    -----------
    x1 : numpy.array
        This is a vector of time series.
    sigma : double
        This is used for RBF kernel.
    sub_len : int
        This is length of subsequence. It is also called window-size.
    '''

    d_len = len(x1) // width 
    km = np.zeros([d_len, d_len])
    em1 = np.zeros([d_len, d_len])
    for i in range(0, d_len):
        for j in range(0, d_len):
            if i < j: continue
            element1 = 0
            for l in range(width):
                element1 += (x1[i*width+l] - x1[j*width+l])**2
            km[i,j] = np.sqrt(element1) * np.exp(-(element1)/(sigma**2))
            km[j,i] = km[i,j]

    return km

def make_kernel_tensor(mat, width=10):

    '''
    Paramters
    --------------
    mat : numpy.ndarray
        multivariate time series data (T x K) where K is the number
        of variables and T is the number of time series
    width : int
        length of subsequences of multivariate time series
    sigma : double
        the sigma is used in RBF kernel, exp{-(x_i - x_j)^2/sigma}

    Return
    ----------
    km : numpy.ndarry
        tensor of a kernel
        three-dimensional array (third-order tensor)
    '''

    _,K = mat.shape
    for dim in range(K):
        km_tmp = make_kernel_matrix(mat[:,dim], sigma=1, width=width)
        #km_tmp = km_tmp.reshape(1, km_tmp.shape[0], km_tmp.shape[1])
        if dim==0:
            km = np.array(km_tmp)
        else:
            km = np.hstack([km, km_tmp])
    return km

class RBFDEXTER_Detector:
    def __init__(self, 
                 n_dimensions,
                 batch_size,
                 sliding = False):

        self.n_dimensions = n_dimensions
        self.batch_size = batch_size

        self.detector = None
        self.imputer = None
        self.num_features_per_dim = None

    def train(
            self,
            args):
        """
        """

        #if extracted features available, load them
        if args.TF_train_data_feature_path != "" and args.TF_imputer_path != "":
            #load the extracted features      
            if os.path.exists(args.TF_train_data_feature_path):
                print("Loading feature extractions data from: ", args.TF_train_data_feature_path)

                with open(args.TF_train_data_feature_path, 'rb') as file:
                    features_imputed = pickle.load(file)    

            else:
                raise ValueError("the specified extrated train data feature path does not exist!")
            
            #load the imputer
            if os.path.exists(args.TF_imputer_path):
                print("Loading imputer from: ", args.TF_imputer_path)

                with open(args.TF_imputer_path, 'rb') as file:
                    imputer = pickle.load(file) 

            else:
                raise ValueError("the specified imputer path does not exist!")
            
        else:
            #load rollout data
            if os.path.exists(args.train_data_path):
                print("loading rollout data")
                ep_data = load_object(args.train_data_path)
            
            else:
                raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")
            
            train_ep_data, val_ep_data = split_train_test_episodes(episodes=ep_data)
            
            # initialize the detector
            print("")
            print("Extracting features from train data...")
            
            states_train = [ep.states for ep in train_ep_data]  # dims: (45, 151, 17), dim of 1 obs is 17, 151 timesteps
            action_train = [ep.actions for ep in train_ep_data]

            processed_train_data = states_train
            settings_efficient = settings.EfficientFCParameters()

            features = []
            train_ep_ctr = 0

            if args.TF_sliding == True:
                print("Using sliding window for feature extraction")
            else:
                print("Not using sliding window for feature extraction")

            for episode in processed_train_data:
                print("Episode shape:", np.array(episode).shape)
                print("Episode: ", train_ep_ctr)
                X = make_kernel_tensor(episode)
                print("X shape", np.array(X).shape)
                train_ep_ctr += 1

                # Replace infinities with NaN
                ind = np.isinf(X)
                X[ind] = np.nan

                # Impute missing values
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(X)

                # Transform the training data
                features_imputed = self.imputer.transform(X)
                features.append(features_imputed)

            print("TRAIN DATA PROCESSING FINISHED")

        #Train the detector
        ISOFOREST_MODELS = []
        
        if self.n_dimensions == 1:
            model = IsolationForest(random_state=2023)
            model.fit(features_imputed)
            ISOFOREST_MODELS.append(model)
            self.num_features_per_dim = features_imputed.shape[1] // self.n_dimensions 
            print("num_features_per_dim", self.num_features_per_dim)

        else:
            #One model per dimension
            print(np.array(features_imputed).shape)
            self.num_features_per_dim = features_imputed.shape[1] // self.n_dimensions 
            print("nb of features per dim", self.num_features_per_dim)
            for dim in range(self.n_dimensions):
                start_idx = dim * self.num_features_per_dim
                end_idx = (dim + 1) * self.num_features_per_dim
                features_imputed_dim = features_imputed[:, start_idx:end_idx]
                model = IsolationForest(random_state=2023)
                model.fit(features_imputed_dim)
                ISOFOREST_MODELS.append(model)

        self.detector = ISOFOREST_MODELS
        print("DETECTOR FITTED")


    def test(self, 
             args,
             observations, 
             actions):
        
        test_data = [observations]
        all_features_test = []
        print("test data", np.array(test_data).shape)
        for episode in test_data:
            print("test episode shape", np.array(episode).shape)
            X = make_kernel_tensor(episode)
            print("X shape", np.array(X).shape)
            # Replace infinities with NaN
            ind = np.isinf(X)
            X[ind] = np.nan

            # Impute missing values
            self.imputer.fit(X)
            features_imputed_test = self.imputer.transform(X)
            all_features_test.append(features_imputed_test)

        print("TEST DATA PROCESSING FINISHED")

        for _, episode in enumerate(all_features_test):
            anom_scores = []

            for i in range(episode.shape[0]):
                feats = episode[i,:]
                anomaly_scores_dim = []
                for dim in range(self.n_dimensions):
                    start_idx = dim * self.num_features_per_dim
                    end_idx = (dim + 1) * self.num_features_per_dim
                    feats_dim = feats[start_idx:end_idx].reshape(1, -1)
                    anomaly_scores_dim.append(-1 * self.detector[dim].decision_function(feats_dim)[0])
                anomaly_score = np.mean(anomaly_scores_dim)
                
                if args.TF_sliding == True:
                    #For the first 10 obs, add the same score (no sliding window)
                    if i == 0:
                        # Append the initial score for the first 10 steps
                        for _ in range(self.batch_size):
                            anom_scores.append(anomaly_score)
                    #if it's the last state, skip (to conform with other detectors)
                    elif i == episode.shape[0] - 1:
                        pass
                    else:
                        anom_scores.append(anomaly_score)

                else:
                    for _ in range(self.batch_size):
                        anom_scores.append(anomaly_score)

        return anom_scores