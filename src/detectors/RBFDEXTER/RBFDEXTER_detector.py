import os
import argparse
import numpy as np
import time 
import hashlib
import torch
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.feature_extraction import settings
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import skew
import pickle
import concurrent.futures
import random
from statsmodels.tsa.stattools import acf, pacf
from utils.data import (
    load_object,
    n_policy_rollouts,
    save_object,
    split_train_test_episodes,
)
from utils.stats import eval_metrics

def make_one_dim_kernel_vector(batch0, x1, sigma=1, width=10):
    x1 = np.flip(x1)
    d_len = len(x1) - width 
    km = np.zeros([d_len])
    for i in range(0, d_len):
        element1 = 0
        for l in range(width):
                element1 += (x1[l] - x1[i+l])**2
        km[i] = np.exp(-(element1)/(sigma**2)) # np.sqrt(element1) *
    return x1*100
    
    """
    last_element = x1[-1]  # Get the last element of x1
    squared_diffs = [(last_element - x)**2 for x in x1[:-1]]
    squared_diffs = sum(squared_diffs)
    return [1.5*np.exp(-(squared_diffs)/sigma**2), np.mean(x1)]
    """

def make_kernel_vector(batch0, mat, sigma=1, width=10):
    # takes a TxN matrix and return NxT' where T' is the nb of features
    _,K = mat.shape
    for dim in range(K):
        km_tmp = make_one_dim_kernel_vector(batch0[:,dim], mat[:,dim], sigma=sigma, width=width)
        if dim==0:
            km = np.array(km_tmp)
        else:
            km = np.vstack([km, km_tmp])
    return km

def preprocess_data(list_data, batch_size = 10, step_size=1, sliding = False, window_size = 5):
    processed_data = []

    if sliding == True:
        for episode in list_data:
            batched_array = [episode[i:i+batch_size] for i in range(0, len(episode) - batch_size + 1, step_size)]
            processed_data.append(batched_array)

    else:
        for episode in list_data:
            num_batches = episode.shape[0] // batch_size
            batched_array = [episode[i:i+batch_size] for i in range(0, num_batches * batch_size, batch_size)]
            processed_data.append(batched_array)

    return processed_data

class RBFDEXTER_Detector:
    def __init__(self, n_dimensions, batch_size, sliding = False):

        self.n_dimensions = n_dimensions
        self.window_size = 10

        self.detector = None
        self.imputer = None
        self.num_features_per_dim = None

        self.batch_size = 10
        self.sigma = 1.5
    def train(self, args):
        if os.path.exists(args.train_data_path):
            print("loading rollout data")
            ep_data = load_object(args.train_data_path) 
        else:
            raise ValueError("the specified data rollout path does not exist! Please specify a proper policy path with --train-data-path 'path_to_data.pkl'")
        
        train_ep_data, _ = split_train_test_episodes(episodes=ep_data)            
        states_train = [ep.states for ep in train_ep_data]  
        processed_train_data = preprocess_data(list_data = states_train, 
                                                batch_size = self.batch_size,
                                                sliding = args.TF_sliding,
                                                window_size = self.window_size)

        features = []
        print(torch.tensor(processed_train_data).shape)
        for i, episode in enumerate(processed_train_data):
            print("Episode: ", i)

            for j, batch in enumerate(episode):
                if j==0:
                    batch0 = batch
                X = make_kernel_vector(batch0, batch, sigma=self.sigma, width=self.window_size)
                # Replace infinities with NaN
 
                ind = np.isinf(X)
                X[ind] = np.nan
                X = torch.tensor(X)
                # Impute missing values
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(X.unsqueeze(0)) 

                # Transform the training data
                features_imputed = self.imputer.transform(X.unsqueeze(0))

                if j == 0 and i == 0:
                    features = features_imputed.flatten()  # 4432, 32
                else:
                    features = np.vstack([features, features_imputed.flatten()])
        print("TRAIN DATA PROCESSING FINISHED")

        #Train the detector
        ISOFOREST_MODELS = []
        
        if self.n_dimensions == 1:
            model = IsolationForest(random_state=2023)
            print(features_imputed.shape)
            model.fit(features_imputed)
            ISOFOREST_MODELS.append(model)
            self.num_features_per_dim = features_imputed.shape[1] // self.n_dimensions 
            print("num_features_per_dim", self.num_features_per_dim)

        else:
            #One model per dimension
            features = np.array(features)
            self.num_features_per_dim = features.shape[1] // self.n_dimensions 
            print(self.num_features_per_dim)
            print(self.num_features_per_dim)
            for dim in range(self.n_dimensions):
                start_idx = dim * self.num_features_per_dim
                end_idx = (dim + 1) * self.num_features_per_dim
                features_imputed_dim = features[:, start_idx:end_idx]
                model = IsolationForest(random_state=2023)
                print(features_imputed_dim.shape)
                model.fit(features_imputed_dim)
                ISOFOREST_MODELS.append(model)

        self.detector = ISOFOREST_MODELS
        print("DETECTOR FITTED")


    def test(self, 
             args,
             observations, 
             actions):
        
        test_data = test_data = preprocess_data(list_data = [observations], 
                                    batch_size = self.batch_size, 
                                    sliding = args.TF_sliding)
        all_features_test = []
        features = []
        for i, episode in enumerate(test_data):
            for j, batch in enumerate(episode):
                if j==0:
                    batch0=batch
                X = make_kernel_vector(batch0, batch, sigma=self.sigma, width=self.window_size)

                # Replace infinities with NaN
                ind = np.isinf(X)
                X[ind] = np.nan
                # Impute missing values
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(X.reshape(-1, 1))

                # Transform the training data
                features_imputed = self.imputer.transform(X.reshape(-1, 1))

                if j == 0 and i==0:
                    features = [features_imputed.flatten()] # 4432, 32
                else:
                    features = np.vstack([features, features_imputed.flatten()])

        print("TEST DATA PROCESSING FINISHED")
        all_features_test = np.array(features)
        print(all_features_test.shape)
        anom_scores = []
        for i, episode in enumerate(all_features_test):
            #one model for each dim
            anomaly_scores_dim = []
            for dim in range(self.n_dimensions):
                start_idx = dim * self.num_features_per_dim
                end_idx = (dim + 1) * self.num_features_per_dim
                if self.n_dimensions != 1:
                    feats_dim = episode[start_idx:end_idx].reshape(1, -1)
                else:
                    feats_dim = episode.reshape(1, -1)
                anomaly_scores_dim.append(-1 * self.detector[dim].decision_function(feats_dim)[0])
            anomaly_score = np.max(anomaly_scores_dim)
                
            if args.TF_sliding == True:
                #For the first 10 obs, add the same score (no sliding window)
                if i < self.batch_size // self.window_size:
                    # Append the initial score for the first 10 steps
                    for _ in range(self.window_size):
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
