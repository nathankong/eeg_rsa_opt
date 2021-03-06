# The code in this file performs cross validated Euclidean distance and Pearson 
# correlation analysis using the pseudo trials (part of the code is from the 
# Guggenmos et al. 2018 tutorial).

import numpy as np
from sklearn.discriminant_analysis import _cov
from scipy.linalg import fractional_matrix_power

from cv import ShuffleBinLeaveOneOut

def convert_data(data):
    # data: (num_images, num_trials, num_timepoints, num_electrodes)
    # Returns eeg_data: (num_images*num_trials, num_timepoints, num_electrodes),
    # and labels: (num_images*num_trials,)
    num_labels = data.shape[0]
    num_trials = data.shape[1]
    labels = np.array([])
    for image in range(num_labels):
        for trial in range(num_trials):
            labels = np.append(labels, [image])

    eeg_data = data.reshape(data.shape[0]*data.shape[1],data.shape[2], data.shape[3])
    return eeg_data, labels

def compute_pseudo_trials(X, train_indices, test_indices, num_electrodes, num_timepoints):
    # Returns: (num_pseudo_trials, num_timepoints, num_electrodes)
    Xpseudo_train = np.full((len(train_indices), num_timepoints, num_electrodes), np.nan)
    Xpseudo_test = np.full((len(test_indices), num_timepoints, num_electrodes), np.nan)
    for i, ind in enumerate(train_indices):
        Xpseudo_train[i, :, :] = np.mean(X[ind, :, :], axis=0)
    for i, ind in enumerate(test_indices):
        Xpseudo_test[i, :, :] = np.mean(X[ind, :, :], axis=0)

    return Xpseudo_train, Xpseudo_test

def perform_analyses(data, n_perm, n_pseudo):
    # data: (num_subjects, num_images, num_trials, num_timepoints, num_electrodes)
    # Returns a numpy "dictionary": (keys: ec_cv and ps_cv), (values: rdms of shape:
    # (num_subjects, num_images, num_images, num_timepoints))
    num_subjects = data.shape[0]
    num_images = data.shape[1]
    num_timepoints = data.shape[3]
    num_electrodes = data.shape[4]

    results_cv = np.full((num_subjects, n_perm, num_images, num_images, num_timepoints), np.nan,
                        dtype={'names': ['ec_cv', 'ps_cv'], 'formats': 2*['f8']})

    for i in range(0, num_subjects):
        print("Subject %d" % (i+1))
        X, y = convert_data(data[i,:,:,:,:])

        cv = ShuffleBinLeaveOneOut(y, n_iter=n_perm, n_pseudo=n_pseudo)
        for f, (train_indices, test_indices) in enumerate(cv.split(X)):
            #print train_indices
            #print test_indices

            print('\tPermutation %g / %g' % (f + 1, n_perm))

            Xpseudo_train, Xpseudo_test = compute_pseudo_trials(X, train_indices, test_indices, num_electrodes, num_timepoints)
            sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
            #print sigma_conditions
            sigma_ = np.empty((num_images, num_electrodes, num_electrodes))
            for c in range(num_images):
                # compute sigma for each time point, then average across time
                sigma_[c] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, t, :], shrinkage='auto')
                                     for t in range(num_timepoints)], axis=0)
            sigma = sigma_.mean(axis=0)  # average across conditions
            sigma_inv = fractional_matrix_power(sigma, -0.5)
            Xpseudo_train = np.dot(Xpseudo_train, sigma_inv)
            Xpseudo_test = np.dot(Xpseudo_test, sigma_inv)

            print Xpseudo_train.shape, Xpseudo_test.shape

            for t in range(num_timepoints):
                for c1 in range(num_images-1):
                    for c2 in range(min(c1 + 1, num_images-1), num_images):
                        data_train = Xpseudo_train[cv.ind_pseudo_train[c1,c2], t, :]
                        data_test = Xpseudo_test[cv.ind_pseudo_test[c1,c2], t, :]
                        classes = np.unique(cv.labels_pseudo_train[c1, c2])

                        data_train_c1 = data_train[cv.labels_pseudo_train[c1,c2]==classes[0]]
                        data_train_c2 = data_train[cv.labels_pseudo_train[c1,c2]==classes[1]]
                        data_test_c1 = data_test[cv.labels_pseudo_test[c1,c2]==classes[0]]
                        data_test_c2 = data_test[cv.labels_pseudo_test[c1,c2]==classes[1]]

                        #print data_train_c1.shape, data_train_c2.shape
                        euclidean_dist = perform_cv_euclidean(data_train_c1, data_train_c2, data_test_c1, data_test_c2)
                        pearson_dist = perform_cv_pearson(data_train_c1, data_train_c2, data_test_c1, data_test_c2)
                        results_cv['ec_cv'][i,f,c1,c2,t] = euclidean_dist
                        results_cv['ps_cv'][i,f,c1,c2,t] = pearson_dist
                        results_cv['ec_cv'][i,f,c2,c1,t] = euclidean_dist
                        results_cv['ps_cv'][i,f,c2,c1,t] = pearson_dist

    # average across permutations
    results_cv_ = np.full((num_subjects, num_images, num_images, num_timepoints), np.nan,
                      dtype={'names': ['ec_cv', 'ps_cv'], 'formats': 2*['f8']})
    results_cv_['ec_cv'] = np.nanmean(results_cv['ec_cv'], axis=1)
    results_cv_['ps_cv'] = np.nanmean(results_cv['ps_cv'], axis=1)
    results_cv = results_cv_

    return results_cv

def perform_cv_euclidean(train_c1, train_c2, test_c1, test_c2):
    dist_train = np.mean(train_c1, axis=0) - np.mean(train_c2, axis=0)
    dist_test = np.mean(test_c1, axis=0) - np.mean(test_c2, axis=0)
    return np.dot(dist_train, dist_test)

def perform_cv_pearson(train_c1, train_c2, test_c1, test_c2):
    train_image1_data = np.mean(train_c1, axis=0)
    train_image2_data = np.mean(train_c2, axis=0)
    test_image1_data = np.mean(test_c1, axis=0)
    test_image2_data = np.mean(test_c2, axis=0)

    assert train_image1_data.ndim == 1
    assert train_image2_data.ndim == 1
    assert test_image1_data.ndim == 1
    assert test_image2_data.ndim == 1
    
    train_image1_var = np.var(train_image1_data)
    train_image2_var = np.var(train_image2_data)
    denom_noncv = np.sqrt(train_image1_var * train_image2_var)
    
    cov_train1test2 = np.cov(train_image1_data, test_image2_data)[0,1]
    cov_train2test1 = np.cov(train_image2_data, test_image1_data)[0,1]
    cov_12 = (cov_train1test2 + cov_train2test1) / 2.0
    var_1_traintest = np.cov(train_image1_data, test_image1_data)[0,1]
    var_2_traintest = np.cov(train_image2_data, test_image2_data)[0,1]
    
    # Regularize variance
    denom = np.sqrt(max(var_1_traintest, 0.1*train_image1_var) * max(var_2_traintest, 0.1*train_image2_var))
    
    # Regularize denom
    denom = max(denom, 0.25 * denom_noncv)
    
    # Bound r values
    r = cov_12 / denom
    r = min(max(-1.0, r), 1.0)
    
    corr = 1 - r
    return corr

if __name__ == "__main__":

    eeg_data = np.load("eeg_data/X_all.npy")
    eeg_data = np.transpose(eeg_data, (0,1,2,4,3))

    np.random.seed(10)
    res = perform_analyses(eeg_data, 20, 5)
    np.save("results_dump.npy", res)


