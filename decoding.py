# Some code is derived from the Guggenmos et al., 2018 tutorial
import argparse
import numpy as np
import scipy
from sklearn.discriminant_analysis import _cov
from dissimilarity import LDA
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


def perform_decoding_analyses(data, n_perm, n_pseudo):
    # data: (num_subjects, num_images, num_trials, num_timepoints, num_electrodes)
    num_subjects = data.shape[0]
    num_images = data.shape[1]
    num_timepoints = data.shape[3]
    num_electrodes = data.shape[4]

    lda = LDA(sigma=np.eye(num_electrodes))  # passing identitity covariance matrix to LDA, since data is pre-whitened

    result = np.full((num_subjects, n_perm, num_images, num_images, num_timepoints), np.nan,
                     dtype={'names': ['lda'], 'formats': 1*['f8']})

    for i in range(0, num_subjects):
        print("Subject %d" % (i+1))
        X, y = convert_data(data[i,:,:,:,:])

        cv = ShuffleBinLeaveOneOut(y, n_iter=n_perm, n_pseudo=n_pseudo)
        for f, (train_indices, test_indices) in enumerate(cv.split(X)):
            print('\tPermutation %g / %g' % (f + 1, n_perm))

            Xpseudo_train, Xpseudo_test = compute_pseudo_trials(X, train_indices, test_indices, num_electrodes, num_timepoints)
            print Xpseudo_train.shape, Xpseudo_test.shape

            sigma_conditions = cv.labels_pseudo_train[0, :, n_pseudo-1:].flatten()
            sigma_ = np.empty((num_images, num_electrodes, num_electrodes))
            for c in range(num_images):
                # Compute sigma for each time point, then average across time
                sigma_[c] = np.mean([_cov(Xpseudo_train[sigma_conditions==c, t, :], shrinkage='auto')
                                     for t in range(num_timepoints)], axis=0)
            sigma = sigma_.mean(axis=0)  # average across conditions
            sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
            Xpseudo_train = np.dot(Xpseudo_train, sigma_inv)
            Xpseudo_test = np.dot(Xpseudo_test, sigma_inv)

            print Xpseudo_train.shape, Xpseudo_test.shape

            for t in range(num_timepoints):
                for c1 in range(num_images-1):
                    for c2 in range(min(c1 + 1, num_images-1), num_images):
                        # 3. Fit the classifier using training data
                        data_train = Xpseudo_train[cv.ind_pseudo_train[c1, c2], t, :]
                        lda.fit(data_train, cv.labels_pseudo_train[c1, c2])

                        # 4. Compute and store classification accuracies
                        data_test = Xpseudo_test[cv.ind_pseudo_test[c1, c2], t, :]
                        result['lda'][i, f, c1, c2, t] = np.mean(lda.predict(data_test) == cv.labels_pseudo_test[c1, c2]) - 0.5

    # Average across permutations
    result_ = np.full((num_subjects, num_images, num_images, num_timepoints), np.nan,
                      dtype={'names': ['lda'], 'formats': 1*['f8']})    
    result_['lda'] = np.nanmean(result['lda'], axis=1)
    result = result_

    return result


