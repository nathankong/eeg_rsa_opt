import os
import time
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
np.random.seed(0)
KFOLD_RAND_SEED = 0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from models import LinearModel, ElementWiseScalingModel
from compute_noise_ceiling import compute_noise_ceiling
from utils import load_features, compute_correlation

torch.manual_seed(0)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
print "Device:", DEVICE

def compute_correlation_timecourse(
    eeg_rdms,
    features,
    n_latent,
    lr,
    max_iter,
    eps,
    corr_func,
    model,
    n_folds=4
):
    """
    Performs optimization for each time point in the time series and outputs the
    optimized correlations for each time point for each fold.

    Inputs:
        eeg_rdms  : (numpy.ndarray) of dimensions (n_images, n_images, n_time)
        features  : (numpy.ndarray) of dimensions (n_images, n_feats)
        n_latent  : (int) number of dimensions for new features
        lr        : (float) learning rate for gradient descent
        max_iter  : (int) number of optimization iterations
        eps       : (float) tolerance to indicate when to stop optimization
        corr_func : a function that takes in two matrices and outputs the correlation of
                    the upper triangular portion of each matrix
        model     : (string) model type for the optimization. It is one of
                    ["elementwise", "linearcombo"]
        n_folds   : (int) number of training/testing image splits.

    Outputs:
        corrs : (numpy.ndarray) of train/test results. The dimensions are:
                (num_timepoints, num_folds, 2). First index in the last dimension
                contains the training set correlations and the second index in the
                last dimension contains the testing set correlations.
    """

    n_time = eeg_rdms.shape[2]
    corrs = np.zeros((n_time,n_folds,2))
    all_losses = dict()
    all_weights = dict()
    for i in range(n_time):
        print "  Timepoint:", i

        # Obtain time point EEG RDM and set diagonal to 0 (it is currently NaN)
        eeg_rdm = eeg_rdms[:,:,i]
        d_idx = np.diag_indices(eeg_rdm.shape[0])
        eeg_rdm[d_idx] = 0.0

        kf = KFold(n_splits=n_folds, random_state=KFOLD_RAND_SEED)
        for j, (train, test) in enumerate(kf.split(features)):
            print("    Fold {}/{}".format(j+1, n_folds))

            # Select stimuli to use in train/test set
            tr_features = features[train,:]
            te_features = features[test,:]
            tr_eeg_rdm = eeg_rdm[train,:][:,train]
            te_eeg_rdm = eeg_rdm[test,:][:,test]

            # Define model (different for each fold and time point)
            if model == "linearcombo":
                m = LinearModel(tr_features.shape[1], n_latent, DEVICE)
            elif model == "elementwise":
                m = ElementWiseScalingModel(tr_features.shape[1], DEVICE)
            else:
                raise ValueError("{} is not a supported model.".format(model))

            # Optimize weight matrix
            m, losses = optimize(m, tr_features, tr_eeg_rdm, n_latent, lr=lr, max_iter=max_iter, eps=eps)

            with torch.no_grad():
                tr_rdm_hat = 1.0 - m.forward_prop(tr_features).cpu().numpy()
                te_rdm_hat = 1.0 - m.forward_prop(te_features).cpu().numpy()

            # Compute the correlation between EEG RDM and layer RDM
            tr_r = compute_correlation(tr_eeg_rdm, tr_rdm_hat, corr_func)
            te_r = compute_correlation(te_eeg_rdm, te_rdm_hat, corr_func)
            corrs[i,j,:] = np.array([tr_r, te_r])
    
    return corrs

def pearson_corr_loss(mat1, mat2):
    """
    Computes the Pearson correlation loss for two RDMs called mat1 and mat2.
    """
    # mat1 and mat2 are torch variables
    # mat1: predict
    # mat2: target

    assert list(mat1.size()) == list(mat2.size())
    n_image = list(mat1.size())[0]
    row_idx, col_idx = np.triu_indices(n_image)

    row_idx = torch.LongTensor(row_idx).to(DEVICE)
    col_idx = torch.LongTensor(col_idx).to(DEVICE)

    R_hat = mat1[row_idx, col_idx]
    R_ = mat2[row_idx, col_idx]

    loss = -1.0 * pearson_corr(R_hat, R_)
    return loss

def pearson_corr(A, B):
    """
    Computes the Pearson correlation between two vectors, A and B.
    """
    # A, B: torch.Variable vectors
    A = A - torch.mean(A)
    B = B - torch.mean(B)
    c = torch.sum(A*B) / (torch.sqrt(torch.sum(A**2)) * torch.sqrt(torch.sum(B**2)))
    return c

def optimize(m, X, R, l, lr=1e-5, max_iter=10, eps=1e-4):
    """
    Main function to perform optimization on the weights.

    Inputs:
        m      : (BaseModel) instance for the model to use
        X      : (numpy.ndarray) feature matrix
        R      : (numpy.ndarray) rdm matrix
        l      : (int) number of latent features to learn
        n_iter : (int) number of optimization iterations

    Outputs:
        m      : (BaseModel) optimized model
        losses : (numpy.ndarray) loss per iteration
    """
    R = Variable(torch.from_numpy(R).type(torch.FloatTensor), requires_grad=False).to(DEVICE)

    loss_func = pearson_corr_loss
    optimizer = optim.SGD(m.optimization_params(), lr=lr)

    losses = list()
    prev_loss = np.Inf
    for i in range(max_iter):
        optimizer.zero_grad()

        # Forward propagation
        R_hat = 1.0 - m.forward_prop(X)

        # Compute loss
        loss = loss_func(R_hat, R)
        curr_loss = loss.data.cpu().numpy()

        # Back propagation
        loss.backward()
        optimizer.step()

        # Check convergence and record loss
        if np.abs(curr_loss - prev_loss) <= eps:
            break
        losses.append(curr_loss)
        prev_loss = curr_loss

    return m, np.array(losses)

def main(dataset, rdms_dir, results_dir, distance, layer_features_fpath, n_folds, optimizer_params, layer, model):
    """
    Main function that loops through each time point in the time series to perform optimization for
    a specific VGG19 layer, a specific distance that was used to previously compute RDMs, a specific
    model (e.g., linear combination or element-wise). It then automatically saves the results into
    the directories defined by the user.

    Inputs:
        dataset              : (string) dataset name (currently only accepts "kaneshiro")
        rdms_dir             : (string) base directory for RDMs computed previously
        results_dir          : (string) base directory to save correlation results
        distance             : (string) one of ["pearson", "decoding", "euclidean"]
        layer_features_fpath : (string) directory of VGG19 activations for layers in
                               ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"]
        n_folds              : (int) number of folds to perform optimization
        optimizer_params     : (dict) with keys in ["n_latent", "max_iterations", "learning_rate", "eps"]
        layer                : (string) which layer of ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"]
                               to do optimization for
        model                : (string) which optimization model to use. One of ["elementwise", "linearcombo"]
    """
    # rdms_dir: PATH/TO/RESULTS/rdms/kaneshiro/cv_euclidean_rdms.npy (for distance == "euclidean")
    rdms_fname = "cv_{}_rdms.npy".format(distance)
    eeg_rdms_fname = "{}/{}".format(rdms_dir, rdms_fname)
    eeg_rdms = np.load(eeg_rdms_fname)

    print "Dataset:", dataset
    print "Distance metric:", distance
    print "Model:", model
    print "RDMs File:", eeg_rdms_fname
    print "Number of folds:", n_folds
    print "Optimizer parameters:", optimizer_params
    print "Layer:", layer

    # Compute the noise ceiling of the test set RDMs
    print "Computing noise ceiling..."
    bounds = compute_noise_ceiling(eeg_rdms)

    # Some constants
    n_subjects = eeg_rdms.shape[0]
    n_time = eeg_rdms.shape[3]
    n_latent = optimizer_params["n_latent"]
    lr = optimizer_params["learning_rate"]
    max_iter = optimizer_params["max_iterations"]
    eps = optimizer_params["eps"]
    corr_func = spearmanr

    # Retrieve VGG19 activations and reshape into vector
    layer_features_template = layer_features_fpath + "/%s_feats.npy"
    if layer not in ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"]:
        raise ValueError("{} is not a supported layer.".format(layer))
    feats = load_features(layer_features_template % layer)
    feats = np.reshape(feats, [feats.shape[0], -1])

    # Perform optimization for each subject.
    # Last dimension is 2 because of train set correlation and test set correlation
    all_subj_timing = np.zeros((n_subjects,))
    all_subj_corrs = np.zeros((n_subjects, n_time, n_folds, 2))
    for i in range(n_subjects):
        print "Subject: {}/{}".format(i+1, n_subjects)

        start = time.time()

        corrs = compute_correlation_timecourse(
            eeg_rdms[i,:,:,:],
            feats,
            n_latent,
            lr,
            max_iter,
            eps,
            corr_func,
            model,
            n_folds=n_folds
        )
        all_subj_corrs[i,:,:,:] = corrs

        curr_subj_time = time.time() - start
        all_subj_timing[i] = curr_subj_time

    # Store the correlations
    layer_results = dict()
    layer_results[layer] = dict()
    layer_results[layer]["corrs"] = all_subj_corrs

    # Store timing
    layer_results["timing"] = all_subj_timing

    # Store the noise ceiling
    layer_results["noise_ceiling"] = bounds

    # Store the optimization run parameters and information
    layer_results["optimizer_params"] = optimizer_params
    layer_results["distance_metric"] = distance
    layer_results["dataset"] = dataset
    layer_results["rdm_file_used"] = eeg_rdms_fname
    layer_results["num_folds"] = n_folds

    # To load the pickled file: pickle.load(open("results.pkl", "rb"))
    results_dir = results_dir + "/{}/{}/".format(dataset, distance)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_fname = "{}/{}_{}.pkl".format(results_dir, layer, model)
    pickle.dump(layer_results, open(results_fname, "wb"))
    print "Optimal correlations results file:", results_fname

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kaneshiro", choices=["kaneshiro"])
    parser.add_argument('--distance', type=str, default="euclidean", choices=["euclidean", "pearson", "decoding"])
    parser.add_argument('--model', type=str, default="elementwise", choices=["linearcombo", "elementwise"])
    parser.add_argument('--rdms-dir', type=str, default="./")
    parser.add_argument('--layer-features-path', type=str, default="./layer_features/")
    parser.add_argument('--results-dir', type=str, default="./")
    parser.add_argument('--nfolds', type=int, default=4)
    parser.add_argument('--layer', type=str, default="fc2")
    parser.add_argument('--latentdim', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('--eps', type=float, default=1e-5)
    args = parser.parse_args()
    print("RDMS directory: {}".format(args.rdms_dir))

    # Layer to do analysis?
    layer = args.layer.lower()
    assert layer in ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"], \
        "Layer {} is not supported.".format(layer)

    # Optimizer parameters
    optimizer_params = dict()
    optimizer_params["n_latent"] = int(args.latentdim)
    optimizer_params["learning_rate"] = float(args.lr)
    optimizer_params["max_iterations"] = int(args.maxiter)
    optimizer_params["eps"] = float(args.eps)

    main(
        args.dataset.lower(),
        args.rdms_dir,
        args.results_dir,
        args.distance.lower(),
        args.layer_features_path,
        args.nfolds,
        optimizer_params,
        layer,
        args.model.lower()
    )


