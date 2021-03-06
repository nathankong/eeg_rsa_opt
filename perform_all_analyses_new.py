import os

import numpy as np

from scipy.stats import spearmanr
from collections import defaultdict

from compute_pwlf import do_pwlf
from decoding import perform_decoding_analyses
from cv_distance_metric import perform_analyses
from compute_noise_ceiling import compute_noise_ceiling

# Supported VGG19 layer names
LAYERS = ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"]

def combine_layer_corrs(all_correlations, num_timepoints):
    # all_correlations: dict: key=layer, value=list of np arrays
    # Returns a dictionary like np array where each key is the layer and value is the correlations of
    # shape (num_subjects, num_timepoints)
    num_subjects = np.unique([np.array(all_correlations[layer]).shape[0] for layer in LAYERS])
    assert num_subjects.size == 1 and num_subjects.ndim == 1
    all_layer_corrs = np.full(
        (num_subjects[0], num_timepoints),
        np.nan,
        dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']}
    )

    for layer in LAYERS:
        correlations = np.array(all_correlations[layer])
        assert correlations.ndim == 2
        assert correlations.shape[0] == num_subjects
        assert correlations.shape[1] == num_timepoints
        all_layer_corrs[layer] = correlations
        print "Layer {}; Correlations shape {}".format(layer, correlations.shape)

    return all_layer_corrs

def _compute_rdm_spearmanr(mat1, mat2):
    """
    Computes Spearmanr between two matrices, mat1 and mat2.
    """
    assert mat1.shape == mat2.shape
    num_images = mat1.shape[0]
    mat1_upr = mat1[np.triu_indices(num_images, 1)]
    mat2_upr = mat2[np.triu_indices(num_images, 1)]
    my_corr = spearmanr(mat1_upr, mat2_upr)

    # Spearman R, p-val
    return my_corr[0], my_corr[1]

def _get_layer_rdm(layer_rdm_dir, layer_name, num_images):
    """
    Helper function to load previously computed RDMs for a VGG19 layer in
    ["pool1", "pool2", "pool3", "pool4", "pool5", "fc2"].
    """
    assert layer_name in LAYERS
    layer_rdm_fname = layer_rdm_dir + '/' + layer_name + ".npy"
    rdm = np.load(layer_rdm_fname)
    assert rdm.shape == (num_images, num_images)
    return rdm

def _compute_correlations(rdms, layer_rdm_dir, num_images, inverse=True):
    """
    Compute the correlation between the VGG19 layer RDMs and the time-resolved RDMs.
    The VGG19 layer RDMs are computed using the Pearson correlation metric.

    Inputs:
        rdms : (num_images, num_images, num_timepoints)

    Outputs:
        corrs : (dict) where the key is layer name and the value is the numpy array 
                of correlations of length num_timepoints.
    """
    corrs = dict()
    num_timepoints = rdms.shape[2]

    for l in LAYERS:
        layer_rdm = _get_layer_rdm(layer_rdm_dir, l, num_images)
        layer_corrs = np.zeros((num_timepoints,))
        for i in range(0, num_timepoints):
            if inverse:
                # 1-layer_rdm since high correlation = similarity, but low distance means similar
                current_corr, _ = _compute_rdm_spearmanr(1-layer_rdm, rdms[:,:,i])
            else:
                # Don't do 1-layer_rdm for comparing confusion matrices and layer RDMs
                current_corr, _ = _compute_rdm_spearmanr(layer_rdm, rdms[:,:,i])
            layer_corrs[i] = current_corr
        corrs[l] = layer_corrs

    return corrs

def compute_bootstrap_peaks(ec_corrs, ps_corrs, dec_corrs, num_bootstrap):
    """
    Compute the time at which the peak correlation occurs in the correlation time course.
    """
    ec_peaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    ps_peaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    dec_peaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    for layer in LAYERS:
        assert ec_corrs[layer].shape == ps_corrs[layer].shape == dec_corrs[layer].shape
        print ec_corrs[layer].shape
        num_subjects = ec_corrs[layer].shape[0]
        ec_peaks_layer = np.zeros((num_bootstrap,))
        ps_peaks_layer = np.zeros((num_bootstrap,))
        dec_peaks_layer = np.zeros((num_bootstrap,))
        for i in range(num_bootstrap):
            np.random.seed(i) # So that bootstrap sample is same across layers
            if i == 0:
                # Start off with the subject pool
                print "Subject pool peak."
                ec_data = ec_corrs[layer][:,:].mean(axis=0)
                ps_data = ps_corrs[layer][:,:].mean(axis=0)
                dec_data = dec_corrs[layer][:,:].mean(axis=0)
            else:
                # Select with replacement num_subjects -- "bootstrap sample"
                idx = np.random.choice(num_subjects, size=num_subjects)
                print "Bootstrap:", idx
                ec_data = ec_corrs[layer][idx,:].mean(axis=0)
                ps_data = ps_corrs[layer][idx,:].mean(axis=0)
                dec_data = dec_corrs[layer][idx,:].mean(axis=0)

            # Convert index of maximum correlation to a time in ms
            ec_peaks_layer[i] = np.argmax(ec_data) * 16. + (-112.)
            ps_peaks_layer[i] = np.argmax(ps_data) * 16. + (-112.)
            dec_peaks_layer[i] = np.argmax(dec_data) * 16. + (-112.)

        ec_peaks[layer] = ec_peaks_layer
        ps_peaks[layer] = ps_peaks_layer
        dec_peaks[layer] = dec_peaks_layer

    return ec_peaks, ps_peaks, dec_peaks


def compute_bootstrap_pwlf(ec_corrs, ps_corrs, dec_corrs, num_bootstrap, dataset):
    """
    Compute the time at which correlatio starts to increase in the correlation time course.
    """
    if dataset == "kaneshiro":
        start = 7
        end = 14
    else:
        raise ValueError("{} is not supported.".format(dataset))

    ec_breaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    ps_breaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    dec_breaks = np.full((num_bootstrap,), np.nan, dtype={'names': LAYERS, 'formats': len(LAYERS)*['f8']})
    for layer in LAYERS:
        assert ec_corrs[layer].shape == ps_corrs[layer].shape == dec_corrs[layer].shape
        print ec_corrs[layer].shape
        num_subjects = ec_corrs[layer].shape[0]
        ec_breaks_layer = np.zeros((num_bootstrap,))
        ps_breaks_layer = np.zeros((num_bootstrap,))
        dec_breaks_layer = np.zeros((num_bootstrap,))
        for i in range(num_bootstrap):
            np.random.seed(i) # So that bootstrap sample is same across layers

            # Acquire bootstrap sample and then average across the sample
            if i == 0:
                # Start off with the subject pool
                print "Subject pool pwlf."
                ec_data = ec_corrs[layer][:,start:end+1].mean(axis=0)
                ps_data = ps_corrs[layer][:,start:end+1].mean(axis=0)
                dec_data = dec_corrs[layer][:,start:end+1].mean(axis=0)
            else:
                # Select with replacement num_subjects -- "bootstrap sample"
                idx = np.random.choice(num_subjects, size=num_subjects)
                print "Bootstrap:", idx
                ec_data = ec_corrs[layer][idx,start:end+1].mean(axis=0)
                ps_data = ps_corrs[layer][idx,start:end+1].mean(axis=0)
                dec_data = dec_corrs[layer][idx,start:end+1].mean(axis=0)

            ec_breaks_layer[i] = do_pwlf(ec_data, start, end)
            ps_breaks_layer[i] = do_pwlf(ps_data, start, end)
            dec_breaks_layer[i] = do_pwlf(dec_data, start, end)

        ec_breaks[layer] = ec_breaks_layer
        ps_breaks[layer] = ps_breaks_layer
        dec_breaks[layer] = dec_breaks_layer

    return ec_breaks, ps_breaks, dec_breaks

def perform_analyses_across_subj(
    data_matrix,
    layer_rdm_dir,
    results_prefix,
    rdms_prefix,
    breaks_prefix,
    bounds_prefix,
    peaks_prefix,
    dataset
):
    # This function does analyses across subjects only
    # data_matrix: (num_subjects, num_images, num_trials, num_timepoints, num_electrodes)

    num_permutations = 20
    num_pseudo_trials = 5
    num_subjects = data_matrix.shape[0]
    num_images = data_matrix.shape[1]
    num_timepoints = data_matrix.shape[3]

    # Compute RDMs with the cross-validated Pearson and Euclidean distances.
    res = perform_analyses(data_matrix, num_permutations, num_pseudo_trials)
    # Compute RDMs with decoding metric.
    decoding_res = perform_decoding_analyses(data_matrix, num_permutations, num_pseudo_trials)

    cv_euclidean_rdms = res["ec_cv"]
    cv_pearson_rdms = res["ps_cv"]
    decoding_rdms = decoding_res["lda"]
    del res, decoding_res

    assert cv_euclidean_rdms.shape == (num_subjects, num_images, num_images, num_timepoints)
    assert cv_pearson_rdms.shape == (num_subjects, num_images, num_images, num_timepoints)
    assert decoding_rdms.shape == (num_subjects, num_images, num_images, num_timepoints)

    all_subj_ec_corrs = defaultdict(list)
    all_subj_ps_corrs = defaultdict(list)
    all_subj_decoding_corrs = defaultdict(list)
    for i in range(0, num_subjects):
        ec_corrs = _compute_correlations(cv_euclidean_rdms[i,:,:,:], layer_rdm_dir, num_images)
        ps_corrs = _compute_correlations(cv_pearson_rdms[i,:,:,:], layer_rdm_dir, num_images)
        decoding_corrs = _compute_correlations(decoding_rdms[i,:,:,:], layer_rdm_dir, num_images)
        for k, v in ec_corrs.items():
            assert v.shape[0] == num_timepoints == v.size
            all_subj_ec_corrs[k].append(v)
        for k, v in ps_corrs.items():
            assert v.shape[0] == num_timepoints == v.size
            all_subj_ps_corrs[k].append(v)
        for k, v in decoding_corrs.items():
            assert v.shape[0] == num_timepoints == v.size
            all_subj_decoding_corrs[k].append(v)

    ec_corrs = combine_layer_corrs(all_subj_ec_corrs, num_timepoints)
    ps_corrs = combine_layer_corrs(all_subj_ps_corrs, num_timepoints)
    dec_corrs = combine_layer_corrs(all_subj_decoding_corrs, num_timepoints)

    # Now do piecewise linear fit for each bootstrap sample for each layer correlation time course
    num_bootstrap = 1000
    ec_breaks, ps_breaks, dec_breaks = compute_bootstrap_pwlf(ec_corrs, ps_corrs, dec_corrs, num_bootstrap, dataset)

    # Now compute time of peak correlation for each bootstrap sample for each layer time course
    ec_peaks, ps_peaks, dec_peaks = compute_bootstrap_peaks(ec_corrs, ps_corrs, dec_corrs, num_bootstrap)

    # Now compute noise ceiling bounds
    ec_bounds = compute_noise_ceiling(cv_euclidean_rdms)
    ps_bounds = compute_noise_ceiling(cv_pearson_rdms)
    dec_bounds = compute_noise_ceiling(decoding_rdms)

    # Make directories to put results
    if not os.path.exists(results_prefix):
        os.makedirs(results_prefix)
    if not os.path.exists(rdms_prefix):
        os.makedirs(rdms_prefix)
    if not os.path.exists(breaks_prefix):
        os.makedirs(breaks_prefix)
    if not os.path.exists(peaks_prefix):
        os.makedirs(peaks_prefix)
    if not os.path.exists(bounds_prefix):
        os.makedirs(bounds_prefix)

    # Save correlations
    np.save("{}/cv_euclidean_corrs.npy".format(results_prefix), ec_corrs)
    np.save("{}/cv_pearson_corrs.npy".format(results_prefix), ps_corrs)
    np.save("{}/cv_decoding_corrs.npy".format(results_prefix), dec_corrs)
    # Save RDMs
    np.save("{}/cv_euclidean_rdms.npy".format(rdms_prefix), cv_euclidean_rdms)
    np.save("{}/cv_pearson_rdms.npy".format(rdms_prefix), cv_pearson_rdms)
    np.save("{}/cv_decoding_rdms.npy".format(rdms_prefix), decoding_rdms)
    # Save breaks (i.e.onsets)
    np.save("{}/cv_euclidean_breaks.npy".format(breaks_prefix), ec_breaks)
    np.save("{}/cv_pearson_breaks.npy".format(breaks_prefix), ps_breaks)
    np.save("{}/cv_decoding_breaks.npy".format(breaks_prefix), dec_breaks)
    # Save peaks
    np.save("{}/cv_euclidean_peaks.npy".format(peaks_prefix), ec_peaks)
    np.save("{}/cv_pearson_peaks.npy".format(peaks_prefix), ps_peaks)
    np.save("{}/cv_decoding_peaks.npy".format(peaks_prefix), dec_peaks)
    # Save bounds
    np.save("{}/cv_euclidean_bounds.npy".format(bounds_prefix), ec_bounds)
    np.save("{}/cv_pearson_bounds.npy".format(bounds_prefix), ps_bounds)
    np.save("{}/cv_decoding_bounds.npy".format(bounds_prefix), dec_bounds)


if __name__ == "__main__":
    np.random.seed(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file-path', type=str, default="")
    parser.add_argument('--dataset', type=str, default="kaneshiro")
    parser.add_argument('--results-dir', type=str, default="./")
    parser.add_argument('--layer-rdm-dir', type=str, default="./layer_rdm_ps")
    args = parser.parse_args()

    # Data are of shape: (num_subjects, num_images, num_trials, num_components, num_timepoints)
    fname = args.data_file_path
    print "Data file:", fname
    data_matrix = np.load(fname)
    dataset = args.dataset
    base_results_dir = args.results_dir
    results_fname_prefix = "{}/results/{}".format(base_results_dir, dataset)
    rdms_fname_prefix = "{}/rdms/{}".format(base_results_dir, dataset)
    breaks_fname_prefix = "{}/breaks/{}".format(base_results_dir, dataset)
    bounds_fname_prefix = "{}/bounds/{}".format(base_results_dir, dataset)
    peaks_fname_prefix = "{}/peaks/{}".format(base_results_dir, dataset)

    # Data are of shape: (num_subjects, num_images, num_trials, num_components, num_timepoints)
    # Reshape to: (num_subjects, num_images, num_trials, num_timepoints, num_components)
    data_matrix = np.transpose(data_matrix, (0,1,2,4,3))
    layer_rdm_dir = args.layer_rdm_dir

    perform_analyses_across_subj(
        data_matrix,
        layer_rdm_dir,
        results_fname_prefix,
        rdms_fname_prefix,
        breaks_fname_prefix,
        bounds_fname_prefix,
        peaks_fname_prefix,
        dataset
    )


