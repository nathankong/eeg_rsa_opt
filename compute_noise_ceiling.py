import numpy as np
from scipy.stats import rankdata, spearmanr

def _vectorize_upper_triangular_and_rank(rdms):
    # rdms: (num_subjects, num_images, num_images)
    # Returns vectorized upper triangular portion of rdms
    # Dimensions: (num_subjects, upper triangular length)

    assert rdms.shape[1] == rdms.shape[2]
    num_images = rdms.shape[1]

    num_subjects = rdms.shape[0]
    subj_upr = np.zeros((num_subjects, (num_images*(num_images-1)/2)))
    for i in range(num_subjects):
        curr_rdm = rdms[i,:,:]
        upr = curr_rdm[np.triu_indices(num_images, 1)]
        subj_upr[i,:] = rankdata(upr)
    return subj_upr

def compute_upper_bound(rdms):
    # Correlations: (num_subjects, num_images, num_images)
    # Have to convert to rank to obtain "best" rdm
    vectorized_rdms_ranked = _vectorize_upper_triangular_and_rank(rdms)
    group_avg = vectorized_rdms_ranked.mean(axis=0)
    num_subjects = rdms.shape[0]

    corrs = np.zeros((num_subjects,))
    for i in range(num_subjects):
        corrs[i], _ = spearmanr(group_avg, vectorized_rdms_ranked[i,:])

    return np.mean(corrs)

def compute_lower_bound(rdms):
    # Correlations: (num_subjects, num_images, num_images)
    # Have to convert to rank to obtain "best" rdm
    vectorized_rdms_ranked = _vectorize_upper_triangular_and_rank(rdms)
    num_subjects = rdms.shape[0]

    # Do leave one out
    subject_indices = np.arange(num_subjects)
    corrs = np.zeros((num_subjects,))
    for i in range(num_subjects):
        # Group avg leaves out current subject
        idx = np.delete(subject_indices, i)
        assert idx.shape[0] == num_subjects - 1
        group_avg = vectorized_rdms_ranked[idx,:].mean(axis=0)
        corrs[i], _ = spearmanr(group_avg, vectorized_rdms_ranked[i,:])

    return np.mean(corrs)

def compute_noise_ceiling(rdms):
    # rdms dimension: (num_subjects, num_images, num_images, num_timepoints)
    num_timepoints = rdms.shape[3]
    bounds = np.zeros((num_timepoints,2))
    for t in range(num_timepoints):
        print "Timepoint {}".format(t)
        curr_rdms = rdms[:,:,:,t]
        lwr = compute_lower_bound(curr_rdms)
        upr = compute_upper_bound(curr_rdms)

        bounds[t,0] = lwr
        bounds[t,1] = upr
    return bounds

