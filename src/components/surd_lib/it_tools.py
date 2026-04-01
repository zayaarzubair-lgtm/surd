"""Information theory tools from the official SURD repository.

Source: https://github.com/ALD-Lab/SURD (MIT License)
Authors: Martínez-Sánchez, Á., Arranz, G., Lozano-Durán, A. (2024)

I have not modified the algorithm logic in this file. I only added
this attribution header and removed the pymp import.
"""

import numpy as np


def myhistogram(x, nbins):
    """Build a histogram and normalise it to a probability distribution."""
    hist, _ = np.histogramdd(x, nbins)
    hist += 1e-14
    hist /= hist.sum()
    return hist


def mylog(x):
    """Compute log base 2, returning 0 for zero or invalid inputs."""
    valid_indices = (x != 0) & (~np.isnan(x)) & (~np.isinf(x))
    log_values = np.zeros_like(x)
    log_values[valid_indices] = np.log2(x[valid_indices])
    return log_values


def entropy(p):
    """Shannon entropy of a probability distribution, in bits."""
    return -np.sum(p * mylog(p))


def entropy_nvars(p, indices):
    """Joint entropy for specific dimensions of a probability distribution."""
    excluded_indices = tuple(set(range(p.ndim)) - set(indices))
    marginalized_distribution = p.sum(axis=excluded_indices)
    return entropy(marginalized_distribution)


def cond_entropy(p, target_indices, conditioning_indices):
    """Conditional entropy H(target | conditioning)."""
    joint_entropy = entropy_nvars(p, set(target_indices) | set(conditioning_indices))
    conditioning_entropy = entropy_nvars(p, conditioning_indices)
    return joint_entropy - conditioning_entropy


def mutual_info(p, set1_indices, set2_indices):
    """Mutual information I(set1 ; set2)."""
    entropy_set1 = entropy_nvars(p, set1_indices)
    conditional_entropy = cond_entropy(p, set1_indices, set2_indices)
    return entropy_set1 - conditional_entropy


def cond_mutual_info(p, ind1, ind2, ind3):
    """Conditional mutual information I(ind1 ; ind2 | ind3)."""
    combined_indices = tuple(set(ind2) | set(ind3))
    return cond_entropy(p, ind1, ind3) - cond_entropy(p, ind1, combined_indices)


def transfer_entropy(p, target_var):
    """Transfer entropy from each input variable to the target."""
    num_vars = len(p.shape) - 1
    TE = np.zeros(num_vars)

    for i in range(1, num_vars + 1):
        present_indices = tuple(range(1, num_vars + 1))
        conditioning_indices = tuple(
            [target_var] + [j for j in range(1, num_vars + 1) if j != i and j != target_var]
        )
        cond_ent_target_given_past = cond_entropy(p, (0,), conditioning_indices)
        cond_ent_target_given_past_and_input = cond_entropy(p, (0,), present_indices)
        TE[i - 1] = cond_ent_target_given_past - cond_ent_target_given_past_and_input

    return TE
