"""SURD decomposition algorithm from the official repository.

Source: https://github.com/ALD-Lab/SURD (MIT License)
Authors: Martínez-Sánchez, Á., Arranz, G., Lozano-Durán, A. (2024)
Paper:  'Decomposing causality into its synergistic, unique, and
        redundant components', Nature Communications, 15, 9296.

I stripped out the matplotlib plotting functions and the pymp parallel
runner because they are not needed in my Streamlit dashboard and pymp
does not work on Windows. The core surd() algorithm is unchanged.
"""

import numpy as np
from itertools import combinations as icmb
from typing import Tuple, Dict

from src.components.surd_lib import it_tools as it

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def surd(p: np.ndarray) -> Tuple[Dict, Dict, Dict, float]:
    """Decompose causality into redundant, unique, and synergistic parts.

    Takes a joint histogram where dimension 0 is the target's future
    and remaining dimensions are agent variables' past states.

    Returns:
        I_R:       Redundancy and unique info per variable combination.
        I_S:       Synergy per variable combination.
        MI:        Mutual information per combination.
        info_leak: Fraction of target uncertainty unexplained by agents.
    """
    # Avoid log(0) and normalise to a proper probability distribution.
    p = p + 1e-14
    p = p / p.sum()

    Ntot = p.ndim
    Nvars = Ntot - 1
    Nt = p.shape[0]
    inds = range(1, Ntot)

    # Information leak: what fraction of the target's future is unexplained.
    H = it.entropy_nvars(p, (0,))
    Hc = it.cond_entropy(p, (0,), range(1, Ntot))
    info_leak = Hc / H

    # Marginal distribution of the target variable.
    p_s = p.sum(axis=(*inds,), keepdims=True)

    # Compute specific mutual information for every subset of agents.
    combs, Is = [], {}

    for i in inds:
        for j in list(icmb(inds, i)):
            combs.append(j)
            noj = tuple(set(inds) - set(j))

            p_a = p.sum(axis=(0, *noj), keepdims=True)
            p_as = p.sum(axis=noj, keepdims=True)

            p_a_s = p_as / p_s
            p_s_a = p_as / p_a

            Is[j] = (p_a_s * (it.mylog(p_s_a) - it.mylog(p_s))).sum(axis=j).ravel()

    # Total mutual information per agent combination.
    MI = {k: (Is[k] * p_s.squeeze()).sum() for k in Is.keys()}

    # Initialise output dictionaries.
    I_R = {cc: 0 for cc in combs}
    I_S = {cc: 0 for cc in combs[Nvars:]}

    # Walk through each target outcome and assign information increments.
    for t in range(Nt):
        I1 = np.array([ii[t] for ii in Is.values()])

        i1 = np.argsort(I1)
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l + 1)[0]
            Il1max = I1[lens == l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        i1 = np.argsort(I1)
        lab = [lab[i_] for i_ in i1]

        Di = np.diff(I1[i1], prepend=0.)
        red_vars = list(inds)

        for i_, ll in enumerate(lab):
            info = Di[i_] * p_s.squeeze()[t]
            if len(ll) == 1:
                I_R[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                I_S[ll] += info

    return I_R, I_S, MI, info_leak


def nice_print(r_, s_, mi_, leak_):
    """Print the normalised redundancy, unique, and synergy values."""
    r_ = {key: value / max(mi_.values()) for key, value in r_.items()}
    s_ = {key: value / max(mi_.values()) for key, value in s_.items()}

    print('    Redundant (R):')
    for k_, v_ in r_.items():
        if len(k_) > 1:
            print(f'        {str(k_):12s}: {v_:5.4f}')

    print('    Unique (U):')
    for k_, v_ in r_.items():
        if len(k_) == 1:
            print(f'        {str(k_):12s}: {v_:5.4f}')

    print('    Synergistic (S):')
    for k_, v_ in s_.items():
        print(f'        {str(k_):12s}: {v_:5.4f}')

    print(f'    Information Leak: {leak_ * 100:5.2f}%')
