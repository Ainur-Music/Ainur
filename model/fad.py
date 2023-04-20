"""
Adapted from https://github.com/gudgud96/frechet-audio-distance
"""
import os

import torch
import torchaudio.transforms as T
from torch import nn
from torchmetrics import Metric
from tqdm import tqdm

SAMPLE_RATE = 16000

class FAD(Metric):
    def __init__(self, use_pca=False, use_activation=False, verbose=False):
        super().__init__()
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.add_state("embds_lst", [], dist_reduce_fx="cat")
    
    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
    
    def get_embeddings(self, x, sr=SAMPLE_RATE):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of torch.tensor audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        for audio in tqdm(x, disable=(not self.verbose)):
            embd = self.model.forward(audio, sr)
            embd = embd.clone().detach()
            embd_lst.append(embd)
        return torch.cat(embd_lst, dim=0)
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = torch.cat(embd_lst, dim=0)
        mu = torch.mean(embd_lst, dim=0)
        sigma = torch.cov(embd_lst.T)
        return [mu, sigma]
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
        
        Torch implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Tensor containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = torch.atleast_1d(mu1)
        mu2 = torch.atleast_1d(mu2)

        sigma1 = torch.atleast_2d(sigma1)
        sigma2 = torch.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = sqrtm((sigma1 @ sigma2).cpu(), disp=False)
        covmean = torch.tensor(covmean)
        if not torch.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = torch.eye(sigma1.shape[0]) * eps
            covmean, _ = sqrtm(((sigma1 + offset) @ (sigma2 + offset)).cpu())
            covmean = torch.tensor(covmean)

        # Numerical error might give slight imaginary component
        if torch.is_complex(covmean):
            if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0.), atol=1e-3):
                m = torch.max(torch.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = torch.trace(covmean)

        return (diff @ diff + torch.trace(sigma1)
                + torch.trace(sigma2) - 2 * tr_covmean)
    
    
    def calculate_embd_statistics_background(self, background=None, save_path=".tmp/"):
        save_path = os.path.join(save_path, "background_statistics.ptc")
        if os.path.exists(save_path):
            return torch.load(save_path)
        
        if background:
            resample = T.Resample(48_000, SAMPLE_RATE)
            embds_background = self.get_embeddings([torch.mean(resample(sample.detach().squeeze()), dim=0).cpu().numpy()  for sample in tqdm(background, disable=(not self.verbose))])
            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set dir is empty, exitting...")
                return -1
            
            background_statistics = self.calculate_embd_statistics(embds_background)
            torch.save(background_statistics, save_path)
            return background_statistics
        else:
            print(f"Background statistics are not pre-computed in directory '{save_path}' and background is None.\nProvide a background dataset for computing the statistics.")
            return None
    

    def update(self, preds, target=None):
        resample = T.Resample(48_000, SAMPLE_RATE)
        self.embds_lst.append(self.get_embeddings([torch.mean(resample(sample.detach().squeeze()), dim=0).cpu().numpy() for sample in tqdm(preds, disable=(not self.verbose))]))
        self.target = target
        if len(self.embds_lst) == 0:
            print("[Frechet Audio Distance] eval set dir is empty, exitting...")
            return -1
        

            
    def compute(self):
        mu_eval, sigma_eval = self.calculate_embd_statistics(self.embds_lst)
        mu_background, sigma_background = self.calculate_embd_statistics_background(self.target)

        return self.calculate_frechet_distance(
            mu_background, 
            sigma_background, 
            mu_eval, 
            sigma_eval
        )
    

"""
Matrix square root for general matrices and for upper triangular matrices.

This module exists to avoid cyclic imports.

"""
__all__ = ['sqrtm']

import numpy as np

from scipy._lib._util import _asarray_validated


# Local imports
from scipy.linalg._misc import norm
from scipy.linalg.lapack import ztrsyl, dtrsyl
from scipy.linalg._decomp_schur import schur, rsf2csf


class SqrtmError(np.linalg.LinAlgError):
    pass


from scipy.linalg._matfuncs_sqrtm_triu import within_block_loop


def _sqrtm_triu(T, blocksize=64):
    """
    Matrix square root of an upper triangular matrix.

    This is a helper function for `sqrtm` and `logm`.

    Parameters
    ----------
    T : (N, N) array_like upper triangular
        Matrix whose square root to evaluate
    blocksize : int, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `T`

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    """
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0

    # Cast to complex as necessary + ensure double precision
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order="C")
        T_diag = np.asarray(T_diag, dtype=np.complex128)
    else:
        T = np.asarray(T, dtype=np.float64, order="C")
        T_diag = np.asarray(T_diag, dtype=np.float64)

    R = np.diag(np.sqrt(T_diag))

    # Compute the number of blocks to use; use at least one block.
    n, n = T.shape
    nblocks = max(n // blocksize, 1)

    # Compute the smaller of the two sizes of blocks that
    # we will actually use, and compute the number of large blocks.
    bsmall, nlarge = divmod(n, nblocks)
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')

    # Define the index range covered by each block.
    start_stop_pairs = []
    start = 0
    for count, size in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size

    # Within-block interactions (Cythonized)
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e

    # Between-block interactions (Cython would give no significant speedup)
    for j in range(nblocks):
        jstart, jstop = start_stop_pairs[j]
        for i in range(j-1, -1, -1):
            istart, istop = start_stop_pairs[i]
            S = T[istart:istop, jstart:jstop]
            if j - i > 1:
                S = S - R[istart:istop, istop:jstart].dot(R[istop:jstart,
                                                            jstart:jstop])

            # Invoke LAPACK.
            # For more details, see the solve_sylvester implemention
            # and the fortran dtrsyl and ztrsyl docs.
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                x, scale, info = dtrsyl(Rii, Rjj, S)
            else:
                x, scale, info = ztrsyl(Rii, Rjj, S)
            R[istart:istop, jstart:jstop] = x * scale

    # Return the matrix square root.
    return R


def sqrtm(A, disp=True, blocksize=64):
    """
    Matrix square root.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose square root to evaluate
    disp : bool, optional
        Print warning if error in the result is estimated large
        instead of returning estimated error. (Default: True)
    blocksize : integer, optional
        If the blocksize is not degenerate with respect to the
        size of the input array, then use a blocked algorithm. (Default: 64)

    Returns
    -------
    sqrtm : (N, N) ndarray
        Value of the sqrt function at `A`. The dtype is float or complex.
        The precision (data size) is determined based on the precision of
        input `A`. When the dtype is float, the precision is same as `A`.
        When the dtype is complex, the precition is double as `A`. The
        precision might be cliped by each dtype precision range.

    errest : float
        (if disp == False)

        Frobenius norm of the estimated error, ||err||_F / ||A||_F

    References
    ----------
    .. [1] Edvin Deadman, Nicholas J. Higham, Rui Ralha (2013)
           "Blocked Schur Algorithms for Computing the Matrix Square Root,
           Lecture Notes in Computer Science, 7782. pp. 171-182.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import sqrtm
    >>> a = np.array([[1.0, 3.0], [1.0, 4.0]])
    >>> r = sqrtm(a)
    >>> r
    array([[ 0.75592895,  1.13389342],
           [ 0.37796447,  1.88982237]])
    >>> r.dot(r)
    array([[ 1.,  3.],
           [ 1.,  4.]])

    """
    byte_size = np.asarray(A).dtype.itemsize
    A = _asarray_validated(A, check_finite=True, as_inexact=True)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    if blocksize < 1:
        raise ValueError("The blocksize should be at least 1.")
    keep_it_real = np.isrealobj(A)
    if keep_it_real:
        T, Z = schur(A)
        if not np.array_equal(T, np.triu(T)):
            T, Z = rsf2csf(T, Z)
    else:
        T, Z = schur(A, output='complex')
    failflag = False
    try:
        R = _sqrtm_triu(T, blocksize=blocksize)
        ZH = np.conjugate(Z).T
        X = Z.dot(R).dot(ZH)
        if not np.iscomplexobj(X):
            # float byte size range: f2 ~ f16
            X = X.astype(f"f{np.clip(byte_size, 2, 16)}", copy=False)
        else:
            # complex byte size range: c8 ~ c32.
            # c32(complex256) might not be supported in some environments.
            if hasattr(np, 'complex256'):
                X = X.astype(f"c{np.clip(byte_size*2, 8, 32)}", copy=False)
            else:
                X = X.astype(f"c{np.clip(byte_size*2, 8, 16)}", copy=False)
    except SqrtmError:
        failflag = True
        X = np.empty_like(A)
        X.fill(np.nan)

    if disp:
        if failflag:
            print("Failed to find a square root.")

            
    return X
