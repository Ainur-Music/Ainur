"""
Adapted from https://github.com/gudgud96/frechet-audio-distance
"""
import os

import tensorflow_hub as hub
import torch
import torchaudio.transforms as T
from scipy.linalg import sqrtm
from torch import nn
from torchmetrics import Metric
from tqdm import tqdm

SAMPLE_RATE = 16000

class FAD(Metric):
    def __init__(self, use_pca=False, use_activation=False, verbose=False, model="vggish", path=".tmp/"):
        super().__init__()
        self.__get_model(model, use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.path = path
        self.add_state("embds_lst", [], dist_reduce_fx="cat")
    
    def __get_model(self, model, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.name = model
        # VGGish model
        if model == 'vggish':
            self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
            if not use_pca:
                self.model.postprocess = False
            if not use_activation:
                self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
            self.model.eval()

        # Trill model
        elif model == 'trill':
            self.model = hub.load("https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3")

        # YAMNet model
        elif model == 'yamnet':
            self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        else:
            raise ValueError(f'Specified model "{model}" is not available for evaluation. Try vggish/trill/yamnet.')

    def get_embeddings(self, x, sr=SAMPLE_RATE):
        """
        Get embeddings using VGGish, Trill or YAMNet model.
        Params:
        -- x    : Either 
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of torch.tensor audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []

        for audio in tqdm(x, disable=(not self.verbose)):
            if self.name == 'vggish':
                embd = self.model.forward(audio, sr).clone().detach()
            elif self.name == 'trill':
                embd = self.model(samples=audio, sample_rate=sr)['embedding'] #pre-ReLU output of the first 512-depth convolutional layer (no activation)
                embd = torch.tensor(embd.numpy()).detach()
            elif self.name == 'yamnet':
                _, embd, _ = self.model(audio)
                embd = torch.tensor(embd.numpy()).detach()
            else:
                raise ValueError(f'Specified model "{self.name}" is not available for evaluation. Try vggish/trill/yamnet.')

            embd_lst.append(embd)
        return torch.cat(embd_lst, dim=0)
    
    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list) or isinstance(embd_lst, tuple):
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
        covmean, _ = sqrtm((sigma1 @ sigma2).cpu().float(), disp=False)
        covmean = torch.tensor(covmean)
        if not torch.isfinite(covmean).all():
            msg = ('FAD calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = torch.eye(sigma1.shape[0]) * eps
            covmean = sqrtm(((sigma1 + offset) @ (sigma2 + offset)).cpu().float())
            covmean = torch.tensor(covmean)

        # Numerical error might give slight imaginary component
        if torch.is_complex(covmean):
            if not torch.allclose(torch.diagonal(covmean).imag, torch.tensor(0.), atol=1e-3):
                m = torch.max(torch.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = torch.trace(covmean.float())

        return (diff @ diff + torch.trace(sigma1.float())
                + torch.trace(sigma2.float()) - 2 * tr_covmean)
    
    
    def calculate_embd_statistics_background(self, background=None):
        save_path = os.path.join(self.path, f"background_statistics_{self.name}.ptc")
        
        if os.path.exists(save_path):
            return torch.load(save_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        elif background:
            resample = T.Resample(48_000, SAMPLE_RATE)
            embds_background = self.get_embeddings([torch.mean(resample(sample.detach().squeeze()), dim=0).cpu().numpy()  for sample in tqdm(background, disable=(not self.verbose))])
            if len(embds_background) == 0:
                print("[Frechet Audio Distance] background set is empty, exitting...")
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