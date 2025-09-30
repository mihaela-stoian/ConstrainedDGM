import torch
from torch.nn.functional import cross_entropy

from sklearn.neighbors import NearestNeighbors

from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder



def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


def _loss_function_MMD(version, recon_x, x, sigmas, mean, std, output_info, factor, kernel_choice='rbf'):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                if version == "constrained":
                    eq = x[:, st] - recon_x[:, st]
                else:
                    eq = x[:, st] - torch.tanh(recon_x[:, st])
                # eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]

    eps = torch.randn_like(std)
    z = eps * std + mean

    N = z.shape[0]

    z_prior = torch.randn_like(z)#.to(device)

    if kernel_choice == "rbf":
        k_z = rbf_kernel(z, z)
        k_z_prior = rbf_kernel(z_prior, z_prior)
        k_cross = rbf_kernel(z, z_prior)

    else:
        k_z = imq_kernel(z, z)
        k_z_prior = imq_kernel(z_prior, z_prior)
        k_cross = imq_kernel(z, z_prior)

    mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
    mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
    mmd_cross = k_cross.sum() / (N ** 2)

    mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

    return sum(loss) * factor / x.size()[0] + mmd_loss

def imq_kernel(z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    kernel_bandwidth = 1.0
    scales = 1.0
    latent_dim=z1.shape[1]
    Cbase = (2.0 * latent_dim * kernel_bandwidth ** 2)

    k = 0

    for scale in scales:
        C = scale * Cbase
        k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)

    return k

def rbf_kernel(z1, z2):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
    kernel_bandwidth = 1.0
    latent_dim=z1.shape[1]
    C = 2.0 * latent_dim * kernel_bandwidth ** 2

    k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)

    return k


def z_gen(embeddings,n_to_sample,metric='minkowski',interpolation_method='SMOTE'):
    # fitting the model
    n_neighbors = 5 +1
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1,metric= metric)
    nn.fit(embeddings)
    dist, ind = nn.kneighbors(embeddings)

    # generating samples
    base_indices = np.random.choice(list(range(len(embeddings))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neighbors)),n_to_sample)

    embeddings_base = embeddings[base_indices]

    if interpolation_method =='SMOTE': ## randomly generate synthetic latent point between 2 real latent points
      embeddings_neighbor = embeddings[ind[base_indices, neighbor_indices]]
      deviations = np.multiply(np.random.rand(n_to_sample,1), embeddings_neighbor - embeddings_base)

      embeddings_samples = embeddings_base + deviations

    elif interpolation_method == 'rectangle':
      embeddings_neighbor = embeddings[ind[base_indices, neighbor_indices]]
      embeddings_samples = np.random.uniform()*embeddings_neighbor + (1-np.random.uniform())*embeddings_base

    elif interpolation_method == 'triangle': ## permutation all latent points in one neighborhood based on their inverse distance as weight
      deviations=0
      for i in range(1,n_neighbors):
        embeddings_neighbor = embeddings[ind[base_indices, i]]
        weight = (n_neighbors -i)/(n_neighbors*(n_neighbors-1)/2)
        deviation = np.multiply(np.random.rand(n_to_sample,1), embeddings_neighbor - embeddings_base)
        deviations += weight*deviation
        embeddings_samples = embeddings_base + deviations

    else:
      mean = torch.zeros(n_to_sample, embeddings.shape[1])
      std = mean + 1
      embeddings_samples = reparameterize(mean, std)

    return embeddings_samples