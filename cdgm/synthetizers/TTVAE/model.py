from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import os
import time
from packaging import version

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, functional
from torch.nn.functional import cross_entropy, softmax
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.autograd import Variable
# from ttvae.util import DataTransformer, reparameterize, _loss_function_MMD,z_gen
from ttvae.util import reparameterize, _loss_function_MMD, z_gen

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering


from data_processors.ctgan.data_sampler import DataSampler
from data_processors.ctgan.data_transformer import DataTransformer
# from evaluation.constraints import constraint_satisfaction
# from ttvae.base import BaseSynthesizer, random_state
from base_ctgan import BaseSynthesizer, random_state



class Encoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Encoder_T, self).__init__()
      # Input data to Transformer
      self.linear = nn.Linear(input_dim,embedding_dim)
      # Transformer Encoder
      self.transformerencoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.encoder = nn.TransformerEncoder(self.transformerencoder_layer, num_layers=2)
      # Latent Space Representation
      self.fc_mu = nn.Linear(embedding_dim, latent_dim)
      self.fc_log_var = nn.Linear(embedding_dim, latent_dim)

    def forward(self, x):
      # Encoder
      x = self.linear(x)
      enc_output = self.encoder(x)
      # Latent Space Representation
      mu = self.fc_mu(enc_output)
      logvar = self.fc_log_var(enc_output)
      std = torch.exp(0.5 * logvar)
      return mu, std, logvar, enc_output


class Decoder_T(nn.Module):
    def __init__(self, input_dim, latent_dim, embedding_dim, nhead, dim_feedforward=2048, dropout=0.1):
      super(Decoder_T, self).__init__()
      # Linear layer for mapping latent space to decoder input size
      self.latent_to_decoder_input = nn.Linear(latent_dim, embedding_dim)
      # Transformer Decoder
      self.transformerdecoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward, dropout)
      self.decoder = nn.TransformerDecoder(self.transformerdecoder_layer, num_layers=2)
      # Transformer Embedding to input
      self.linear = nn.Linear(embedding_dim,input_dim)
      self.sigma = Parameter(torch.ones(input_dim) * 0.1)

    def forward(self, z, enc_output):
      # Encoder
      z_decoder_input = self.latent_to_decoder_input(z)
      # Decoder
      # Note: Pass enc_output (memory) to the decoder
      dec_output = self.decoder(z_decoder_input, enc_output)

      return self.linear(dec_output), self.sigma


class TTVAE(BaseSynthesizer):
    """TTVAE."""

    def __init__(
        self,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        latent_dim =32,# Example latent dimension
        embedding_dim=128,# Transformer embedding dimension
        nhead=8,# Number of attention heads
        dim_feedforward=1028,# Feedforward layer dimension
        dropout=0.1,
        cuda=False,
        verbose=False,
        device='cuda',
        version = 'unconstrained',  # 'unconstrained', 'constrained', 'postprocessing'

    ):
        self.latent_dim=latent_dim
        self.embedding_dim = embedding_dim
        self.nhead=nhead
        self.dim_feedforward=dim_feedforward
        self.dropout=dropout
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._device = torch.device(device)
        self._verbose = verbose
        self.version = version

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse('1.2.0'):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError('gumbel_softmax returning NaN.')

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2, hard=True)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    def get_sets_constraints(self, label_ordering_choice, constraints_file):
            ordering, constraints = parse_constraints_file(constraints_file)
            # set ordering
            ordering = set_ordering(self.args.use_case, ordering, label_ordering_choice, 'tvae')

            sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
            return constraints, sets_of_constr, ordering
    
    def _apply_constrained(self, dec_data, data_bef, sigmas):

        self.inverse = self.transformer.inverse_transform(dec_data, None)
        self.constrained = correct_preds(self.inverse, self.ordering, self.sets_of_constr)
        self.transformed = self.transformer.transform(self.constrained, dec_data)

        data_t = []
        st = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(self.transformed[:, st:ed])
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    data_t.append(data_bef[:, st:ed])
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        data = torch.cat(data_t, dim=1)
        return data

    # @random_state
    def fit(self, args, train_data, discrete_columns=(),save_path=''):
        self.train_data = train_data
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = self.get_sets_constraints(args.label_ordering, args.constraints_file)

        self.transformer = DataTransformer()
        self.transformer.fit(self.train_data, discrete_columns)
        self.train_data = self.transformer.transform(self.train_data, None).astype('float32')

        dataset = TensorDataset(torch.from_numpy(self.train_data))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dimensions
        print(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout)

        self.encoder = Encoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout)
        self.decoder = Decoder_T(data_dim, self.latent_dim, self.embedding_dim, self.nhead, self.dim_feedforward, self.dropout)

        optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

        self.encoder.train()
        self.decoder.train()

        best_loss = float('inf')
        patience = 0
        start_time = time.time()

        for epoch in range(self.epochs):
        
            pbar = tqdm(loader, total=len(loader))
            pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

            batch_loss = 0.0
            len_input = 0

            for id_, data in enumerate(pbar):
                # print("Batch: ", id_)
                optimizer.zero_grad()
                real_x = data[0]
                mean, std, logvar, enc_output = self.encoder(real_x)
                z = reparameterize(mean, logvar)
                recon_x, sigmas = self.decoder(z,enc_output)

                recon_x.retain_grad()
                if self.version == "constrained":
                    rec_act = self._apply_activate(recon_x)
                    rec = self._apply_constrained(rec_act, recon_x, sigmas)
                else:
                    rec = recon_x.clone()
                loss = _loss_function_MMD(self.version, rec, real_x, sigmas, mean, logvar, self.transformer.output_info_list, self.loss_factor)

                batch_loss += loss.item() * len(real_x)
                len_input += len(real_x)

                loss.backward()
                optimizer.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                # pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss/len_input
            scheduler.step(curr_loss)
            
            wandb.log({'epochs/epoch': epoch, 'epochs/loss':curr_loss})
            if self._verbose:
                print(f'Epoch {epoch+1}, Loss: {curr_loss: .4f}',flush=True)
                
            # cons_rate, batch_rate, ind_score = self.eval_cons_layer(columns)
            # wandb.log({'constraints/mean_ind_score': ind_score.mean(), 'constraints/batch_rate': batch_rate, 'constraints/cons_rate': cons_rate})
            # wandb.log({f'constraints/ind_score_{epoch}': ind_score[epoch] for epoch in range(len(ind_score))})
            
            # if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
            #     torch.save(self.decoder, f"{self._path}/model_{epoch}.pt")

            if curr_loss < best_loss:
              best_loss = loss.item()
              patience = 0
            #   torch.save(self.decoder, save_path+'/model.pt')
            else:
                patience += 1
                if patience == 500:
                    print('Early stopping')
                    break
                        
    # @random_state
    def sample(self,  n_samples=100):
        """Sample data similar to the training data.

        """
        self.encoder.eval()
        steps = n_samples // self.batch_size + 1
        data = []

        if self.args.use_case == 'lcld':

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.Tensor(self.train_data)),
                batch_size=self.batch_size,
                shuffle=False)

            all_means = []
            all_stds = []
            all_enc_embeds = []

            with torch.no_grad():
                for batch_x, in train_loader:
                    mean, std, logvar, enc_embed = self.encoder(batch_x)
                    all_means.append(mean)
                    all_stds.append(std)
                    all_enc_embeds.append(enc_embed)

            all_means = torch.cat(all_means, dim=0)
            all_stds = torch.cat(all_stds, dim=0)
            all_enc_embeds = torch.cat(all_enc_embeds, dim=0)

        else:
            with torch.no_grad():
                mean, std, logvar, enc_embed = self.encoder(torch.Tensor(self.train_data))

        for _ in range(steps):
            if self.args.use_case == 'lcld':

                idx = torch.randint(0, all_means.size(0), (self.batch_size,))
                mean = all_means[idx]
                std = all_stds[idx]
                enc_embed = all_enc_embeds[idx]

            embeddings = torch.normal(mean=mean, std=std).cpu().detach().numpy()
            synthetic_embeddings=z_gen(embeddings,n_to_sample=self.batch_size, metric='minkowski',interpolation_method='SMOTE')
            noise = torch.Tensor(synthetic_embeddings)

            self.decoder.eval()
            with torch.no_grad():
                fake, sigmas = self.decoder(noise,enc_embed)
                fake_act = self._apply_activate(fake)
                data.append(fake_act)
            # fake = torch.tanh(fake).cpu().detach().numpy()

        data = torch.concat(data, axis=0)
        data = data[:n_samples]
        inverse = self.transformer.inverse_transform(data, None)

        unconstrained_output = inverse.clone()
        if self.version == "constrained" or self.version == "postprocessing":
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        return inverse.detach().numpy(), unconstrained_output

        # return self.transformer.inverse_transform(fake)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        device = "cpu"
        self._device = device
        self.decoder.to(self._device)