"""TVAE module."""

import numpy as np
import torch
from packaging import version
from torch.nn import Linear, Module, Parameter, ReLU, Sequential, functional
from torch.nn.functional import cross_entropy, softmax, nll_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering
from data_processors.ctgan.data_sampler import DataSampler
from data_processors.ctgan.data_transformer import DataTransformer
from synthetizers.CTGAN.base_ctgan import BaseSynthesizer, random_state
from evaluation.constraints import constraint_satisfaction
#from synthetizers.manual_url_constrained_layer import get_constr_out


import wandb



class Encoder(Module):
    """Encoder for the TVAE.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma


def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor, version):
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
                loss.append((eq ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                # if version == "constrained":
                #     loss.append(nll_loss(
                #             recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                # else:
                #     loss.append(cross_entropy(
                #         recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                loss.append(cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]



class TVAE(BaseSynthesizer):
    """TVAE."""

    def __init__(
        self,
        test_data,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        path='',
        bin_cols_idx=[], 
        version="unconstrained",
        verbose=True
    ):

        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self._version = version

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        self._path = path
        self._verbose = verbose

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
        # if self.args.use_case == 'botnet':
        #         self.inverse = self.inverse.clamp(-1000, 1000)  # TODO: for botnet?

        self.constrained = correct_preds(self.inverse, self.ordering, self.sets_of_constr)

        # if self.args.use_case != 'botnet':
        #     sat = check_all_constraints_sat(self.constrained.clone(), self.constraints)

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
        #return self.transformed

    @random_state
    def fit(self, args, train_data, discrete_columns=()):
        """Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = self.get_sets_constraints(args.label_ordering, args.constraints_file)

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data, None)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for epoch in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
  
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec_, sigmas = self.decoder(emb)
                rec_.retain_grad()
                if args.version=="constrained":
                    rec_act = self._apply_activate(rec_)
                    rec = self._apply_constrained(rec_act, rec_, sigmas)
                else:
                    rec = rec_.clone()
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar,
                    self.transformer.output_info_list, self.loss_factor, self._version
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

            wandb.log({'epochs/epoch': epoch, 'epochs/loss':loss.item()})
            if self._verbose:
                print(f'Epoch {epoch+1}, Loss: {loss.detach().cpu(): .4f}',flush=True)
                
            # cons_rate, batch_rate, ind_score = self.eval_cons_layer(columns)
            # wandb.log({'constraints/mean_ind_score': ind_score.mean(), 'constraints/batch_rate': batch_rate, 'constraints/cons_rate': cons_rate})
            # wandb.log({f'constraints/ind_score_{epoch}': ind_score[epoch] for epoch in range(len(ind_score))})
            
            if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
                torch.save(self.decoder, f"{self._path}/model_{epoch}.pt")

        PATH = f"{self._path}/model.pt"
        torch.save(self.decoder, PATH)


    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake_act = self._apply_activate(fake)
            data.append(fake_act)
        data = torch.concat(data, axis=0)
        data = data[:samples]
        inverse = self.transformer.inverse_transform(data, None)

        unconstrained_output = inverse.clone()
        if self._version == "constrained" or self._version == "postprocessing":
            # inverse = get_constr_out(inverse)
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        return inverse.detach().numpy(), unconstrained_output

        # unconstrained_output = inverse.clone()
        # if self._version=="constrained" or self._version=="postprocessing":
        #     self.constrained = correct_preds(inverse, self.ordering, self.sets_of_constr)
        #     fake_cons = self.transformer.transform(self.constrained, fake_act)
        # else: 
        #     fake_cons = fake_act.clone()
        # ##Commented out because this is a bug.  Tanh should not be applied to binary features
        # #fake = torch.tanh(fake)
        # data.append(fake_cons.detach().cpu())
        # return self.transformer.inverse_transform(data, sigmas).detach().cpu().numpy(), unconstrained_output

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)