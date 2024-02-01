"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from torch.optim import Adam, RMSprop, SGD
from tqdm import tqdm

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering
from data_processors.ctgan.data_sampler import DataSampler
from data_processors.ctgan.data_transformer import DataTransformer
from synthetizers.CTGAN.base_ctgan import BaseSynthesizer, random_state
from evaluation.constraints import constraint_satisfaction
# from synthetizers.manual_url_constrained_layer import get_constr_out
import wandb

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim, discrete_cols):
        super(Generator, self).__init__()
        self.discrete_cols = discrete_cols
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)


    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data



class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, test_data, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=True, epochs=300, pac=10, cuda=True, path="", bin_cols_idx=[], version="unconstrained", feats_in_constraints=[]):

        assert batch_size % 2 == 0
        self.feats_in_constraints = feats_in_constraints
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self._path = path
        self._bin_cols_idx = bin_cols_idx
        self._version = version
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.test_data = test_data
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
        for column_info in self._transformer.output_info_list:
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

    def _apply_constrained(self, data):

        self.inverse = self._transformer.inverse_transform(data)

        # if self.args.use_case == 'botnet':
        #         self.inverse = self.inverse.clamp(-1000, 1000)  # TODO: for botnet?

        self.constrained = correct_preds(self.inverse, self.ordering, self.sets_of_constr)

        if self.args.use_case != 'botnet':
            sat = check_all_constraints_sat(self.constrained.clone(), self.constraints)

        #output_transformer = self._transformer.transform(self.constrained, data)
        #self.transformed = self._transformer.transform(self.constrained, data.clone().detach())
        self.transformed = self._transformer.transform(self.constrained, data)

        # for i in range(self.transformed.shape[1]):
        #     print(i)
        #     np.testing.assert_almost_equal(data[:,i].detach().numpy(), self.transformed[:,i].detach().numpy(), decimal=4)


        # get the indices of the feats that appear in constraints, as they are mapped by the transformer into nn space
        # st = 0
        # mask = []
        # for i, column_info in enumerate(self._transformer._column_transform_info_list):
        #     num_dim = column_info.output_dimensions
        #     if i in self.feats_in_constraints:
        #         mask.extend(list(range(st, st+num_dim)))
        #     st += num_dim
        # self.transformed[:, mask] = output_transformer.clone()[:, mask]


        # st = 0
        # ed = 0
        # data_t = []
        # for column_info in self._transformer._column_transform_info_list:
        #     if column_info.column_type=="continuous":
        #         ed = st + column_info.output_dimensions
        #         data_t.append(self.transformed[:,st:st+1])
        #         data_t.append(data[:,st+1:ed])
        #         st = ed
        #     elif column_info.column_type=="discrete":
        #         ed = st + column_info.output_dimensions
        #         data_t.append(self.transformed[:,st:ed])
        #         st = ed
        # only_partial = torch.cat(data_t, dim=1)


        if self._generator.training:
            return self.transformed
            #return only_partial

        else:
            return self.inverse

    def eval_cons_layer(self, columns, global_condition_vec=None):
        self._generator.eval()
        fakez = self.generate_noise(global_condition_vec, self.test_data.shape[0])
        fake = self._generator(fakez)
        generated_data = self._apply_activate(fake)
        if self._version=="constrained":
            generated_data = self._apply_constrained(generated_data)
        else:
            generated_data = self._transformer.inverse_transform(generated_data)

        #lr, ada = batch_eval(generated_data, self.train_data, self.test_data)
        #lr, ada = 0, 0
        features = generated_data.detach().numpy()[:, :-1]
        cons_rate, batch_rate, ind_score = constraint_satisfaction(features,"url")
        self._generator.train()

        return cons_rate, batch_rate, ind_score
    
    def get_sets_constraints(self, label_ordering_choice, constraints_file):
        ordering, constraints = parse_constraints_file(constraints_file)

        # set ordering
        ordering = set_ordering(self.args.use_case, ordering, label_ordering_choice, 'ctgan')

        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        return constraints, sets_of_constr, ordering

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
        

    def get_discrete_col(self):
        discrete_cols = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    #not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    discrete_cols.append((st,ed))
            st += span_info.dim
        return discrete_cols


    @random_state
    def fit(self, args, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

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

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )
        self._transformer = DataTransformer()

        print('Start fit transformer', discrete_columns, train_data.shape)
        # if self.args.use_case == 'lcld' or self.args.use_case == 'botnet':
        #     from utils import read_csv
        #     print('Using a tiny train data set to fit the transformer')
        #     X_train_tiny, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(
        #         f"data/{args.use_case}/tiny/train_data.csv", args.use_case)
        #     self._transformer.fit(X_train_tiny, discrete_columns)
        # else:
        #     self._transformer.fit(train_data, discrete_columns)
        self._transformer.fit(train_data, discrete_columns)

        print('End fit transformer')

        columns = train_data.columns.values.tolist()
        train_data = self._transformer.transform(train_data, None)
        print('End fit data')

        discrete_cols= self.get_discrete_col()
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim, discrete_cols
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        if args.optimiser == "adam":
            optimizerG = Adam(self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9), weight_decay=self._generator_decay)
            optimizerD = Adam(discriminator.parameters(), lr=self._discriminator_lr, betas=(0.5, 0.9), weight_decay=self._discriminator_decay)
        elif args.optimiser == "rmsprop":
            optimizerG = RMSprop(self._generator.parameters(), lr=self._generator_lr, alpha=0.9, momentum=0, eps=1e-3, weight_decay=self._generator_decay)
            optimizerD = RMSprop(discriminator.parameters(), lr=self._discriminator_lr, alpha=0.9, momentum=0, eps=1e-3, weight_decay=self._discriminator_decay)
        elif args.optimiser == "sgd":
            optimizerG = SGD(self._generator.parameters(), lr=self._generator_lr, momentum=0, weight_decay=self._generator_decay)
            optimizerD = SGD(discriminator.parameters(), lr=self._discriminator_lr, momentum=0, weight_decay=self._discriminator_decay)
        else:
            pass

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        loss_g_all, loss_d_syn_all,  loss_d_real_all, loss_d_all = [], [], [], []

        for epoch in range(epochs):
            loss_g_running,  loss_d_syn_running, loss_d_real_running, loss_d_running = 0, 0, 0, 0
            for id_ in tqdm(range(steps_per_epoch), total=steps_per_epoch):
                mean_d = 0
                mean_d_syn = 0
                mean_d_real = 0

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        idx, real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        idx, real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]


                    fake = self._generator(fakez)
                    fake_act = self._apply_activate(fake)
                    if self._version=="unconstrained" or self._version == "postprocessing":
                        fakecons = fake_act.clone()
                    else:
                        fakecons = self._apply_constrained(fake_act)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)
                    if c1 is not None:
                        fake_cat = torch.cat([fakecons, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakecons

                    y_fake = discriminator(fake_cat.squeeze())
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_syn_d = torch.mean(y_fake)
                    loss_real_d = torch.mean(y_real)

                    loss_d = -(loss_real_d - loss_syn_d)

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()
                    mean_d_syn += loss_syn_d
                    mean_d_real += loss_real_d
                    mean_d += loss_d
                    #wandb.log({'steps/1step_disc_real': loss_real_d, 'steps/1step_disc_syn': loss_real_d, 'steps/1step_disc': loss_d})


                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                
                fakeact = self._apply_activate(fake)
                if self._version=="unconstrained" or self._version == "postprocessing":
                    fakecons = fakeact.clone()
                else:
                    fakecons = self._apply_constrained(fakeact)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakecons, c1], dim=1))
                else:
                    y_fake = discriminator(fakecons)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)
                #self.inverse.retain_grad()
                loss_g = -torch.mean(y_fake) + cross_entropy
                optimizerG.zero_grad()
                loss_g.backward()
                #grad = self._generator.seq[-1].weight.grad
                #self.test_gradient(grad)


                optimizerG.step()


                loss_d_syn = mean_d_syn/self._discriminator_steps
                loss_d_real = mean_d_real/self._discriminator_steps
                loss_d = -(loss_d_real - loss_d_syn)
                # wandb.log({'steps/gen_loss': loss_g, 'steps/disc_loss': loss_d})
                loss_g_running += loss_g
                loss_d_syn_running += loss_d_syn
                loss_d_real_running += loss_d_real
                loss_d_running += loss_d

            # loss_g_all.append(loss_g_running.item()/steps_per_epoch)
            # loss_d_syn_all.append(loss_d_syn_running.item()/steps_per_epoch)
            # loss_d_real_all.append(loss_d_real_running.item()/steps_per_epoch)
            # loss_d_all.append(loss_d_running.item()/steps_per_epoch)
            wandb.log({'epochs/epoch': epoch, 'epochs/loss_gen': loss_g_running/steps_per_epoch, 'epochs/loss_disc_syn': loss_d_syn_running/steps_per_epoch, 'epochs/loss_disc_real': loss_d_real_running/steps_per_epoch, 'epochs/loss_disc': loss_d_running/steps_per_epoch})
            #wandb.log({'learning_rates/g_lr': self._generator_lr, 'learning_rates/d_lr': self._discriminator_lr})

            if self._verbose:
                print(f'Epoch {epoch+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)
                
            # cons_rate, batch_rate, ind_score = self.eval_cons_layer(columns)
            # wandb.log({'constraints/mean_ind_score': ind_score.mean(), 'constraints/batch_rate': batch_rate, 'constraints/cons_rate': cons_rate})
            # wandb.log({f'constraints/ind_score_{epoch}': ind_score[epoch] for epoch in range(len(ind_score))})
            
            if epoch >= 25 and epoch % args.save_every_n_epochs == 0:
                torch.save(self._generator, f"{self._path}/model_{epoch}.pt")
                # self.save(f"{self._path}/ctgan_model_{epoch}.pt")

        PATH = f"{self._path}/model.pt"
        torch.save(self._generator, PATH)
        # PATH = f"{self._path}/ctgan_model.pt"
        # self.save(PATH)


    @random_state
    def generate_noise(self,  global_condition_vec, eval_size=None):
        size = self._batch_size
        if eval_size:
            size = eval_size

        mean = torch.zeros(size, self._embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(self._device)

        if global_condition_vec is not None:
            condvec = global_condition_vec.copy()
        else:
            condvec = self._data_sampler.sample_original_condvec(size)

        if condvec is None:
            pass
        else:
            c1 = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)
        return fakez

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            self._generator.eval()
            fakez = self.generate_noise(global_condition_vec)
            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact)
        data = torch.concat(data, axis=0)
        data = data[:n]
        inverse = self._transformer.inverse_transform(data)
        # if self.args.use_case == 'botnet':
        #         inverse = inverse.clamp(-1000, 1000)  # TODO: for botnet?
        unconstrained_output = inverse.clone()
        if self._version == "constrained" or self._version == "postprocessing":
            # inverse = get_constr_out(inverse)
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        return inverse.detach().numpy(), unconstrained_output


    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
