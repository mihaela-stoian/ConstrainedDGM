import torch
import numpy as np
import wandb
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.nn import functional
from packaging import version
from tqdm import tqdm

from synthetizers.OCTGAN.utils import calc_gradient_penalty
from synthetizers.OCTGAN.base_octgan import BaseSynthesizer
#from octgan.synthesizers.transformers import BGMTransformer
from synthetizers.OCTGAN.networks import Generator, Discriminator
from data_processors.ctgan.data_sampler import DataSampler
from data_processors.ctgan.data_transformer import DataTransformer

from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering
from evaluation.constraints import constraint_satisfaction

class OCTGAN(BaseSynthesizer):


    def __init__ (self, test_data, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                  lr=2e-4, l2scale=1e-06, batch_size=500, discriminator_steps=1,
                  verbose=True, epochs=300, num_split=3, cuda=True, path="", bin_cols_idx=[], version="unconstrained"):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.lr = lr
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_split = num_split

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path = path
        self.bin_cols_idx = bin_cols_idx
        self.version = version
        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self.device = torch.device(device)

        self.transformer = None
        self.data_sampler = None
        self.generator = None
        self.test_data = test_data
        self.log_frequency = True
        self.verbose = verbose


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

    def apply_activate(self, data):
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

    def apply_constrained(self, data):

        self.inverse = self.transformer.inverse_transform(data)

        # if self.args.use_case == 'botnet':
        #         self.inverse = self.inverse.clamp(-1000, 1000)  # TODO: for botnet?

        self.constrained = correct_preds(self.inverse, self.ordering, self.sets_of_constr)

        if self.args.use_case != 'botnet':
            sat = check_all_constraints_sat(self.constrained.clone(), self.constraints)
        self.transformed = self.transformer.transform(self.constrained, data)
        if self.generator.training:
            return self.transformed
            #return only_partial

        else:
            return self.inverse



    
    def get_sets_constraints(self, label_ordering_choice, constraints_file):
        ordering, constraints = parse_constraints_file(constraints_file)
        # set ordering
        ordering = set_ordering(self.args.use_case, ordering, label_ordering_choice, 'octgan')

        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        return constraints, sets_of_constr, ordering
    
    def cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self.transformer.output_info_list:
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



    def odetime(self, num_split):
        return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True) for
                    i in range(1, num_split)]


    def fit(self, args, train_data, discrete_columns=(), epochs=None):
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = self.get_sets_constraints(args.label_ordering, args.constraints_file)
        self.train = train_data.copy()
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data, None)

        if len(train_data) <= self.batch_size:
            self.batch_size = (len(train_data) // 10)*10

        self.data_sampler = DataSampler(train_data, self.transformer.output_info_list, log_frequency=self.log_frequency)
        data_dim = self.transformer.output_dimensions

        self.generator = Generator(
            self.embedding_dim + self.data_sampler.dim_cond_vec(),
            self.gen_dim,
            data_dim)
        self.generator = nn.DataParallel(self.generator).to(self.device)

        discriminator = Discriminator(
            data_dim + self.data_sampler.dim_cond_vec(),
            self.dis_dim, self.num_split)
        discriminator = nn.DataParallel(discriminator).to(self.device)

        optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        all_time = self.odetime(self.num_split)
        optimizerT = optim.Adam(
            all_time, lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size

        iter = 0

        for epoch in range(self.epochs):
            print("epoch", epoch)
            loss_g_running,  loss_d_syn_running, loss_d_real_running, loss_d_running = 0, 0, 0, 0

            for id_ in tqdm(range(steps_per_epoch), total=steps_per_epoch):
                iter += 1
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.data_sampler.sample_condvec(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    idx, real = self.data_sampler.sample_data(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    idx, real = self.data_sampler.sample_data(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self.apply_activate(fake)
                if self.version=="unconstrained" or self.version == "postprocessing":
                    fakecons = fakeact.clone()
                else:
                    fakecons = self.apply_constrained(fakeact)
                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakecons, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                ######## update discriminator #########
                y_fake = discriminator([fake_cat,all_time])
                y_real = discriminator([real_cat,all_time])


                loss_d_syn = torch.mean(y_fake)
                loss_d_real = torch.mean(y_real)
                loss_d = -(loss_d_real - loss_d_syn)
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, all_time, self.device, lambda_=10)                
                
                loss_d = loss_d + pen 
                optimizerD.zero_grad()
                optimizerT.zero_grad()

                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerT.step()

                # clipping time points t.
                with torch.no_grad():
                    for j in range(len(all_time)):
                        if j == 0:
                            start = 0 + 0.00001
                        else:
                            start = all_time[j - 1].item() + 0.00001

                        if j == len(all_time) - 1:
                            end = 1 - 0.00001
                        else:
                            end = all_time[j + 1].item() - 0.00001
                        all_time[j] = all_time[j].clamp_(min=start, max=end)

                ######### update generator ##########
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.data_sampler.sample_condvec(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self.apply_activate(fake)
                if self.version=="unconstrained" or self.version == "postprocessing":
                    fakecons = fakeact.clone()
                else:
                    fakecons = self.apply_constrained(fakeact)
                if c1 is not None:
                    y_fake = discriminator([torch.cat([fakecons, c1], dim=1), all_time])
                else:
                    y_fake = discriminator([fakecons,all_time])

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self.cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()
                loss_g_running += loss_g
                loss_d_syn_running += loss_d_syn
                loss_d_real_running += loss_d_real
                loss_d_running += loss_d
            wandb.log({'epochs/epoch': epoch, 'epochs/loss_gen': loss_g_running/steps_per_epoch, 'epochs/loss_disc_syn': loss_d_syn_running/steps_per_epoch, 'epochs/loss_disc_real': loss_d_real_running/steps_per_epoch, 'epochs/loss_disc': loss_d_running/steps_per_epoch})

            if self.verbose:
                print(f'Epoch {epoch+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)
                

            if epoch >= 19 and epoch % args.save_every_n_epochs == 0:
                torch.save(self.generator, f"{self.path}/model_{epoch}.pt")

        PATH = f"{self.path}/model.pt"
        torch.save(self.generator, PATH)

    def sample(self, n):

        self.generator.eval()
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.data_sampler.sample_condvec(self.batch_size)
            if condvec is None:
                pass
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self.apply_activate(fake)
            data.append(fakeact)

        data = torch.concat(data, axis=0)
        data = data[:n]
        inverse = self.transformer.inverse_transform(data)
        unconstrained_output = inverse.clone()

        if self.version=="constrained" or self.version == "postprocessing":
            inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)
        return inverse.detach().numpy(), unconstrained_output