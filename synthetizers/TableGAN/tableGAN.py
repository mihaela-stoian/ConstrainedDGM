#!/usr/bin/env python
# coding: utf-8

### Commit 37a217f
#https://github.com/sdv-dev/SDGym/commit/37a217f9bbd5ec7cd09b33b9c067566019caceb9
import numpy as np
import torch
import wandb
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, Tanh, init)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from synthetizers.TableGAN.base_tableGAN import LegacySingleTableBaseline
# from data_processors.tableGAN.utils import TableganTransformer
from data_processors.wgan.tab_scaler import TabScaler
# from synthetizers.manual_url_constrained_layer import get_constr_out
from constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
from constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from constraints_code.parser import parse_constraints_file
from constraints_code.feature_orderings import set_ordering






class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        #self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        #self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        return self.seq(input_)


class Classifier(Module):
    def __init__(self, data_len, side, layers, use_case, device):
        super(Classifier, self).__init__()
        #self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        self.use_case = use_case
        if self.use_case=="news" or self.use_case=="faults":
            self.valid = False
        ### Only handles binary classification otherwise self.valid = False
        # if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
        #     self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        #index = len(self.meta) - 1
        index = data_len -1
    
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    layers_G += [Tanh()]

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

def _apply_constrained(use_case, version, fake, side, transformer, col_len, sets_of_constr, ordering, constraints):
    if version == "constrained":
        fake_re = fake.reshape(-1, side * side)
        fake_re = fake_re[:, :col_len]
        inverse = transformer.inverse_transform(fake_re)
        # cons = get_constr_out(inverse)
        # if use_case == 'botnet':
        #     inverse = inverse.clamp(-1000, 1000)
        cons = correct_preds(inverse, ordering, sets_of_constr)
        if use_case != 'botnet':
            sat = check_all_constraints_sat(cons, constraints)
        # check_all_constraints_sat(cons_layer, self.constraints)
        fake_cons = transformer.transform(cons)

        if side * side > len(fake_cons[0]):
            padding = torch.zeros((len(fake_cons), side**2 - len(fake_cons[0])))
            fake_cons = torch.concat([fake_cons, padding], axis=1)  # TODO: is it right to do the padding like this???

        fake_cons = fake_cons.reshape(-1, 1, side, side)
    else:
        fake_cons = fake.clone()
    return fake_cons

class TableGAN(LegacySingleTableBaseline):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 test_data,
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 verbose = True,
                 epochs=300,
                 path="",
                 bin_cols_idx=[],
                 version="unconstrained"):

        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale
        self._verbose = verbose

        self.batch_size = batch_size
        self.epochs = epochs
        self.version = version
        self.path = path
        self.device = 'cpu' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, args, train_data, discrete_columns_idx):
        self.args = args
        self.constraints, self.sets_of_constr, self.ordering = self.get_sets_constraints(args.label_ordering, args.constraints_file)

        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= train_data.shape[1]:
                self.side = i
                break
        self.transformer = TabScaler(out_min=-1.0, out_max=1.0, one_hot_encode=False)
        #self.transformer = TableganTransformer(self.side)
        train_data = torch.from_numpy(train_data.values.astype('float32')).to(self.device)
        self.transformer.fit(train_data)
        data = self.transformer.transform(train_data)
        if self.side * self.side > len(data[0]):
            padding = torch.zeros((len(data), self.side * self.side - len(data[0])))
            data = torch.concat([data, padding], axis=1)
        data = data.reshape(-1, 1, self.side, self.side)

        #data = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.num_channels)
        self.generator = Generator(None, self.side, layers_G).to(self.device)
        discriminator = Discriminator(None, self.side, layers_D).to(self.device)
        classifier = Classifier(train_data.shape[1], self.side, layers_C, self.args.use_case, self.device).to(self.device)
        if args.optimiser == "adam":
            optimizer_params = dict(lr=args.lr, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
            optimizerG = Adam(self.generator.parameters(), **optimizer_params)
            optimizerD = Adam(discriminator.parameters(), **optimizer_params)
            optimizerC = Adam(classifier.parameters(), **optimizer_params)
        elif args.optimiser == "rmsprop":
            optimizer_params = dict(lr=args.lr, alpha=0.9, momentum=0, eps=1e-3, weight_decay=self.l2scale)
            optimizerG = RMSprop(self.generator.parameters(), **optimizer_params)
            optimizerD = RMSprop(discriminator.parameters(), **optimizer_params)
            optimizerC = RMSprop(classifier.parameters(), **optimizer_params)
        elif args.optimiser == "sgd":
            optimizer_params = dict(lr=args.lr, momentum=0, weight_decay=self.l2scale)
            optimizerG = SGD(self.generator.parameters(), **optimizer_params)
            optimizerD = SGD(discriminator.parameters(), **optimizer_params)
            optimizerC = SGD(classifier.parameters(), **optimizer_params)
        else:
            pass

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)
        classifier.apply(weights_init)

        for epoch in range(self.epochs):
            loss_g_running,  loss_d_running, loss_class_c_running, loss_class_g_running = 0, 0, 0, 0

            for id_, data in tqdm(enumerate(loader), total=len(loader)):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                fake_cons = _apply_constrained(self.args.use_case, self.version, fake, self.side, self.transformer, train_data.shape[1], self.sets_of_constr, self.ordering, self.constraints)
                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake_cons)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                fake_cons = _apply_constrained(self.args.use_case, self.version, fake, self.side, self.transformer, train_data.shape[1], self.sets_of_constr, self.ordering, self.constraints)
                optimizerG.zero_grad()
                y_fake = discriminator(fake_cons)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake_cons, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake_cons, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                fake_cons = _apply_constrained(self.args.use_case, self.version, fake, self.side, self.transformer, train_data.shape[1], self.sets_of_constr, self.ordering, self.constraints)

                if classifier.valid:
                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fake_cons)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_cc = 0
                    loss_cg = 0
                    loss_c = 0
                loss_g_running += loss_g
                loss_d_running += loss_d
                loss_class_c_running += loss_cc
                loss_class_g_running += loss_cg
                # if((id_ + 1) % 1 == 0):
                #     print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g, loss_c)
            wandb.log({'epochs/epoch': epoch, 'epochs/loss_gen': loss_g_running/len(loader), 'epochs/loss_disc_syn': loss_d_running/len(loader), 
                       'epochs/loss_class_c': loss_class_c_running/len(loader), 'epochs/loss_class_g': loss_class_g_running/len(loader)})

            if self._verbose:
                print(f'Epoch {epoch+1}, Loss G: {loss_g.detach().cpu(): .4f}, '  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}, ')  
                if classifier.valid:
                    print(f'Loss classifier: {loss_cc.detach().cpu(): .4f}, {loss_cg.detach().cpu(): .4f}',
                      flush=True)

            if epoch >= 5 and epoch % args.save_every_n_epochs == 0:
                torch.save(self.generator, f"{self.path}/model_{epoch}.pt")

        PATH = f"{self.path}/model.pt"
        torch.save(self.generator, PATH)
        import pickle as pkl
        pkl.dump(self, open(f"{self.path}/model_tablegan.pt", 'wb'), -1)

    def sample_unconstrained(self, n, col_len):
        self.generator.eval()

        steps = n // self.batch_size + 1
        data = []
        uncons_data = []


        for i in range(steps):
            noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            fake_re = fake.reshape(-1, self.side * self.side)
            fake_re = fake_re[:, :col_len]
            inverse = self.transformer.inverse_transform(fake_re.clone())

            unconstrained_output = inverse.clone()
            if self.version == "constrained" or self.version == "postprocessing":
                # inverse = get_constr_out(inverse)
                if self.args.use_case == 'botnet':
                    inverse = inverse.clamp(-1000, 1000)  # TODO: for botnet?
                inverse = correct_preds(inverse, self.ordering, self.sets_of_constr)

            data.append(inverse.detach().cpu().numpy())
            uncons_data.append(unconstrained_output.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)[:n,:col_len]
        uncons_data = np.concatenate(uncons_data, axis=0)[:n,:col_len]
        return data, uncons_data

    def get_sets_constraints(self, label_ordering_choice, constraints_file):
        ordering, constraints = parse_constraints_file(constraints_file)

        # set ordering
        ordering = set_ordering(self.args.use_case, ordering, label_ordering_choice, 'tablegan')

        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        return constraints, sets_of_constr, ordering
