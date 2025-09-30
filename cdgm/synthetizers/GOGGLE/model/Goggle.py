# 3rd party
import torch
from torch import nn

# Goggle
from cdgm.synthetizers.GOGGLE.model.Encoder import Encoder
from cdgm.synthetizers.GOGGLE.model.GraphDecoder import GraphDecoderHet, GraphDecoderHomo
from cdgm.synthetizers.GOGGLE.model.GraphInputProcessor import (
    GraphInputProcessorHet,
    GraphInputProcessorHomo,
)
from cdgm.synthetizers.GOGGLE.model.LearnedGraph import LearnedGraph
from cdgm.constraints_code.correct_predictions import correct_preds


class Goggle(nn.Module):
    def __init__(
        self,
        input_dim,
        column_transformer=None,
        constrained=False,
        constraints=None, sets_of_constr=None, ordering=None,
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=64,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device="cpu",
    ):
        super(Goggle, self).__init__()
        self.input_dim = input_dim
        self.column_transformer = column_transformer
        self.constrained = constrained
        self.constraints = constraints
        self.sets_of_constr = sets_of_constr
        self.ordering = ordering
        self.device = device
        self.learned_graph = LearnedGraph(
            input_dim, graph_prior, prior_mask, threshold, device
        )
        self.encoder = Encoder(input_dim, encoder_dim, encoder_l, device)
        if decoder_arch == "het":
            n_edge_types = input_dim * input_dim
            self.graph_processor = GraphInputProcessorHet(
                input_dim, decoder_dim, n_edge_types, het_encoding, device
            )
            self.decoder = GraphDecoderHet(decoder_dim, decoder_l, n_edge_types, device)
        else:
            self.graph_processor = GraphInputProcessorHomo(
                input_dim, decoder_dim, het_encoding, device
            )
            self.decoder = GraphDecoderHomo(
                decoder_dim, decoder_l, decoder_arch, device
            )

    def forward(self, x, iter):
        z, (mu_z, logvar_z) = self.encoder(x)
        b_size, _ = z.shape
        adj = self.learned_graph(iter)
        graph_input = self.graph_processor(z, adj)
        x_hat = self.decoder(graph_input, b_size)
        if self.constrained:
            initial_x_hat = x_hat.clone()
            x_hat = self.apply_CL(initial_x_hat)
            # print((initial_x_hat - x_hat)[0])
        return x_hat, adj, mu_z, logvar_z

    def apply_CL(self, x, training=True):  # to be called only at training time
        if training:
            x_inverse = self.column_transformer.inverse_transform(x)
            x_constrained = correct_preds(x_inverse, self.ordering, self.sets_of_constr)
            x_transformed = self.column_transformer.transform(x_constrained)
            return x_transformed
        else:
            # assumes x is inversed already (i.e. in the original space)
            x_constrained = correct_preds(x, self.ordering, self.sets_of_constr)
            return x_constrained

    def sample(self, count):
        with torch.no_grad():
            mu = torch.zeros(self.input_dim)
            sigma = torch.ones(self.input_dim)
            q = torch.distributions.Normal(mu, sigma)
            z = q.rsample(sample_shape=torch.Size([count])).squeeze().to(self.device)

            self.learned_graph.eval()
            self.graph_processor.eval()
            self.decoder.eval()

            adj = self.learned_graph(100)
            graph_input = self.graph_processor(z, adj)
            synth_x = self.decoder(graph_input, count)
            # if self.constrained:
            #     unconstrained_synth_x = synth_x.clone()
            #     synth_x = self.apply_CL(unconstrained_synth_x, training=False)
            # else:
            #     unconstrained_synth_x = synth_x
        # return synth_x, unconstrained_synth_x
        return synth_x
