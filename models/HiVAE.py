from __future__ import annotations
from functools import partial, reduce
from typing import List, Iterable

import torch
import torch.nn as nn
import torch.distributions as dists
import torch.distributions as dist
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.utils import logits_to_probs, probs_to_logits

import pytorch_lightning as pl
import numpy as np
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical

def get_distribution_by_name(name):
    return {
        'normal': Normal, 'lognormal': LogNormal, 'gamma': Gamma, 'exponential': Exponential,
        'bernoulli': Bernoulli, 'poisson': Poisson, 'categorical': Categorical,
        }[name]


class Base(object):
    def __init__(self):
        self._weight = torch.tensor([1.0])
        self.arg_constraints = {}
        self.size = 1

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        if not isinstance(value, torch.Tensor) and isinstance(value, Iterable):
            assert len(value) == 1, value
            value = iter(value)

        self._weight = value if isinstance(value, torch.Tensor) else torch.tensor([value])

    @property
    def expanded_weight(self):
        return reduce(list.__add__, [[w] * len(self[i].f) for i,w in enumerate(self.weight)])

    @property
    def parameters(self):
        return list(self.dist.arg_constraints.keys())

    @property
    def real_parameters(self):
        return self.real_dist.real_parameters if id(self) != id(self.real_dist) else self.parameters

    def __getitem__(self, item):
        assert item == 0
        return self

    def preprocess_data(self, x, mask=None):
        return x,

    def scale_data(self, x, weight=None):
        weight = weight or self.weight
        return x * weight

    def unscale_data(self, x, weight=None):
        weight = weight or self.weight
        return x / weight

    @property
    def f(self):
        raise NotImplementedError()

    def sample(self, size, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).sample(torch.Size([size]))

    def impute(self, etas):
        raise NotImplementedError()
        # real_params = self.to_real_params(etas)
        # real_params = dict(zip(self.real_parameters, real_params))
        # return self.real_dist.dist(**real_params).mean

    def mean(self, etas):
        params = self.to_params(etas)
        params = dict(zip(self.parameters, params))
        return self.dist(**params).mean

    def to_text(self, etas):
        params = self.to_real_params(etas)
        params = [x.cpu().tolist() for x in params]
        params = dict(zip(self.real_parameters, params))
        try:
            mean = self.mean(etas).item()
        except NotImplementedError:
            mean = None

        return f'{self.real_dist} params={params}' + (f' mean={mean}' if mean is not None else '')

    def params_from_data(self, x):
        raise NotImplementedError()

    def real_params_from_data(self, x):
        etas = self.real_dist.params_from_data(x)
        return self.real_dist.to_real_params(etas)

    @property
    def real_dist(self) -> Base:
        return self

    def to_real_params(self, etas):
        return self.to_params(etas)

    @property
    def num_params(self):
        return len(self.arg_constraints)

    @property
    def size_params(self):
        return [1] * self.num_params

    @property
    def num_suff_stats(self):
        return self.num_params

    @property
    def num_dists(self):
        return 1

    def log_prob(self, x, etas):
        params = self.to_params(etas)
        params = dict(zip(self.parameters, params))
        return self.dist(**params).log_prob(x)

    def real_log_prob(self, x, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).log_prob(x)

    @property
    def dist(self):
        raise NotImplementedError()

    def unscale_params(self, etas):
        c = torch.ones_like(etas)
        for i, f in enumerate(self.f):
            c[i].mul_(f(self.expanded_weight[i]).item())
        return etas * c

    def scale_params(self, etas):
        c = torch.ones_like(etas)
        for i, f in enumerate(self.f):
            c[i].mul_(f(self.expanded_weight[i]).item())
        return etas / c

    def __str__(self):
        raise NotImplementedError()

    def to_params(self, etas):
        raise NotImplementedError()

    def to_naturals(self, params):
        raise NotImplementedError()

    @property
    def is_discrete(self):
        raise NotImplementedError()

    @property
    def is_continuous(self):
        return not self.is_discrete

    def __rshift__(self, data):
        return self.scale_data(data)

    def __lshift__(self, etas):
        return self.unscale_params(etas)


class Normal(Base):
    def __init__(self):
        super(Normal, self).__init__()

        self.arg_constraints = [
            constraints.real,  # eta1
            constraints.less_than(0)  # eta2
        ]

    @property
    def is_discrete(self):
        return False

    @property
    def dist(self):
        return dist.Normal

    @property
    def f(self):
        return [lambda w: w, lambda w: w**2]

    def params_from_data(self, x):
        return self.to_naturals([x.mean(), x.std()])

    def to_params(self, etas):
        eta1, eta2 = etas
        return -0.5 * eta1 / eta2, torch.sqrt(-0.5 / eta2)

    def to_naturals(self, params):
        loc, std = params

        eta2 = -0.5 / std ** 2
        eta1 = -2 * loc * eta2

        return eta1, eta2

    def impute(self, etas):
        return self.mean(etas)

    def __str__(self):
        return 'normal'


class LogNormal(Normal):
    def scale_data(self, x, weight=None):
        weight = self.weight if weight is None else weight
        return torch.clamp(torch.pow(x, weight), min=1e-20, max=1e20)

    def unscale_data(self, x, weight=None):
        weight = self.weight if weight is None else weight
        return torch.clamp(torch.pow(x, 1./weight), min=1e-20, max=1e20)

    @property
    def dist(self):
        return dist.LogNormal

    def params_from_data(self, x):
        return super().params_from_data(torch.log(x))

    def sample(self, size, etas):
        return torch.clamp(super().sample(size, etas), min=1e-20, max=1e20)

    def impute(self, etas):
        mu, sigma = self.to_real_params(etas)
        return torch.clamp(torch.exp(mu - sigma**2), min=1e-20, max=1e20)

    def __str__(self):
        return 'lognormal'


class Gamma(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.greater_than(-1),  # eta1
            constraints.less_than(0)  # eta2
        ]

    @property
    def dist(self):
        return dist.Gamma

    @property
    def f(self):
        return [lambda w: torch.ones_like(w), lambda w: w]

    @property
    def is_discrete(self):
        return False

    def params_from_data(self, x):
        mean, meanlog = x.mean(), x.log().mean()
        s = mean.log() - meanlog

        shape = (3 - s + ((s-3)**2 + 24*s).sqrt()) / (12 * s)
        for _ in range(50):
            shape = shape - (shape.log() - torch.digamma(shape) - s) / (1 / shape - torch.polygamma(1, shape))

        concentration = shape
        rate = shape / mean

        eta1 = concentration - 1
        eta2 = -rate

        return eta1, eta2

    def to_params(self, etas):
        eta1, eta2 = etas

        return eta1 + 1, -eta2

    def impute(self, etas):
        alpha, beta = self.to_real_params(etas)
        return torch.clamp((alpha - 1) / beta, min=0.0)

    def __str__(self):
        return 'gamma'


class Exponential(Base):
    def __init__(self):
        super(Exponential, self).__init__()

        self.arg_constraints = [
            constraints.less_than(0)  # eta1
        ]

    @property
    def dist(self):
        return dist.Exponential

    @property
    def is_discrete(self):
        return False

    @property
    def f(self):
        return [lambda w: w]

    def params_from_data(self, x):
        mean = x.mean()
        return -1 / mean,

    def to_params(self, etas):
        return -etas[0],

    def impute(self, etas):
        raise NotImplementedError()

    def __str__(self):
        return "exponential"


class Bernoulli(Base):
    def __init__(self):
        super().__init__()
        self.size = 2
        self.arg_constraints = [
            constraints.real
        ]

    @property
    def dist(self):
        return dist.Bernoulli

    @property
    def is_discrete(self):
        return True

    @property
    def parameters(self):
        return 'logits',

    @property
    def real_parameters(self):
        return 'probs',

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return probs_to_logits(x.mean(), is_binary=True),

    def to_params(self, etas):
        return etas[0],

    def to_real_params(self, etas):
        return logits_to_probs(self.to_params(etas)[0], is_binary=True),

    def impute(self, etas):
        probs = self.to_real_params(etas)[0]

        return (probs >= 0.5).float()

    def __str__(self):
        return 'bernoulli'


class Poisson(Base):
    def __init__(self):
        super().__init__()

        self.arg_constraints = [
            constraints.real
        ]

    @property
    def dist(self):
        return dist.Poisson

    @property
    def is_discrete(self):
        return True

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return torch.log(torch.clamp(x.mean(), min=1e-20)),

    def to_params(self, etas):
        return torch.exp(etas[0]).clamp(min=1e-6, max=1e20),  # TODO

    def impute(self, etas):
        rate = self.to_real_params(etas)[0]
        return rate.floor()

    def __str__(self):
        return 'poisson'


class Categorical(Base):
    def __init__(self, size):
        super().__init__()
        self.arg_constraints = [constraints.real_vector]
        self.size = size

    @property
    def dist(self):
        return dist.Categorical

    @property
    def parameters(self):
        return 'logits',

    @property
    def is_discrete(self):
        return True

    @property
    def real_parameters(self):
        return 'probs',

    @property
    def size_params(self):
        return [self.size]

    def scale_data(self, x, weight=None):
        return x

    @property
    def f(self):
        return [lambda w: torch.ones_like(w)]

    def impute(self, etas):
        real_params = self.to_real_params(etas)
        real_params = dict(zip(self.real_parameters, real_params))
        return self.real_dist.dist(**real_params).probs.max(dim=-1)[1]

    def params_from_data(self, x):
        new_x = to_one_hot(x, self.size)
        return probs_to_logits(new_x.sum(dim=0) / x.size(0)),

    def mean(self, etas):
        raise NotImplementedError()

    def to_params(self, etas):
        return etas[0],

    def to_real_params(self, etas):
        return logits_to_probs(self.to_params(etas)[0]),

    def __str__(self):
        return f'categorical({self.size})'

def _get_distributions(num_vars, x_train) -> List[Base]:
    dists = []
    
    '''
    names = []
    num_vars_temp = num_vars//42
    for i in range(num_vars_temp):
        for j in range(7):
            names += ['normal', 'lognormal', 'normal', 'normal', 'normal', 'poisson']
            
    for i in range(num_vars_temp*42, num_vars):
        if len(torch.unique(x_train[:, i])) == 2:
            names.append('bernoulli')
        else:
            names.append('lognormal')
    '''
    names = num_vars*['normal']
        
    for i in range(num_vars):
        dist_i = get_distribution_by_name(names[i])()
        dists += [dist_i]
        
    return dists


class ProbabilisticModel(object):
    def __init__(self, num_vars, x_train):
        self.dists = _get_distributions(num_vars, x_train)
        self.indexes = reduce(list.__add__, [[[i, j] for j in range(d.num_dists)] for i, d in enumerate(self.dists)])

    def to(self, device):
        for d in self:
            d._weight = d._weight.to(device)
        return self

    @property
    def weights(self):
        return [d.weight for d in self]

    @weights.setter
    def weights(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().tolist()

        for w, d in zip(values, self):
            d.weight = w

    def scale_data(self, x):
        new_x = []
        for i, d in enumerate(self):
            new_x.append(d >> x[:, i])
        return torch.stack(new_x, dim=-1)

    def __rshift__(self, data):
        return self.scale_data(data)

    def params_from_data(self, x, mask):
        params = []
        for i, d in enumerate(self):
            pos = self.gathered_index(i)
            data = x[..., i] if mask is None or mask[..., pos].all() else torch.masked_select(x[..., i], mask[..., pos])
            params += d.params_from_data(data)
        return params

    def preprocess_data(self, x, mask=None):
        new_x = []
        for i, dist_i in enumerate(self.dists):
            new_x += dist_i.preprocess_data(x[:, i], mask)

        for i in range(len(self.dists), x.size(1)):
            new_x += [x[:, i]]

        return torch.stack(new_x, 1)

    def gathered_index(self, index):
        return self.indexes[index][0]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item) -> Base:
        if isinstance(item, int):
            return self.__getitem__(self.indexes[item])

        return self.dists[item[0]][item[1]]

    @property
    def gathered(self):
        class GatherProbabilisticModel(object):
            def __init__(self, model):
                self.model = model

            def __len__(self):
                return len(self.model.dists)

            def __getitem__(self, item):
                offset = sum([d.num_dists for d in self.model.dists[: item]])
                idxs = range(offset, offset + self.model.dists[item].num_dists)

                return idxs, self.model.dists[item]

            @property
            def weights(self):
                return [d.weight for [_, d] in self]

            @weights.setter
            def weights(self, values):
                if isinstance(values, torch.Tensor):
                    values = values.detach().tolist()

                for w, [_, d] in zip(values, self):
                    d.weight = w

            def __iter__(self):
                offset = 0
                for i, d in enumerate(self.model.dists):
                    yield list(range(offset, offset + d.num_dists)), d
                    offset += d.num_dists

            def get_param_names(self):
                names = []
                for i, dist_i in enumerate(self.model.dists):
                    if dist_i.num_dists > 1 or dist_i.size_params[0] > 1:
                        param_name = dist_i.real_parameters[0]
                        num_classes = dist_i.size_params[0] if dist_i.num_dists == 1 else dist_i.num_dists
                        names += [f'{dist_i}_{param_name}{j}_dim{i}' for j in range(num_classes)]
                    else:
                        names += [f'{dist_i}_{v}_dim{i}' for v in dist_i.real_parameters]

                return names

            def scale_data(self, x):
                new_x = []
                for i, [_, d] in enumerate(self):
                    new_x.append(d >> x[:, i])
                return torch.stack(new_x, dim=-1)

            def __rshift__(self, data):
                return self.scale_data(data)

        return GatherProbabilisticModel(self)
        
        
def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()
    return x_one_hot


class GumbelDistribution(ExpRelaxedCategorical):
    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        return OneHotCategorical(probs=self.probs).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return torch.exp(super().rsample(sample_shape))

    @property
    def mean(self):
        return self.probs

    def expand(self, batch_shape, _instance=None):
        return super().expand(batch_shape[:-1], _instance)

    def log_prob(self, value):
        return OneHotCategorical(probs=self.probs).log_prob(value)
        
        
def init_weights(m, gain=1.):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.05)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class Encoder(nn.Module):
    def __init__(self, prob_model, size_s, size_z, input_size):
        super().__init__()
        #input_size = sum([d.size for d in prob_model])

        # Encoder
        self.encoder_s = nn.Linear(input_size, size_s)

        self.encoder_z = nn.Identity()  # Just in case we want to increase this part
        self.q_z_loc = nn.Linear(input_size + size_s, size_z)
        self.q_z_log_scale = nn.Linear(input_size + size_s, size_z)

        self.encoder_z.apply(partial(init_weights))
        self.q_z_loc.apply(init_weights)
        self.q_z_log_scale.apply(init_weights)

        self.temperature = 1.

    def q_z(self, loc, log_scale):
        scale = torch.exp(log_scale)
        scale = torch.clamp(scale, min=1e-6, max=1e6)
        return dists.Normal(loc, scale)

    def q_s(self, logits):
        return GumbelDistribution(logits=logits, temperature=self.temperature)

    def forward(self, x, mode=True):
        s_logits = self.encoder_s(x)
        if mode:
            s_samples = to_one_hot(torch.argmax(s_logits, dim=-1), s_logits.size(-1)).float()
        else:
            s_samples = self.q_s(s_logits).rsample() if self.training else self.q_s(s_logits).sample()

        x_and_s = torch.cat((x, s_samples), dim=-1)  # batch_size x (input_size + latent_s_size)

        h = self.encoder_z(x_and_s)
        z_loc = self.q_z_loc(h)
        z_log_scale = self.q_z_log_scale(h)
        # z_log_scale = torch.clamp(z_log_scale, -7.5, 7.5)

        return s_samples, [s_logits, z_loc, z_log_scale]


class HIVAEHead(nn.Module):
    def __init__(self, dist, size_s, size_z, size_y):
        super().__init__()
        self.dist = dist

        # Generates its own y from z
        self.net_y = nn.Linear(size_z, size_y)

        # First parameter generated with y and s
        self.head_y_and_s = nn.Linear(size_y + size_s, self.dist.size_params[0], bias=False)

        # Next parameters (if any) generated only with s
        self.head_s = None
        if len(self.dist.size_params) > 1:
            self.head_s = nn.Linear(size_s, sum(self.dist.size_params[1:]), bias=False)
            self.head_s.apply(partial(init_weights))

        self.net_y.apply(partial(init_weights))
        self.head_y_and_s.apply(partial(init_weights))

    def unpack_params(self, theta, first_parameter):
        noise = 1e-15

        params = []
        pos = 0
        for i in ([0] if first_parameter else range(1, self.dist.num_params)):
            value = theta[..., pos: pos + self.dist.size_params[i]]
            value = value.squeeze(-1)

            if isinstance(self.dist.arg_constraints[i], constraints.greater_than):
                lower_bound = self.dist.arg_constraints[i].lower_bound
                value = lower_bound + noise + softplus(value)

            elif isinstance(self.dist.arg_constraints[i], constraints.less_than):
                upper_bound = self.dist.arg_constraints[i].upper_bound
                value = upper_bound - noise - softplus(value)

            elif self.dist.arg_constraints[i] == constraints.simplex:
                value = logits_to_probs(value)

            elif self.dist.size > 1:
                value[..., 0] = value[..., 0] * 0.

            params += [value]
            pos += self.dist.size_params[i]

        return torch.stack(params, dim=0)

    def forward(self, z, s):
        y = self.net_y(z)
        y_and_s = torch.cat((y, s), dim=-1)  # batch_size x (hidden_size + latent_s_size)

        raw_params = self.head_y_and_s(y_and_s)  # First parameter
        params = self.unpack_params(raw_params, first_parameter=True)

        if self.head_s is not None:  # Other parameters (if any)
            raw_params = self.head_s(s)
            params_s = self.unpack_params(raw_params, first_parameter=False)
            params = torch.cat((params, params_s), dim=0)

        return params


class HIVAE(pl.LightningModule):
    def __init__(self, x_train):
        super().__init__()
        
        num_vars = x_train.shape[1]
        prob_model = ProbabilisticModel(num_vars, x_train)
        self.prob_model = prob_model
        self.samples = 1
        
        latent_size = max(1, int(len(prob_model.gathered) * 0.75 + 0.5))

        # Parameters for the normalization layers
        self.mean_data = [0. for _ in range(len(prob_model))]
        self.std_data = [1. for _ in range(len(prob_model))]

        # Priors
        self.prior_s_pi = torch.ones(latent_size) / latent_size
        self.p_z_loc = nn.Linear(latent_size, latent_size)
        self.p_z_loc.apply(partial(init_weights))

        # Encoder
        self.encoder = Encoder(prob_model, latent_size, latent_size, num_vars)

        # Decoder
        self.decoder_shared = nn.Identity()  # In case we want to increase this part
        self.decoder_shared.apply(partial(init_weights))
        
        size_y = 200
        self.heads = nn.ModuleList([
            HIVAEHead(dist, latent_size, latent_size, size_y) for dist in prob_model
        ])
    
    def mytrain(self, x_train,  batch_size, n_epochs):
        n_epochs = n_epochs//2
        trainer = pl.Trainer(max_epochs=n_epochs)
        train_dataset = x_train.float()
        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

        # Train
        trainer.fit(self, train_data)

    def prior_z(self, loc):
        return dists.Normal(loc, 1.)

    @property
    def prior_s(self):
        return dists.OneHotCategorical(probs=self.prior_s_pi, validate_args=False)

    def normalize_data(self, x, mask, epsilon=1e-6):
        assert len(self.prob_model) == x.size(-1)
        
        new_x = []
        for i, d in enumerate(self.prob_model):
            x_i = torch.masked_select(x[..., i], mask[..., i].bool()) if mask is not None else x[..., i]
            new_x_i = torch.unsqueeze(x[..., i], 1)

            if str(d) == 'normal':
                self.mean_data[i] = x_i.mean()
                self.std_data[i] = x_i.std()
                self.std_data[i] = torch.clamp(self.std_data[i], 1e-6, 1e6)

                new_x_i = (new_x_i - self.mean_data[i]) / (self.std_data[i] + epsilon)
            elif str(d) == 'lognormal':
                x_i = torch.log1p(x_i)
                self.mean_data[i] = x_i.mean()
                self.std_data[i] = x_i.std()
                self.std_data[i] = torch.clamp(self.std_data[i], 1e-10, 1e20)

                new_x_i = (torch.log1p(new_x_i) - self.mean_data[i]) / (self.std_data[i] + epsilon)
            elif str(d) == 'poisson':
                new_x_i = torch.log1p(new_x_i)  # x[..., i] can have 0 values (just as a poisson distribution)

            #elif 'categorical' in str(d) or 'bernoulli' in str(d):
            #    new_x_i = to_one_hot(torch.squeeze(new_x_i, 1), d.size)


            new_x.append(new_x_i)

        # new_x = torch.stack(new_x, dim=-1)
        new_x = torch.cat(new_x, 1)

        def broadcast_mask(mask, prob_model):
            if all([d.size == 1 for d in prob_model]):
                return mask

            new_mask = []
            for i, d in enumerate(self.prob_model):
                new_mask.append(mask[:, i].unsqueeze(-1).expand(-1, d.size))

            return torch.cat(new_mask, dim=-1)

        mask = broadcast_mask(mask, self.prob_model)

        #if mask is not None:
        #    new_x = new_x * mask
        return new_x

    def denormalize_params(self, etas):
        new_etas = []
        for i, d in enumerate(self.prob_model):
            etas_i = etas[i]

            if str(d) == 'normal':
                mean_data, std_data = self.mean_data[i], self.std_data[i]
                std_data = torch.clamp(std_data, min=1e-3)

                mean, std = d.to_params(etas_i)
                mean = mean * std_data + mean_data
                std = torch.clamp(std, min=1e-3, max=1e20)
                std = std * std_data

                etas_i = d.to_naturals([mean, std])
                etas_i = torch.stack(etas_i, dim=0)
            elif str(d) == 'lognormal':
                mean_data, std_data = self.mean_data[i], self.std_data[i]
                # std_data = torch.clamp(std_data, min=1e-10)

                mean, std = d.to_params(etas_i)
                mean = mean * std_data + mean_data
                # std = torch.clamp(std, min=1e-6) #, max=1)
                std = std * std_data

                etas_i = d.to_naturals([mean, std])
                etas_i = torch.stack(etas_i, dim=0)

            new_etas.append(etas_i)
        return new_etas

    def _run_step(self, x, mask):
        # Normalization layer
        new_x = self.normalize_data(x, mask)

        # Sampling s and obtaining z and s parameters
        s_samples, params = self.encoder(new_x)
        s_logits, z_loc, z_log_scale = params

        # Sampling z
        z = self.encoder.q_z(z_loc, z_log_scale).rsample()

        # Obtaining the parameters of x
        y_shared = self.decoder_shared(z)
        x_params = [head(y_shared, s_samples) for head in self.heads]
        x_params = self.denormalize_params(x_params)  # Denormalizing parameters

        # Compute all the log-likelihoods

        # batch_size x D
        log_px_z = [self.log_likelihood(x, mask, i, params_i) for i, params_i in enumerate(x_params)]

        pz_loc = self.p_z_loc(s_samples)
        log_pz = self.prior_z(pz_loc).log_prob(z).sum(dim=-1)  # batch_size
        log_qz_x = self.encoder.q_z(z_loc, z_log_scale).log_prob(z).sum(dim=-1)  # batch_size
        kl_z = log_qz_x - log_pz

        # batch_size
        try:
            log_ps = self.prior_s.log_prob(s_samples.cpu()).to(s_samples.device)
        except:
            log_ps = self.prior_s.log_prob(s_samples).to(s_samples.device)
            
        log_qs_x = dists.OneHotCategorical(logits=s_logits.cpu(), validate_args=False).log_prob(s_samples.cpu()).to(log_ps.device)
        kl_s = log_qs_x - log_ps

        return log_px_z, kl_z, kl_s

    def _step(self, batch, batch_idx):
        x = batch
        mask = torch.ones_like(x)
        log_px_z, kl_z, kl_s = self._run_step(x, mask)

        elbo = sum(log_px_z) - kl_z - kl_s
        loss = -elbo.sum(dim=0)
        assert loss.size() == torch.Size([])

        logs = dict()
        logs['loss'] = loss / x.size(0)

        with torch.no_grad():
            log_prob = (self.log_likelihood_real(x, mask) * mask).sum(dim=0) / mask.sum(dim=0)
            logs['re'] = -log_prob.mean(dim=0)
            logs['kl_z'] = kl_z.mean(dim=0)
            logs['kl_s'] = kl_s.mean(dim=0)
            logs.update({f'll_{i}': l_i.item() for i, l_i in enumerate(log_prob)})

            if self.training:
                logs['temperature'] = self.encoder.temperature

        return loss, logs

    def training_step(self, batch, batch_idx):
        self.encoder.temperature = max(1e-3, 1. - 0.01 * self.trainer.current_epoch)
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'training/{k}': v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, batch_idx)
        self.log_dict({f'validation/{k}': v for k, v in logs.items()})
        return loss
    
    def get_novelty_score(self, model_fake, data):
        self.eval()
        
        x = data.to('cpu')
        mask = torch.ones_like(x)
        log_px_z, kl_z, kl_s = self._run_step(x, mask)
        elbo = sum(log_px_z) - kl_z - kl_s
        loss = -elbo
        
        #print(log_px_z, sum(log_px_z))
        conf = sum(log_px_z)
        pred = np.zeros(conf.shape)

        return pred, conf.detach().cpu().numpy()

    def _infer_step(self, x, mask, mode):
        new_x = self.normalize_data(x, mask)
        s_samples, params = self.encoder(new_x, mode=mode)
        s_logits, z_loc, z_log_scale = params

        if mode:
            z = z_loc  # Mode of a Normal distribution
        else:
            z = self.encoder.q_z(z_loc, z_log_scale).sample()

        y_shared = self.decoder_shared(z)
        x_params = [head(y_shared, s_samples) for head in self.heads]
        x_params = self.denormalize_params(x_params)  # Denormalizing parameters

        return x_params

    def _impute_step(self, x, mask, mode):
        x_params = self._infer_step(x, mask, mode=mode)

        new_x = []
        for idxs, dist_i in self.prob_model.gathered:
            params = torch.cat([x_params[i] for i in idxs], dim=0)
            new_x_i = dist_i.impute(params).float().flatten()
            if str(dist_i) == 'lognormal':
                # new_x_i = torch.where(new_x_i > 20, new_x_i, new_x_i.expm1().log())
                new_x_i = torch.clamp(new_x_i, 1e-20, 1e20)
            new_x.append(new_x_i)

        return torch.stack(new_x, dim=-1), x_params

    def forward(self, batch, mode=True):
        x, mask, _ = batch
        return self._impute_step(x, mask, mode=mode)[0]

    # Measures
    def log_likelihood(self, x, mask, i, params_i):
        x_i = x[..., i]

        log_prob_i = self.prob_model[i].log_prob(x_i, params_i)
        if mask is not None:
            log_prob_i = log_prob_i * mask[..., i].float()
        return log_prob_i

    def _log_likelihood(self, x, x_params):
        log_prob = []
        for i, [idxs, dist_i] in enumerate(self.prob_model.gathered):
            x_i = x[..., i]

            params = torch.cat([x_params[i] for i in idxs], dim=0)
            log_prob_i = dist_i.real_log_prob(x_i, params)
            log_prob.append(log_prob_i)

        return torch.stack(log_prob, dim=-1).squeeze(dim=0)  # batch_size x num_dimensions

    def log_likelihood_real(self, x, mask):
        x_params = self._infer_step(x, mask, mode=True)
        return self._log_likelihood(x, x_params)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.parameters(), 'lr': 0.001},
        ])

     
        return optimizer

        # We cannot set different schedulers if we want to avoid manual optimization
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'  # Alternatively: "step"
            },
        }
