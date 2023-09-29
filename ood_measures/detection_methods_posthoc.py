"""
All the post-hoc detectors are defined separately in this file, and function 'detection_method' calls each of them based on their name. 
The density-based models are similarly called in this function.
"""

from __future__ import division, print_function

from models.predictive_models import apply_model
from models.AE import AE
from models.VAE import VAE
from models.flow import Flow
from models.ppca import PPCA
from models.lof import LOF
from models.due import DUE
from models.HiVAE import HIVAE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
from numpy.linalg import norm, pinv
import scipy
from scipy.special import logsumexp
import scipy.spatial.distance as spd

import faiss
import libmr
from tqdm import tqdm
from typing import Any
from copy import deepcopy

import sklearn.covariance
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min
from openood.preprocessors.transform import normalization_dict

@torch.no_grad()
def MSP(model, data):
    logits, features = apply_model(model, data, return_features=True)
    conf, pred = F.softmax(logits, dim=1).max(dim=1)
    return pred.cpu().numpy(), conf.cpu().numpy()

###########

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNPostprocessor:
    def __init__(self, k, device):
        self.K = k
        self.setup_flag = False
        self.device = device

    def setup(self, net, x_train, batch_size):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    _, feature = apply_model(net, data, return_features=True)
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.index = faiss.IndexFlatL2(feature.shape[1])
            self.index.add(self.activation_log)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net, data):
        output, feature = apply_model(net, data, return_features=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred.cpu().numpy(), torch.from_numpy(kth_dist).cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K


###########

class OpenMax:
    def __init__(self, num_classes, device):
        self.nc = num_classes
        self.weibull_alpha = 2
        self.weibull_threshold = 0.9
        self.weibull_tail = 20
        self.setup_flag = False
        self.device=device

    def setup(self, net, x_train, y_train, batch_size):
        if not self.setup_flag:
            # Fit the weibull distribution from training data.
            print('Fittting Weibull distribution...')
            _, mavs, dists = compute_train_score_and_mavs_and_dists(
                self.nc, x_train, y_train, device=self.device, net=net, batch_size=batch_size)
            categories = list(range(0, self.nc))
            self.weibull_model = fit_weibull(mavs, dists, categories,
                                             self.weibull_tail, 'euclidean')
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data):
        net.eval()
        scores = apply_model(net, data, return_features=False).cpu().numpy()
        scores = np.array(scores)[:, np.newaxis, :]
        categories = list(range(0, self.nc))

        pred_openmax = []
        score_openmax = []
        for score in scores:
            so, _ = openmax(self.weibull_model, categories, score, 0.5,
                            self.weibull_alpha,
                            'euclidean')  # openmax_prob, softmax_prob
            pred_openmax.append(
                np.argmax(so) if np.max(so) >= self.weibull_threshold else (
                    self.nc - 1))

            score_openmax.append(so)

        pred = torch.tensor(pred_openmax)
        conf = -1 * torch.from_numpy(np.array(score_openmax))[:, -1]

        return pred.cpu().numpy(), conf.cpu().numpy()


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV
        for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append(
            [spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([
            spd.euclidean(mcv, feat[channel]) * eu_weight +
            spd.cosine(mcv, feat[channel]) for feat in features
        ])

    return {
        'eucos': np.array(eucos_dists),
        'cosine': np.array(cos_dists),
        'euclidean': np.array(eu_dists)
    }


def compute_train_score_and_mavs_and_dists(train_class_num, x_train, y_train,
                                           device, net, batch_size):
    scores = [[] for _ in range(train_class_num)]

    with torch.no_grad():
        #for train_step in tqdm(range(1, len(train_dataiter) + 1),
                               #desc='Progress: ',
                               #position=0,
                               #leave=True):
        for batch_idx in range(len(x_train) // batch_size):
            data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
            target = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)

            # this must cause error for cifar
            outputs = apply_model(net, data, return_features=False)
            for score, t in zip(outputs, target):

                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [
        compute_channel_distances(mcv, score)
        for mcv, score in zip(mavs, scores)
    ]
    return scores, mavs, dists


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances
                        and save weibull model parameters for re-adjusting
                        softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(
            distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su)).clip(min=-1e10, max=1e10)

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [
        weibull_model[category_name]['mean_vec'],
        weibull_model[category_name]['distances_{}'.format(distance_type)],
        weibull_model[category_name]['weibull_model']
    ]


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print('distance type not known: enter either of eucos, \
               euclidean or cosine')
    return query_distance


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model,
            categories,
            input_score,
            eu_weight,
            alpha=10,
            distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [((alpha + 1) - i) / float(alpha)
                     for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model,
                                             distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel],
                                         eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob
    
###########


class MDSPostprocessor():
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.setup_flag = False

    def setup(self, net, x_train, y_train, batch_size):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    labels = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    #data, labels = batch['data'].cuda(), batch['label']
                    logits, features = apply_model(net, data, return_features=True)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats).cpu()
            all_labels = torch.cat(all_labels).cpu()
            all_preds = torch.cat(all_preds).cpu()
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # compute class-conditional statistics
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = apply_model(net, data, return_features=True)
        pred = logits.argmax(1)

        class_scores = torch.zeros((logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()

        conf = torch.max(class_scores, dim=1)[0]
        return pred.cpu().numpy(), conf.cpu().numpy()
        
###########

class RMDSPostprocessor():
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.setup_flag = False

    def setup(self, net, x_train, y_train, batch_size):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    labels = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    logits, features = apply_model(net, data, return_features=True)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats).cpu()
            all_labels = torch.cat(all_labels).cpu()
            all_preds = torch.cat(all_preds).cpu()
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            # compute class-conditional statistics
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples -
                                     self.class_mean[c].view(1, -1))

            self.class_mean = torch.stack(
                self.class_mean)  # shape [#classes, feature dim]

            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(
                torch.cat(centered_data).cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_).float()

            self.whole_mean = all_feats.mean(0)
            centered_data = all_feats - self.whole_mean.view(1, -1)
            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            self.whole_precision = torch.from_numpy(
                group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = apply_model(net, data, return_features=True)
        pred = logits.argmax(1)

        tensor1 = features.cpu() - self.whole_mean.view(1, -1)
        background_scores = -torch.matmul(
            torch.matmul(tensor1, self.whole_precision), tensor1.t()).diag()

        class_scores = torch.zeros((logits.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()
            class_scores[:, c] = class_scores[:, c] - background_scores

        conf = torch.max(class_scores, dim=1)[0]
        return pred.cpu().numpy(), conf.cpu().numpy()

###########

class TemperatureScalingPostprocessor():
    """A decorator which wraps a model with temperature scaling, internalize
    'temperature' parameter as part of a net model."""
    def __init__(self, device):
        self.temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)  # initialize T
        self.setup_flag = False
        self.device = device

    def setup(self, net: nn.Module, x_val, y_val, batch_size):
        if not self.setup_flag:
            nll_criterion = nn.CrossEntropyLoss().to(self.device)

            logits_list = []  # fit in whole dataset at one time to back prop
            labels_list = []
            with torch.no_grad(
            ):  # fix other params of the net, only learn temperature
                for batch_idx in range(len(x_val) // batch_size):
                    data = x_val[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    labels = y_val[batch_idx*batch_size:(batch_idx+1)*batch_size]

                    logits = apply_model(net, data, return_features=False)
                    logits_list.append(logits)
                    labels_list.append(labels)
                # convert a list of many tensors (each of a batch) to one tensor
                logits = torch.cat(logits_list).to(self.device)
                labels = torch.cat(labels_list).to(self.device)
                # calculate NLL before temperature scaling
                before_temperature_nll = nll_criterion(logits, labels)

            print('Before temperature - NLL: %.3f' % (before_temperature_nll))

            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

            # make sure only temperature parameter will be learned,
            # fix other parameters of the network
            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self._temperature_scale(logits), labels)
                loss.backward()
                return loss

            optimizer.step(eval)

            # print learned parameter temperature,
            # calculate NLL after temperature scaling
            after_temperature_nll = nll_criterion(
                self._temperature_scale(logits), labels).item()
            print('Optimal temperature: %.3f' % self.temperature.item())
            print('After temperature - NLL: %.3f' % (after_temperature_nll))
            self.setup_flag = True
        else:
            pass

    def _temperature_scale(self, logits):
        return logits / self.temperature

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = apply_model(net, data, return_features=False)
        logits_ts = self._temperature_scale(logits)
        score = torch.softmax(logits_ts, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()
        
###########

class ODINPostprocessor():
    def __init__(self):

        self.temperature = 1000
        self.noise = 0.0014

    def postprocess(self, net: nn.Module, data: Any):
        data.requires_grad = True
        output = apply_model(net, data, return_features=False)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        criterion = nn.CrossEntropyLoss()

        labels = output.detach().argmax(axis=1)

        # Using temperature scaling
        output = output / self.temperature

        loss = criterion(output, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.detach(), 0)
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        #gradient = gradient/std

        # Adding small perturbations to images
        tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)
        output = apply_model(net, tempInputs, return_features=False)
        output = output / self.temperature

        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)

        conf, pred = nnOutput.max(dim=1)

        return pred.cpu().numpy(), conf.cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.temperature = hyperparam[0]
        self.noise = hyperparam[1]

    def get_hyperparam(self):
        return [self.temperature, self.noise]
        
###########

class GRAMPostprocessor():
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.powers = [1,2,3,4,5]

        self.feature_min, self.feature_max = None, None
        self.setup_flag = False

    def setup(self, net, x_train, y_train, batch_size):
        if not self.setup_flag:
            self.feature_min, self.feature_max = sample_estimator(
                net, x_train, y_train, batch_size, self.num_classes, self.powers, self.device)
            self.setup_flag = True
        else:
            pass

    def postprocess(self, net: nn.Module, data: Any):
        preds, deviations = get_deviations(net, data, self.feature_min,
                                           self.feature_max, self.num_classes,
                                           self.powers, self.device)
        return preds.cpu().numpy(), deviations.cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.powers = hyperparam[0]

    def get_hyperparam(self):
        return self.powers


def tensor2list(x, device):
    return x.data.to(device).tolist()


@torch.no_grad()
def sample_estimator(model, x_train, y_train, batch_size, num_classes, powers, device):

    model.eval()

    num_layer = 1  # 4 for lenet
    num_poles_list = powers
    num_poles = len(num_poles_list)
    feature_class = [[[None for x in range(num_poles)]
                      for y in range(num_layer)] for z in range(num_classes)]
    label_list = []
    mins = [[[None for x in range(num_poles)] for y in range(num_layer)]
            for z in range(num_classes)]
    maxs = [[[None for x in range(num_poles)] for y in range(num_layer)]
            for z in range(num_classes)]

    # collect features and compute gram metrix
    for batch_idx in range(len(x_train) // batch_size):
        data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
        label = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
        _, feature_list = apply_model(model, data, return_features=True)
        feature_list = [feature_list]
        
        label_list = tensor2list(label, device)
        for layer_idx in range(num_layer):

            for pole_idx, p in enumerate(num_poles_list):
                temp = feature_list[layer_idx].detach()

                temp = temp**p
                temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                temp = ((torch.matmul(temp,
                                      temp.transpose(dim0=2,
                                                     dim1=1)))).sum(dim=2)
                temp = (temp.sign() * torch.abs(temp)**(1 / p)).reshape(
                    temp.shape[0], -1)

                temp = tensor2list(temp, device)
                for feature, label in zip(temp, label_list):
                    if isinstance(feature_class[label][layer_idx][pole_idx],
                                  type(None)):
                        feature_class[label][layer_idx][pole_idx] = feature
                    else:
                        feature_class[label][layer_idx][pole_idx].extend(
                            feature)
    # compute mins/maxs
    for label in range(num_classes):
        for layer_idx in range(num_layer):
            for poles_idx in range(num_poles):
                feature = torch.tensor(
                    np.array(feature_class[label][layer_idx][poles_idx]))
                current_min = feature.min(dim=0, keepdim=True)[0]
                current_max = feature.max(dim=0, keepdim=True)[0]

                if mins[label][layer_idx][poles_idx] is None:
                    mins[label][layer_idx][poles_idx] = current_min
                    maxs[label][layer_idx][poles_idx] = current_max
                else:
                    mins[label][layer_idx][poles_idx] = torch.min(
                        current_min, mins[label][layer_idx][poles_idx])
                    maxs[label][layer_idx][poles_idx] = torch.max(
                        current_min, maxs[label][layer_idx][poles_idx])

    return mins, maxs


def get_deviations(model, data, mins, maxs, num_classes, powers, device):
    model.eval()

    num_layer = 1  # 4 for lenet
    num_poles_list = powers
    exist = 1
    pred_list = []
    dev = [0 for x in range(data.shape[0])]

    # get predictions
    logits, feature_list = apply_model(model, data, return_features=True)
    feature_list = [feature_list]
    
    #confs = F.softmax(logits, dim=1).cpu().detach().numpy
    confs = softmax(logits.cpu().detach().numpy())
    preds = np.argmax(confs, axis=1)
    predsList = preds.tolist()
    preds = torch.tensor(preds)

    for pred in predsList:
        exist = 1
        if len(pred_list) == 0:
            pred_list.extend([pred])
        else:
            for pred_now in pred_list:
                if pred_now == pred:
                    exist = 0
            if exist == 1:
                pred_list.extend([pred])

    # compute sample level deviation
    for layer_idx in range(num_layer):
        for pole_idx, p in enumerate(num_poles_list):
            # get gram metirx
            temp = feature_list[layer_idx].detach()
            temp = temp**p
            temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
            temp = ((torch.matmul(temp, temp.transpose(dim0=2,
                                                       dim1=1)))).sum(dim=2)
            temp = (temp.sign() * torch.abs(temp)**(1 / p)).reshape(
                temp.shape[0], -1)
            temp = tensor2list(temp, device)

            # compute the deviations with train data
            for idx in range(len(temp)):
                dev[idx] += (F.relu(mins[preds[idx]][layer_idx][pole_idx] -
                                    sum(temp[idx])) /
                             torch.abs(mins[preds[idx]][layer_idx][pole_idx] +
                                       10**-6)).sum()
                dev[idx] += (F.relu(
                    sum(temp[idx]) - maxs[preds[idx]][layer_idx][pole_idx]) /
                             torch.abs(maxs[preds[idx]][layer_idx][pole_idx] +
                                       10**-6)).sum()
    conf = [i / 50 for i in dev]

    return preds, torch.tensor(conf).clamp(min=-1e10, max=1e10)
    
###########

class EBOPostprocessor():
    def __init__(self):
        self.temperature = 1

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = apply_model(net, data, return_features=False)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()
    
    def set_hyperparam(self,  hyperparam:list):
        self.temperature =hyperparam[0] 
    
    def get_hyperparam(self):
        return self.temperature
        
###########

class GradNormPostprocessor():
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device

    def gradnorm(self, x, w, b):
        fc = torch.nn.Linear(*w.shape[::-1])
        fc.weight.data[...] = torch.from_numpy(w)
        fc.bias.data[...] = torch.from_numpy(b)
        fc.to(self.device)

        targets = torch.ones((1, self.num_classes)).to(self.device)

        confs = []
        for i in x:
            fc.zero_grad()
            loss = torch.mean(
                torch.sum(-targets * F.log_softmax(fc(i[None]), dim=-1),
                          dim=-1))
            loss.backward()
            layer_grad_norm = torch.sum(torch.abs(
                fc.weight.grad.data)).cpu().numpy()
            confs.append(layer_grad_norm)

        return np.array(confs)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        w, b, _ = net.get_fc()
        logits, features = apply_model(net, data, return_features=True)
        with torch.enable_grad():
            scores = self.gradnorm(features, w, b)
        _, preds = torch.max(logits, dim=1)
        return preds.cpu().numpy(), torch.from_numpy(scores).cpu().numpy()

###########

class ReactPostprocessor():
    def __init__(self, device):
        self.percentile = 90
        self.device = device
        self.setup_flag = False

    def setup(self, net, x_val, batch_size):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch_idx in range(len(x_val) // batch_size):
                    data = x_val[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    data = data.float()

                    _, feature = apply_model(net, data, return_features=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.set_hyperparam()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = apply_model(net, data, return_features=True)
        feature = feature.clip(max=self.threshold)
        
        _, _, fc = net.get_fc()
        output = fc(feature)
        #output = net.forward_threshold(data, self.threshold)
        
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred.cpu().numpy(), energyconf.cpu().numpy()

    def set_hyperparam(self):
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        return self.percentile

###########

class MaxLogitPostprocessor():
    def __init__(self):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = apply_model(net, data, return_features=False)
        conf, pred = torch.max(output, dim=1)
        return pred.cpu().numpy(), conf.cpu().numpy()

###########

class KLMatchingPostprocessor():
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.setup_flag = False

    def kl(self, p, q):
        return scipy.stats.entropy(p, q)

    def setup(self, net, x_val, batch_size):
        if not self.setup_flag:
            net.eval()

            print('Extracting id validation softmax posterior distributions')
            all_softmax = []
            preds = []
            with torch.no_grad():
                for batch_idx in range(len(x_val) // batch_size):
                    data = x_val[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    logits = apply_model(net, data, return_features=False)
                    all_softmax.append(F.softmax(logits, 1).cpu())
                    preds.append(logits.argmax(1).cpu())

            all_softmax = torch.cat(all_softmax)
            preds = torch.cat(preds)

            self.mean_softmax_val = []
            for i in tqdm(range(self.num_classes)):
                # if there are no validation samples
                # for this category
                if torch.sum(preds.eq(i).float()) == 0:
                    temp = np.zeros((self.num_classes, ))
                    temp[i] = 1
                    self.mean_softmax_val.append(temp)
                else:
                    self.mean_softmax_val.append(
                        all_softmax[preds.eq(i)].mean(0).numpy())

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = apply_model(net, data, return_features=False)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds.cpu().numpy(), torch.from_numpy(scores).cpu().numpy()

###########

class VIMPostprocessor():
    def __init__(self, device, dim):
        self.dim = dim
        self.device = device
        self.setup_flag = False

    def setup(self, net, x_train, batch_size):
        if not self.setup_flag:
            net.eval()

            with torch.no_grad():
                self.w, self.b, _ = net.get_fc()
                print('Extracting id training feature')
                feature_id_train = []
                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    #data = data.float()
                    _, feature = apply_model(net, data, return_features=True)
                    feature_id_train.append(feature.cpu().numpy())
                feature_id_train = np.concatenate(feature_id_train, axis=0)
                logit_id_train = feature_id_train @ self.w.T + self.b

            self.u = -np.matmul(pinv(self.w), self.b)
            ec = EmpiricalCovariance(assume_centered=True)
            ec.fit(feature_id_train - self.u)
            eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
            self.NS = np.ascontiguousarray(
                (eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim:]]).T)

            vlogit_id_train = norm(np.matmul(feature_id_train - self.u,
                                             self.NS),
                                   axis=-1)
            self.alpha = logit_id_train.max(
                axis=-1).mean() / vlogit_id_train.mean()
            print(f'{self.alpha=:.4f}')

            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature_ood = apply_model(net, data, return_features=True)
        feature_ood = feature_ood.cpu()
        logit_ood = feature_ood @ self.w.T + self.b
        _, pred = torch.max(logit_ood, dim=1)
        energy_ood = logsumexp(logit_ood.numpy(), axis=-1)
        #print(feature_ood.numpy() - self.u, self.NS, self.alpha)
        vlogit_ood = norm(np.matmul(feature_ood.numpy() - self.u, self.NS),
                          axis=-1) * self.alpha
        score_ood = -vlogit_ood + energy_ood
        return pred.cpu().numpy(), torch.from_numpy(score_ood).cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim

###########

class DICEPostprocessor():
    def __init__(self, device):
        self.p = 90
        self.device = device
        self.mean_act = None
        self.masked_w = None
        self.setup_flag = False

    def setup(self, net, x_train, batch_size):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    #data = data.float()

                    _, feature = apply_model(net, data, return_features=True)
                    activation_log.append(feature.data.cpu().numpy())

            activation_log = np.concatenate(activation_log, axis=0)
            self.mean_act = activation_log.mean(0)
            self.setup_flag = True
        else:
            pass

    def calculate_mask(self, w):
        contrib = self.mean_act[None, :] * w.data.squeeze().cpu().numpy()
        self.thresh = np.percentile(contrib, self.p)
        mask = torch.Tensor((contrib > self.thresh)).to(self.device)
        self.masked_w = w * mask

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        fc_weight, fc_bias, _ = net.get_fc()
        if self.masked_w is None:
            self.calculate_mask(torch.from_numpy(fc_weight).to(self.device))
        _, feature = apply_model(net, data, return_features=True)
        vote = feature[:, None, :] * self.masked_w
        output = vote.sum(2) + torch.from_numpy(fc_bias).to(self.device)
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred.cpu().numpy(), energyconf.cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.p = hyperparam[0]

    def get_hyperparam(self):
        return self.p
        
###########

def ash_b(x, percentile=65):
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x
    
def forward_threshold(net, data, percentile, fc):
    _, feature = apply_model(net, data, return_features=True)
    feature = ash_b(feature.view(feature.size(0), -1, 1, 1), percentile)
    feature = feature.view(feature.size(0), -1)
    logits_cls = fc(feature)
    return logits_cls
        
class ASHPostprocessor():
    def __init__(self):
        self.percentile = 90

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, _, fc = net.get_fc()
        output = forward_threshold(net, data, self.percentile, fc)
        _, pred = torch.max(output, dim=1)
        energyconf = torch.logsumexp(output.data.cpu(), dim=1)
        return pred.cpu().numpy(), energyconf.cpu().numpy()

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]

    def get_hyperparam(self):
        return self.percentile
    
###########

def distance(penultimate, target, metric='inner_product'):
    if metric == 'inner_product':
        return torch.sum(torch.mul(penultimate, target), dim=1)
    elif metric == 'euclidean':
        return -torch.sqrt(torch.sum((penultimate - target)**2, dim=1))
    elif metric == 'cosine':
        return torch.cosine_similarity(penultimate, target, dim=1)
    else:
        raise ValueError('Unknown metric: {}'.format(metric))


class SHEPostprocessor():
    def __init__(self, num_classes, device, metric='inner_product'):
        self.num_classes = num_classes
        self.device = device
        self.activation_log = None
        self.setup_flag = False
        self.metric = metric

    def setup(self, net, x_train, y_train, batch_size):
        if not self.setup_flag:
            net.eval()

            all_activation_log = []
            all_labels = []
            all_preds = []
            with torch.no_grad():

                for batch_idx in range(len(x_train) // batch_size):
                    data = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size].to(self.device)
                    labels = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    logits, features = apply_model(net, data, return_features=True)
                    
                    all_labels.append(deepcopy(labels))
                    all_activation_log.append(features.cpu())
                    all_preds.append(logits.argmax(1).cpu())

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            all_activation_log = torch.cat(all_activation_log)

            self.activation_log = []
            for i in range(self.num_classes):
                mask = torch.logical_and(all_labels.cpu() == i, all_preds.cpu() == i)
                class_correct_activations = all_activation_log[mask]
                self.activation_log.append(
                    class_correct_activations.mean(0, keepdim=True))

            self.activation_log = torch.cat(self.activation_log).to(self.device)
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = apply_model(net, data, return_features=True)
        pred = output.argmax(1)
        conf = distance(feature, self.activation_log[pred], self.metric)
        return pred.cpu().numpy(), conf.cpu().numpy()
        
###########

def detection_method(detector, model, device, **arguments):
    """
    Get a postprocessing method or novelty detection model based on the specified detector.

    Parameters:
    -----------
    detector: str
        The name of the detection method or postprocessor.
    model: torch.nn.Module
        The neural network model used for detection or postprocessing.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    **arguments: dict
        Additional arguments specific to the selected detector.

    Returns:
    --------
    Callable:
        Depending on the specified detector, returns a callable function for postprocessing or novelty detection.

    Raises:
    -------
    Exception:
        If the specified detection method is not implemented.
    """
    
    if detector == 'MSP':
        return MSP
    
    elif detector== 'KNN':
        processor = KNNPostprocessor(k=arguments['k_knn'], device=device)
        processor.setup(model, arguments['x_train'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector =='OpenMax':
        processor = OpenMax(num_classes=arguments['n_classes'], device=device)
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'MDS':
        processor = MDSPostprocessor(num_classes=arguments['n_classes'], device=device)
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess

    elif detector == 'RMDS':
        processor = RMDSPostprocessor(num_classes=arguments['n_classes'], device=device)
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'temp_scaling':
        processor = TemperatureScalingPostprocessor(device=device)
        processor.setup(model, arguments['x_val'], arguments['y_val'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'odin':
        processor = ODINPostprocessor()
        return processor.postprocess
    
    elif detector == 'gram':
        processor = GRAMPostprocessor(num_classes=arguments['n_classes'], device=device)
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'ebo':
        processor = EBOPostprocessor()
        return processor.postprocess
    
    elif detector == 'gradnorm':
        processor = GradNormPostprocessor(num_classes=arguments['n_classes'], device=device)
        return processor.postprocess
    
    elif detector == 'react':
        processor = ReactPostprocessor(device=device)
        processor.setup(model, arguments['x_val'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'mls':
        processor = MaxLogitPostprocessor()
        return processor.postprocess
    
    elif detector == 'klm':
        processor = KLMatchingPostprocessor(num_classes=arguments['n_classes'], device=device)
        processor.setup(model, arguments['x_val'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'vim':
        processor = VIMPostprocessor(device=device, dim=arguments['vim_dim'])
        processor.setup(model, arguments['x_train'], arguments['batch_size'])
        return processor.postprocess  

    elif detector == 'dice':
        processor = DICEPostprocessor(device=device)
        processor.setup(model, arguments['x_train'], arguments['batch_size'])
        return processor.postprocess
    
    elif detector == 'ash':
        processor = ASHPostprocessor()
        return processor.postprocess

    elif detector =='she_inner':
        processor = SHEPostprocessor(num_classes=arguments['n_classes'], device=device, metric='inner_product')
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess

    elif detector =='she_euclidean':
        processor = SHEPostprocessor(num_classes=arguments['n_classes'], device=device, metric='euclidean')
        processor.setup(model, arguments['x_train'], arguments['y_train'], arguments['batch_size'])
        return processor.postprocess
        
    elif detector =='AE':
        density_model = AE(hidden_sizes=None, input_size= arguments['x_train'].shape[1], latent_dim=None, device=device, lr=arguments['lr'])
        density_model.train(arguments['x_train'], arguments['batch_size'], arguments['n_epochs'])
        density_model.model.eval()
        return density_model.postprocess

    elif detector =='VAE':
        density_model = VAE(device=device, hidden_sizes=None, input_size=arguments['x_train'].shape[1], latent_dim=None, lr=arguments['lr'])
        density_model.train(arguments['x_train'],  arguments['batch_size'], arguments['n_epochs'])
        density_model.model.eval()
        return density_model.get_novelty_score
    
    elif detector =='Flow':
        density_model =  Flow(device=device, input_size=arguments['x_train'].shape[1], lr=arguments['lr'])
        density_model.train(arguments['x_train'],  arguments['batch_size'], arguments['n_epochs'])
        density_model.model.eval()
        return density_model.get_novelty_score
    
    elif detector == 'ppca':
        density_model = PPCA()
        density_model.train(arguments['x_train'])
        return density_model.get_novelty_score
    
    elif detector == 'lof':
        density_model = LOF()
        density_model.train(arguments['x_train'])
        return density_model.get_novelty_score

    elif detector == 'due':
        density_model = DUE(device=device, lr=arguments['lr'], num_outputs=arguments['n_classes'])
        density_model.train(arguments['x_train'],  arguments['y_train'], arguments['batch_size'], arguments['n_epochs'])
        return density_model.get_novelty_score
    
    elif detector == 'HiVAE':
        density_model = HIVAE(arguments['x_train'])
        density_model.mytrain(arguments['x_train'], arguments['batch_size'], arguments['n_epochs'])
        return density_model.get_novelty_score
        
    else:
        raise Exception("Sorry, this detection method is not implemented. Are you sure about the exact name?")
        
