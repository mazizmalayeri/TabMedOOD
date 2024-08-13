import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from openood.evaluators.metrics import compute_all_metrics
import tqdm

'''
def print_all_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out,
        ccr_4, ccr_3, ccr_2, ccr_1, accuracy] \
        = metrics
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('CCR: {:.2f}, {:.2f}, {:.2f}, {:.2f},'.format(
        ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
            end=' ',
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)
'''

def print_all_metrics(metrics):
    """
    Print multiple evaluation metrics.

    Parameters:
    -----------
    metrics: list of float
        A list of evaluation metrics in the following order: [fpr, auroc, aupr_in, aupr_out, accuracy].
    """
    
    [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
    
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)

    
def eval_ood(postprocess_results, to_print=True, missclass_as_ood=False):
        """
        Calculates the OOD metrics (fpr, auroc, etc.) based on the postprocessing results.
        
        Parameters:
        -----------
        postprocess_results: list
            A list containing the following elements in order:
            [id_pred, id_conf, ood_pred, ood_conf, id_gt, ood_gt].
        to_print: bool, optional
            Whether to print the evaluation metrics or only return the metrics. Default is True.
        missclass_as_ood: bool, optional
            If True, consider misclassified in-distribution samples as OOD. Default is False.
    
        Returns:
        --------
        dict:
            A dictionary containing various OOD detection evaluation metrics.
        """
    
        [id_pred, id_conf, ood_pred, ood_conf, id_gt, ood_gt] = postprocess_results
        
        if missclass_as_ood:
            id_gt_np = np.array(id_gt)
            id_gt_np[np.array(id_pred) != id_gt_np] = -1
            print((id_gt_np == -1).mean())
            id_gt = id_gt_np.tolist()
            

        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        
        check_nan = np.isnan(conf)
        num_nans = check_nan.sum()
        if num_nans>0:
            print(num_nans, 'nan ignored.')
            conf = np.delete(conf, np.where(check_nan))
            pred = np.delete(pred, np.where(check_nan))
            label = np.delete(label, np.where(check_nan))
        
        check_inf = np.isinf(conf)
        num_infs = check_inf.sum()
        if num_infs>0:
            print(num_infs, 'inf ignored.')
            conf = np.delete(conf, np.where(check_inf))
            pred = np.delete(pred, np.where(check_inf))
            label = np.delete(label, np.where(check_inf))

        ood_metrics = compute_all_metrics(conf, label, pred)
        if to_print:
            print_all_metrics(ood_metrics)
        else:
            return ood_metrics
        
def ood_score_calc(inlier_score, ood_score):
    """
    Calculate the Area Under the ROC Curve (AUROC) for OOD detection based on the ID/OOD scores.

    Parameters:
    -----------
    inlier_score: list or array-like
        Scores for in-distribution samples.
    ood_score: list or array-like
        Scores for out-of-distribution samples.

    Returns:
    --------
    float:
        The computed AUC (Area Under the ROC Curve) for OOD detection.
    """
    label_ood = list(np.zeros(len(ood_score)))
    label_inlier = list(np.ones(len(inlier_score)))
    labels = label_ood + label_inlier
    scores = ood_score + inlier_score
    auc = roc_auc_score(labels, scores)
    
    return auc


def get_ood_score(model, in_test_features, in_test_labels, ood_type, score_function, batch_size, device, preprocess, random_sample=None, scales=None, out_features=None, missclass_as_ood=False):
    """
    Calculate the novelty scores that an OOD detector (score_function) assigns to ID and OOD and evaluate them via AUROC and FPR.

    Parameters:
    -----------
    model: torch.nn.Module or None
        The neural network model for applying the post-hoc method.
    in_test_features: torch.Tensor
        In-distribution test features.
    in_test_labels: torch.Tensor
        In-distribution test labels.
    ood_type: str
        The type of out-of-distribution (OOD) data ('other_domain', 'feature_separation', or 'multiplication').
    score_function: callable
        The scoring function that assigns each sample a novelty score.
    batch_size: int
        Batch size for processing data.
    device: str
        The device on which to run the model (e.g., 'cpu' or 'cuda').
    preprocess: object
        The preprocess for normalizing the data if it is needed.
    random_sample: list or None, optional
        List of randomly selected feature indices for 'multiplication'. Default is None.
    scales: list or None, optional
        List of scales for feature multiplication. Default is None.
    out_features: torch.Tensor or None, optional
        Out-of-distribution (OOD) features for 'other_domain' or 'feature_separation'. Default is None.
    missclass_as_ood: bool, optional
        If True, consider misclassified in-distribution samples as OOD. Default is False.
    """
    
    if model is not None:
        model.eval() 
        
    preds_in, confs_in, gt_in = [], [], []
    for batch_idx in range(len(in_test_features) // batch_size):
        x_batch = in_test_features[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
        pred, conf = score_function(model, x_batch)
        preds_in += list(pred)
        confs_in += list(conf)
        gt_in += list(in_test_labels[batch_idx*batch_size:(batch_idx+1)*batch_size].cpu().detach().numpy())

    
    if ood_type in ['other_domain', 'feature_seperation']:
        preds_out, confs_out, gt_out = [], [], []
        for batch_idx in range(len(out_features) // batch_size):
              x_batch = out_features[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
              pred, conf = score_function(model, x_batch)
              preds_out += list(pred)
              confs_out += list(conf)
              gt_out += list(np.ones(conf.shape[0])*-1)
        
        eval_ood([preds_in, confs_in, preds_out, confs_out, gt_in, gt_out], missclass_as_ood=missclass_as_ood)
        #auc = ood_score_calc(scores_inlier, scores_ood)
        #print('AUC:', auc)

    elif ood_type == 'multiplication':
        X_test_adjusted = torch.clone(in_test_features).cpu().numpy()
        X_test_adjusted = preprocess.inverse_transform(X_test_adjusted)
        for scale_adjustment in scales:
            scores_per_scale = np.zeros(5)
            X_test_adjusted_scaled = np.copy(X_test_adjusted)*scale_adjustment
            X_test_adjusted_scaled = preprocess.transform(X_test_adjusted_scaled)

            for r in random_sample:
                out_features = torch.clone(in_test_features)
                out_features[:,r] = torch.tensor(X_test_adjusted_scaled[:, r])

                preds_out, confs_out, gt_out = [], [], []
                for batch_idx in range(len(out_features) // batch_size):
                      x_batch = out_features[batch_idx*batch_size:(batch_idx+1)*batch_size].to(device)
                      pred, conf = score_function(model, x_batch)
                      preds_out += list(pred)
                      confs_out += list(conf)
                      gt_out += list(np.ones(conf.shape[0])*-1)
                
                ood_metrics = eval_ood([preds_in, confs_in, preds_out, confs_out, gt_in, gt_out], to_print=False, missclass_as_ood=missclass_as_ood)
                #print(ood_metrics)
                scores_per_scale += np.array(ood_metrics)
                #scores_per_scale += ood_score_calc(scores_inlier, scores_ood)
            
            print('Scale:', scale_adjustment)
            print_all_metrics(list(scores_per_scale/len(random_sample)))
            #print('Scale:', scale_adjustment, 'Average AUC:', scores_per_scale/len(random_sample))
