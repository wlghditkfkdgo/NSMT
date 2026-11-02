"""Anomaly-detection metrics (2026-11-02, exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration).

The TSLib protocol (neorecall_ad_v1/anomaly_detection/test.py of this repository): the score of a time step is the channel mean of
the squared reconstruction error; the test score vector is the concatenation of the (overlapping, step 1) test windows; the
threshold is the (100 - anomaly_ratio) percentile of the training scores and the test scores together; raw F1 = binary F1 of
score > threshold; point-adjusted F1 = the same after `adjustment` (one hit inside a labelled segment marks the whole segment).
User decision 11-02: all three are produced, the comparisons use AUC-PR (threshold-free, no point adjustment).
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def adjustment(gt, pred):
    """Point adjustment, verbatim from neorecall_ad_v1/anomaly_detection/utils.py (this repository, git blob c0eb878a9, last
    changed in commit c66ffcee9); modifies pred in place and returns (gt, pred)."""
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def threshold(train_score, test_score, anomaly_ratio):
    return float(np.percentile(np.concatenate([train_score, test_score], axis=0), 100 - anomaly_ratio))


def ad_metrics(train_score, test_score, labels, anomaly_ratio):
    """Every metric of one run; scores and labels are 1-D arrays over the concatenated windows."""
    s = np.asarray(test_score, dtype=np.float64)
    gt = np.asarray(labels).astype(int)
    if s.shape != gt.shape or s.ndim != 1:
        raise ValueError(f'scores {s.shape} vs labels {gt.shape}')
    if not np.isfinite(s).all():
        raise ValueError('non-finite test scores')
    thr = threshold(np.asarray(train_score, dtype=np.float64), s, anomaly_ratio)
    pred = (s > thr).astype(int)
    rp, rr, rf, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)
    g2, p2 = adjustment(gt.copy(), pred.copy())
    pp, pr, pf, _ = precision_recall_fscore_support(np.array(g2), np.array(p2), average='binary', zero_division=0)
    return dict(aucpr=float(average_precision_score(gt, s)), aucroc=float(roc_auc_score(gt, s)), thr=thr,
                raw_p=float(rp), raw_r=float(rr), raw_f1=float(rf), pa_p=float(pp), pa_r=float(pr), pa_f1=float(pf),
                n_points=int(gt.size), anomaly_frac=float(gt.mean()))
