import numpy as np
from sed_scores_eval.utils.curves import get_curve_idx_for_threshold


def precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
):
    """Compute precision-recall curve from intermediate_statistics

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided, keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.

    Returns: (all arrays sorted by corresponding recall)
        precisions ((dict of) 1d np.ndarray): precision values for all operating points
        recalls ((dict of) 1d np.ndarray): recall values for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding precision-recall pairs
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events

    """
    if isinstance(scores_intermediate_statistics, dict):
        p, r, scores, intermediate_stats = {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                p[class_name], r[class_name], scores[class_name],
                intermediate_stats[class_name]
            ) = precision_recall_curve_from_intermediate_statistics(scores_stats)
        return p, r, scores, intermediate_stats

    if not isinstance(scores_intermediate_statistics, (list, tuple)):
        raise ValueError(
            f'scores_intermediate_statistics must be list/tuple of length 2, '
            f'but {type(scores_intermediate_statistics)} was given.'
        )
    if len(scores_intermediate_statistics) != 2:
        raise ValueError(
            f'scores_intermediate_statistics must be list/tuple of length 2, '
            f'but list/tuple of length {len(scores_intermediate_statistics)}'
            f'was given.'
        )
    scores, stats = scores_intermediate_statistics
    if not isinstance(stats, dict):
        raise ValueError(
            f'intermediate_statistics must be dict but {type(stats)} was given.')
    if not all([key in stats for key in ['tps', 'fps', 'n_ref']]):
        raise ValueError(
            f'intermediate_statistics must contain keys "tps", "fps" and "n_ref". '
            f'Provided keys: {sorted(stats.keys())}.'
        )
    p = stats['tps'] / np.maximum(stats['tps']+stats['fps'], 1)
    p[(stats['tps']+stats['fps']) == 0] = 1.
    if stats['n_ref'] == 0:
        r = np.ones_like(p)
    else:
        r = stats['tps'] / stats['n_ref']
    sort_idx = sorted(np.arange(len(r)).tolist(), key=lambda i: (r[i], p[i]))

    def sort_stat(stat):
        if np.isscalar(stat):
            return stat
        if isinstance(stat, (list, tuple)):
            return [sort_stat(stat_i) for stat_i in stat]
        return stat[sort_idx]

    return (
        p[sort_idx], r[sort_idx], scores[sort_idx],
        {key: sort_stat(stat) for key, stat in stats.items()}
    )


def fscore_from_precision_recall(precision, recall, *, beta=1.):
    """Compute f-score from precision and recall.

    Args:
        precision (scalar or np.ndarray):
        recall (scalar or np.ndarray):
        beta: \beta parameter for f-score computation

    Returns: f-score

    """
    return (
        (1 + beta**2) * precision * recall
        / np.maximum(beta**2 * precision + recall, 1e-15)
    )


def fscore_curve_from_intermediate_statistics(
        scores_intermediate_statistics, beta=1.
):
    """Compute f-scores with corresponding precisions, recalls and
    intermediate statistics for various operating points

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        beta: \beta parameter of f-score computation

    Returns: (all arrays sorted by corresponding score)
        f_beta ((dict of) 1d np.ndarray): f-score values  for all operating
            points
        precisions ((dict of) 1d np.ndarray): precision values for all operating points
        recalls ((dict of) 1d np.ndarray): recall values for all operating points
        scores ((dict of) 1d np.ndarray): score values that the threshold has to
            fall below to obtain corresponding precision-recall pairs
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps': 1d np.ndarray of true positive counts for each score
            'fps': 1d np.ndarray of false positive counts for each score
            'n_ref': integer number of ground truth events


    """
    if isinstance(scores_intermediate_statistics, dict):
        f, p, r, scores, intermediate_stats = {}, {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                f[class_name], p[class_name], r[class_name],
                scores[class_name], intermediate_stats[class_name]
            ) = fscore_curve_from_intermediate_statistics(
                scores_stats, beta=beta,
            )
        return f, p, r, scores, intermediate_stats
    p, r, scores, intermediate_stats = precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    sort_idx = np.argsort(scores)
    scores = scores[sort_idx]
    p = p[sort_idx]
    r = r[sort_idx]
    f_beta = fscore_from_precision_recall(p, r, beta=beta)
    intermediate_stats = {
        key: stat if np.isscalar(stat) else stat[sort_idx]
        for key, stat in intermediate_stats.items()
    }
    return f_beta, p, r, scores, intermediate_stats


def best_fscore_from_intermediate_statistics(
        scores_intermediate_statistics,
        beta=1., min_precision=0., min_recall=0.
):
    """Get the best possible (macro-average) f-score with corresponding
    precision, recall, intermediate statistics and decision threshold

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        beta: \beta parameter of f-score computation
        min_precision: the minimum precision that must be achieved.
        min_recall: the minimum recall that must be achieved. If the
            constraint(s) cannot be achieved at any threshold, however,
            fscore, precision, recall and threshold of 0,1,0,inf are returned.

    Returns:
        f_beta ((dict of) float): best achievable f-score value
        precision ((dict of) float): precision value at best fscore
        recall ((dict of) float): recall value at best fscore
        threshold ((dict of) float): threshold to obtain best fscore which is
            centered between the score that the threshold has to fall below
            and the next smaller score which results in different intermediate
            statistics.
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count at best fscore
            'fps' (int): false positive count at best fscore
            'n_ref' (int): number of ground truth events

    """
    if isinstance(scores_intermediate_statistics, dict):
        f, p, r, thresholds, intermediate_stats = {}, {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                f[class_name], p[class_name], r[class_name],
                thresholds[class_name], intermediate_stats[class_name]
            ) = best_fscore_from_intermediate_statistics(
                scores_stats, beta=beta,
                min_precision=min_precision,
                min_recall=min_recall,
            )
        f['macro_average'] = np.mean([f[class_name] for class_name in f])
        p['macro_average'] = np.mean([p[class_name] for class_name in p])
        r['macro_average'] = np.mean([r[class_name] for class_name in r])
        (
            f['micro_average'], p['micro_average'], r['micro_average']
        ) = micro_average(intermediate_stats, beta=beta)
        return f, p, r, thresholds, intermediate_stats
    f, p, r, scores, intermediate_stats = \
        fscore_curve_from_intermediate_statistics(
            scores_intermediate_statistics, beta=beta
        )
    f[p < min_precision] = 0.
    f[r < min_recall] = 0.
    best_idx = len(f) - 1 - np.argmax(f[::-1], axis=0)
    threshold = (
        (scores[best_idx] + scores[best_idx-1])/2 if best_idx > 0 else -np.inf
    )
    return (
        f[best_idx], p[best_idx], r[best_idx], threshold,
        {
            key: stat if np.isscalar(stat) else stat[best_idx]
            for key, stat in intermediate_stats.items()
        }
    )


def single_fscore_from_intermediate_statistics(
        scores_intermediate_statistics, threshold, *, beta=1., ):
    """Get f-score with corresponding precision, recall and intermediate
    statistics for a specific decision threshold

    Args:
        scores_intermediate_statistics ((dict of) tuple): tuple of scores array
            and dict of intermediate_statistics with the following key value
            pairs:
             'tps' (1d np.ndarray): true positive counts for each score
             'fps' (1d np.ndarray): false positive counts for each score
             'n_ref' (int): number of ground truth events
            If dict input is provided keys are expected to be class names with
            corresponding scores/intermediate_statistics as values.
        threshold ((dict of) float): threshold that is to be evaluated.
        beta: \beta parameter of f-score computation

    Returns:
        fscore ((dict of) float): fscore value for threshold
        precision ((dict of) float): precision value for threshold
        recall ((dict of) float): recall value for threshold
        intermediate_statistics ((dict of) dict): dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count for threshold
            'fps' (int): false positive count for threshold
            'n_ref' (int): number of ground truth events

    """
    if isinstance(threshold, dict):
        if not all([np.isscalar(thr) for thr in threshold]):
            raise ValueError('All values of thresholds dict must be scalars')
    elif not np.isscalar(threshold):
        raise ValueError(
            f'threshold must be (dict of) scalar(s) but '
            f'{type(threshold)} was given.'
        )
    if isinstance(scores_intermediate_statistics, dict):
        f, p, r, intermediate_stats = {}, {}, {}, {}
        for class_name, scores_stats in scores_intermediate_statistics.items():
            (
                f[class_name], p[class_name], r[class_name],
                intermediate_stats[class_name]
            ) = single_fscore_from_intermediate_statistics(
                scores_stats,
                threshold[class_name]
                if isinstance(threshold, dict)
                else threshold,
                beta=beta,
            )
        f['macro_average'] = np.mean([f[class_name] for class_name in f])
        p['macro_average'] = np.mean([p[class_name] for class_name in p])
        r['macro_average'] = np.mean([r[class_name] for class_name in r])
        (
            f['micro_average'], p['micro_average'], r['micro_average']
        ) = micro_average(intermediate_stats, beta=beta)
        return f, p, r, intermediate_stats
    pr_curve = precision_recall_curve_from_intermediate_statistics(
        scores_intermediate_statistics
    )
    p, r, scores, intermediate_stats = pr_curve
    return _single_fscore_from_precision_recall_curve(
        p, r, scores, threshold, beta=beta,
        intermediate_statistics=intermediate_stats
    )


def _single_fscore_from_precision_recall_curve(
        precisions, recalls, scores, threshold, beta=1.,
        intermediate_statistics=None
):
    idx = get_curve_idx_for_threshold(scores, threshold)
    p = precisions[idx]
    r = recalls[idx]
    f = fscore_from_precision_recall(p, r, beta=beta)
    if intermediate_statistics is None:
        return f, p, r
    else:
        return f, p, r, {
            key: stat if np.isscalar(stat) else stat[idx]
            for key, stat in intermediate_statistics.items()
        }


def micro_average(intermediate_stats, beta=1.):
    """Compute the mirco averaged f-score, where the intermediate statistics
    are summed up before computation of precision, recall and f-score

    Args:
        intermediate_stats (dict of dict): contains a dict of
            intermediate_statistics with the following key value pairs:
            'tps' (int): true positive count for threshold
            'fps' (int): false positive count for threshold
            'n_ref' (int): number of ground truth events
            for each event class
        beta: \beta parameter of f-score computation

    Returns:
        fscore (float): micro-average fscore
        precision (float): micro-average precision
        recall (float): micro-average recall

    """
    tps = np.sum([
        intermediate_stats[class_name]['tps']
        for class_name in intermediate_stats
    ])
    fps = np.sum([
        intermediate_stats[class_name]['fps']
        for class_name in intermediate_stats
    ])
    n_ref = np.sum([
        intermediate_stats[class_name]['n_ref']
        for class_name in intermediate_stats
    ])
    p = tps / np.maximum(tps + fps, 1)
    r = tps / np.maximum(n_ref, 1)
    f = fscore_from_precision_recall(p, r, beta=beta)
    return f, p, r


def fscore_from_sed_eval_metrics(sed_eval_metrics):
    """extract class-wise and averaged fscores, precisions and recalls from
    sed_eval metrics object

    Args:
        sed_eval_metrics:

    Returns:
        fscore (dict of float): fscore values
        precision (dict of float): precision values
        recall (dict of float): recall values

    """
    f = {}
    p = {}
    r = {}
    sed_eval_results_classwise = sed_eval_metrics.results_class_wise_metrics()
    for key in sed_eval_results_classwise:
        f[key] = sed_eval_results_classwise[key]['f_measure']['f_measure']
        p[key] = sed_eval_results_classwise[key]['f_measure']['precision']
        r[key] = sed_eval_results_classwise[key]['f_measure']['recall']
    sed_eval_results_macro = sed_eval_metrics.results_class_wise_average_metrics()
    f['macro_average'] = sed_eval_results_macro['f_measure']['f_measure']
    p['macro_average'] = sed_eval_results_macro['f_measure']['precision']
    r['macro_average'] = sed_eval_results_macro['f_measure']['recall']
    sed_eval_results_micro = sed_eval_metrics.results_overall_metrics()
    f['micro_average'] = sed_eval_results_micro['f_measure']['f_measure']
    p['micro_average'] = sed_eval_results_micro['f_measure']['precision']
    r['micro_average'] = sed_eval_results_micro['f_measure']['recall']
    return f, p, r
