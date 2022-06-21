import numpy as np
from pb_sed.utils import correlate
from pb_sed.evaluation.instance_based import f1_curve, er_curve
import pdb

def alignments_to_event_list(alignment_mat):
    """
    Detects event on- and offsets from multi-hot alignment

    Args:
        alignment_mat: multi-hot alignment (num_examples, num_frames, num_labels)

    Returns: list of tuples (example_idx, onset-frame, offset-frame, label_idx)

    >>> score_mat_1 = np.array([[0,0,0,0,1,1,0,0,1,1],[1,1,1,1,1,0,0,0,0,0]])[..., None]
    >>> score_mat_2 = np.array([[1,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,1,1]])[..., None]
    >>> score_mat = np.concatenate((score_mat_1, score_mat_2), axis=-1)
    >>> from pprint import pprint
    >>> pprint(alignments_to_event_list(score_mat))
    [(0, 0, 1, 1),
     (0, 3, 6, 1),
     (0, 4, 6, 0),
     (0, 8, 10, 0),
     (1, 0, 5, 0),
     (1, 8, 10, 1)]
    """
    zeros = np.zeros_like(alignment_mat[:, :1, :])
    alignment_mat = np.concatenate((zeros, alignment_mat, zeros), axis=1).astype(np.float)
    alignment_boundaries = alignment_mat[:, 1:] - alignment_mat[:, :-1]
    event_list = []
    for n in range(alignment_boundaries.shape[0]):
        for k in range(alignment_boundaries.shape[-1]):
            onsets = np.argwhere(alignment_boundaries[n, :, k] > .5).flatten().tolist()
            offsets = np.argwhere(alignment_boundaries[n, :, k] < -.5).flatten().tolist()
            for i, (on, off) in enumerate(zip(onsets, offsets)):
                event_list.append((n, on, off, k))
    #pdb.set_trace()
    return sorted(event_list)


def fscore(
        target_mat, decision_mat, collar, offset_collar_rate,
        beta=1., event_wise=False
):
    """Computes event-based fscore from multi-hot target and decision matrices

    Args:
        target_mat: multi-hot matrix indicating ground truth events/labels
            (num_examples, num_frames, num_labels)
        decision_mat: multi-hot matrix indicating detected events/labels
            (num_examples, num_frames, num_labels)
        collar:
        offset_collar_rate:
        beta:
        event_wise:

    Returns:
        fscore:
        precision:
        recall:

    >>> target_mat_1 = np.array([[0,0,1,1,1,0,0,0,1,1],[1,1,1,1,1,0,0,0,0,0]])[..., None]
    >>> score_mat_1 = np.array([[0,0,0,0,1,1,0,0,1,1],[1,1,1,1,1,0,0,0,0,0]])[..., None]
    >>> fscore(target_mat_1, score_mat_1, 1, .5)
    (0.6666666666666666, 0.6666666666666666, 0.6666666666666666)
    >>> target_mat_2 = np.array([[0,0,0,1,1,1,0,0,0,0],[0,0,0,1,1,1,0,0,0,0]])[..., None]
    >>> score_mat_2 = np.array([[0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])[..., None]
    >>> score_mat = np.concatenate((score_mat_1, score_mat_2), axis=-1)
    >>> target_mat = np.concatenate((target_mat_1, target_mat_2), axis=-1)
    >>> fscore(target_mat, score_mat, 1, .5, event_wise=True)
    (array([0.66666667, 0.66666667]), array([0.66666667, 1.        ]), array([0.66666667, 0.5       ]))
    """
    target_event_list = alignments_to_event_list(target_mat)
    decision_event_list = alignments_to_event_list(decision_mat)

    tp = np.zeros(target_mat.shape[-1])
    fp = np.zeros(target_mat.shape[-1])
    fn = np.zeros(target_mat.shape[-1])
    i = 0
    for n, ton, toff, k in target_event_list:
        while (
            i < len(decision_event_list)
            and (
                (decision_event_list[i][0] < n)
                or (
                    decision_event_list[i][0] == n
                    and decision_event_list[i][1] <= (ton - collar)
                )
            )
        ):
            fp[decision_event_list[i][3]] += 1
            i += 1
        j = 0
        hit = False
        offset_collar = max(
            collar, offset_collar_rate * (toff - ton)
        )
        while (
            not hit and ((i+j) < len(decision_event_list))
            and decision_event_list[i+j][0] == n
            and decision_event_list[i+j][1] < (ton + collar)
        ):
            if (
                decision_event_list[i+j][3] == k
                and np.abs(decision_event_list[i+j][2] - toff) < offset_collar
            ):
                tp[k] += 1
                decision_event_list.pop(i+j)
                hit = True
            j += 1
        if not hit:
            fn[k] += 1
    while i < len(decision_event_list):
        fp[decision_event_list[i][3]] += 1
        i += 1

    if not event_wise:
        tp = tp.sum(-1)
        fp = fp.sum(-1)
        fn = fn.sum(-1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f_beta = (1 + beta**2) * precision * recall / np.maximum(
        beta**2 * precision + recall, 1e-15
    )
    return f_beta, precision, recall


def get_optimal_thresholds(
        target_mat, score_mat, metric, collar, offset_collar_rate, decimals=4
):
    """Given multi-class soft scores returns the optimal threshold for each class w.r.t. a provided metric.

    Args:
        target_mat: multi-hot matrix indicating ground truth
            (num_clips, num_frames, num_classes)
        score_mat: classification scores for multi-label classification
            (num_clips, num_frames, num_classes)
        metric: metric to be optimized \in {'f1', 'er'}
        collar: absolute on-/offset collar in frames
        offset_collar_rate: percentage of event length with
            offset_collar=max(collar, int(offset_collar_rate * t_event))

    Returns:
        thresholds: opitmal thresholds (num_classes,)
        metric values: optimal metric values

    >>> target_mat_1 = np.array([[0.,0.,1.,1.,1.,0.,0.,0.,1.,1.],[1.,1.,1.,1.,1.,0.,0.,0.,0.,0.]])[..., None]
    >>> score_mat_1 = np.array([[0.,0.,0.,0.,.92,.81,0.,0.,.80,.64],[.32,.67,.44,.75,.22,.11,.11,0.,0.,0.]])[..., None]
    >>> get_optimal_thresholds(target_mat_1, score_mat_1, 'f1', 1, .5)
    (array([0.165]), array([0.66666667]))
    >>> target_mat_2 = np.array([[0,0,0,1,1,1,0,0,0,0],[0,0,0,1,1,1,0,0,0,0]])[..., None]
    >>> score_mat_2 = np.array([[0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])[..., None]
    >>> score_mat = np.concatenate((score_mat_1, score_mat_2), axis=-1)
    >>> target_mat = np.concatenate((target_mat_1, target_mat_2), axis=-1)
    >>> get_optimal_thresholds(target_mat, score_mat, 'f1', 1, .5)
    (array([0.165, 0.5  ]), array([0.66666667, 0.66666667]))

    """
    assert collar > 0, collar
    thresholds = []
    values = []
    b, t, k = target_mat.shape
    for label_idx in range(target_mat.shape[-1]):  # compute for each class individually
        # similarly to instance_based._metric_curve we aim to compute the
        # metric value, e.g., fscore, for decision thresholds between each
        # adjacent score pairs, i.e., we like to compute the metric value
        # when the the decision threshold is below the current score but above
        # the next smaller score in the score_mat.
        # For that purpose we like to compute the number of true positives,
        # the total number of positive predictions n_sys
        # and the ground truth number of events n_ref at each score in score_mat.

        cur_targets = target_mat[:, :, label_idx]
        cur_scores = score_mat[:, :, label_idx]
        prev_scores = np.concatenate(
            (-np.inf*np.ones((b, 1)), cur_scores[:, :-1]), axis=1
        )
        next_scores = np.concatenate(
            (cur_scores[:, 1:], -np.inf*np.ones((b, 1))), axis=1
        )
        onset_counts = (cur_scores > prev_scores).astype(np.int) - (next_scores > cur_scores).astype(np.int)
        # onset_counts allows to count the number of onsets (which equals number of positive predictions n_sys):
        # 1) onset_counts == 1 if cur_scores > prev_scores and next_scores < cur_scores,
        # 2) onset_counts == -1 if cur_scores < prev_scores and next_scores > cur_scores (cur_score represent a local maximum),
        # else 0.
        # In case 1) cur_score is a local maximum and if the decision threshold
        # falls below that score this results in a new onset.
        # In case 2) cur_score is a local minimum and if the decision threshold
        # falls below that score two separate events are merged to one and one
        # onset has to be substracted.

        # it remains to compute number of true positives and n_ref
        target_boundary_detection = correlate(cur_targets, [-1, 1], mode='full')
        target_onset_indices = np.argwhere(target_boundary_detection > 0.5)  # len is n_ref
        target_offset_indices = np.argwhere(target_boundary_detection < -0.5)
        tp_counts = np.zeros_like(onset_counts)
        # iterate each target event
        for (n_on, t_on), (n_off, t_off) in zip(target_onset_indices, target_offset_indices):  # n represents clip indices and t frame indices
            assert n_on == n_off, (n_on, n_off)
            offset_collar = int(max(
                collar, offset_collar_rate * (t_off - t_on)
            ))
            # determine the allowed onset and offset collar and split the score accordingly:
            onset_collar_onset = max(t_on - collar, 0)
            onset_collar_offset = t_on + collar - 1
            offset_collar_onset = max(t_off - offset_collar + 1, 0)
            offset_collar_offset = t_off + offset_collar
            if offset_collar_onset <= onset_collar_offset:
                onset_collar_offset = offset_collar_onset + np.argmax(cur_scores[n_on, max(offset_collar_onset-1, 0):onset_collar_offset+1]) - 1
                offset_collar_onset = onset_collar_offset + 1
            scores_inner = cur_scores[n_on, onset_collar_offset:offset_collar_onset]
            scores_onset_collar = cur_scores[n_on, onset_collar_onset:onset_collar_offset]
            scores_offset_collar = cur_scores[n_on, offset_collar_onset:offset_collar_offset]

            # target can only be true positive if the decision threshold falls
            # below all scores in scores_inner:
            max_detection_score_idx = onset_collar_offset+np.argmin(scores_inner)

            # decision threshold must exceed at least one score in onset collar
            # otherwise there is no onset in the collar
            # (if the collar does not include the first frame of the clip)
            min_onset_detection_score_idx = None if t_on < collar else onset_collar_onset + np.argmin(scores_onset_collar)

            # decision threshold must exceed at least one score in offset collar
            # otherwise there is no offset in the collar
            # (if the collar does not include the last frame of the clip)
            min_offset_detection_score_idx = None if t-t_off < offset_collar else offset_collar_onset+np.argmin(scores_offset_collar)
            if min_onset_detection_score_idx is not None and min_offset_detection_score_idx is not None:
                if cur_scores[n_on, min_onset_detection_score_idx] > cur_scores[n_on, min_offset_detection_score_idx]:
                    min_detection_score_idx = min_onset_detection_score_idx
                else:
                    min_detection_score_idx = min_offset_detection_score_idx
            elif min_onset_detection_score_idx is None:
                min_detection_score_idx = min_offset_detection_score_idx
            elif min_offset_detection_score_idx is None:
                min_detection_score_idx = min_onset_detection_score_idx
            else:
                min_detection_score_idx = None

            if min_detection_score_idx is None or cur_scores[n_on, max_detection_score_idx] > cur_scores[n_on, min_detection_score_idx]:
                # when the decision threshold falls below
                # cur_scores[n_on, max_detection_score_idx]
                # the target is detected true positive
                tp_counts[n_on, max_detection_score_idx] += 1
                if min_detection_score_idx is not None:
                    # As soon as the decision threshold falls below
                    # cur_scores[n_on, min_detection_score_idx]
                    # the target is not detected true positive anymore
                    tp_counts[n_on, min_detection_score_idx] -= 1
            # else: cannot be detected as true positive

        cur_scores = cur_scores.flatten()
        cur_targets = cur_targets.flatten()
        onset_counts = onset_counts.flatten()
        tp_counts = tp_counts.flatten()
        sort_indices = np.argsort(cur_scores)
        cur_scores = np.concatenate((cur_scores[sort_indices], [np.inf]))
        onset_counts = np.concatenate((onset_counts[sort_indices], [0]))
        tp_counts = np.concatenate((tp_counts[sort_indices], [0]))

        # cumulative sum of the true positives and n_sys for scores > cur_score
        # (note that tp_counts and onset_counts may also include negative values,
        # i.e., subtractions if the decision threshold falls below such scores)
        tps = np.cumsum(tp_counts[::-1])[::-1]
        n_sys = np.cumsum(onset_counts[::-1])[::-1]

        _, valid_idx = np.unique(cur_scores, return_index=True)
        tps = tps[valid_idx]
        n_sys = n_sys[valid_idx]
        n_ref = len(target_onset_indices)

        if metric == 'f1':
            p = tps / np.maximum(n_sys, 1)
            r = tps / np.maximum(n_ref, 1)
            event_based_value = 2*p*r / (p + r + 1e-15)
            instance_based_f1, cur_thresholds = f1_curve(cur_targets, cur_scores[:-1])
            assert len(cur_thresholds) == len(event_based_value), (len(cur_thresholds), len(event_based_value))
            idx = np.argmax(np.round(event_based_value, decimals=decimals) + 10**(-decimals) * instance_based_f1)
        elif metric == 'er':
            i = n_sys - tps
            d = n_ref - tps
            event_based_value = (i+d) / n_ref
            instance_based_er, cur_thresholds = er_curve(cur_targets, cur_scores[:-1])
            assert len(cur_thresholds) == len(event_based_value), (len(cur_thresholds), len(event_based_value))
            idx = np.argmin(np.round(event_based_value, decimals=decimals) + 10**(-decimals) * instance_based_er)
        else:
            raise NotImplementedError

        values.append(event_based_value[idx])
        thresholds.append(cur_thresholds[idx])
    return np.array(thresholds), np.array(values)
