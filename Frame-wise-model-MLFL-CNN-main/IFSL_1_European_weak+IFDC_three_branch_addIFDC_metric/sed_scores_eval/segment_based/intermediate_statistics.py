import numpy as np
from pathlib import Path
import multiprocessing
from sed_scores_eval.base_modules.io import parse_inputs, read_audio_durations
from sed_scores_eval.utils.scores import validate_score_dataframe
from sed_scores_eval.utils.array_ops import get_first_index_where
from sed_scores_eval.base_modules.ground_truth import multi_label_to_single_label_ground_truths


def intermediate_statistics(
        scores, ground_truth, audio_durations, *,
        segment_length=1., time_decimals=6, num_jobs=1,
):
    """

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.
        audio_durations: The duration of each audio file in the evaluation set.
        segment_length: the segment length of the segments that are to be
            evaluated.
        time_decimals (int): the decimal precision used for evaluation. If
            chosen to high detected or ground truth events that have
            onsets or offsets right on a segment boundary may swap over to the
            adjacent segment because of small deviations due to limited
            floating point precision.
        num_jobs (int): the number of processes to use. Default is 1 in which
            case no multiprocessing is used.

    Returns:

    """
    if not isinstance(num_jobs, int) or num_jobs < 1:
        raise ValueError(
            f'num_jobs has to be an integer greater or equal to 1 but '
            f'{num_jobs} was given.'
        )
    scores, ground_truth, audio_ids = parse_inputs(scores, ground_truth)
    if isinstance(audio_durations, (str, Path)):
        audio_durations = Path(audio_durations)
        assert audio_durations.is_file(), audio_durations
        audio_durations = read_audio_durations(audio_durations)

    if audio_durations is not None and not audio_durations.keys() == set(audio_ids):
        raise ValueError(
            f'audio_durations audio ids do not match audio ids in scores. '
            f'Missing ids: {set(audio_ids) - audio_durations.keys()}. '
            f'Additional ids: {audio_durations.keys() - set(audio_ids)}.'
        )

    _, event_classes = validate_score_dataframe(scores[audio_ids[0]])
    single_label_ground_truths = multi_label_to_single_label_ground_truths(
        ground_truth, event_classes)

    def worker(audio_ids, output_queue=None):
        segment_scores = None
        segment_targets = None
        for audio_id in audio_ids:
            scores_k = scores[audio_id]
            timestamps, _ = validate_score_dataframe(
                scores_k, event_classes=event_classes)
            timestamps = np.round(timestamps, time_decimals)
            if segment_scores is None:
                segment_scores = {class_name: [] for class_name in event_classes}
                segment_targets = {class_name: [] for class_name in event_classes}
            scores_k = scores_k[event_classes].to_numpy()
            if audio_durations is None:
                duration = max(
                    [timestamps[-1]] + [t_off for t_on, t_off, _ in ground_truth[audio_id]]
                )
            else:
                duration = audio_durations[audio_id]
            n_segments = int(np.ceil(duration / segment_length))
            segment_boundaries = np.round(
                np.arange(n_segments+1) * segment_length,
                time_decimals
            )
            segment_onsets = segment_boundaries[:-1]
            segment_offsets = segment_boundaries[1:]
            for class_name in event_classes:
                gt = single_label_ground_truths[class_name][audio_id]
                if len(gt) == 0:
                    segment_targets[class_name].append(
                        np.zeros(n_segments, dtype=np.bool_))
                else:
                    segment_targets[class_name].append(
                        np.any([
                            (segment_onsets < gt_offset)
                            * (segment_offsets > gt_onset)
                            * (segment_offsets > segment_onsets)
                            for gt_onset, gt_offset in
                            single_label_ground_truths[class_name][audio_id]
                        ], axis=0)
                    )
            for i in range(n_segments):
                idx_on = get_first_index_where(
                    timestamps, "gt", segment_onsets[i]) - 1
                idx_on = max(idx_on, 0)
                idx_off = get_first_index_where(
                    timestamps, "geq", segment_offsets[i])
                idx_off = min(idx_off, len(timestamps)-1)
                if idx_off <= idx_on:
                    scores_ki = np.zeros(scores_k.shape[-1])
                else:
                    scores_ki = np.max(scores_k[idx_on:idx_off], axis=0)
                for c, class_name in enumerate(event_classes):
                    segment_scores[class_name].append(scores_ki[c])
        if output_queue is not None:
            output_queue.put((segment_scores, segment_targets))
        return segment_scores, segment_targets

    if num_jobs == 1:
        segment_scores, segment_targets = worker(audio_ids)
    else:
        queue = multiprocessing.Queue()
        shard_size = int(np.ceil(len(audio_ids) / num_jobs))
        shards = [
            audio_ids[i*shard_size:(i+1)*shard_size] for i in range(num_jobs)
            if i*shard_size < len(audio_ids)
        ]
        processes = [
            multiprocessing.Process(
                target=worker, args=(shard, queue), daemon=True,
            )
            for shard in shards
        ]
        try:
            for p in processes:
                p.start()
            segment_scores, segment_targets = None, None
            count = 0
            while count < len(shards):
                seg_scores_i, seg_targets_i = queue.get()
                if segment_scores is None:
                    segment_scores = seg_scores_i
                    segment_targets = seg_targets_i
                else:
                    for class_name in segment_scores:
                        segment_scores[class_name].extend(seg_scores_i[class_name])
                        segment_targets[class_name].extend(seg_targets_i[class_name])
                count += 1
        finally:
            for p in processes:
                p.terminate()
    stats = {}
    for class_name in event_classes:
        segment_scores[class_name] = np.array(segment_scores[class_name]+[np.inf])
        sort_idx = np.argsort(segment_scores[class_name])
        segment_scores[class_name] = segment_scores[class_name][sort_idx]
        segment_targets[class_name] = np.concatenate(
            segment_targets[class_name]+[np.zeros(1)])[sort_idx]
        tps = np.cumsum(segment_targets[class_name][::-1])[::-1]
        n_sys = np.arange(len(tps))[::-1]
        segment_scores[class_name], unique_idx = np.unique(segment_scores[class_name], return_index=True)
        n_ref = tps[0]
        fns = n_ref - tps
        tns = n_sys[0] - n_sys - fns
        stats[class_name] = {
            'tps': tps[unique_idx],
            'fps': n_sys[unique_idx] - tps[unique_idx],
            'tns': tns,
            'n_ref': n_ref,
        }
    return {
        class_name: (segment_scores[class_name], stats[class_name])
        for class_name in event_classes
    }
