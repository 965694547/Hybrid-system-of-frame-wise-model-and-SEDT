import os.path
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import lazy_dataset
from sed_scores_eval.utils.scores import (
    create_score_dataframe,
    validate_score_dataframe,
    get_unique_thresholds,
)
from sed_scores_eval.base_modules.detection import scores_to_event_list
from sed_scores_eval.base_modules.ground_truth import (
    onset_offset_times_to_indices
)
from urllib.request import urlretrieve
import pdb

def parse_inputs(scores, ground_truth, *, tagging=False):
    """read scores and ground_truth from files if string or path provided and
    validate audio ids

    Args:
        scores (dict, str, pathlib.Path): dict of SED score DataFrames
            (cf. sed_scores_eval.utils.scores.create_score_dataframe)
            or a directory path (as str or pathlib.Path) from where the SED
            scores can be loaded.
        ground_truth (dict, str or pathlib.Path): dict of lists of ground truth
            event tuples (onset, offset, event label) for each audio clip or a
            file path from where the ground truth can be loaded.

    Returns:
        scores:
        ground_truth:
        audio_ids:

    """
    if not isinstance(scores, (dict, str, Path, lazy_dataset.Dataset)):
        raise ValueError(
            f'scores must be dict, str, pathlib.Path or lazy_dataset.Dataset '
            f'but {type(scores)} was given.'
        )
    if not isinstance(ground_truth, (dict, str, Path, object)):
        raise ValueError(
            f'ground_truth must be dict, str or Path but {type(ground_truth)} '
            f'was given.'
        )
    if isinstance(scores, (str, Path)):
        scores = Path(scores)
        scores = lazy_sed_scores_loader(scores)
    audio_ids = sorted(scores.keys())
    if isinstance(ground_truth, (str, Path, object)):
        if isinstance(ground_truth, (str, Path)):
            ground_truth = Path(ground_truth)
        if tagging:
            ground_truth, _ = read_ground_truth_tags(ground_truth)
        else:
            ground_truth = read_ground_truth_events(ground_truth)
    if not ground_truth.keys() == set(audio_ids):
        """
        for key in (ground_truth.keys() - set(audio_ids)):
            ground_truth.pop(key)
        """
        #pdb.set_trace()
        raise ValueError(
            f'ground_truth audio ids do not match audio ids in scores. '
            f'Missing ids: {set(audio_ids) - ground_truth.keys()}. '
            f'Additional ids: {ground_truth.keys() - set(audio_ids)}.'
        )
    return scores, ground_truth, audio_ids

def make_sed_scores(strong_preds, filenames, timestamp, event_classes, score_file):
    batch_size, class_num, timeslen = strong_preds.shape
    timestamps = np.round([t * timestamp for t in range(0, timeslen + 1)], 3)
    for i in range(batch_size):
        filepath = os.path.join(score_file, os.path.basename(filenames[i].replace('.wav', '.tsv')))
        c_preds = strong_preds[i]
        pred = c_preds.transpose(0, 1).detach().cpu().numpy()
        write_sed_scores(pred, filepath, timestamps=timestamps, event_classes=event_classes)

def write_sed_scores(scores, filepath, *, timestamps=None, event_classes=None):
    """write sound event detection scores to tsv file

    Args:
        scores (pandas.DataFrame): containing onset and offset times
            of a score window in first two columns followed by sed score
            columns for each event class.
        filepath (str or pathlib.Path): path to file that is to be written
        timestamps (np.ndarray or list of float): optional list of timestamps
            to be compared with timestamps in scores DataFrame
        event_classes (list of str): optional list of event classes used to
            assert correct event labels in scores DataFrame

    """
    if not isinstance(scores, (np.ndarray, pd.DataFrame)):
        raise ValueError(
            f'scores must be np.ndarray or pd.DataFrame but {type(scores)}'
            f'was given.'
        )
    if isinstance(scores, np.ndarray):
        if timestamps is None:
            raise ValueError(
                f'timestamps must not be None if scores is np.ndarray'
            )
        if event_classes is None:
            raise ValueError(
                f'event_classes must not be None if scores is np.ndarray'
            )
        scores = create_score_dataframe(scores, timestamps, event_classes)
    validate_score_dataframe(scores, timestamps=timestamps, event_classes=event_classes)
    scores.to_csv(filepath, sep='\t', index=False)


def read_sed_scores(filepath):
    scores = pd.read_csv(filepath, sep='\t')
    validate_score_dataframe(scores)
    return scores


def lazy_sed_scores_loader(dir_path):
    """lazy loader for sound event detection files in a directory. This is
    particularly useful if scores do not fit in memory for all audio files
    simultaneously.

    Args:
        dir_path (str or pathlib.Path): path to directory with sound event
            detection files
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(str(dir_path))
    score_files = {}
    for file in sorted(dir_path.iterdir()):
        """
        if not file.is_file() or not file.name.endswith('.tsv'):
            raise ValueError('dir_path must only contain tsv files.')
        
        score_files[file.name[:-len('.tsv')]] = str(file)
        """
        score_files[file.name] = str(file)
    scores = lazy_dataset.new(score_files)
    return scores.map(read_sed_scores)


def read_ground_truth_events(filepath):
    """read ground truth events from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event class) for each audio clip.

    """
    ground_truth = {}
    if isinstance(filepath, (Path)):
        file = pd.read_csv(filepath, sep='\t')
    else:
        file = filepath
    if not all([
        name in list(file.columns)
        for name in ['filename', 'onset', 'offset', 'event_label']
    ]):
        raise ValueError(
            f'ground_truth events file must contain columns "filename", '
            f'"onset", "offset" and "event_label" but only columns '
            f'{list(file.columns)} were found.'
        )
    for filename, onset, offset, event_label in zip(
        file['filename'], file['onset'], file['offset'], file['event_label']
    ):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        if example_id not in ground_truth:
            ground_truth[example_id] = []
        if isinstance(event_label, str):
            assert len(event_label) > 0
            ground_truth[example_id].append([
                float(onset), float(offset), event_label
            ])
        else:
            # file without active events
            #assert np.isnan(event_label), event_label
            assert event_label == None, event_label
    return ground_truth


def read_ground_truth_tags(filepath):
    """read ground truth tags from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        tags (dict of lists): list of active events for each audio file.
        class_counts (dict of ints): number of files in which event_class is
            active for each event_class

    """
    tags = {}
    if isinstance(filepath, (Path)):
        file = pd.read_csv(filepath, sep='\t')
    else:
        file = filepath
    if not all([
        name in list(file.columns) for name in ['filename', 'event_label']
    ]):
        raise ValueError(
            f'ground_truth tags file must contain columns "filename", '
            f'and "event_label" but only columns {list(file.columns)} were '
            f'found.'
        )
    class_counts = {}
    for filename, event_labels in zip(file['filename'], file['event_label']):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        if example_id not in tags:
            tags[example_id] = []
        if isinstance(event_labels, str):
            event_labels = event_labels.split(',')
            for label in event_labels:
                tags[example_id].append(label)
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1
        else:
            # file without active events
            assert np.isnan(event_labels), event_labels
    return tags, class_counts


def read_audio_durations(filepath):
    """read audio clip durations from tsv file

    Args:
        filepath (str or pathlib.Path): path to file that is to be read.

    Returns:
        audio_duration (dict of floats): audio duration in seconds for each
            audio file

    """
    audio_duration = {}
    file = pd.read_csv(filepath, sep='\t')
    assert [
        name in list(file.columns) for name in ['filename', 'duration']
    ], list(file.columns)
    for filename, duration in zip(file['filename'], file['duration']):
        example_id = filename.rsplit('.', maxsplit=1)[0]
        audio_duration[example_id] = float(duration)
    return audio_duration


def write_detection(
        scores, threshold, filepath, audio_format='wav'
):
    """perform thresholding of sound event detection scores and write detected
    events to tsv file

    Args:
        scores (dict of pandas.DataFrame): each DataFrame containing onset and
            offset times of a score window in first two columns followed by
            sed score columns for each event class. Dict keys have to be
            filenames without audio format ending.
        threshold ((dict of) float): threshold that is to be evaluated.
        filepath (str or pathlib.Path): path to file that is to be written/extended.
        audio_format: the audio format that is required to reconstruct the
            filename from audio ids/keys.

    """
    if not hasattr(scores, 'keys') or not callable(scores.keys):
        raise ValueError('scores must implement scores.keys()')
    keys = sorted(scores.keys())
    _, event_classes = validate_score_dataframe(scores[keys[0]])
    if isinstance(threshold, dict):
        threshold = [threshold[event_class] for event_class in event_classes]
        if not all([np.isscalar(thr) for thr in threshold]):
            raise ValueError('All values of thresholds dict must be scalars')
        threshold = np.asanyarray(threshold)
    elif not np.isscalar(threshold):
        raise ValueError(
            f'threshold must be (dict of) scalar(s) but {type(threshold)} '
            f'was given.'
        )
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        with Path(filepath).open('w') as fid:
            fid.write('filename\tonset\toffset\tevent_label\n')

    with filepath.open('a') as fid:
        event_lists = scores_to_event_list(scores, thresholds=threshold)
        for key, event_list in event_lists.items():
            for t_on, t_off, event_label in event_list:
                fid.write(
                    f'{key}.{audio_format}\t{t_on}\t{t_off}\t{event_label}\n')


def write_detections_for_multiple_thresholds(
        scores, thresholds, dir_path, audio_format='wav', score_transform=None,
):
    """writes a detection for multiple thresholds (operating points) as
    required by the psds_eval package (https://github.com/audioanalytic/psds_eval).
    This function is primarily used for testing purposes.

    Args:
        scores (dict of pandas.DataFrame): each DataFrame containing onset and
            offset times of a score window in first two columns followed by
            sed score columns for each event class. Dict keys have to be
            filenames without audio format ending.
        thresholds (np.array): an array of decision thresholds for each of
            which a detection file is written.
        dir_path (str or pathlib.Path): path to directory where to save
            detection files.
        audio_format: the audio format that is required to reconstruct the
            filename from audio ids/keys.
        score_transform:

    """
    if not hasattr(scores, 'keys') or not callable(scores.keys):
        raise ValueError('scores must implement scores.keys()')
    keys = sorted(scores.keys())
    thresholds = np.asanyarray(thresholds)
    if thresholds.ndim != 1:
        raise ValueError(
            f'thresholds must be a 1-dimensional array but has shape '
            f'{thresholds.shape}.'
        )
    dir_path = Path(dir_path)

    if score_transform is not None:
        if isinstance(score_transform, (str, Path)):
            score_transform = read_score_transform(score_transform)
        if not callable(score_transform):
            raise ValueError('score_transform must be callable.')
        if isinstance(scores, lazy_dataset.Dataset):
            scores = scores.map(score_transform)
        else:
            scores = {
                key: score_transform(scores_i)
                for key, scores_i in scores.items()
            }
    for key in keys:
        scores_i = scores[key]
        for threshold in thresholds:
            write_detection(
                {key: scores_i}, threshold,
                dir_path / '{:.3f}.tsv'.format(threshold),
                audio_format=audio_format,
            )


def write_score_transform(
        scores, ground_truth, filepath,
        num_breakpoints=1001, min_score=0., max_score=1.
):
    """compute and save a piecewise-linear score transform which is supposed
    to uniformly distribute scores from within ground truth events between 0
    and 1. This allows to obtain smoother PSD-ROC curve approximations when
    using the psds_eval package (https://github.com/audioanalytic/psds_eval)
    with linearly spaced thresholds.
    This function is primarily used for testing purposes.

    Args:
        scores (dict of pandas.DataFrames): score DataFrames for each audio
            clip of a data set. Each DataFrame contains onset and offset times
            of a score window  in first two columns followed by sed score
            columns for each event class.
        ground_truth (dict of lists of tuples): list of ground truth event
            tuples (onset, offset, event class) for each audio clip.
        filepath (str or pathlib.Path): path to file that is to be written.
        num_breakpoints: the number of breakpoints in the piecewise-linear
            transformation function.
        min_score: the first value (where y=x) in the transformation.
        max_score: the last value (where y=x) in the transformation.

    """
    scores, ground_truth, keys = parse_inputs(scores, ground_truth)
    scores_at_positives = {}
    for key in keys:
        scores_for_key = scores[key]
        validate_score_dataframe(scores_for_key)
        onset_times = scores_for_key['onset'].to_numpy()
        offset_times = scores_for_key['offset'].to_numpy()
        timestamps = np.concatenate((onset_times, offset_times[-1:]))
        for (t_on, t_off, event_label) in ground_truth[key]:
            idx_on, idx_off = onset_offset_times_to_indices(
                onset_time=t_on, offset_time=t_off, timestamps=timestamps
            )
            if event_label not in scores_at_positives:
                scores_at_positives[event_label] = []
            scores_at_positives[event_label].append(
                scores_for_key[event_label].to_numpy()[idx_on:idx_off])
    output_scores = np.linspace(min_score, max_score, num_breakpoints)
    score_transform = [output_scores]
    #pdb.set_trace()
    event_classes = sorted(scores_at_positives.keys())
    for event_class in event_classes:
        scores_k = np.concatenate(scores_at_positives[event_class])
        thresholds, *_ = get_unique_thresholds(scores_k)
        assert len(thresholds) >= num_breakpoints, (len(thresholds), num_breakpoints)
        breakpoint_indices = np.linspace(
            0, len(thresholds), num_breakpoints)[1:-1].astype(np.int)
        #pdb.set_trace()
        assert (thresholds[breakpoint_indices] >= min_score).all(), (
            np.min(thresholds[breakpoint_indices]), min_score)
        assert (thresholds[breakpoint_indices] <= max_score).all(), (
            np.max(thresholds[breakpoint_indices]), max_score)
        breakpoints = np.concatenate((
            [min_score], thresholds[breakpoint_indices], [max_score]
        ))
        score_transform.append(breakpoints)
    score_transform = pd.DataFrame(
        np.array(score_transform).T, columns=['y', *event_classes])
    score_transform.to_csv(filepath, sep='\t', index=False)
    return score_transform


def read_score_transform(filepath):
    """read a piecewise linear score transform from tsv file

    Args:
        filepath: path to tsv file as written by write_score_transform

    Returns:
        score_transform: function which takes scores as pd.DataFrame and
            returns the transformed scores as pd.DataFrame

    """
    transform = pd.read_csv(filepath, sep='\t')
    column_names = list(transform.columns)
    assert len(column_names) > 1, column_names
    assert column_names[0] == 'y', column_names
    event_classes = column_names[1:]
    y = transform['y'].to_numpy()

    def score_transform(scores):
        validate_score_dataframe(scores, event_classes=event_classes)
        transformed_scores = [
            scores['onset'].to_numpy(), scores['offset'].to_numpy()
        ]
        for event_class in event_classes:
            x = transform[event_class].to_numpy()
            transformed_scores.append(interp1d(
                x, y, kind='linear',
            )(scores[event_class]))
        transformed_scores = pd.DataFrame(
            np.array(transformed_scores).T,
            columns=['onset', 'offset', *event_classes],
        )
        return transformed_scores

    return score_transform


def download_test_data():
    from sed_scores_eval import package_dir
    import zipfile
    tests_dir_path = package_dir / 'tests'
    if (tests_dir_path / 'data').exists():
        print('Test data already exists.')
        return
    print('Download test data')
    zip_file_path = tests_dir_path / 'data.zip'
    urlretrieve(
        'http://go.upb.de/sed_scores_eval_test_data',
        filename=str(zip_file_path)
    )
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(tests_dir_path)
    zip_file_path.unlink()
    print('Download successful')
