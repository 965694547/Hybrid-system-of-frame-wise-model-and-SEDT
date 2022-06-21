"""data preparation functions

We use the lazy_dataset package to perform data loading and transformation.
See https://github.com/fgnt/lazy_dataset for more information aboutlazy_dataset.

Note that we here only load and normalize waveforms and extract STFTs, while
log mel-band energy extraction and normalization is performed in the pytorch models.
Similarly we here only perform random scaling of waveforms and mixup at
STFT-level, while other data augmentations are performed in the pytorch models.
"""
from pathlib import Path
from copy import copy

import lazy_dataset
import numpy as np
from lazy_dataset.core import DynamicTimeSeriesBucket
from padertorch.contrib.je.data.mixup import MixUpDataset, \
    SampleMixupComponents, SuperposeEvents
from padertorch.contrib.je.data.transforms import (
    AudioReader, STFT, MultiHotLabelEncoder, MultiHotAlignmentEncoder, Collate
)
from padertorch.contrib.je.modules.augment import LogTruncNormalSampler
from pb_sed.database.desed.database import DESED

db = DESED()


def get_train(
        dataset_repetitions,
        audio_reader, stft,
        num_workers, prefetch_buffer,
        batch_size, max_padding_rate, bucket_expiration,
        storage_dir,
        add_alignment=False,
        mixup_probs=(1/3, 2/3), max_mixup_length=None,
        min_examples=None,
        cached_datasets=None,
        unlabeled=False,
        max_chunk_len=None, chunk_overlap=0,
):
    """prepare dataset

    Args:
        dataset_repetitions: dict with dataset names as keys and an integer
            value indicating the number of repetitions to balance the datasets
        audio_reader: AudioReader kwargs
        stft: STFT kwargs
        num_workers: number of workers used in prefetching
        prefetch_buffer: number of batches to be prefetched
        batch_size:
        max_padding_rate: maximum amount of padding allowed. Determines which
            clips can be joined to a batch (see lazy_dataset.DynamicTimeSeriesBucket)
        bucket_expiration: maximum number of examples to wait for a bucket to
            complete (see lazy_dataset.DynamicTimeSeriesBucket)
        storage_dir: location to store event label mapping
        add_alignment: whether frame-level labels should be added to example
            dicts
        mixup_probs: probabilities for the number of components in a mixture
            (superpposition in our case). E.g., for mixup_prob=(1,) no mixup is
            performed and for mixup_prob=(.5,.5) no mixup is performed with a
            probability of .5 and mixup of two clips is performed with a
            probability of .5 .
        max_mixup_length: In our mixup implementation audio clips may be
            shifted to each other before superposition. max_mixup_length limits
            the shift such that the superposition is not longer than
            max_mixup_length frames.
        min_examples: a dict with dataset names as keys and an integer value
            indicating the minimum number of examples from a dataset in each
            batch.
        cached_datasets: list of dataset_names to be cached in memory.
        unlabeled: whether to discard all labels
        max_chunk_len: the maximum number of frames allowed in a clip. Clips
            exceeding max_chunk_len will be split into chunks.
        chunk_overlap: overlap rate of adjacent chunks

    Returns:
        dataset: lazy_dataset (iteratable) lazily loading example dicts
            containing an stft plus meta data.

    """
    def maybe_remove_start_stop_times(example):
        if not add_alignment:
            if "events_start_times" in example:
                example.pop("events_start_times")
            if "events_stop_times" in example:
                example.pop("events_stop_times")
        return example

    def random_scale(example):
        c = example['audio_data'].shape[0]
        scales = LogTruncNormalSampler(scale=1., truncation=3.)(c)[:, None]
        example['audio_data'] *= scales
        return example

    datasets = {
        name: get_dataset(
            name, audio_reader, cache=(
                cached_datasets is not None and name in cached_datasets
            )
        ).map(maybe_remove_start_stop_times).map(random_scale)
        for name in dataset_repetitions if dataset_repetitions[name] > 0
    }

    # interleave
    training_set = lazy_dataset.intersperse(
        *[
            ds.shuffle(reshuffle=True).tile(dataset_repetitions[name])
            for name, ds in datasets.items()
        ]
    )
    print('Total train set length:', len(training_set))

    return prepare_dataset(
        training_set,
        storage_dir=storage_dir,
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration,
        min_examples=min_examples,
        add_alignment=add_alignment,
        local_shuffle=True,
        drop_incomplete=True,
        mixup_probs=mixup_probs,
        max_mixup_length=max_mixup_length,
        unlabeled=unlabeled,
        max_chunk_len=max_chunk_len,
        chunk_overlap=chunk_overlap,
    )


def get_dataset(name, audio_reader, cache=False):
    """get a dataset from DESED database which lazily loads audio data (or caches it in memory).

    Args:
        name: Name of the requested dataset
        audio_reader: AudioReader kwargs
        cache: whether to cache the dataset in memory

    Returns:
        dataset: lazy_dataset (iteratable) lazily loading (cached) example dicts
            containing an audio waveform plus meta data.

    """
    ds = db.get_dataset(name)
    ds = ds.filter(lambda ex: ex['audio_length'] > 1., lazy=False)
    print(f'Data set length {name}:', len(ds))

    audio_reader = AudioReader(**audio_reader)
    if name == "synthetic":
        # load background and event audio separately
        def load_data(example):
            event_files = Path(example['audio_path'][:-len('.wav')] + '_events').glob('*.wav')
            audio_data = []
            for i, event_file in enumerate(sorted(event_files)):
                signal = audio_reader.read_file(event_file)
                if 'background' in event_file.name:
                    assert i == 0, i
                    onset, offset = 0, len(signal[0])
                else:
                    idx = sorted(np.argwhere(signal[0]**2 > 0).flatten())
                    onset, offset = min(idx), max(idx)
                    signal = signal[:, onset:offset]
                signal -= signal.mean()
                audio_data.append((signal, onset, offset))
            return example, audio_data

        ds = ds.map(load_data)
        if cache:
            ds = ds.cache(lazy=False).map(lambda ex: copy(ex))

        # perform on-the-fly reverberation
        rooms = db.get_dataset('rir_data_train')

        def reverberate(example_audio_data):
            example, audio_data = example_audio_data
            room = rooms.random_choice()
            T = audio_data[0][0].shape[-1]
            for signal, onset, offset in audio_data:
                rir = room['rirs'][int(np.random.choice(len(room['rirs'])))]
                rir = audio_reader.read_file(rir)[0, :2000]
                if 'audio_data' not in example:
                    # first item is background which is not convolved but only
                    # scaled here as it is already a real recording
                    example['audio_data'] = (np.sqrt((rir**2).sum()) * signal)
                else:
                    sound = np.convolve(signal[0], rir)
                    offset = onset + len(sound)
                    example['audio_data'][:, onset:offset] += sound[:T-onset]
            example['audio_data'] *= 70
            return example

        ds = ds.map(reverberate)
    else:
        ds = ds.map(audio_reader)
        if cache:
            ds = ds.cache(lazy=False).map(lambda ex: copy(ex))

    def normalize(example):
        example['audio_data'] -= example['audio_data'].mean(-1, keepdims=True)
        example['audio_data'] = example['audio_data'].mean(0, keepdims=True)
        example['audio_data'] /= np.abs(example['audio_data']).max() + 1e-3
        return example

    return ds.map(normalize)


def prepare_dataset(
        dataset, storage_dir,
        audio_reader, stft,
        num_workers, prefetch_buffer,
        batch_size, max_padding_rate, bucket_expiration,
        unlabeled=False,
        add_alignment=False,
        local_shuffle=False,
        drop_incomplete=False,
        mixup_probs=(1., 0.), max_mixup_length=None, min_mixup_overlap=.5,
        min_examples=None,
        max_chunk_len=None, chunk_overlap=0,
):
    """

    Args:
        dataset: lazy_dataset as returned by get_dataset
        storage_dir: location to store event label mapping
        audio_reader: AudioReader kwargs
        stft: STFT kwargs
        num_workers: number of workers used in prefetching
        prefetch_buffer: number of batches to be prefetched
        batch_size:
        max_padding_rate: maximum amount of padding allowed. Determines which
            clips can be joined to a batch (see lazy_dataset.DynamicTimeSeriesBucket)
        bucket_expiration: maximum number of examples to wait for a bucket to
            complete (see lazy_dataset.DynamicTimeSeriesBucket)
        unlabeled: whether to discard labels (if available)
        add_alignment: whether frame-level labels should be added to example
            dicts
        local_shuffle: whether to perform local shuffle when max_chunk_len is
            not None. This serves the purpose to not always have the chunks of
            a clip in the same batch.
        drop_incomplete: whether to drop expired buckets or process it with a
            smaller batch_size
        mixup_probs: probabilities for the number of components in a mixture
            (superpposition in our case). E.g., for mixup_prob=(1,) no mixup is
            performed and for mixup_prob=(.5,.5) no mixup is performed with a
            probability of .5 and mixup of two clips is performed with a
            probability of .5 .
        max_mixup_length: In our mixup implementation audio clips may be
            shifted to each other before superposition. max_mixup_length limits
            the shift such that the superposition is not longer than
            max_mixup_length frames.
        min_mixup_overlap: limits the shift explained above such that
            superposed clips overlap at least at a rate of min_mixup_overlap
        min_examples: a dict with dataset names as keys and an integer value
            indicating the minimum number of examples from a dataset in each
            batch.
        max_chunk_len: the maximum number of frames allowed in a clip. Clips
            exceeding max_chunk_len will be split into chunks.
        chunk_overlap: overlap rate of adjacent chunks

    Returns:
        dataset: lazy_dataset (iteratable) lazily loading example dicts
            containing an stft plus meta data.

    """

    stft = STFT(**stft)
    dataset = dataset.map(stft)

    if unlabeled:
        assert not add_alignment, add_alignment
    else:
        if add_alignment:
            event_encoder = MultiHotAlignmentEncoder(
                label_key='events', storage_dir=storage_dir,
                sample_rate=audio_reader['target_sample_rate'], stft=stft,
            )
        else:
            event_encoder = MultiHotLabelEncoder(
                label_key='events', storage_dir=storage_dir,
            )
        event_encoder.initialize_labels(dataset=db.get_dataset("weak"), verbose=True)
        dataset = dataset.map(event_encoder)

    def finalize(example):
        # print(example['stft'].shape[1])
        example_ = {
            'example_id': example['example_id'],
            'stft': example['stft'].astype(np.float32),
            'seq_len': example['stft'].shape[1],
            'dataset': example['dataset'],
        }
        if not unlabeled:
            example_['events'] = example['events'].T.astype(np.float32)
        if "events_alignment" in example:
            example_["events_alignment"] = example['events_alignment'].T.astype(np.float32)
        if max_chunk_len is not None and example_['seq_len'] > max_chunk_len:
            # print('.')
            n = int(np.ceil((example_['seq_len']-chunk_overlap) / (max_chunk_len - chunk_overlap)))
            chunk_len = (example_['seq_len']-chunk_overlap) / n + chunk_overlap
            examples = []
            for onset in np.arange(0, example_['seq_len']-chunk_overlap, chunk_len - chunk_overlap):
                stft_chunk = example_['stft'][:, int(onset):int(onset + chunk_len)]
                chunk = {
                    'example_id': f'{example_["example_id"]}_!chunk!_{onset}',
                    'stft': stft_chunk,
                    'seq_len': stft_chunk.shape[1],
                    'dataset': example['dataset'],
                }
                if "events_alignment" in example_:
                    chunk["events_alignment"] = example_['events_alignment'][int(onset):int(onset + chunk_len)]
                    chunk["events"] = chunk["events_alignment"].max(0)
                examples.append(chunk)
            return examples
        return [example_]

    dataset = dataset.map(finalize)\
        .prefetch(num_workers, prefetch_buffer, catch_filter_exception=True).unbatch()

    if local_shuffle and max_chunk_len is not None:
        print('Shuffle')
        dataset = dataset.shuffle(reshuffle=True, buffer_size=80*batch_size)

    if mixup_probs[0] < 1.:
        print('Mixup')
        dataset = MixUpDataset(
            dataset,
            sample_fn=SampleMixupComponents(mixup_probs),
            mixup_fn=SuperposeEvents(min_overlap=min_mixup_overlap, max_length=max_mixup_length),
            buffer_size=80*batch_size,
        )
    if min_examples is None:
        return dataset.batch_dynamic_time_series_bucket(
            batch_size=batch_size, len_key="seq_len",
            max_padding_rate=max_padding_rate, expiration=bucket_expiration,
            drop_incomplete=drop_incomplete, sort_key="seq_len", reverse_sort=True
        ).map(Collate())
    return dataset.batch_dynamic_bucket(
        bucket_cls=DatasetBalancedTimeSeriesBucket, min_examples=min_examples,
        batch_size=batch_size, len_key="seq_len",
        max_padding_rate=max_padding_rate, expiration=bucket_expiration,
        drop_incomplete=drop_incomplete, sort_key="seq_len", reverse_sort=True
    ).map(Collate())


class DatasetBalancedTimeSeriesBucket(DynamicTimeSeriesBucket):
    def __init__(self, init_example, min_examples, **kwargs):
        """
        Extension of the DynamicTimeSeriesBucket such that examples are
        balanced with respect to the dataset they originate from

        Args:
            init_example: first example in the bucket
            min_examples: a dict with dataset_names as keys and integer values
                indicating the minimum number of examples from that dataset
                within each batch
            **kwargs: kwargs of DynamicTimeSeriesBucket
        """
        super().__init__(init_example, **kwargs)
        self.missing = {key: value for key, value in min_examples.items()}

    def assess(self, example):
        names = example['dataset'].split('+')  # '+' indicates mixtures
        assert all([name in self.missing for name in names]), (
            names, sorted(self.missing.keys())
        )
        return (
            super().assess(example) and (
                (self.batch_size - len(self.data)) > sum(self.missing.values())
                or
                any([self.missing[name] > 0 for name in names])
            )
        )

    def _append(self, example):
        super()._append(example)
        for name in example['dataset'].split('+'):
            if self.missing[name] > 0:
                self.missing[name] -= 1
