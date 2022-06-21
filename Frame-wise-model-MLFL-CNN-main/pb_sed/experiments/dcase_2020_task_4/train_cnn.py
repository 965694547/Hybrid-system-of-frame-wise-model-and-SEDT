"""
This script trains a (tag-conditioned) CNN model.

For model details see
http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Ebbers_69.pdf

CNN training relies on (pseudo) strong labels.
Available datasets:

strong pseudo labels used in paper generated by five different FBCRNN ensemble:
weak_pseudo_strong_2020-07-04-13-10-05_best_frame_f1_crnn
weak_pseudo_strong_2020-07-04-13-10-19_best_frame_f1_crnn
weak_pseudo_strong_2020-07-04-13-10-33_best_frame_f1_crnn
weak_pseudo_strong_2020-07-04-13-11-09_best_frame_f1_crnn
weak_pseudo_strong_2020-07-04-13-12-06_best_frame_f1_crnn
unlabel_in_domain_pseudo_strong_2020-07-04-13-10-05_best_frame_f1_crnn
unlabel_in_domain_pseudo_strong_2020-07-04-13-10-19_best_frame_f1_crnn
unlabel_in_domain_pseudo_strong_2020-07-04-13-10-33_best_frame_f1_crnn
unlabel_in_domain_pseudo_strong_2020-07-04-13-11-09_best_frame_f1_crnn
unlabel_in_domain_pseudo_strong_2020-07-04-13-12-06_best_frame_f1_crnn

strong pseudo labels generated by five different hybrid models (4 FBCRNN + 4 tag-conditioned CNNs):
weak_pseudo_strong_2020-07-05-12-37-18_best_frame_f1_hybrid
weak_pseudo_strong_2020-07-05-12-37-26_best_frame_f1_hybrid
weak_pseudo_strong_2020-07-05-12-37-35_best_frame_f1_hybrid
weak_pseudo_strong_2020-07-05-12-37-45_best_frame_f1_hybrid
weak_pseudo_strong_2020-07-05-12-37-54_best_frame_f1_hybrid
unlabel_in_domain_pseudo_strong_2020-07-05-12-37-18_best_frame_f1_hybrid
unlabel_in_domain_pseudo_strong_2020-07-05-12-37-26_best_frame_f1_hybrid
unlabel_in_domain_pseudo_strong_2020-07-05-12-37-35_best_frame_f1_hybrid
unlabel_in_domain_pseudo_strong_2020-07-05-12-37-45_best_frame_f1_hybrid
unlabel_in_domain_pseudo_strong_2020-07-05-12-37-54_best_frame_f1_hybrid

Example calls:
train a tag-conditioned CNN model:
python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with 'pseudo_strong_suffix=2020-07-05-12-37-18_best_frame_f1_hybrid'

train an unconditioned CNN model:
python -m pb_sed.experiments.dcase_2020_task_4.train_cnn with trainer.model.tag_conditioning=False 'pseudo_strong_suffix=2020-07-05-12-37-18_best_frame_f1_hybrid'

Training checkpoints and meta data are stored in a directory
/path/to/storage_root/dcase_2020_cnn/<timestamp>
"""
from pathlib import Path

import numpy as np
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.modules.augment import (
    MelWarping, LogTruncNormalSampler, TruncExponentialSampler
)
from padertorch.train.hooks import LRAnnealingHook
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.cnn import CNN
from pb_sed.paths import storage_root
from sacred import Experiment as Exp
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from codecarbon import EmissionsTracker
import os
ex_name = 'dcase_2020_cnn'
ex = Exp(ex_name)


@ex.config
def config():
    debug = False

    # Data configuration
    pseudo_strong_suffix = ''

    weak_pseudo_strong_suffix = pseudo_strong_suffix
    assert len(weak_pseudo_strong_suffix) > 0, 'Set weak_pseudo_strong_suffix on the command line.'
    weak_pseudo_strong = \
        f'weak_pseudo_strong_{weak_pseudo_strong_suffix}'

    unlabel_in_domain_pseudo_strong_suffix = pseudo_strong_suffix
    assert len(unlabel_in_domain_pseudo_strong_suffix) > 0, 'Set unlabel_in_domain_pseudo_strong_suffix on the command line.'
    unlabel_in_domain_pseudo_strong = \
        f'unlabel_in_domain_pseudo_strong_{unlabel_in_domain_pseudo_strong_suffix}'

    dataset_repetitions = {
        weak_pseudo_strong: 10,
        'synthetic': 2,
        unlabel_in_domain_pseudo_strong: 1,
    }
    audio_reader = {
        'source_sample_rate': None,
        'target_sample_rate': 16000,
    }
    cached_datasets = [] if debug else ['synthetic', weak_pseudo_strong]
    stft = {
        'shift': 320,
        'window_length': 960,
        'size': 1024,
        'fading': None,
        'pad': False,
    }

    mixup_probs = (.5, .5)
    max_mixup_length = int(12.*audio_reader['target_sample_rate']/stft['shift']) + 1
    batch_size = 24
    min_examples = {
        **{ds: 0 for ds in dataset_repetitions},
        weak_pseudo_strong: int(batch_size/3),
    }
    num_workers = 8
    prefetch_buffer = 10 * batch_size
    max_total_size = None
    max_padding_rate = 0.05
    bucket_expiration = 2000 * batch_size

    # Trainer configuration
    subdir = str(Path(ex_name) / timeStamped('')[1:])
    trainer = {
        'model': {
            'factory':  CNN,
            'tag_conditioning': True,
            'feature_extractor': {
                'sample_rate': audio_reader['target_sample_rate'],
                'fft_length': stft['size'],
                'n_mels': 128,
                'warping_fn': {
                    'factory': MelWarping,
                    'alpha_sampling_fn': {
                        'factory': LogTruncNormalSampler,
                        'scale': .08,
                        'truncation': np.log(1.3),
                    },
                    'fhi_sampling_fn': {
                        'factory': TruncExponentialSampler,
                        'scale': .5,
                        'truncation': 5.,
                    },
                },
                'max_resample_rate': 1.,
                'blur_sigma': .5,
                'n_time_masks': 0,
                'max_masked_time_steps': 70,
                'max_masked_time_rate': .2,
                'n_mel_masks': 1,
                'max_masked_mel_steps': 20,
                'max_masked_mel_rate': .2,
                'max_noise_scale': .2,
            },
            'cnn_2d': {
                'out_channels': [16, 16, 32, 32, 64, 64, 128, 128, 256],
                'pool_size': [1, (2, 1), 1, (2, 1), 1, (2, 1), 1, (2, 1), (2, 1)],
                'output_layer': False,
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
            'cnn_1d': {
                'out_channels': 2*[256] + [10],
                'kernel_size': 3,
                'norm': 'batch',
                'activation_fn': 'relu',
                'dropout': .0,
            },
        },
        'optimizer': {
            'factory': Adam,
            'lr': 5e-4,
            'gradient_clipping': 20.,
            'weight_decay': 1e-6,
        },
        'storage_dir': str(storage_root / subdir),
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'stop_trigger': (2, 'iteration')
        #'stop_trigger': (40000, 'iteration')
    }
    Trainer.get_config(trainer)
    resume = False
    rampup_steps = 1000
    lr_decay_step = 15000

    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))


@ex.automain
def train(
        _run,
        dataset_repetitions, audio_reader, cached_datasets, stft,
        mixup_probs, max_mixup_length,
        num_workers, prefetch_buffer,
        batch_size, max_padding_rate, bucket_expiration, min_examples,
        rampup_steps, lr_decay_step,
        trainer, resume,
):

    print_config(_run)
    trainer = Trainer.from_config(trainer)

    train_iter = data.get_train(
        dataset_repetitions=dataset_repetitions,
        audio_reader=audio_reader, stft=stft,
        mixup_probs=mixup_probs, max_mixup_length=max_mixup_length,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration, min_examples=min_examples,
        storage_dir=trainer.storage_dir,
        add_alignment=True,
        cached_datasets=cached_datasets,
    )
    validation_set = data.get_dataset(
        'validation', audio_reader=audio_reader,
    )
    validation_iter = data.prepare_dataset(
        validation_set,
        storage_dir=trainer.storage_dir,
        audio_reader=audio_reader, stft=stft,
        num_workers=num_workers, prefetch_buffer=prefetch_buffer,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        bucket_expiration=bucket_expiration,
        add_alignment=True,
    )

    trainer.test_run(train_iter, validation_iter)

    trainer.register_validation_hook(
        validation_iter, metric='mean_fscore', maximize=True
    )
    trainer.register_hook(LRAnnealingHook(
        trigger=(100, 'iteration'),
        breakpoints=[
            (0, 0.),
            (rampup_steps, 1.),
            (lr_decay_step, 1.),
            (lr_decay_step, 1/5),
        ],
        unit='iteration',
    ))
    os.makedirs("training_codecarbon", exist_ok=True)
    tracker_train = EmissionsTracker("DCASE Task 4 SEDT TRAINING", output_dir="training_codecarbon")
    tracker_train.start()
    trainer.train(train_iter, resume=resume)
    tracker_train.stop()
    print(40000)

