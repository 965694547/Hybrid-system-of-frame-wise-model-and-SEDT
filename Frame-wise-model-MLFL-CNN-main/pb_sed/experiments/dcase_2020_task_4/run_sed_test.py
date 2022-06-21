"""
Given tuned hyper parameters this script performs tagging and SED on new
data sets, stores prediction files and prints evaluation if a reference file
(ground truth) is given.

Example call:
run inference on all data sets and print evaluation for validation and
eval_dcase2020 data sets:
python -m pb_sed.experiments.dcase_2020_task_4.run_inference with 'hyper_params_dir=/path/to/storage_root/dcase_2020_hyper_params/<timestamp>' 'dataset_names=["validation", "eval_dcase2019", "weak", "unlabel_in_domain", "eval_dcase2020"]' 'reference_files=["/path/to/desed/real/metadata/validation/validation.tsv", "/path/to/desed/real/metadata/eval/eval_dcase2019.tsv", None, None, None]'

prediction files are stored in a directory
/path/to/storage_root/dcase_2020_inference/<timestamp>
"""
import os
from pathlib import Path

import dcase_util
import numpy as np
import sed_eval
import torch
from paderbox.io.json_module import load_json
from paderbox.transform.module_stft import STFT
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data.transforms import Collate
from padertorch.data import example_to_device
from pb_sed.evaluation import instance_based
from pb_sed.evaluation.event_based import alignments_to_event_list
from pb_sed.experiments.dcase_2020_task_4 import data
from pb_sed.models.cnn import CNN
from pb_sed.models.crnn import CRNN
from pb_sed.paths import storage_root
from pb_sed.utils import join_tsv_files
from pb_sed.utils import medfilt
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
import pdb
ex_name = 'dcase_2020_inference'
ex = Exp(ex_name)
ts = timeStamped('')[1:]


@ex.config
def config():
    subdir = str(Path(ex_name) / ts)
    storage_dir = str(storage_root / subdir)

    hyper_params_dir = ''

    tuning_config = load_json(Path(hyper_params_dir) / '1' / 'config.json')
    crnn_dirs = tuning_config['crnn_dirs']
    crnn_config = tuning_config['crnn_config']
    crnn_checkpoints = tuning_config['crnn_checkpoints']
    cnn_dirs = tuning_config['cnn_dirs']
    cnn_config = tuning_config['cnn_config']
    cnn_checkpoints = tuning_config['cnn_checkpoints']
    tagging_thresholds = str(Path(hyper_params_dir) / 'tagging_thresholds_best_f1.json')
    ensembles = tuning_config['ensembles']
    detection_threshold_names = ['best_frame_f1', 'best_event_f1']
    del tuning_config

    dataset_names = ["validation", "eval_dcase2019"]
    reference_files = None

    batch_size = 16
    max_padding_rate = .1
    num_workers = 8
    device = 0

    debug = False

    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(
    storage_dir,
    crnn_dirs, crnn_config, crnn_checkpoints, cnn_dirs, cnn_checkpoints,
    dataset_names, batch_size, max_padding_rate,
    num_workers, device,
    hyper_params_dir, tagging_thresholds,
    detection_threshold_names, ensembles,
    reference_files, debug
):
    assert all([ensemble in ['hybrid', 'crnn', 'cnn'] for ensemble in ensembles]), ensembles

    best_configs = {
        f'{detection_threshold_name}_{ensemble}': load_json(Path(hyper_params_dir) / f'{detection_threshold_name}_{ensemble}_config.json')
        for detection_threshold_name in detection_threshold_names for ensemble in ensembles
    }

    if reference_files is None:
        reference_files = len(dataset_names) * [None]
    else:
        assert len(reference_files) == len(dataset_names)
    for dataset_name, reference_file in zip(dataset_names, reference_files):
        ds = data.get_dataset(
            dataset_name, audio_reader=crnn_config['audio_reader'],
        )
        ds = data.prepare_dataset(
            ds,
            audio_reader=crnn_config['audio_reader'], stft=crnn_config['stft'],
            num_workers=num_workers, prefetch_buffer=4 * batch_size,
            batch_size=batch_size, max_padding_rate=max_padding_rate,
            bucket_expiration=None,
            min_examples={dataset_name: 0},
            storage_dir=crnn_dirs[0],
            unlabeled=dataset_name in ["unlabel_in_domain", "eval_dcase2020"],
            max_chunk_len=1000,
        )

        if 'weak' in dataset_name:
            # use provided tags
            tags = []
            example_ids = []
            for batch in ds:
                tags.append(batch['events'])
                example_ids.extend(batch['example_id'])
                if debug and len(tags) >= 64/batch_size:
                    break
            tags = np.concatenate(tags)
        else:
            # run tagging
            example_ids = []

            for batch in ds:
                    batch = example_to_device(batch, device)
                    example_ids.extend(batch['example_id'])
            labels = load_json(Path(crnn_dirs[0]) / 'events.json')
            
        labels = load_json(Path(crnn_dirs[0]) / 'events.json')
        medfilt_sizes = sorted(set([
            int(entry.split('_')[-1])
            for detection_threshold_name in detection_threshold_names
            for ensemble in ensembles
            for entry in best_configs[f'{detection_threshold_name}_{ensemble}']
        ]), reverse=True)

        np.save('example_ids.npy', example_ids)
        os.system('python run_sed_tag.py')
        tags = np.load('tags.npy')

        def sweep_medfilts(score_mat, name):
            """given a score_mat sweep median-filter sizes and save SED for
            each label to individual files (to join class-wise SEDs from
            different hyper-parameter sets later on)

            Args:
                score_mat: ensemble scores (num_clips, num_frames, num_classes)
                name: identifier of the model, e.g., crnn_10 for a CRNN with a
                    one-sided context length of 10

            Returns:

            """
            for medfilt_size in medfilt_sizes:
                print(f'  Medfilt size: {medfilt_size}')
                if medfilt_size > 1:
                    score_mat_filtered = medfilt(
                        score_mat.astype(np.float), medfilt_size, axis=1
                    )
                else:
                    assert medfilt_size == 1, medfilt_size
                    score_mat_filtered = score_mat
                for detection_threshold_name in detection_threshold_names:
                    detection_threshold = np.array(load_json(str(Path(hyper_params_dir) / f'detection_thresholds_{detection_threshold_name}_{name}_{medfilt_size}.json')))
                    decision_mat = score_mat_filtered > np.array(detection_threshold)
                    event_list = alignments_to_event_list(decision_mat)
                    # If audio clips have been to long they may have been
                    # cut into smaller chunks (see data.py)
                    # Therefore chunks from the same clip have to be merged.

                    # We first fix the example_id and onset and offset frames
                    # according to the chunk onset which is part of the
                    # example_id:
                    fixed_event_list = []
                    for n, onset, offset, k in event_list:
                        example_id = example_ids[n]
                        if '_!chunk!_' in example_id:
                            example_id, c = example_id.split('_!chunk!_')
                            onset += int(c)
                            offset += int(c)
                        fixed_event_list.append((example_id, onset, offset, k))
                    event_list = sorted(fixed_event_list)

                    # We now merge overlapping events from different chunks
                    i = 0
                    while i < len(event_list):
                        (example_id, onset, offset, k) = event_list[i]
                        j = 1
                        while i+j < len(event_list) and event_list[i+j][0] == example_id and event_list[i+j][1] < offset:
                            if event_list[i+j][3] == k:
                                offset = max(offset, event_list[i+j][2])
                                event_list.pop(i+j)
                            else:
                                j += 1
                        event_list[i] = (example_id, onset, offset, k)
                        i += 1

                    # convert onset-/offset-frames to onset-/offset-times and
                    # save to file
                    sr = crnn_config['audio_reader']['target_sample_rate']
                    stft = STFT(**crnn_config['stft'])
                    for i, label in enumerate(labels):
                        file = Path(storage_dir) / f'{dataset_name}_pseudo_strong_{label}_{detection_threshold_name}_{name}_{medfilt_size}.tsv'
                        # print(file)
                        with Path(file).open('w') as fid:
                            fid.write('filename\tonset\toffset\tevent_label\n')
                            for example_id, onset, offset, k in event_list:
                                if k != i:
                                    continue
                                label = labels[k]
                                onset = stft.frame_index_to_sample_index(
                                    onset, mode='center'
                                ) / sr
                                offset = stft.frame_index_to_sample_index(
                                    offset - 1, mode='center'
                                ) / sr
                                fid.write(f'{example_id}.wav\t{onset}\t{offset}\t{label}\n')

        # if required compute CNN scores of each sub model in CNN ensemble
        cnn_detection_score_mat = None
        if any([ensemble != 'crnn' for ensemble in ensembles]):
            assert len(cnn_dirs) > 0
            cnn_detection_score_mat = []
            for exp_dir, cnn_checkpoint in zip(cnn_dirs, cnn_checkpoints):
                cnn: CNN = CNN.from_storage_dir(
                    storage_dir=exp_dir, config_name='1/config.json',
                    checkpoint_name=cnn_checkpoint
                )
                print('#Params:', sum(p.numel() for p in cnn.parameters() if p.requires_grad))
                cnn.to(device)
                cnn.eval()

                score_mat_i = []
                example_ids_i = []
                i = 0
                #pdb.set_trace()
                with torch.no_grad():
                    for batch in ds:
                        batch = example_to_device(batch, device)
                        x = batch['stft']
                        b = x.shape[0]
                        seq_len = batch['seq_len']
                        example_ids_i.extend(batch['example_id'])
                        (y, seq_len), _ = cnn.predict(
                            x,
                            torch.Tensor(tags[i:i + b].astype(np.float32)).to(x.device),
                            seq_len
                        )
                        y = y.data.cpu().numpy() * tags[i:i + b, :, None]
                        score_mat_i.extend(
                            [y[j, :, :l].T for j, l in enumerate(seq_len)]
                        )
                        if debug and len(score_mat_i) >= 64:
                            break
                        i += b
                cnn.to('cpu')
                cnn_detection_score_mat.append(Collate()(score_mat_i))
                assert example_ids_i == example_ids, (len(example_ids_i), len(example_ids))
            cnn_detection_score_mat = np.mean(cnn_detection_score_mat, axis=0)  # average over sub models in CNN ensemble

            if 'cnn' in ensembles:
                print(f'Ensemble: cnn')
                np.save('cnn_detection_score_mat.npy', cnn_detection_score_mat)
                #sweep_medfilts(cnn_detection_score_mat, 'cnn')

        # if required run CRNN SED for each sub model in CRNN ensemble
        if any([ensemble != 'cnn' for ensemble in ensembles]):
            contexts = sorted(set([
                int(entry.split('_')[-2])
                for detection_threshold_name in detection_threshold_names
                for ensemble in ensembles
                for entry in best_configs[f'{detection_threshold_name}_{ensemble}']
                if ensemble != 'cnn'
            ]), reverse=True)
            context = list()
            for key in best_configs:
                if 'cnn' not in key:
                    for subkey in best_configs[key]:
                        context.append(int(subkey.split('_')[1]))
            contexts = [int(np.mean(context))]
            for context in contexts:
                crnn_detection_score_mat = []
                for exp_dir, crnn_checkpoint in zip(crnn_dirs, crnn_checkpoints):
                    crnn: CRNN = CRNN.from_storage_dir(
                        storage_dir=exp_dir, config_name='1/config.json',
                        checkpoint_name=crnn_checkpoint
                    )
                    crnn.to(device)
                    crnn.eval()

                    score_mat_i = []
                    example_ids_i = []
                    i = 0
                    with torch.no_grad():
                        for batch in ds:
                            batch = example_to_device(batch, device)
                            x = batch['stft']
                            seq_len = batch['seq_len']
                            example_ids_i.extend(batch['example_id'])
                            y_sed, seq_len = crnn.sed(x, context, seq_len)
                            b, t, _ = y_sed.shape
                            y_sed = y_sed.detach().cpu().numpy() * tags[i:i+b, None]
                            r = np.ceil(x.shape[2]/t)
                            y_sed = np.repeat(y_sed, r, axis=1)
                            score_mat_i.extend([y_sed[j, :l, :] for j, l in enumerate(seq_len)])
                            if debug and len(score_mat_i) >= 64:
                                break
                            i += b
                    crnn.to('cpu')
                    crnn_detection_score_mat.append(Collate()(score_mat_i))
                    assert example_ids_i == example_ids
                crnn_detection_score_mat = np.mean(crnn_detection_score_mat, axis=0)  # average over sub models in CRNN ensemble

                if 'hybrid' in ensembles:
                    print(f'Ensemble: hybrid_{context}')
                    assert cnn_detection_score_mat is not None
                    mean_detection_score_mat = (crnn_detection_score_mat + cnn_detection_score_mat) / 2
                    #sweep_medfilts(mean_detection_score_mat, f'hybrid_{context}')
                if 'crnn' in ensembles:
                    print(f'Ensemble: crnn_{context}')
                    np.save('crnn_detection_score_mat.npy', crnn_detection_score_mat)
                    #sweep_medfilts(crnn_detection_score_mat, f'crnn_{context}')

        """
        for name in best_configs.keys():
            # join class-wise SEDs from different hyper-parameter sets
            detection_threshold_name = '_'.join(name.split('_')[:3])
            files = [
                Path(storage_dir) / f'{dataset_name}_pseudo_strong_{label}_{detection_threshold_name}_{conf}.tsv'
                for label, conf in zip(labels, best_configs[name])
            ]
            filename = f'{dataset_name}_pseudo_strong_{ts}_{name}.tsv'
            output_file = Path(storage_dir) / filename
            # print('Join sub-files:')
            # for file in files:
            #     print(' ', file)
            print('Output file:', output_file)
            join_tsv_files(files, output_file)
            if reference_file is not None:
                # perform and print evaluation using sed_eval package
                reference_event_list = sed_eval.io.load_event_list(
                    filename=reference_file
                )
                reference_event_list = dcase_util.containers.MetaDataContainer(
                    [entry for entry in reference_event_list if
                     entry['event_label'] is not None]
                )
                estimated_event_list = sed_eval.io.load_event_list(
                    filename=str(output_file)
                )

                all_data = dcase_util.containers.MetaDataContainer()
                all_data += reference_event_list
                all_data += estimated_event_list

                event_labels = all_data.unique_event_labels

                # Start evaluating
                # Create metrics classes, define parameters
                event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                    event_label_list=event_labels,
                    t_collar=.2, percentage_of_length=.2
                )

                # Go through files
                for filename in all_data.unique_files:
                    reference_event_list_for_current_file = reference_event_list.filter(
                        filename=filename
                    )

                    estimated_event_list_for_current_file = estimated_event_list.filter(
                        filename=filename
                    )
                    event_based_metrics.evaluate(
                        reference_event_list=reference_event_list_for_current_file,
                        estimated_event_list=estimated_event_list_for_current_file
                    )

                # print all metrics as reports
                print(f'{name} report:')
                print(event_based_metrics)
        """


