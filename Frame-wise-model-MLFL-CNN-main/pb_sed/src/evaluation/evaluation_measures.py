# !/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# Initial software, Nicolas Turpault, Romain Serizel, Hamid Eghbal-zadeh, Ankit Parag Shah
# Copyright © INRIA, 2018, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

from dcase_util.data import ProbabilityEncoder
import numpy
import os
import pandas as pd
import sed_eval
from psds_eval import PSDSEval, plot_psd_roc

def get_f_measure_by_class(keras_model, nb_tags, generator, steps, thresholds=None):
    """ get f measure for each class given a model and a generator of data (X, y)

    Parameters
    ----------

    keras_model : Model, model to get predictions

    nb_tags : int, number of classes which are represented

    generator : generator, data generator used to get f_measure

    steps : int, number of steps the generator will be used before stopping

    thresholds : int or list, thresholds to apply to each class to binarize probabilities

    Return
    ------

    macro_f_measure : list, f measure for each class

    """

    # Calculate external metrics
    TP = numpy.zeros(nb_tags)
    TN = numpy.zeros(nb_tags)
    FP = numpy.zeros(nb_tags)
    FN = numpy.zeros(nb_tags)
    for counter, (X, y) in enumerate(generator):
        if counter == steps:
            break
        predictions = keras_model.predict(X)

        if len(predictions.shape) == 3:
            # average data to have weak labels
            predictions = numpy.mean(predictions, axis=1)
            y = numpy.mean(y, axis=1)

        if thresholds is None:
            binarization_type = 'global_threshold'
            thresh = 0.5
        else:
            binarization_type = "class_threshold"
            assert type(thresholds) is list
            thresh = thresholds

        predictions = ProbabilityEncoder().binarization(predictions,
                                                        binarization_type=binarization_type,
                                                        threshold=thresh,
                                                        time_axis=0
                                                        )

        TP += (predictions + y == 2).sum(axis=0)
        FP += (predictions - y == 1).sum(axis=0)
        FN += (y - predictions == 1).sum(axis=0)
        TN += (predictions + y == 0).sum(axis=0)

    macro_f_measure = numpy.zeros(nb_tags)
    mask_f_score = 2*TP + FP + FN != 0
    macro_f_measure[mask_f_score] = 2*TP[mask_f_score] / (2*TP + FP + FN)[mask_f_score]

    return macro_f_measure


def based_evaluation(reference_event_list, estimated_event_list,unique_event_labels,metrics):
    t_collar=0.2
    percentage_of_length=0.2
    time_resolution=1.0
    files = {}
    for event in reference_event_list:
        files[event['filename']] = event['filename']

    evaluated_files = sorted(list(files.keys()))
    if metrics=='EventBasedMetrics':
        based_metric = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=unique_event_labels,
            t_collar=t_collar,
            percentage_of_length=percentage_of_length,
            empty_system_output_handling='zero_score'
        )
    elif metrics=='SegmentBasedMetrics':
        based_metric = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=unique_event_labels,
            time_resolution=1.0
        )

    for file in evaluated_files:
        reference_event_list_for_current_file = []
        # events = []
        for event in reference_event_list:
            if event['filename'] == file:
                reference_event_list_for_current_file.append(event)
                # events.append(event.event_label)
        estimated_event_list_for_current_file = []
        for event in estimated_event_list:
            if event['filename'] == file:
                estimated_event_list_for_current_file.append(event)

        based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return based_metric

def compute_psds_from_operating_points(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    gt = ground_truth_file
    durations = durations_file
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        pred_dir = os.path.join(
            save_dir,
            f"predictions_dtc{dtc_threshold}_gtc{gtc_threshold}_cttc{cttc_threshold}",
        )
        os.makedirs(pred_dir, exist_ok=True)
        for k in prediction_dfs.keys():
            prediction_dfs[k].to_csv(
                os.path.join(pred_dir, f"predictions_th_{k:.2f}.tsv"),
                sep="\t",
                index=False,
            )

        plot_psd_roc(
            psds_score,
            filename=os.path.join(save_dir, f"PSDS_ct{alpha_ct}_st{alpha_st}_100.png"),
        )

    return psds_score.value

def compute_psds_from_operating_points_tune(
    prediction_dfs,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    save_dir=None,
):

    gt = ground_truth_file
    durations = durations_file
    psds_eval = PSDSEval(
        ground_truth=gt,
        metadata=durations,
        dtc_threshold=dtc_threshold,
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold,
    )

    for i, k in enumerate(prediction_dfs.keys()):
        det = prediction_dfs[k]
        # see issue https://github.com/audioanalytic/psds_eval/issues/3
        det["index"] = range(1, len(det) + 1)
        det = det.set_index("index")
        psds_eval.add_operating_point(
            det, info={"name": f"Op {i + 1:02d}", "threshold": k}
        )

    psds_score = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=max_efpr)

    tpr_fpr_curve, tpr_ctr_curve, tpr_efpr_curve = psds_eval.psd_roc_curves(alpha_ct=alpha_ct)
    if max_efpr is None:
        max_efpr = numpy.max(tpr_efpr_curve.xp)
    etpr = effective_tp_ratio(tpr_efpr_curve, alpha_st)

    return psds_score.value, numpy.array([auc(tpr_efpr_curve.xp, etpr[i], max_efpr, alpha_st > 0) / max_efpr for i in range(etpr.shape[0])])

def auc(x, y, max_x=None, decreasing_y=False):
    """Compute area under curve described by the given x, y points.

            To avoid an overestimate the area in case of large gaps between
            points, the area is computed as sums of rectangles rather than
            trapezoids (np.trapz).

            Both x and y must be non-decreasing 1-dimensional numpy.ndarray. In
            particular cases it is necessary to relax such constraint for y. This
            can be done by setting allow_decrease_y to True.
            The non-decreasing property is verified if
            for all i in {2, ..., x.size}, x[i-1] <= x[i]

            Args:
                x (numpy.ndarray): 1-D array containing non-decreasing
                    values for x-axis
                y (numpy.ndarray): 1-D array containing non-decreasing
                    values for y-axis
                max_x (float): maximum x-coordinate for area computation
                decreasing_y (bool): controls the check for non-decreasing property
                    of y

            Returns:
                 A float that represents the area under curve

            Raises:
                PSDSEvalError: If there is an issue with the input data
            """
    _x = numpy.array(x)
    _y = numpy.array(y)

    if max_x is None:
        max_x = _x.max()
    if max_x not in _x:
        # add max_x to x and the correspondent y value
        _x = numpy.sort(numpy.concatenate([_x, [max_x]]))
        max_i = int(numpy.argwhere(_x == max_x))
        _y = numpy.concatenate([_y[:max_i], [_y[max_i - 1]], _y[max_i:]])
    valid_idx = _x <= max_x
    dx = numpy.diff(_x[valid_idx])
    _y = numpy.array(_y[valid_idx])[:-1]
    return numpy.sum(dx * _y)

def effective_tp_ratio(tpr_efpr, alpha_st):
    """Calculates the effective true positive rate (eTPR)

            Reduces a set of class ROC curves into a single Polyphonic
            Sound Detection (PSD) ROC curve. If NaN values are present they
            will be converted to zero.

            Args:
                tpr_efpr (PSDROC): A ROC that describes the PSD-ROC for
                    all classes
                alpha_st (float): A weighting applied to the
                    inter-class variability

            Returns:
                PSDROC: A namedTuple that describes the PSD-ROC used for the
                    calculation of PSDS.
            """
    etpr = tpr_efpr.yp - alpha_st * tpr_efpr.std
    numpy.nan_to_num(etpr, copy=False, nan=0.0)
    etpr = numpy.where(etpr < 0, 0.0, etpr)
    return etpr

