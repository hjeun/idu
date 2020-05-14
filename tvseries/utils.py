import numpy as np
import random
from copy import deepcopy
from scipy.ndimage import label


def get_actionness(ti_anno):
    class_line = np.argmax(ti_anno, axis=1)
    actionness_line = (class_line > 0).astype(np.float32)
    return actionness_line


def get_relevance(ti_anno):
    class_line = np.argmax(ti_anno, axis=1)
    binary_line = (class_line == class_line[-1]).astype(np.int32)
    indexed_line, _ = label(binary_line)
    t0_idx = indexed_line[-1]
    relation_line = (indexed_line == t0_idx).astype(np.float32)
    return relation_line


def next_batch_for_test(step, samples, test_data, batch_size, n_inputs):
    rgb_feat_batch, flow_feat_batch, class_batch, relevance_batch = list(), list(), list(), list()
    t0class_batch, vid_name_batch = list(), list()

    next_samples = deepcopy(samples[step * batch_size:min((step + 1) * batch_size, len(samples))])
    for sample in next_samples:
        vid_name = sample[0]
        s_end = sample[1]
        s_start = s_end - (n_inputs - 1)
        cls_idx = sample[2]

        rgb_feat = test_data[vid_name]['rgb']
        flow_feat = test_data[vid_name]['flow']
        anno = test_data[vid_name]['anno']

        ti_rgb_feat = rgb_feat[s_start:s_end + 1, :]
        ti_flow_feat = flow_feat[s_start:s_end + 1, :]
        ti_classes = anno[s_start:s_end + 1, :]
        ti_relavance = get_relevance(ti_classes)
        ti_actionness = get_actionness(ti_classes)

        rgb_feat_batch.append(ti_rgb_feat)
        flow_feat_batch.append(ti_flow_feat)
        class_batch.append(ti_classes)
        relevance_batch.append(ti_relavance)
        t0class_batch.append(cls_idx)
        vid_name_batch.append([vid_name, s_end])

    return np.asarray(rgb_feat_batch), np.asarray(flow_feat_batch), np.asarray(class_batch), \
           np.asarray(relevance_batch), t0class_batch, vid_name_batch


def frame_level_map_n_cap(results):
    all_probs = results['probs']
    all_labels = results['labels']    
    for i in range(1, all_probs.shape[1] - 1):        
        all_probs[:, i] = np.mean(all_probs[:, i - 1:i + 1], axis=1)
    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(1, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum/np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)
    return map, all_cls_ap, cap, all_cls_acp



