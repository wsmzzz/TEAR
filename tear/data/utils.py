import numpy as np
import h5py
import os
import sys
import logging

import torch


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def pad_1d(datas, max_len, pad_val=0):
    res = []
    for data in datas:
        res.append(np.pad(data, (0, max_len - data.shape[0]),
                          mode='constant',
                          constant_values=pad_val))
    res = np.stack(res, axis=0)

    return res


def pad_2d(datas, max_len, pad_val=0):
    res = []
    for data in datas:
        res.append(np.pad(data, ((0, max_len - data.shape[0]), (0, 0)),
                          mode='constant',
                          constant_values=pad_val))
    res = np.stack(res, axis=0)
    return res


def pad_weight(datas, max_x_len, max_y_len, pad_val=0):
    res = []
    for data in datas:
        res.append(np.pad(data, ((0, max_x_len - data.shape[0]), (0, max_y_len - data.shape[1])),
                          mode='constant',
                          constant_values=pad_val))
    res = np.stack(res, axis=0)

    return res


def dur_to_align(durs, max_len=None):
    max_len = max_len if max_len is not None else np.sum(durs)
    weight = []
    start_frame = []
    end_frame = []
    start_mat =[]
    end_mat = []
    dur_cumsum = 0
    for i, dur in enumerate(durs):
        weight.append([.0] * dur_cumsum + [1 / dur] * dur + [.0] * (max_len - dur_cumsum - dur))
        start_mat.append([.0] * dur_cumsum + [1.] * 1 + [.0] * (max_len - dur_cumsum - 1))
        end_mat.append([.0] * (dur_cumsum+dur-1) + [1.] * 1 + [.0] * (max_len - dur_cumsum -dur))
        start_frame +=[dur_cumsum+1]+ (dur-1)*[0.]
        end_frame +=(dur-1)*[0.]+[dur_cumsum+dur]
        dur_cumsum += dur

    weight = [np.asarray(i) for i in weight]
    start_mat = [np.asarray(i) for i in start_mat]
    end_mat = [np.asarray(i) for i in end_mat]
    start_frame = np.asarray(start_frame)
    end_frame = np.asarray(end_frame)
    return np.stack(weight, axis=0),np.stack(start_mat, axis=0) ,np.stack(end_mat, axis=0), start_frame , end_frame

def sequence_mask(lens, max_len=None):
    max_len = max_len if max_len is not None else max(lens)
    res = []
    for i in lens:
        res.append(np.pad(np.arange(1, i + 1), (0, max_len - i)))
    return np.stack(res, axis=0)


def seq_pos_encoding(phoneme_id, sil_id=[0,265,257,258,259]):
    pos = 0
    seg_pos = []
    sp_pos = []
    for i, id in enumerate(phoneme_id):
        seg_pos.append(pos)
        sp_pos.append(i + pos * 10)
        if id in sil_id:
            pos += 1
    return np.asarray(seg_pos, dtype=np.int64), np.asarray(sp_pos, dtype=np.int64)


def longformer_mask(durs,win=8,max_len=None):

    #casual mask
    masks = []
    dur_cumsum =np.sum(durs)
    for dur in durs:
        for i in range(1,dur+1) :
            future_masks = [1]*(min(dur-i,win))+(max(win-dur+i,0))*[0]
            past_masks = [0]*(max(win-i+1,0))+min(i,win+1)*[1]
            masks.append(past_masks+future_masks)

    masks = [np.asarray(mask) for mask in masks]


    masks = np.stack(masks,axis=0)

    if max_len and max_len>dur_cumsum:
        masks = np.concatenate([masks,np.zeros([max_len-dur_cumsum,2*win+1])],axis=0)
    return masks


if __name__=='__main__':
    duration = [10,2,7,11,5]
    l_mask = longformer_mask(duration)
    s=0

