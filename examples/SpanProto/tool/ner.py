# -*- coding: utf-8 -*-
# @Time    : 2022/2/16 6:03 pm.
# @Author  : JianingWang
# @File    : ner
def position_2_bio(positions, seq_len, max_len=None):
    label = [0] * seq_len
    for start, end in positions:
        for i in range(start, end):
            if i == start:
                label[i] = 1
            else:
                label[i] = 2
    if max_len and max_len > seq_len:
        label.extend([-100] * (max_len - seq_len))
    return label


def bio_2_position(bio):
    positions = []
    start = None
    for i, l in enumerate(bio):
        if l == 0:
            if start is not None:
                positions.append([start, i])
                start = None
        elif l == 1:
            if start is not None:
                positions.append([start, i])
            start = i

    return positions
