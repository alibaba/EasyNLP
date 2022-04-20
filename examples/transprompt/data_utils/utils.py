# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 10:10 pm
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @Github  : https://github.com/alibaba/EasyTransfer, https://github.com/wjn1996

# 每一组对应的task名称（小写）
groups = {
    'g1': ['sst-2', 'mr', 'cr'],
    'g2': ['mnli', 'snli'],
    'g3': ['mrpc', 'qqp'],
    # 'g4': ['qnli', 'rte'],
}

# 每个group对应的标签体系
group_to_label = {
    'g1': ['0', '1'],
    'g2': ['contradiction', 'entailment', 'neutral'],
    'g3': ['0', '1'],
}

# 在 cross-task中，每个group的task需要使用对应的prompt-encoder，因此需要为每个task给予一个group内部的下标编号，用于指定对应的prompt-encoder
task_to_id = {
    'SST-2': 0,
    'mr': 1,
    'cr': 2,
    'MNLI': 0,
    'SNLI': 1,
    'MRPC': 0,
    'QQP': 1,
}

# 每个group或task对应的标签个数
label_to_num = {
    'g1': 2,
    'g2': 3,
    'g3': 2,
    'g4': 2,
    'SST-2': 2,
    'mr': 2,
    'cr': 2,
    'MNLI': 3,
    'SNLI': 3,
    'MRPC': 2,
    'QQP': 2,
}

# 名称对齐
data_to_name = {
    'SST-2': 'SST-2',
    'sst-5': 'sst-5',
    'mr': 'mr',
    'cr': 'cr',
    'mpqa': 'mpqa',
    'subj': 'subj',
    'trec': 'trec',
    'CoLA': 'CoLA',
    'MRPC': 'MRPC',
    'QQP': 'QQP',
    'STS-B': 'STS-B',
    'MNLI': 'MNLI',
    'SNLI': 'SNLI',
    'QNLI': 'QNLI',
    'RTE': 'RTE',
    'sst-2': 'SST-2',
    'cola': 'CoLA',
    'mrpc': 'MRPC',
    'qqp': 'QQP',
    'sts-b': 'STS-B',
    'mnli': 'MNLI',
    'snli': 'SNLI',
    'qnli': 'QNLI',
    'rte': 'RTE',
    'g1': 'g1',
    'g2': 'g2',
    'g3': 'g3',
    'g4': 'g4'
}

# 每一个task如果获取全量数据时，对应的k的取值
full_k_to_num = {
    'g1': 4096,
    'SST-2': 2048,
    'mr': 4096,
    'cr': 650,
    'g2': 167000,
    'MNLI': 104700,
    'SNLI': 167000,
    'g3': 145000,
    'MRPC': 1467,
    'QQP': 145000
}


def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'SST-2':
            return line[-1]
        elif task == 'STS-B':
            return 0 if float(line[-1]) < 2.5 else 1
        elif task == 'WNLI':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

## add by wjn
def get_text(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
        # GLUE style
        line = line.strip().split('\t')
        if task == 'CoLA':
            pass
        elif task == 'MNLI':
            pass
        elif task == 'MRPC':
            pass
        elif task == 'QNLI':
            pass
        elif task == 'QQP':
            pass
        elif task == 'RTE':
            pass
        elif task == 'SNLI':
            pass
        elif task == 'SST-2':
            return line[0]
        elif task == 'STS-B':
            pass
        elif task == 'WNLI':
            pass
        else:
            raise NotImplementedError
    else:
        return line[-1]



def load_single_data(task, train_lines):
    '''
    生成
    '''
    if task in ['SST-2', 'mr', 'cr']:
        label_list = {}  # {'<label>':[xx, xx, ..], '<label>': [..],}
        for line in train_lines:
            label = get_label(task, line)
            label = str(label)
            text = get_text(task, line)
            if label not in label_list:
                label_list[label] = [[text.replace('\t', ' '), task, label]]
            else:
                label_list[label].append([text.replace('\t', ' '), task, label])
        return label_list
    if task in ['MNLI', 'SNLI']:
        # 对于SMLI和MNLI都是sentence pair，但是MNLI比SNLI多了一列“genre”，需要将其删除，并统一列数
        # MNLI的两个sentence分别对应于列8和9，SNLI则对应于7和8，需要将MNLI的“genre”一列删除
        label_list = {}  # {'<label>':[xx, xx, ..], '<label>': [..],}
        for line in train_lines:
            label = get_label(task, line)
            line_list = line.strip().split('\t')
            if task == 'MNLI':
                sent1, sent2 = line_list[8].replace('\t', ' '), line_list[9].replace('\t', ' ')
            else:
                sent1, sent2 = line_list[7].replace('\t', ' '), line_list[8].replace('\t', ' ')
            if sent1 == "n/a":
                sent1 = "None"
            if sent2 == "n/a":
                sent2 = "None"
            if label not in label_list:
                label_list[label] = [[sent1, sent2, task, label]]
            else:
                label_list[label].append([sent1, sent2, task, label])
        return label_list
    if task in ['MRPC', 'QQP']:
        label_list = {}  # {'<label>':[xx, xx, ..], '<label>': [..],}
        for line in train_lines:
            label = get_label(task, line)
            line_list = line.strip().split('\t')
            if task == 'MRPC':
                sent1, sent2 = line_list[-1].replace('\t', ' '), line_list[-2].replace('\t', ' ')
            else:
                sent1, sent2 = line_list[-2].replace('\t', ' '), line_list[-3].replace('\t', ' ')
            if label not in label_list:
                label_list[label] = [[sent1, sent2, task, label]]
            else:
                label_list[label].append([sent1, sent2, task, label])
        return label_list