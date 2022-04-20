# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None, guid=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.guid = guid


class InputFeatures(object):
    """A single set of features of data for text classification/match."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id
        self.guid = guid


class LabelingFeatures(object):
    """A single set of features of data for sequence labeling."""

    def __init__(self, input_ids, input_mask, segment_ids, all_tokens, label_ids,
                 tok_to_orig_index, seq_length=None, guid=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.all_tokens = all_tokens
        self.seq_length = seq_length
        self.label_ids = label_ids
        self.tok_to_orig_index = tok_to_orig_index
        self.guid = guid


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def bert_cls_convert_example_to_feature(example, tokenizer, max_seq_length, label_map=None):
    """ Convert `InputExample` into `InputFeature` For classification task

        Args:
            example (`InputExample`): an input example
            tokenizer (`BertTokenizer`): BERT Tokenizer
            max_seq_length (`int`): Maximum sequence length while truncating
            label_map (`dict`): a map from label_value --> label_idx,
                                "regression" task if it is None else "classification"
        Returns:
            feature (`InputFeatures`): an input feature
    """

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    seq_length = len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_type = "classification" if label_map else None
    if label_type == "classification":
        label_id = label_map[example.label]
    else:
        try:
            label_id = float(example.label)
        except:
            label_id = None

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            seq_length=seq_length,
                            guid=example.guid)

    return feature


def bert_labeling_convert_example_to_feature(example, tokenizer, max_seq_length, label_map=None):
    """ Convert `InputExample` into `InputFeature` For sequence labeling task

        Args:
            example (`InputExample`): an input example
            tokenizer (`BertTokenizer`): BERT Tokenizer
            max_seq_length (`int`): Maximum sequence length while truncating
            label_map (`dict`): a map from label_value --> label_idx,
                                "regression" task if it is None else "classification"
        Returns:
            feature (`InputFeatures`): an input feature
    """

    content_tokens = example.text_a.split(" ")
    if example.label is not None:
        label_tags = example.label.split(" ")
    else:
        label_tags = None

    all_tokens = ["[CLS]"]
    all_labels = [""]
    tok_to_orig_index = [-100]
    for i, token in enumerate(content_tokens):
        sub_tokens = tokenizer.tokenize(token)
        all_tokens.extend(sub_tokens)
        tok_to_orig_index.extend([i] * len(sub_tokens))
        if label_tags is None:
            all_labels.extend(["" for _ in range(len(sub_tokens))])
        else:
            all_labels.extend([label_tags[i] for _ in range(len(sub_tokens))])
    all_tokens = all_tokens[:max_seq_length - 1]
    all_labels = all_labels[:max_seq_length - 1]
    all_tokens.append("[SEP]")
    all_labels.append("")
    tok_to_orig_index.append(-100)

    input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    label_ids = [label_map[label] if label else -100 for label in all_labels]

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(-100)

    feature = LabelingFeatures(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_ids=label_ids,
                               all_tokens=all_tokens,
                               seq_length=max_seq_length,
                               tok_to_orig_index=tok_to_orig_index,
                               guid=example.guid)

    return feature