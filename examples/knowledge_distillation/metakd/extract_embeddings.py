import argparse
import csv
import os
import sys
import uuid
import torch
from tqdm import tqdm
from easynlp.modelzoo.models.bert import BertTokenizer, BertModel


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, domain=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.domain = domain


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, example):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.example = example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_examples(self, data_path, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "train", domain)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, domain=None):
        """Creates examples for the training and dev sets."""
        examples = []
        cnt = 0
        domain_list = domain.split(",") if domain else None
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if domain and line[3] not in domain_list:
                continue
            if cnt == 0:
                print(line[0], line[1], line[2], line[3], line[8], line[9], line[-1])
                cnt += 1
            guid = line[2]
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, domain=line[3]))
        return examples


class SentiProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_examples(self, data_path, domain=None):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "train", domain)

    def get_labels(self):
        """See base class."""
        return ["positive", "negative"]

    def _create_examples(self, lines, set_type, genre=None):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 3:
                import pdb
                pdb.set_trace()
            review, domain, sentiment = line
            if genre and genre != "mix" and domain != genre:
                continue
            guid = uuid.uuid4()
            text_a = review
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=sentiment, domain=domain))
        return examples



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


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):

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

        if ex_index < 1:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: {}".format(example.label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          example=example))
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", default="bert-base-uncased",
                        type=str, required=True, help="bert embedding path")
    parser.add_argument("--input", default="./inputs.tsv",
                        type=str, required=True, help="bert embedding path")
    parser.add_argument("--output", default="./output.tsv",
                        type=str, required=True, help="bert embedding path")
    parser.add_argument("--task_name", default="mnli",
                        type=str, required=False, help="bert embedding path")
    parser.add_argument("--max_seq_length", default=128,
                        type=int, required=False, help="bert embedding path")
    parser.add_argument("--batch_size", default=128,
                        type=int, required=False, help="bert embedding path")
    parser.add_argument("--gpu", default=0,
                        type=int, required=False, help="bert embedding path")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    root_path = os.path.dirname(args.input)
    dev_path = os.path.join(root_path, "dev.tsv")
    test_path = os.path.join(root_path, "test.tsv")
    processors = {
        "mnli": MnliProcessor,
        "senti": SentiProcessor
    }

    processor = processors[args.task_name]()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    examples = processor.get_examples(args.input)
    features = convert_examples_to_features(examples, args.max_seq_length, tokenizer)
    bert_model = BertModel.from_pretrained(args.bert_path)
    bert_model.to(device)

    total_steps = len(examples) // args.batch_size + 1
    fout = open(args.output, "w")
    fout.write("\t".join(["guid", "text_a", "text_b", "label", "domain", "embeddings"]) + "\n")
    for step in tqdm(range(total_steps)):
        batch_features = features[step * args.batch_size: (step + 1) * args.batch_size]
        input_ids = torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(device)
        input_mask = torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(device)
        segment_ids = torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(device)
        with torch.no_grad():
            # sequence_output = bert_model(input_ids, input_mask, segment_ids)[0]
            sequence_output = bert_model(input_ids, input_mask, segment_ids).last_hidden_state
        content_embeddings = torch.mean(sequence_output[:, 1:, :], dim=1).tolist()
        for idx, content_embedding in enumerate(content_embeddings):
            feat = batch_features[idx]
            line = "\t".join([
                str(feat.example.guid),
                str(feat.example.text_a),
                str(feat.example.text_b),
                str(feat.example.label),
                str(feat.example.domain),
                str(" ".join([str(t) for t in content_embedding]) + "\n")
            ])
            fout.write(line)
    fout.close()