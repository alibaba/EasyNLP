import sys
import os

sys.path.append('./')
sys.path.append('./easynlp/appzoo/')
sys.path.append('./easynlp/appzoo/sequence_classification/')

print('*'*50)
print('running local main...\n')

import torch
from easynlp.utils import initialize_easynlp, get_args
from easynlp.appzoo.dataset import BaseDataset
from easynlp.modelzoo import AutoTokenizer
from easynlp.utils import io

class ClassificationDataset(BaseDataset):
    def __init__(self,
                 pretrained_model_name_or_path,
                 data_file,
                 max_seq_length,
                 input_schema,
                 first_sequence,
                 label_name=None,
                 second_sequence=None,
                 label_enumerate_values=None,
                 multi_label=False,
                 *args,
                 **kwargs):
        super().__init__(data_file,
                         input_schema=input_schema,
                         output_format="dict",
                         *args,
                         **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        self.max_seq_length = max_seq_length
        self.multi_label = multi_label

        if label_enumerate_values is None:
            self._label_enumerate_values = "0,1".split(",")
        else:
            if io.exists(label_enumerate_values):
                with io.open(label_enumerate_values) as f:
                    self._label_enumerate_values = [line.strip() for line in f]
            else:
                self._label_enumerate_values = label_enumerate_values.split(",")
        self.max_num_labels = len(self._label_enumerate_values)
        assert first_sequence in self.column_names, \
            "Column name %s needs to be included in columns" % first_sequence
        self.first_sequence = first_sequence

        if second_sequence:
            assert second_sequence in self.column_names, \
                "Column name %s needs to be included in columns" % second_sequence
            self.second_sequence = second_sequence
        else:
            self.second_sequence = None

        if label_name:
            assert label_name in self.column_names, \
                "Column name %s needs to be included in columns" % label_name
            self.label_name = label_name
        else:
            self.label_name = None

        self.label_map = dict({value: idx for idx, value in enumerate(self.label_enumerate_values)})

    @property
    def label_enumerate_values(self):
        return self._label_enumerate_values

    def convert_single_row_to_example(self, row):
        text_a = row[self.first_sequence]
        text_b = row[self.second_sequence] if self.second_sequence else None
        label = row[self.label_name] if self.label_name else None

        encoding = self.tokenizer(text_a,
                                  text_b,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_seq_length)
        if not self.multi_label:
            encoding['label_ids'] = self.label_map[label]
        else:
            label_id = [self.label_map[x] for x in label.split(",") if x]
            new_label_id = [0] * self.max_num_labels
            for idx in label_id:
                new_label_id[idx] = 1
            encoding['label_ids'] = new_label_id

        return encoding

    def batch_fn(self, features):
        return {k: torch.tensor([dic[k] for dic in features]) for k in features[0]}


if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()

    print('log: starts to process dataset...\n')
    train_dataset = ClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        user_defined_parameters=None,
        is_training=True)

    if args.read_odps:
        train_sampler = None
    else:
        if args.n_gpu <= 1:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
        else:
            train_sampler = torch.utils.data.DistributedSampler(train_dataset)

    if getattr(train_dataset, 'batch_fn', None) is not None:
        # self._train_loader = DataLoader(
        _train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.micro_batch_size,
            collate_fn=train_dataset.batch_fn,
            num_workers=args.data_threads)
    else:
        # self._train_loader = DataLoader(
        _train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.micro_batch_size,
            num_workers=args.data_threads)
    
    worker_info = torch.utils.data.get_worker_info()
    print('\n\n' + '*'*10 )
    print('batch_fn?', hasattr(train_dataset, 'batch_fn'))
    print(args.data_threads)
    print(worker_info)
    
    for _epoch in range(int(args.epoch_num)):
        # self.before_epoch(_epoch)
        # start_time = time.time()

        for _step, batch in enumerate(_train_loader):
            if _step % 50 == 0:
                print(f"proocessed {_step} steps.")

        print(f'epoch {_epoch} finished.')
