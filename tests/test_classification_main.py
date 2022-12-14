import os
import sys

from easynlp.appzoo import ClassificationDataset, get_application_evaluator
from easynlp.appzoo.sequence_classification.evaluator import \
    SequenceClassificationEvaluator
from easynlp.appzoo.sequence_classification.model import SequenceClassification
from easynlp.core import Trainer
from easynlp.utils import get_args, get_pretrain_model_path, initialize_easynlp
from easynlp.utils.global_vars import parse_user_defined_parameters

print('*' * 50)
print('running local main...\n')

if __name__ == '__main__':
    print('log: starts to init...\n')

    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(
        args.user_defined_parameters)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    print('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    print('log: starts to process dataset...\n')
    train_dataset = ClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(',')[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        user_defined_parameters=user_defined_parameters,
        is_training=True)

    valid_dataset = ClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(',')[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        is_training=False)

    model = SequenceClassification(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path)
    evaluator = None

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      user_defined_parameters=user_defined_parameters,
                      evaluator=evaluator)
    trainer.train()
