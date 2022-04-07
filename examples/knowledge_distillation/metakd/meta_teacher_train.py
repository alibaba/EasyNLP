# import sys
# sys.path.append('./')
# sys.path.append('./easynlp/appzoo/')
# sys.path.append('./easynlp/appzoo/sequence_classification/')

print('*'*50)
print('running local main...\n')

from easynlp.distillation.distill_metakd_dataset import MetakdSentiClassificationDataset
from easynlp.core.distiller import MetaTeacherTrainer
from easynlp.appzoo import SequenceClassificationEvaluator
from easynlp.core.distiller import MetaTeacherTrainer
from easynlp.distillation.distill_metakd_application import MetaTeacherForSequenceClassification
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters


if __name__ == "__main__":
    print('log: starts to init...\n')
    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    
    print('log: starts to process dataset...\n')
    train_dataset = MetakdSentiClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        # user_defined_parameters=user_defined_parameters,
        is_training=True,
        skip_first_line=True)

    valid_dataset = MetakdSentiClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        is_training=False,
        skip_first_line=True)

    print('log: starts to run...\n')

    model = MetaTeacherForSequenceClassification(pretrained_model_name_or_path=args.pretrained_model_name_or_path, num_labels=2, 
        num_domains=4)
    evaluator = SequenceClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)

    trainer = MetaTeacherTrainer(model=model, train_dataset=train_dataset,
                      evaluator=evaluator, user_defined_parameters=user_defined_parameters)
                      
    trainer.train()
