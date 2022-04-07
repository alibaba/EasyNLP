# import sys
# import os

from easynlp.core.distiller import MetaDistillationTrainer
from easynlp.distillation.distill_metakd_application import MetaStudentForSequenceClassification

print('*'*50)
print('running local main...\n')

from easynlp.distillation.distill_metakd_dataset import MetakdSentiClassificationDataset
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.appzoo import SequenceClassificationEvaluator


if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"

    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    path = "logits_and_feature_16.pt"
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
        genre=user_defined_parameters["genre"],
        is_training=True,
        skip_first_line=True)
    
    # train_dataset = MetaDistillationDataset(path, user_defined_parameters["genre"])

    valid_dataset = MetakdSentiClassificationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        enre=user_defined_parameters["genre"],
        is_training=False,
        skip_first_line=True)

    print('log: starts to run...\n')

    print('log: start to load teacher model...\n')
    teacher = MetaStudentForSequenceClassification.from_pretrained(user_defined_parameters["teacher_model_path"])
    print('log: start to load student model...\n')
    if user_defined_parameters["distill_stage"] == "first":
        student = MetaStudentForSequenceClassification(args.pretrained_model_name_or_path, num_labels=2, num_domains=4)
    else:
        student = MetaStudentForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)
    
    evaluator = SequenceClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)

    if user_defined_parameters["distill_stage"] == "first":
        evaluator = None
    else:
        evaluator = SequenceClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)
    trainer = MetaDistillationTrainer(student_model=student,
                        teacher_model=teacher, 
                        train_dataset=train_dataset,
                        evaluator=evaluator, user_defined_parameters=user_defined_parameters)
                      
    trainer.train()
