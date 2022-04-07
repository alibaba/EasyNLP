import torch.cuda

from easynlp.distillation.distill_metakd_dataset import MetakdSentiClassificationDataset
from easynlp.distillation.distill_metakd_application import MetaStudentForSequenceClassification
from easynlp.appzoo import get_application_evaluator
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    args.pretrained_model_name_or_path = args.checkpoint_dir
    
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

    pretrained_model_name_or_path = args.pretrained_model_name_or_path \
        if args.pretrained_model_name_or_path else args.checkpoint_dir

    model = MetaStudentForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
    evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,user_defined_parameters=user_defined_parameters,
                                            eval_batch_size=args.micro_batch_size)
    model.to(torch.cuda.current_device())
    evaluator.evaluate(model=model)
