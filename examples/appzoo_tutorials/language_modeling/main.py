import sys
import os

from easynlp.core import evaluator
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from easynlp.appzoo.api import get_application_model, get_application_dataset, get_application_evaluator
from easynlp.core.trainer import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    print('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    model = get_application_model(app_name=args.app_name,
                                  pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                  user_defined_parameters=user_defined_parameters)

    train_dataset = get_application_dataset(app_name=args.app_name,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                data_file=args.tables.split(",")[0],
                                max_seq_length=args.sequence_length,
                                user_defined_parameters=user_defined_parameters)

    valid_dataset = get_application_dataset(app_name=args.app_name,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                data_file=args.tables.split(",")[1],
                                max_seq_length=args.sequence_length,
                                user_defined_parameters=user_defined_parameters)
    evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,user_defined_parameters=user_defined_parameters,
                              eval_batch_size=args.micro_batch_size)
    # Training
    trainer = Trainer(model=model, train_dataset=train_dataset,
                      evaluator=evaluator)
    trainer.train()

