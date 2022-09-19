import imp
import sys
import os
import torch.cuda

sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from easynlp.core import Trainer

from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.appzoo import get_application_model_for_evaluation
from easynlp.appzoo import MachineReadingComprehensionDataset
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.core import PredictorManager
from easynlp.utils import get_pretrain_model_path

if __name__ == "__main__":
    print('log: starts to init...\n')

    initialize_easynlp()
    args = get_args()

    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "train" or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir
    args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)

    if args.mode == "predict":
        predictor = get_application_predictor(app_name=args.app_name,
                                              model_dir=args.checkpoint_dir,
                                              first_sequence=args.first_sequence,
                                              second_sequence=args.second_sequence,
                                              max_seq_length=args.sequence_length,
                                              output_file=args.outputs,
                                              user_defined_parameters=user_defined_parameters
                                              )
        predictor_manager = PredictorManager(predictor=predictor,
                                             input_file=args.tables.split(",")[-1],
                                             input_schema=args.input_schema,
                                             output_file=args.outputs,
                                             output_schema=args.output_schema,
                                             append_cols=args.append_cols,
                                             batch_size=args.micro_batch_size
                                             )
        predictor_manager.run()
        exit()

    print('log: starts to process dataset...\n')
    valid_dataset = MachineReadingComprehensionDataset(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       data_file=args.tables.split(",")[-1],
                                                       max_seq_length=args.sequence_length,
                                                       input_schema=args.input_schema,
                                                       first_sequence=args.first_sequence,
                                                       second_sequence=args.second_sequence,
                                                       user_defined_parameters=user_defined_parameters,
                                                       is_training=False
                                                       )

    if args.mode == "train":
        train_dataset = MachineReadingComprehensionDataset(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                           data_file=args.tables.split(",")[0],
                                                           max_seq_length=args.sequence_length,
                                                           input_schema=args.input_schema,
                                                           first_sequence=args.first_sequence,
                                                           second_sequence=args.second_sequence,
                                                           user_defined_parameters=user_defined_parameters,
                                                           is_training=True
                                                           )

        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters
                                      )
        evaluator = get_application_evaluator(app_name=args.app_name,
                                              valid_dataset=valid_dataset,
                                              user_defined_parameters=user_defined_parameters,
                                              pretrained_model_name_or_path=args.pretrained_model_name_or_path
                                              )

        trainer = Trainer(model=model,
                          train_dataset=train_dataset,
                          user_defined_parameters=user_defined_parameters,
                          evaluator=evaluator
                          )
        trainer.train()

    elif args.mode == "evaluate":
        model = get_application_model_for_evaluation(app_name=args.app_name,
                                                     pretrained_model_name_or_path=args.checkpoint_dir,
                                                     user_defined_parameters=user_defined_parameters
                                                     )
        evaluator = get_application_evaluator(app_name=args.app_name,
                                              valid_dataset=valid_dataset,
                                              user_defined_parameters=user_defined_parameters,
                                              eval_batch_size=args.micro_batch_size
                                              )

        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)
