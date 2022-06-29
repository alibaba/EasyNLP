import sys
sys.path.append('./')
sys.path.append('./easynlp/appzoo/')
sys.path.append('./easynlp/appzoo/sequence_labeling/')

import torch.cuda

from easynlp.appzoo import SequenceLabelingDataset
from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.appzoo import get_application_model_for_evaluation
from easynlp.core import PredictorManager
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp, get_args, get_pretrain_model_path
from easynlp.utils.global_vars import parse_user_defined_parameters


if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == 'train' or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path')
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir
    args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)

    if args.mode == "predict":
        predictor = get_application_predictor(
            app_name=args.app_name, model_dir=args.checkpoint_dir,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            sequence_length=args.sequence_length)
        predictor_manager = PredictorManager(
            predictor=predictor,
            input_file=args.tables.split(",")[-1],
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size
        )
        predictor_manager.run()
        exit()

    valid_dataset = SequenceLabelingDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        # multi_label=args.multi_label,
        label_enumerate_values=args.label_enumerate_values,
        is_training=False,
        user_defined_parameters=user_defined_parameters)

    model = get_application_model(app_name=args.app_name,
                                  pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                  num_labels=len(valid_dataset.label_enumerate_values),
                                  user_defined_parameters=user_defined_parameters
                                  # multi_label=args.multi_label
                                  )

    if args.mode == "train":

        train_dataset = SequenceLabelingDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[0],
            max_seq_length=args.sequence_length,
            input_schema=args.input_schema,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            label_name=args.label_name,
            label_enumerate_values=args.label_enumerate_values,
            # multi_label=args.multi_label,
            is_training=True,
            user_defined_parameters=user_defined_parameters)

        trainer = Trainer(model=model, train_dataset=train_dataset,
                          evaluator=get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,
                                                              eval_batch_size=args.micro_batch_size,
                                                              user_defined_parameters=user_defined_parameters
                                                              # multi_label=args.multi_label
                                                              ))
        trainer.train()

    elif args.mode == "evaluate":
        model = get_application_model_for_evaluation(app_name=args.app_name,
                                                     pretrained_model_name_or_path=args.checkpoint_dir,
                                                     multi_label=args.multi_label)
        evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,
                                              eval_batch_size=args.micro_batch_size, multi_label=args.multi_label)
        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)