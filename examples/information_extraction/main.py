import os
import torch
from easynlp.core import Trainer
from easynlp.core import PredictorManager
from easynlp.utils import initialize_easynlp, get_args
from easynlp.appzoo import InformationExtractionDataset
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.appzoo import get_application_model, get_application_evaluator, get_application_predictor, get_application_model_for_evaluation

if __name__ == "__main__":
    args = initialize_easynlp()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)

    if args.mode == "predict":

        predictor = get_application_predictor(
            app_name=args.app_name,
            model_dir=args.checkpoint_dir,
            input_schema=args.input_schema,
            sequence_length=args.sequence_length,
            output_file=args.outputs,
            user_defined_parameters=user_defined_parameters)
        predictor_manager = PredictorManager(
            predictor=predictor,
            input_file=args.tables.split(",")[0],
            skip_first_line=args.skip_first_line,
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size
        )
        predictor_manager.run()
        exit()

    elif args.mode == "train":

        train_dataset = InformationExtractionDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[0],
            input_schema=args.input_schema,
            max_seq_length=args.sequence_length
        )

        valid_dataset = InformationExtractionDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[-1],
            input_schema=args.input_schema,
            max_seq_length=args.sequence_length
        )

        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters)
                                                                      
        trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                        evaluator=get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, 
                                                                user_defined_parameters=user_defined_parameters,
                                                                few_shot_anchor_args=args,
                                                                eval_batch_size=args.micro_batch_size))
        
        trainer.train()

    elif args.mode == "evaluate":

        valid_dataset = InformationExtractionDataset(
            pretrained_model_name_or_path=args.checkpoint_dir,
            data_file=args.tables,
            input_schema=args.input_schema,
            max_seq_length=args.sequence_length
        )
        
        model = get_application_model_for_evaluation(app_name=args.app_name,
                                                    pretrained_model_name_or_path=args.checkpoint_dir, 
                                                    user_defined_parameters=user_defined_parameters)

        evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, 
                                              few_shot_anchor_args=args, eval_batch_size=args.micro_batch_size)
        
        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)


