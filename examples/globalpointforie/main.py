import os
import torch
from easynlp.core import Trainer
from easynlp.utils import initialize_easynlp
from easynlp.appzoo import GlobalPointForIEDataset
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.appzoo import get_application_model, get_application_evaluator, get_application_predictor, get_application_model_for_evaluation

if __name__ == "__main__":

    args = initialize_easynlp()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    args.data_dir = user_defined_parameters.get('data_dir', None)

    assert args.data_dir is not None

    if args.mode == "predict":

        predictor = get_application_predictor(
            app_name=args.app_name,
            model_dir=args.checkpoint_dir,
            input_file=os.path.join(args.data_dir, "input.json"),
            output_file=os.path.join(args.data_dir, "output.json"),
            max_seq_length=args.sequence_length,
            user_defined_parameters=user_defined_parameters)
        predictor.run()
        exit()

    elif args.mode == "train":

        train_dataset = GlobalPointForIEDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=os.path.join(args.data_dir, "train.json"),
            max_seq_length=args.sequence_length
        )

        valid_dataset = GlobalPointForIEDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=os.path.join(args.data_dir, "dev.json"),
            max_seq_length=args.sequence_length
        )

        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters)

        trainer = Trainer(model=model, train_dataset=train_dataset, evaluator=get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, 
                                                                                                        user_defined_parameters=user_defined_parameters,
                                                                                                        sequence_length=args.sequence_length))
        
        trainer.train()

    elif args.mode == "evaluate":

        valid_dataset = GlobalPointForIEDataset(
            pretrained_model_name_or_path=args.checkpoint_dir,
            data_file=os.path.join(args.data_dir, "dev.json"),
            max_seq_length=args.sequence_length
        )
        
        model = get_application_model_for_evaluation(app_name=args.app_name,
                                                    pretrained_model_name_or_path=args.checkpoint_dir, 
                                                    user_defined_parameters=[user_defined_parameters])

        evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, 
                                              sequence_length=args.sequence_length)
        
        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)


