from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.appzoo import get_application_predictor, get_application_model, get_application_evaluator
from easynlp.appzoo import get_application_model_for_evaluation
from easynlp.utils import get_pretrain_model_path
from easynlp.appzoo import OpenDomainDialogueDataset
from easynlp.core import Trainer
import torch

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.checkpoint_dir == None:
        args.checkpoint_dir = args.pretrained_model_name_or_path

    if args.mode == "predict":
        predictor = get_application_predictor(
            app_name=args.app_name,
            model_dir=args.checkpoint_dir,
            data_dir = args.tables.split(",")[-1],
            user_defined_parameters=user_defined_parameters
        )
        predictor.run()
        exit()
    
    if args.mode == "train" or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir
    args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)
    args.label_length = int(user_defined_parameters.get('label_length', 128))
    valid_dataset = OpenDomainDialogueDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_text_length=args.sequence_length,
        max_label_length=args.label_length,
        origin_model_name=user_defined_parameters.get('pretrain_model_name_or_path', None))

    pretrained_model_name_or_path = args.pretrained_model_name_or_path \
        if args.pretrained_model_name_or_path else args.checkpoint_dir
    pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)
    if args.mode == "train":

        model = get_application_model(app_name=args.app_name,
                                      pretrained_model_name_or_path=pretrained_model_name_or_path,
                                      user_defined_parameters=user_defined_parameters)
        
        train_dataset = OpenDomainDialogueDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[0],
            max_text_length=args.sequence_length,
            max_label_length=args.label_length,
            origin_model_name=user_defined_parameters.get('pretrain_model_name_or_path', None))
        
        trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                          evaluator=get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters,
                                                              eval_batch_size=args.micro_batch_size))
        trainer.train()
    
    elif args.mode == "evaluate":
        model = get_application_model_for_evaluation(app_name=args.app_name,
                                      pretrained_model_name_or_path=args.checkpoint_dir, user_defined_parameters=user_defined_parameters)
        evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters,
                                              eval_batch_size=args.micro_batch_size)
        
        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)