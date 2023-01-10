import imp
import sys
import os

sys.path.append('./')

print('*'*50)
print('running local main...\n')
from easynlp.core import Trainer

from easynlp.appzoo.latent_diffusion.data import LdmDataset
from easynlp.appzoo.latent_diffusion.evaluator import LatentDiffusionModelEvaluator
from easynlp.appzoo.latent_diffusion.model import LatentDiffusion
from easynlp.appzoo.latent_diffusion.predictor import LatentDiffusionPredictor
from easynlp.utils import initialize_easynlp, get_args,get_pretrain_model_path,get_dir_name
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.core import PredictorManager 
import shutil

if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"

    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "train" or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir
    pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)

    if args.mode == "predict":
        predictor = LatentDiffusionPredictor(model_dir=args.checkpoint_dir, model_cls=LatentDiffusion,
                                       args=args,user_defined_parameters=user_defined_parameters)
        predictor_manager = PredictorManager(
            predictor=predictor,
            input_file=args.tables.split(",")[0],
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size
        )
        predictor_manager.run()
        exit()
    if not os.path.exists(os.path.join(pretrained_model_name_or_path, 'vocab.txt')):
        raise ValueError("Lack Vocab.txt")

    print('log: starts to process dataset...\n')
    train_dataset = LdmDataset(
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=True)

    valid_dataset = LdmDataset(
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=False)

    if args.mode =='train':
        if not os.path.exists(os.path.join(get_dir_name(args.checkpoint_dir),'RRDB_ESRGAN_x4.pth')):
            shutil.copy(os.path.join(pretrained_model_name_or_path,'RRDB_ESRGAN_x4.pth'),os.path.join(get_dir_name(args.checkpoint_dir),'RRDB_ESRGAN_x4.pth'))
        if not os.path.exists(os.path.join(get_dir_name(args.checkpoint_dir),'vocab.txt')):
            shutil.copy(os.path.join(pretrained_model_name_or_path,'vocab.txt'),os.path.join(get_dir_name(args.checkpoint_dir),'vocab.txt'))
    
    model = LatentDiffusion(pretrained_model_name_or_path=pretrained_model_name_or_path,args=args,user_defined_parameters=user_defined_parameters)
    del model.config.json_data["model"]["params"]["first_stage_model"]
    del model.config.json_data["model"]["params"]["cond_stage_model"]   

    evaluator = LatentDiffusionModelEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)

    trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                      evaluator=evaluator)
    trainer.train()
