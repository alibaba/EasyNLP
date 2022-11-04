import imp
import sys
import os
import torch.cuda

sys.path.append('./')

from easynlp.core import Trainer

from easynlp.appzoo.sequence_generation.data import SequenceGenerationDataset
from easynlp.appzoo.sequence_generation.model import SequenceGeneration
from easynlp.appzoo.sequence_generation.evaluator import SequenceGenerationEvaluator
from easynlp.appzoo.sequence_generation.predictor import SequenceGenerationPredictor
from easynlp.appzoo import get_application_model_for_evaluation
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.core import PredictorManager
from easynlp.utils import get_pretrain_model_path

from easynlp.utils.arguments import get_ds_args

if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"
    args = get_ds_args()

    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    model_info = user_defined_parameters.get('pretrain_model_name_or_path', '').split('/')
    pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    args.pretrained_model_name_or_path = pretrained_model_name_or_path or args.checkpoint_dir
    
    args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)
    checkpoint_files = os.listdir(args.pretrained_model_name_or_path)
    if args.mode != 'train':
        checkpoint_files += os.listdir(args.checkpoint_dir)
    if 'mg' in model_info or args.mg_model or ('latest_checkpointed_iteration.txt' in checkpoint_files and 'pytorch_model.bin' not in checkpoint_files):
        args.model_name = model_info[-1]
        is_mg = True
        try:
            from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
        except ModuleNotFoundError:
            print('APEX is required but not found. Installing Apex...')
            os.system('git clone https://github.com/NVIDIA/apex')
            os.system('cd apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')
            os.system('rm -rf apex')
            print('*'*80)
            print('APEX is installed. Please run the code again.')
            print('*'*80)
        from easynlp.appzoo.sequence_generation.mg_seq2seq.finetune import main
        from easynlp.modelzoo.mg_utils.pretrain_glm import initialize_distributed, set_random_seed
    else:
        is_mg = False
    
    if is_mg:
        torch.backends.cudnn.enabled = False
        initialize_distributed(args)
        set_random_seed(args.seed)
        main(args, user_defined_parameters)
        exit()

    initialize_easynlp()
    args = get_args()
    if args.mode == "predict":
        predictor = SequenceGenerationPredictor(model_dir=args.checkpoint_dir, model_cls=SequenceGeneration,
                                      first_sequence=args.first_sequence, user_defined_parameters=user_defined_parameters)
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

    print('log: starts to process dataset...\n')
    valid_dataset = SequenceGenerationDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=False)
    if args.mode == "train":
        train_dataset = SequenceGenerationDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[0],
            max_seq_length=args.sequence_length,
            input_schema=args.input_schema,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            user_defined_parameters=user_defined_parameters,
            is_training=True)

        #model = SequenceGeneration(pretrained_model_name_or_path=pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, from_config=transformer_config)
        model = SequenceGeneration(pretrained_model_name_or_path=args.pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters)
        extra_para = {'pretrained_model_name_or_path':args.pretrained_model_name_or_path}
        evaluator = SequenceGenerationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **extra_para)

        trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                        evaluator=evaluator)
        trainer.train()

    elif args.mode == "evaluate":
        extra_para = {'pretrained_model_name_or_path':args.pretrained_model_name_or_path}
        model = SequenceGeneration(pretrained_model_name_or_path=args.pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters)
        evaluator = SequenceGenerationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **extra_para)

        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)