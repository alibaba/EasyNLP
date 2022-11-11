import imp
import sys
import os

sys.path.append('./')

print('*'*50)
print('running local main...\n')
from easynlp.core import Trainer
# from easynlp.appzoo import get_application_evaluator

from easynlp.appzoo.text2video_retrieval.data import Text2VideoRetrievalDataset
from easynlp.appzoo.text2video_retrieval.model import Text2VideoRetrieval
from easynlp.appzoo.text2video_retrieval.evaluator import Text2VideoRetrievalEvaluator
from easynlp.appzoo.text2video_retrieval.predictor import Text2VideoRetrievalPredictor
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.core import PredictorManager
from easynlp.utils import get_pretrain_model_path


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
        predictor = Text2VideoRetrievalPredictor(model_dir=args.checkpoint_dir, model_cls=Text2VideoRetrieval,
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

    train_dataset = Text2VideoRetrievalDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=True)

    valid_dataset = Text2VideoRetrievalDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=False)
    
    
    model = Text2VideoRetrieval(pretrained_model_name_or_path=pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters)
    evaluator = Text2VideoRetrievalEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)

    trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                      evaluator=evaluator)
    trainer.train()
