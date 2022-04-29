import sys
import os

sys.path.append('./')
sys.path.append('./easynlp/appzoo/')
sys.path.append('./examples/hf_adapter_easynlp/')

from easynlp.appzoo import ClassificationDataset
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path
from transformers import BertForSequenceClassification
from examples.hf_adapter_easynlp.hf_ez_nlp_evaluator import SequenceClassificationEvaluator
from examples.hf_adapter_easynlp.hf_ez_nlp_trainer import HfTrainer
from examples.hf_adapter_easynlp.hf_ez_nlp_predictor import Predictor

from types import MethodType
from examples.hf_adapter_easynlp.hf_ez_nlp_user_defined import compute_loss, forward_repre

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    if args.mode == "predict":
        predictor = Predictor(
            model_dir=args.checkpoint_dir,
            user_defined_parameters=args.user_defined_parameters,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            sequence_length=args.sequence_length,
            input_file=args.tables.split(",")[-1],
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size,
            args=args
        )
        predictor.run()

    else:
        user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
        if args.mode == "train" or not args.checkpoint_dir:
            args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
        else:
            args.pretrained_model_name_or_path = args.checkpoint_dir
        args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)
        pretrained_model_name_or_path = args.pretrained_model_name_or_path \
            if args.pretrained_model_name_or_path else args.checkpoint_dir
        valid_dataset = ClassificationDataset(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            data_file=args.tables.split(",")[-1],
            max_seq_length=args.sequence_length,
            input_schema=args.input_schema,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            label_name=args.label_name,
            label_enumerate_values=args.label_enumerate_values,
            is_training=False)

        if args.mode == "train":
            model = BertForSequenceClassification.from_pretrained(args.pretrained_model_name_or_path)
            train_dataset = ClassificationDataset(
                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                data_file=args.tables.split(",")[0],
                max_seq_length=args.sequence_length,
                input_schema=args.input_schema,
                first_sequence=args.first_sequence,
                second_sequence=args.second_sequence,
                label_name=args.label_name,
                label_enumerate_values=args.label_enumerate_values,
                user_defined_parameters=user_defined_parameters,
                is_training=True)

            model.compute_loss = MethodType(compute_loss, model)
            model.forward_repre = MethodType(forward_repre, model)
            trainer = HfTrainer(model=model, train_dataset=train_dataset,user_defined_parameters=user_defined_parameters,
                            evaluator=SequenceClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, eval_batch_size=args.micro_batch_size))
            trainer.train()
