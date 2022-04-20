import sys
import os

sys.path.append('./')
sys.path.append('./easynlp/appzoo/')
# sys.path.append('./easynlp/appzoo/sequence_classification/')

print('*'*50)
print('running local main...\n')
from easynlp.core import Trainer
from easynlp.appzoo import get_application_evaluator
# from easynlp.core.trainer_vanilla import VanillaTrainer as Trainer

from easynlp.appzoo.multi_modal.model import TextImageRetrieval
from easynlp.appzoo.multi_modal.data import CLIPDataset



# from easynlp.appzoo import ClassificationDataset
# from easynlp.appzoo.sequence_classification.model import SequenceClassification
# from easynlp.appzoo.sequence_classification.evaluator import SequenceClassificationEvaluator
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import get_pretrain_model_path


if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"

    initialize_easynlp()
    args = get_args()

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    print('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    model = TextImageRetrieval(pretrained_model_name_or_path=args.pretrained_model_name_or_path,user_defined_parameters=user_defined_parameters)
    
    pwd = os.getcwd()
    # print(pwd)
    # raise Exception(44)

    print('log: starts to process dataset...\n')
    train_dataset = CLIPDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        current_path=pwd,
        tokenizer=model.chinese_roberta_tokenizer,
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        text_col=args.first_sequence,
        image_col=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=True)
    


    valid_dataset = CLIPDataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        current_path=pwd,
        tokenizer=model.chinese_roberta_tokenizer,
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        text_col=args.first_sequence,
        image_col=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        is_training=False)



    
    evaluator = None

    trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                      evaluator=evaluator)
    
    
    
    trainer.train()

    raise Exception('stop 43')
