import sys
import os
import logging

from easynlp.core import evaluator
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from typing import Any, Dict, Optional
from easynlp.appzoo.api import get_application_model, get_application_dataset, get_application_evaluator
from easynlp.core.trainer import Trainer
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters


def get_logger(name: Optional[str] = 'knowledge insert') -> Any:
    """single instance

    Args:
        name (Optional[str], optional): logger name. Defaults to 'knowledge insert'.

    Returns:
        Any: return logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = get_logger()

def read_knowledge_txt(path: str) -> Dict[str, Dict[str, str]]:
    """get knowledge graph

    Args:
        path (str): the knowledge data path

    Returns:
        Dict[str, Dict[str, str]]: the knowledge data
    """
    triples = {}
    logger.info('start to read knowledge...')
    with open(path, 'r') as f:
        for line in f:
            try:
                ner_1, relationship, ner_2 = line.strip().split('\t')
                if ner_1 in triples:
                    triples[ner_1][relationship] = ner_2
                else:
                    triples[ner_1] = {relationship: ner_2}
            except ValueError as e:
                # print(e)
                ...
    logger.info('read knowledge over...')
    return triples

if __name__ == "__main__":
    initialize_easynlp()
    args = get_args()
    ckpt_path = Path(args.checkpoint_dir)
    Knowledge_G = None
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True, exist_ok=True)
        logger.info('create checkpoint saved directory')
    else:
        logger.info('checkpoint directory exists')

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    external_mask_flag = user_defined_parameters.get('external_mask_flag', False) == 'True'
    contrast_learning_flag = user_defined_parameters.get('contrast_learning_flag', False) == 'True'
    kg_path = user_defined_parameters.get('kg_path', None)
    
    if contrast_learning_flag:
        Knowledge_G = read_knowledge_txt(kg_path)
    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    print('pretrained_model_name_or_path', args.pretrained_model_name_or_path)

    model = get_application_model(app_name=args.app_name,
                                  pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                  user_defined_parameters=user_defined_parameters)

    train_dataset = get_application_dataset(app_name=args.app_name,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                data_file=args.tables.split(",")[0],
                                max_seq_length=args.sequence_length,
                                user_defined_parameters=user_defined_parameters,
                                external_mask=external_mask_flag,
                                knowledge_graph=Knowledge_G,
                                contrast_learning_flag=contrast_learning_flag)
    model.backbone.resize_token_embeddings(len(train_dataset.tokenizer))
    model.config.vocab_size = len(train_dataset.tokenizer)
    # valid_dataset = get_application_dataset(app_name=args.app_name,
    #                             pretrained_model_name_or_path=args.pretrained_model_name_or_path,
    #                             data_file=args.tables.split(",")[1],
    #                             max_seq_length=args.sequence_length,
    #                             user_defined_parameters=user_defined_parameters,
    #                             external_mask=external_mask_flag,
    #                             knowledge_graph=Knowledge_G,
    #                             contrast_learning_flag=contrast_learning_flag)
    
    # evaluator = get_application_evaluator(app_name=args.app_name, valid_dataset=valid_dataset,user_defined_parameters=user_defined_parameters,
    #                           eval_batch_size=args.micro_batch_size)
    # Training
    trainer = Trainer(model=model, train_dataset=train_dataset,
                      evaluator=None, contrast_learning_flag=contrast_learning_flag)
    trainer.train()

