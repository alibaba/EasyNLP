# coding=utf-8
# Copyright (c) 2020 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import time
import torch
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from easynlp.appzoo import SequenceClassification, SequenceMultiLabelClassification, DistillatorySequenceClassification, FewshotSequenceClassification, CptFewshotSequenceClassification
from easynlp.appzoo import TextMatch, TextMatchTwoTower, DistillatoryTextMatch, FewshotSingleTowerTextMatch, CptFewshotSingleTowerTextMatch
from easynlp.appzoo import SequenceLabeling, LanguageModeling, FeatureVectorization, DataAugmentation, GEEPClassification
from easynlp.appzoo import MultiModal
from easynlp.appzoo import WukongCLIP
from easynlp.appzoo import TextImageGeneration, ImageTextGeneration
# from easynlp.appzoo.sequence_generation.model import SequenceGeneration

from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSingleTowerTextMatchEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSingleTowerTextMatchEvaluator
from easynlp.appzoo import SequenceClassificationEvaluator, SequenceMultiLabelClassificationEvaluator
from easynlp.appzoo import SequenceLabelingEvaluator, LanguageModelingEvaluator, TextMatchEvaluator, GEEPClassificationEvaluator
from easynlp.appzoo import MultiModalEvaluator
from easynlp.appzoo import WukongEvaluator
from easynlp.appzoo import TextImageGenerationEvaluator, ImageTextGenerationEvaluator
# from easynlp.appzoo import SequenceGenerationEvaluator

from easynlp.appzoo import SequenceClassificationPredictor, FewshotSequenceClassificationPredictor, CptFewshotSequenceClassificationPredictor
from easynlp.appzoo import SequenceLabelingPredictor, FeatureVectorizationPredictor
from easynlp.appzoo import TextMatchPredictor, TextMatchTwoTowerPredictor, FewshotSingleTowerTextMatchPredictor, CptFewshotSingleTowerTextMatchPredictor
from easynlp.appzoo import DataAugmentationPredictor, GEEPClassificationPredictor
from easynlp.appzoo import MultiModalPredictor
from easynlp.appzoo import WukongPredictor
from easynlp.appzoo import TextImageGenerationPredictor, ImageTextGenerationPredictor
# from easynlp.appzoo import SequenceGenerationPredictor

from easynlp.appzoo import ClassificationDataset, DistillatoryClassificationDataset, FewshotSequenceClassificationDataset
from easynlp.appzoo import SequenceLabelingDataset, LanguageModelingDataset
from easynlp.appzoo import SingleTowerDataset, TwoTowerDataset, DistillatorySingleTowerDataset, FewshotSingleTowerTextMatchDataset, SiameseDataset
# from easynlp.appzoo import SequenceGenerationDataset
from easynlp.appzoo import GEEPClassificationDataset
from easynlp.appzoo import MultiModalDataset
from easynlp.appzoo import WukongDataset
from easynlp.appzoo import TextImageDataset, ImageTextDataset

from easynlp.core import PredictorManager, Trainer, DistillatoryTrainer
from easynlp.utils.logger import logger
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import initialize_easynlp, get_args


Dataset_Mapping = {
    'text_classify': {
        'enable_distillation': DistillatoryClassificationDataset,
        'enable_fewshot': FewshotSequenceClassificationDataset,
        'others': ClassificationDataset,
    },
    'text_match': {
        'two_tower.siamese': ClassificationDataset,
        'two_tower.others': TwoTowerDataset,
        'enable_distillation': DistillatorySingleTowerDataset,
        'enable_fewshot': FewshotSingleTowerTextMatchDataset,
        'others': SingleTowerDataset,
    },
    'sequence_labeling': SequenceLabelingDataset,
    'language_modeling': LanguageModelingDataset,
    'geep_classify': GEEPClassificationDataset,
    'clip': MultiModalDataset,
    'wukong': WukongDataset,
    'text2image_generation': TextImageDataset,
    'image2text_generation': ImageTextDataset,
}

ModelMapping = {
    'text_classify': {
        'multi_label': SequenceMultiLabelClassification,
        'enable_distillation': DistillatorySequenceClassification,
        'enable_fewshot.pet_fewshot': FewshotSequenceClassification,
        'enable_fewshot.cpt_fewshot': CptFewshotSequenceClassification,
        'others': SequenceClassification
    },
    'text_match': {
        'two_tower': TextMatchTwoTower,
        'enable_distillation': DistillatoryTextMatch,
        'enable_fewshot.pet_fewshot': FewshotSingleTowerTextMatch,
        'enable_fewshot.cpt_fewshot': CptFewshotSingleTowerTextMatch,
        'others': TextMatch
    },
    'language_modeling': LanguageModeling,
    'sequence_labeling': SequenceLabeling,
    'vectorization': FeatureVectorization,
    'data_augmentation': DataAugmentation,
    'geep_classify': GEEPClassification,
    'clip': MultiModal,
    'wukong': WukongCLIP,
    'text2image_generation': TextImageGeneration,
    'image2text_generation': ImageTextGeneration,
}

Eval_Model_Mapping = {
    'text_classify': {
        'multi_label': SequenceMultiLabelClassification,
        'enable_fewshot.pet_fewshot': FewshotSequenceClassification,
        'enable_fewshot.cpt_fewshot': CptFewshotSequenceClassification,
        'others': SequenceClassification
    },
    'text_match': {
        'enable_fewshot.pet_fewshot': FewshotSingleTowerTextMatch,
        'enable_fewshot.cpt_fewshot': CptFewshotSingleTowerTextMatch,
        'others': TextMatch
    },
    'sequence_labeling': SequenceLabeling,
    'geep_classify': GEEPClassification,
    'clip': MultiModal,
    'wukong': WukongCLIP,
    'text2image_generation': TextImageGeneration,
    'image2text_generation': ImageTextGeneration,
}

Evaluator_Mapping = {
    'text_classify': {
        'multi_label': SequenceMultiLabelClassificationEvaluator,
        'enable_fewshot.pet_fewshot': FewshotSequenceClassificationEvaluator,
        'enable_fewshot.cpt_fewshot': CptFewshotSequenceClassificationEvaluator,
        'others': SequenceClassificationEvaluator
    },
    'text_match': {
        'enable_fewshot.pet_fewshot': FewshotSingleTowerTextMatchEvaluator,
        'enable_fewshot.cpt_fewshot': CptFewshotSingleTowerTextMatchEvaluator,
        'others': TextMatchEvaluator
    },
    'language_modeling': LanguageModelingEvaluator,
    'sequence_labeling': SequenceLabelingEvaluator,
    'geep_classify': GEEPClassificationEvaluator,
    'clip': MultiModalEvaluator,
    'wukong': WukongEvaluator,
    'text2image_generation': TextImageGenerationEvaluator,
    'image2text_generation': ImageTextGenerationEvaluator,
}

Predictor_Mapping = {
    'text_classify': {
        'enable_fewshot.pet_fewshot': [FewshotSequenceClassificationPredictor, FewshotSequenceClassification],
        'enable_fewshot.cpt_fewshot': [CptFewshotSequenceClassificationPredictor, CptFewshotSequenceClassification],
        'others': [SequenceClassificationPredictor, SequenceClassification]
    },
    'text_match': {
        'two_tower': [TextMatchTwoTowerPredictor, TextMatchTwoTower],
        'enable_fewshot.pet_fewshot': [FewshotSingleTowerTextMatchPredictor, FewshotSequenceClassification],
        'enable_fewshot.cpt_fewshot': [CptFewshotSingleTowerTextMatchPredictor, CptFewshotSequenceClassification],
        'others': [TextMatchPredictor, TextMatch]
    },
    'sequence_labeling': [SequenceLabelingPredictor, SequenceLabeling],
    'vectorization': [FeatureVectorizationPredictor, FeatureVectorization],
    'data_augmentation': [DataAugmentationPredictor, DataAugmentation],
    'geep_classify': [GEEPClassificationPredictor, GEEPClassification],
    'clip': [MultiModalPredictor, MultiModal],
    'wukong': [WukongPredictor, WukongCLIP],
    'text2image_generation': [TextImageGenerationPredictor, TextImageGeneration],
    'image2text_generation': [ImageTextGenerationPredictor, ImageTextGeneration],
}




def get_application_dataset(app_name, 
                            pretrained_model_name_or_path,
                            data_file,
                            max_seq_length,
                            input_schema=None,
                            first_sequence=None,
                            label_name=None,
                            second_sequence=None,
                            label_enumerate_values=None,
                            user_defined_parameters=None,
                            *args,
                            **kwargs):
    for name, dataset in Dataset_Mapping.items():
        if app_name.startswith(name):
            if type(dataset) != dict:
                return dataset(
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    user_defined_parameters=user_defined_parameters,
                    *args,
                    **kwargs)
            app_parameters = user_defined_parameters.get('app_parameters')
            dataset_keys = set([key.split('.')[0] for key in dataset.keys()])
            union_name = list(dataset_keys & set(app_parameters.keys()))
            assert len(union_name) <= 1, "Only one model can be invoked, but more than one is specified in the app_parameters!"
            if len(union_name) == 0:
                return dataset['others'](
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    user_defined_parameters=user_defined_parameters,
                    *args,
                    **kwargs)
            elif union_name[0] == 'two_tower':
                key = 'two_tower.siamese' if 'siamese' in app_parameters else 'two_tower.others'
                cls_dataset = dataset[key](
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    user_defined_parameters=user_defined_parameters,
                    *args,
                    **kwargs)
                return SiameseDataset(cls_dataset, *args, **kwargs)
                    
            elif union_name[0] == 'enable_distillation' and kwargs.get('is_training', False):
                return dataset['enable_distillation'](
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    user_defined_parameters=user_defined_parameters,
                    *args,
                    **kwargs)
            else:
                return dataset[union_name[0]](
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    user_defined_parameters=user_defined_parameters,
                    *args,
                    **kwargs)
    raise NotImplementedError("application dataset %s is not implemented" % app_name)
            

def get_application_model(app_name, pretrained_model_name_or_path, user_defined_parameters, **kwargs):
    for name, model in ModelMapping.items():
        if app_name.startswith(name):
            if type(model) != dict:
                return model(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            app_parameters = user_defined_parameters.get('app_parameters')
            model_keys = set([key.split('.')[0] for key in model.keys()])
            union_name = list(model_keys & set(app_parameters.keys()))
            assert len(union_name) <= 1, "Only one model can be invoked, but more than one is specified in the app_parameters!"
            if len(union_name) == 0:
                return model['others'](pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            elif union_name[0] == 'enable_fewshot':
                fewshot_type = app_parameters.get('type')
                assert fewshot_type in ['pet_fewshot', 'cpt_fewshot'], "This fewshot model is not implemented!"
                key = union_name[0] + '.' + fewshot_type
                return model[key](pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            else:
                return model[union_name[0]](pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    raise NotImplementedError("application model %s is not implemented" % app_name)


def get_application_model_for_evaluation(app_name, pretrained_model_name_or_path, user_defined_parameters, **kwargs):
    for name, model in Eval_Model_Mapping.items():
        if app_name.startswith(name):
            if type(model) != dict:
                return model.from_pretrained(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            app_parameters = user_defined_parameters.get('app_parameters')
            model_keys = set([key.split('.')[0] for key in model.keys()])
            union_name = list(model_keys & set(app_parameters.keys()))
            assert len(union_name) <= 1, "Only one model can be invoked, but more than one is specified in the app_parameters!"
            if len(union_name) == 0:
                return model['others'].from_pretrained(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            elif union_name[0] == 'enable_fewshot':
                fewshot_type = app_parameters.get('type')
                assert fewshot_type in ['pet_fewshot', 'cpt_fewshot'], "This fewshot model is not implemented!"
                key = union_name[0] + '.' + fewshot_type
                return model[key].from_pretrained(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            else:
                return model[union_name[0]].from_pretrained(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    raise NotImplementedError("application model %s is not implemented" % app_name)     


def get_application_evaluator(app_name, valid_dataset, user_defined_parameters, **kwargs):
    for name, evaluator in Evaluator_Mapping.items():
        if app_name.startswith(name):
            if type(evaluator) != dict:
                return evaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
            app_parameters = user_defined_parameters.get('app_parameters')
            evaluator_keys = set([key.split('.')[0] for key in evaluator.keys()])
            union_name = list(evaluator_keys & set(app_parameters.keys()))
            assert len(union_name) <= 1, "Only one evaluator can be invoked, but more than one is specified in the app_parameters!"
            if len(union_name) == 0:
                return evaluator['others'](valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
            elif union_name[0] == 'enable_fewshot':
                fewshot_type = app_parameters.get('type')
                assert fewshot_type in ['pet_fewshot', 'cpt_fewshot'], "This fewshot evaluator is not implemented!"
                key = union_name[0] + '.' + fewshot_type
                return evaluator[key](valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
            else:
                return evaluator[union_name[0]](valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    raise NotImplementedError("application evaluator %s is not implemented" % app_name)  


def get_application_predictor(app_name, model_dir, user_defined_parameters, **kwargs):
    for name, predictor in Predictor_Mapping.items():
        if app_name.startswith(name):
            if type(predictor) != dict:
                return predictor[0](
                    model_dir=model_dir, 
                    model_cls=predictor[1],
                    user_defined_parameters=user_defined_parameters, **kwargs)
            
            app_parameters = user_defined_parameters.get('app_parameters')
            predictor_keys = set([key.split('.')[0] for key in predictor.keys()])
            union_name = list(predictor_keys & set(app_parameters.keys()))
            assert len(union_name) <= 1, "Only one model can be invoked, but more than one is specified in the app_parameters!"
            
            if len(union_name) == 0:
                return predictor['others'][0](
                    model_dir=model_dir, 
                    model_cls=predictor['others'][1],
                    user_defined_parameters=user_defined_parameters, **kwargs)
            elif union_name[0] == 'enable_fewshot':
                fewshot_type = app_parameters.get('type')
                assert fewshot_type in ['pet_fewshot', 'cpt_fewshot'], "This fewshot predictor is not implemented!"
                key = union_name[0] + '.' + fewshot_type
                return predictor[key][0](
                    model_dir=model_dir, 
                    model_cls=predictor[key][1],
                    user_defined_parameters=user_defined_parameters, **kwargs)
            else:
                return predictor[union_name[0]][0](
                    model_dir=model_dir, 
                    model_cls=predictor[union_name[0]][1],
                    user_defined_parameters=user_defined_parameters, **kwargs)
    
    raise NotImplementedError("application predictor %s is not implemented" % app_name)


def default_main_fn():
    start_time = time.time()
    initialize_easynlp()
    args = get_args()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "predict":
        predictor = get_application_predictor(
            app_name=args.app_name,
            model_dir=args.checkpoint_dir,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            sequence_length=args.sequence_length,
            input_schema=args.input_schema,
            output_file=args.outputs,
            user_defined_parameters=user_defined_parameters,
            few_shot_anchor_args=args)
        predictor_manager = PredictorManager(
            predictor=predictor,
            input_file=args.tables.split(",")[-1],
            skip_first_line=args.skip_first_line,
            input_schema=args.input_schema,
            output_file=args.outputs,
            output_schema=args.output_schema,
            append_cols=args.append_cols,
            batch_size=args.micro_batch_size,
            queue_size=args.predict_queue_size,
            slice_size=args.predict_slice_size,
            thread_num=args.predict_thread_num,
            table_read_thread_num=args.predict_table_read_thread_num)
        predictor_manager.run()
        logger.info("Duration time: {} s".format(time.time() - start_time))
        exit()

    if args.mode != 'train' and args.checkpoint_dir:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    multi_label = user_defined_parameters.get('app_parameters', False).get('multi_label', False)
    valid_dataset = get_application_dataset(
        app_name=args.app_name,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        skip_first_line=args.skip_first_line,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        label_name=args.label_name,
        label_enumerate_values=args.label_enumerate_values,
        user_defined_parameters=user_defined_parameters,
        multi_label=multi_label,
        is_training=False)

    # pretrained_model_name_or_path = args.pretrained_model_name_or_path \
    #     if args.pretrained_model_name_or_path else args.checkpoint_dir
    # pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)

    if args.mode == "train":
        model = get_application_model(
            app_name=args.app_name,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            num_labels=len(valid_dataset.label_enumerate_values),
            user_defined_parameters=user_defined_parameters)
        # Build Data Loader
        train_dataset = get_application_dataset(
            app_name=args.app_name,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            skip_first_line=args.skip_first_line,
            data_file=args.tables.split(",")[0],
            max_seq_length=args.sequence_length,
            input_schema=args.input_schema,
            first_sequence=args.first_sequence,
            second_sequence=args.second_sequence,
            label_name=args.label_name,
            label_enumerate_values=args.label_enumerate_values,
            user_defined_parameters=user_defined_parameters,
            multi_label=multi_label,
            is_training=True)

        evaluator = get_application_evaluator(
            app_name=args.app_name,
            valid_dataset=valid_dataset,
            few_shot_anchor_args=args,
            eval_batch_size=args.micro_batch_size,
            user_defined_parameters=user_defined_parameters)
        enable_distillation = user_defined_parameters.get('app_parameters', False).get('enable_distillation', False)
        # Training
        default_trainer = DistillatoryTrainer if enable_distillation else Trainer
        trainer = default_trainer(
            model=model,
            train_dataset=train_dataset,
            user_defined_parameters=user_defined_parameters,
            evaluator=evaluator)
        
        if args.save_checkpoint_steps is None:
            args.save_checkpoint_steps = trainer._optimizer.total_training_steps // args.epoch_num
        trainer.train()
    elif args.mode == "evaluate":
        model = get_application_model_for_evaluation(
            app_name=args.app_name, 
            pretrained_model_name_or_path=args.checkpoint_dir, 
            user_defined_parameters=user_defined_parameters)

        evaluator = get_application_evaluator(
            app_name=args.app_name,
            valid_dataset=valid_dataset,
            few_shot_anchor_args=args,
            eval_batch_size=args.micro_batch_size,
            user_defined_parameters=user_defined_parameters)
        
        if args.n_gpu > 0:
            model.to(torch.cuda.current_device())
        else:
            model.to("cpu")
        evaluator.evaluate(model=model)

    logger.info("Duration time: {} s".format(time.time() - start_time))

if __name__ == "__main__":
    default_main_fn()
