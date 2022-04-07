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
from ast import literal_eval
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from easynlp.appzoo.sequence_classification.model import SequenceClassification, SequenceMultiLabelClassification, DistillatorySequenceClassification, FewshotSequenceClassification, CptFewshotSequenceClassification
from easynlp.appzoo.text_match.model import TextMatch, TextMatchTwoTower, DistillatoryTextMatch, FewshotSingleTowerTextMatch, CptFewshotSingleTowerTextMatch
from easynlp.appzoo.sequence_labeling.model import SequenceLabeling
from easynlp.appzoo.language_modeling.model import LanguageModeling
from easynlp.appzoo.feature_vectorization.model import FeatureVectorization
from easynlp.appzoo.data_augmentation.model import DataAugmentation
from easynlp.appzoo.geep_classification.model import GEEPClassification
# from easynlp.appzoo.sequence_generation.model import SequenceGeneration

from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSingleTowerTextMatchEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSingleTowerTextMatchEvaluator

from easynlp.appzoo.sequence_classification.evaluator import SequenceClassificationEvaluator, SequenceMultiLabelClassificationEvaluator
from easynlp.appzoo.sequence_labeling.evaluator import SequenceLabelingEvaluator
from easynlp.appzoo.sequence_classification.predictor import SequenceClassificationPredictor, FewshotSequenceClassificationPredictor, CptFewshotSequenceClassificationPredictor
from easynlp.appzoo.language_modeling.evaluator import LanguageModelingEvaluator
from easynlp.appzoo.text_match.evaluator import TextMatchEvaluator
from easynlp.appzoo.geep_classification.evaluator import GEEPClassificationEvaluator
# from easynlp.appzoo.sequence_generation.evaluator import SequenceGenerationEvaluator

from easynlp.appzoo.sequence_labeling.predictor import SequenceLabelingPredictor
from easynlp.appzoo.feature_vectorization.predictor import FeatureVectorizationPredictor
from easynlp.appzoo.text_match.predictor import TextMatchPredictor, TextMatchTwoTowerPredictor, FewshotSingleTowerTextMatchPredictor, CptFewshotSingleTowerTextMatchPredictor
from easynlp.appzoo.data_augmentation.predictor import DataAugmentationPredictor
from easynlp.appzoo.geep_classification.predictor import GEEPClassificationPredictor
# from easynlp.appzoo.sequence_generation.predictor import SequenceGenerationPredictor

from easynlp.appzoo.sequence_classification.data import ClassificationDataset, DistillatoryClassificationDataset, FewshotSequenceClassificationDataset
from easynlp.appzoo.sequence_labeling.data import SequenceLabelingDataset
from easynlp.appzoo.language_modeling.data import LanguageModelingDataset
from easynlp.appzoo.text_match.data import SingleTowerDataset, TwoTowerDataset, DistillatorySingleTowerDataset, FewshotSingleTowerTextMatchDataset, SiameseDataset
# from easynlp.appzoo.sequence_generation.data import SequenceGenerationDataset
from easynlp.appzoo.geep_classification.data import GEEPClassificationDataset

from easynlp.core import PredictorManager
from easynlp.core import Trainer
from easynlp.utils import get_pretrain_model_path
from easynlp.utils.logger import logger
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import initialize_easynlp, get_args
from easynlp.core import DistillatoryTrainer


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

    if app_name.startswith("text_classify"):
        if user_defined_parameters.get('app_parameters', {}).get('enable_distillation', False) \
            and kwargs.get('is_training', False):
            return DistillatoryClassificationDataset(
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
        elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            return FewshotSequenceClassificationDataset(
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
                **kwargs
            )
        else:
            multi_label = user_defined_parameters.get('app_parameters', False).get('multi_label', False)
            return ClassificationDataset(
                data_file=data_file,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                max_seq_length=max_seq_length,
                input_schema=input_schema,
                first_sequence=first_sequence,
                second_sequence=second_sequence,
                label_name=label_name,
                label_enumerate_values=label_enumerate_values,
                multi_label=multi_label,
                *args,
                **kwargs)
    elif app_name.startswith("text_match"):
        if user_defined_parameters.get('app_parameters', False).get('two_tower', False) is True:
            if user_defined_parameters.get('app_parameters', False).get('siamese', False) is True:
                multi_label = user_defined_parameters.get('app_parameters', False).get('multi_label', False)
                cls_dataset = ClassificationDataset(
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    multi_label=multi_label,
                    prefetch_all=True,
                    *args,
                    **kwargs)

                return SiameseDataset(cls_dataset, *args, **kwargs)

            else:
                return TwoTowerDataset(data_file=data_file,
                                       pretrained_model_name_or_path=pretrained_model_name_or_path,
                                       max_seq_length=max_seq_length,
                                       input_schema=input_schema,
                                       first_sequence=first_sequence,
                                       second_sequence=second_sequence,
                                       label_name=label_name,
                                       label_enumerate_values=label_enumerate_values,
                                       *args,
                                       **kwargs)
        else:
            if user_defined_parameters.get('app_parameters', {}).get('enable_distillation', False) \
                and kwargs.get('is_training', False):
                return DistillatorySingleTowerDataset(
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
            elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
                return FewshotSingleTowerTextMatchDataset(
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
                    **kwargs
                )
            else:
                return SingleTowerDataset(
                    data_file=data_file,
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    max_seq_length=max_seq_length,
                    input_schema=input_schema,
                    first_sequence=first_sequence,
                    second_sequence=second_sequence,
                    label_name=label_name,
                    label_enumerate_values=label_enumerate_values,
                    *args,
                    **kwargs)
    elif app_name.startswith("sequence_labeling"):
        return SequenceLabelingDataset(data_file=data_file,
                                       pretrained_model_name_or_path=pretrained_model_name_or_path,
                                       max_seq_length=max_seq_length,
                                       input_schema=input_schema,
                                       first_sequence=first_sequence,
                                       label_name=label_name,
                                       label_enumerate_values=label_enumerate_values,
                                       *args,
                                       **kwargs)
    elif app_name.startswith("language_modeling"):
        return LanguageModelingDataset(data_file=data_file,
                                       pretrained_model_name_or_path=pretrained_model_name_or_path,
                                       user_defined_parameters=user_defined_parameters,
                                       max_seq_length=max_seq_length)
    elif app_name.startswith("geep_classify"):
        multi_label = user_defined_parameters.get('app_parameters', False).get('multi_label', False)
        return GEEPClassificationDataset(
                data_file=data_file,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                max_seq_length=max_seq_length,
                input_schema=input_schema,
                first_sequence=first_sequence,
                second_sequence=second_sequence,
                label_name=label_name,
                label_enumerate_values=label_enumerate_values,
                multi_label=multi_label,
                *args,
                **kwargs)
    # elif app_name.startswith("sequence_generation"):
    #     return SequenceGenerationDataset(data_file=data_file,
    #                                     pretrained_model_name_or_path=pretrained_model_name_or_path,
    #                                     max_seq_length=max_seq_length,
    #                                     input_schema=input_schema,
    #                                     first_sequence=first_sequence,
    #                                     second_sequence=second_sequence,
    #                                     user_defined_parameters=user_defined_parameters,
    #                                     *args,
    #                                     **kwargs)

def get_application_model(app_name, pretrained_model_name_or_path, user_defined_parameters, **kwargs):
    if app_name.startswith("text_classify"):
        if user_defined_parameters.get('app_parameters', False).get('multi_label', False) is True:
            return SequenceMultiLabelClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
        elif user_defined_parameters.get('app_parameters', False).get('enable_distillation', False) is True:
            return DistillatorySequenceClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
        elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSequenceClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSequenceClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            else:
                raise ValueError('This fewshot model is not implemented!')
        else:
            return SequenceClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("text_match"):
        if user_defined_parameters.get('app_parameters', False).get('two_tower', False) is True:
            return TextMatchTwoTower(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
        else:
            if user_defined_parameters.get('app_parameters', False).get('enable_distillation', False) is True:
                return DistillatoryTextMatch(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
            elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
                few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
                if few_shot_model_name == 'pet_fewshot':
                    return FewshotSingleTowerTextMatch(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
                elif few_shot_model_name == 'cpt_fewshot':
                    return CptFewshotSingleTowerTextMatch(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
                else:
                    raise ValueError('This few shot model is not implemented!')
            else:
                return TextMatch(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("language_modeling"):
        return LanguageModeling(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("sequence_labeling"):
        return SequenceLabeling(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("vectorization"):
        return FeatureVectorization(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith('data_augmentation'):
        return DataAugmentation(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith('geep_classify'):
        return GEEPClassification(pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters, **kwargs)
    # elif app_name.startswith("sequence_generation"):
    #     return SequenceGeneration(pretrained_model_name_or_path,user_defined_parameters=user_defined_parameters, **kwargs)
    else:
        raise NotImplementedError("application model %s is not implemented" % app_name)


def get_application_model_for_evaluation(app_name, pretrained_model_name_or_path, user_defined_parameters, **kwargs):
    if app_name.startswith("text_classify"):
        if user_defined_parameters.get('app_parameters', False).get('multi_label', False) is True:
            return SequenceMultiLabelClassification.from_pretrained(pretrained_model_name_or_path)
        elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSequenceClassification.from_pretrained(pretrained_model_name_or_path, **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSequenceClassification.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                raise ValueError('This fewshot model is not implemented!')
        else:
            return SequenceClassification.from_pretrained(pretrained_model_name_or_path)
    elif app_name.startswith("text_match"):
        if user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSingleTowerTextMatch.from_pretrained(pretrained_model_name_or_path, **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSingleTowerTextMatch.from_pretrained(pretrained_model_name_or_path, **kwargs)
            else:
                raise ValueError('This fewshot model is not implemented!')
        else:
            return TextMatch.from_pretrained(pretrained_model_name_or_path)
    elif app_name.startswith("sequence_labeling"):
        return SequenceLabeling.from_pretrained(pretrained_model_name_or_path)
    elif app_name.startswith("geep_classify"):
        return GEEPClassification.from_pretrained(pretrained_model_name_or_path,user_defined_parameters=user_defined_parameters, **kwargs)
    # elif app_name.startswith("sequence_generation"):
    #     return SequenceGeneration.from_pretrained(pretrained_model_name_or_path,user_defined_parameters)
    else:
        raise NotImplementedError("application model %s is not implemented" % app_name)


def get_application_evaluator(app_name, valid_dataset, user_defined_parameters, **kwargs):
    if app_name.startswith("text_classify"):
        if user_defined_parameters.get('app_parameters', False).get('multi_label', False) is True:
            return SequenceMultiLabelClassificationEvaluator(valid_dataset=valid_dataset, **kwargs)
        elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSequenceClassificationEvaluator(valid_dataset=valid_dataset, **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSequenceClassificationEvaluator(valid_dataset=valid_dataset, **kwargs)
            else:
                raise ValueError('This fewshot model is not implemented!')
        else:
            return SequenceClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("text_match"):
        if user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSingleTowerTextMatchEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSingleTowerTextMatchEvaluator(valid_dataset=valid_dataset, **kwargs)
            else:
                raise ValueError('This fewshot model is not implemented!')
        else:
            return TextMatchEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("sequence_labeling"):
        return SequenceLabelingEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("language_modeling"):
        return LanguageModelingEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    elif app_name.startswith("geep_classify"):
        return GEEPClassificationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters, **kwargs)
    # elif app_name.startswith("sequence_generation"):
    #     return SequenceGenerationEvaluator(valid_dataset=valid_dataset, **kwargs)
    else:
        raise NotImplementedError("application evaluator %s is not implemented" % app_name)


def get_application_predictor(app_name, model_dir, user_defined_parameters, **kwargs):
    if app_name.startswith("text_classify"):
        if user_defined_parameters == 'None':
            return SequenceClassificationPredictor(model_dir=model_dir,
                                                model_cls=SequenceClassification,
                                                **kwargs)
        elif user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
            few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
            if few_shot_model_name == 'pet_fewshot':
                return FewshotSequenceClassificationPredictor(model_dir=model_dir,
                                                            model_cls=FewshotSequenceClassification,
                                                            user_defined_parameters=user_defined_parameters,
                                                            **kwargs)
            elif few_shot_model_name == 'cpt_fewshot':
                return CptFewshotSequenceClassificationPredictor(model_dir=model_dir,
                                                            model_cls=CptFewshotSequenceClassification,
                                                            user_defined_parameters=user_defined_parameters,
                                                            **kwargs)
            else:
                raise ValueError("This fewshot model is not implemented!")
        else:
            return SequenceClassificationPredictor(model_dir=model_dir,
                                                model_cls=SequenceClassification,
                                                **kwargs)
    elif app_name.startswith("text_match"):
        if user_defined_parameters.get('app_parameters', False).get('two_tower', False) is True:
            return TextMatchTwoTowerPredictor(model_dir=model_dir, model_cls=TextMatchTwoTower, **kwargs)
        else:
            if user_defined_parameters.get('app_parameters', False).get('enable_fewshot', False) is True:
                few_shot_model_name = user_defined_parameters.get('app_parameters').get('type')
                if few_shot_model_name == 'pet_fewshot':
                    return FewshotSingleTowerTextMatchPredictor(model_dir=model_dir,
                                                                model_cls=FewshotSequenceClassification,
                                                                user_defined_parameters=user_defined_parameters,
                                                                **kwargs)
                elif few_shot_model_name == 'cpt_fewshot':
                    return CptFewshotSingleTowerTextMatchPredictor(model_dir=model_dir,
                                                                model_cls=CptFewshotSequenceClassification,
                                                                user_defined_parameters=user_defined_parameters,
                                                                **kwargs)
                else:
                    raise ValueError("This fewshot model is not implemented!")
            else:
                return TextMatchPredictor(model_dir=model_dir, model_cls=TextMatch, **kwargs)
    elif app_name.startswith("sequence_labeling"):
        return SequenceLabelingPredictor(model_dir=model_dir, model_cls=SequenceLabeling, **kwargs)
    elif app_name.startswith("vectorization"):
        return FeatureVectorizationPredictor(model_dir=model_dir,
                                             model_cls=FeatureVectorization,
                                             **kwargs)
    elif app_name.startswith('data_augmentation'):

        return DataAugmentationPredictor(model_dir=model_dir,
                                         model_cls=DataAugmentation,
                                         user_defined_parameters=user_defined_parameters,
                                         **kwargs)
    elif app_name.startswith("geep_classify"):
        return GEEPClassificationPredictor(model_dir=model_dir, model_cls=GEEPClassification, user_defined_parameters=user_defined_parameters,**kwargs)
    # elif app_name.startswith("sequence_generation"):
    #     return SequenceGenerationPredictor(model_dir=model_dir, model_cls=SequenceGeneration, user_defined_parameters=user_defined_parameters,**kwargs)

    else:
        raise NotImplementedError("application predictor %s is not implemented" % app_name)

def default_main_fn():
    start_time = time.time()
    initialize_easynlp()
    args = get_args()
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "predict":
        predictor = get_application_predictor(app_name=args.app_name,
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
        is_training=False)

    # pretrained_model_name_or_path = args.pretrained_model_name_or_path \
    #     if args.pretrained_model_name_or_path else args.checkpoint_dir
    # pretrained_model_name_or_path = get_pretrain_model_path(pretrained_model_name_or_path)

    if args.mode == "train":
        model = get_application_model(app_name=args.app_name,
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
            is_training=True)

        evaluator = get_application_evaluator(app_name=args.app_name,
                                              valid_dataset=valid_dataset,
                                              few_shot_anchor_args=args,
                                              eval_batch_size=args.micro_batch_size,
                                              user_defined_parameters=user_defined_parameters)
        enable_distillation = user_defined_parameters.get('app_parameters', False).get('enable_distillation', False)
        # Training
        if enable_distillation:
            trainer = DistillatoryTrainer(model=model,
                                          train_dataset=train_dataset,
                                          user_defined_parameters=user_defined_parameters,
                                          evaluator=evaluator)
        else:
            trainer = Trainer(model=model,
                              train_dataset=train_dataset,
                              user_defined_parameters=user_defined_parameters,
                              evaluator=evaluator)

        if args.save_checkpoint_steps is None:
            args.save_checkpoint_steps = trainer._optimizer.total_training_steps // args.epoch_num
        trainer.train()
    elif args.mode == "evaluate":
        model = get_application_model_for_evaluation(
            app_name=args.app_name, pretrained_model_name_or_path=args.checkpoint_dir, user_defined_parameters=user_defined_parameters)

        evaluator = get_application_evaluator(app_name=args.app_name,
                                              valid_dataset=valid_dataset,
                                              few_shot_anchor_args=args,
                                              eval_batch_size=args.micro_batch_size,
                                              user_defined_parameters=user_defined_parameters)
        model.to(torch.cuda.current_device())
        evaluator.evaluate(model=model)

    logger.info("Duration time: {} s".format(time.time() - start_time))

if __name__ == "__main__":
    default_main_fn()