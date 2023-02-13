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

from easynlp.appzoo.video2text_generation.model import CLIPGPTFrameTextGeneration
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")

from easynlp.appzoo import SequenceClassification, SequenceMultiLabelClassification, DistillatorySequenceClassification, FewshotSequenceClassification, CptFewshotSequenceClassification
from easynlp.appzoo import TextMatch, TextMatchTwoTower, DistillatoryTextMatch, FewshotSingleTowerTextMatch, CptFewshotSingleTowerTextMatch
from easynlp.appzoo import SequenceLabeling, LanguageModeling, FeatureVectorization, DataAugmentation, GEEPClassification
from easynlp.appzoo import Text2VideoRetrieval
from easynlp.appzoo import CLIPApp
from easynlp.appzoo import WukongCLIP
from easynlp.appzoo import TextImageGeneration
from easynlp.appzoo import LatentDiffusion, StableDiffusion
from easynlp.appzoo import VQGANGPTImageTextGeneration, CLIPGPTImageTextGeneration
from easynlp.appzoo import CLIPGPTFrameTextGeneration
from easynlp.appzoo.sequence_generation.model import SequenceGeneration
from easynlp.appzoo import MachineReadingComprehension
from easynlp.appzoo import OpenDomainDialogue
from easynlp.appzoo import InformationExtractionModel

from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSequenceClassificationEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import PromptEvaluator as FewshotSingleTowerTextMatchEvaluator
from easynlp.fewshot_learning.fewshot_evaluator import CPTEvaluator as CptFewshotSingleTowerTextMatchEvaluator
from easynlp.appzoo import SequenceClassificationEvaluator, SequenceMultiLabelClassificationEvaluator
from easynlp.appzoo import SequenceLabelingEvaluator, LanguageModelingEvaluator, TextMatchEvaluator, GEEPClassificationEvaluator
from easynlp.appzoo import Text2VideoRetrievalEvaluator
from easynlp.appzoo import CLIPEvaluator
from easynlp.appzoo import WukongCLIPEvaluator
from easynlp.appzoo import TextImageGenerationEvaluator
from easynlp.appzoo import ImageTextGenerationEvaluator
from easynlp.appzoo import FrameTextGenerationEvaluator
from easynlp.appzoo import SequenceGenerationEvaluator
from easynlp.appzoo import MachineReadingComprehensionEvaluator
from easynlp.appzoo import OpenDomainDialogueEvaluator
from easynlp.appzoo import InformationExtractionEvaluator
from easynlp.appzoo import LatentDiffusionModelEvaluator


from easynlp.appzoo import SequenceClassificationPredictor, FewshotSequenceClassificationPredictor, CptFewshotSequenceClassificationPredictor
from easynlp.appzoo import SequenceLabelingPredictor, FeatureVectorizationPredictor
from easynlp.appzoo import TextMatchPredictor, TextMatchTwoTowerPredictor, FewshotSingleTowerTextMatchPredictor, CptFewshotSingleTowerTextMatchPredictor
from easynlp.appzoo import DataAugmentationPredictor, GEEPClassificationPredictor
from easynlp.appzoo import Text2VideoRetrievalPredictor
from easynlp.appzoo import CLIPPredictor
from easynlp.appzoo import WukongCLIPPredictor
from easynlp.appzoo import TextImageGenerationPredictor
from easynlp.appzoo import LatentDiffusionPredictor
from easynlp.appzoo import VQGANGPTImageTextGenerationPredictor, CLIPGPTImageTextGenerationPredictor
from easynlp.appzoo import CLIPGPTFrameTextGenerationPredictor
from easynlp.appzoo import SequenceGenerationPredictor
from easynlp.appzoo import MachineReadingComprehensionPredictor
from easynlp.appzoo import OpenDomainDialoguePredictor
from easynlp.appzoo import InformationExtractionPredictor

from easynlp.appzoo import ClassificationDataset, DistillatoryClassificationDataset, FewshotSequenceClassificationDataset
from easynlp.appzoo import SequenceLabelingDataset, LanguageModelingDataset
from easynlp.appzoo import SingleTowerDataset, TwoTowerDataset, DistillatorySingleTowerDataset, FewshotSingleTowerTextMatchDataset, SiameseDataset
from easynlp.appzoo import SequenceGenerationDataset
from easynlp.appzoo import GEEPClassificationDataset
from easynlp.appzoo import Text2VideoRetrievalDataset
from easynlp.appzoo import CLIPDataset
from easynlp.appzoo import WukongCLIPDataset
from easynlp.appzoo import TextImageDataset
from easynlp.appzoo import CLIPGPTImageTextDataset, VQGANGPTImageTextDataset
from easynlp.appzoo import CLIPGPTFrameTextDataset
from easynlp.appzoo import MachineReadingComprehensionDataset
from easynlp.appzoo import OpenDomainDialogueDataset
from easynlp.appzoo import InformationExtractionDataset
from easynlp.appzoo import LdmDataset


from easynlp.core import PredictorManager, Trainer, DistillatoryTrainer
from easynlp.utils.logger import logger
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.utils import initialize_easynlp, get_args

try:
    from easynlp.utils.arguments import get_ds_args
    from easynlp.utils import get_pretrain_model_path
    import torch
    import os
    from easynlp.appzoo.sequence_generation.mg_seq2seq.finetune import main
    from easynlp.modelzoo.mg_utils.pretrain_glm import initialize_distributed, set_random_seed
except:
    pass

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
    'clip4clip': Text2VideoRetrievalDataset,
    'clip': CLIPDataset,
    'wukong_clip': WukongCLIPDataset,
    'text2image_generation': TextImageDataset,
    'image2text_generation': {
        'enable_vit': CLIPGPTImageTextDataset,
        'enable_vqgan': VQGANGPTImageTextDataset,
        'others': CLIPGPTImageTextDataset,
    },
    'video2text_generation': CLIPGPTFrameTextDataset,
    'sequence_generation': SequenceGenerationDataset,
    'machine_reading_comprehension': MachineReadingComprehensionDataset,
    'open_domain_dialogue': OpenDomainDialogueDataset,
    'information_extraction': InformationExtractionDataset,
    'latent_diffusion':LdmDataset

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
    'clip4clip': Text2VideoRetrieval,
    'clip': CLIPApp,
    'wukong_clip': WukongCLIP,
    'text2image_generation': TextImageGeneration,
    'image2text_generation': {
        'enable_vit': CLIPGPTImageTextGeneration,
        'enable_vqgan': VQGANGPTImageTextGeneration,
        'others': CLIPGPTImageTextGeneration,
    },
    'vqgan_image2text_generation': VQGANGPTImageTextGeneration,
    'video2text_generation': CLIPGPTFrameTextGeneration, 
    'sequence_generation': SequenceGeneration,
    'machine_reading_comprehension': MachineReadingComprehension,
    'open_domain_dialogue': OpenDomainDialogue,
    'information_extraction': InformationExtractionModel,
    'latent_diffusion':LatentDiffusion,

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
    'clip4clip': Text2VideoRetrieval,
    'clip': CLIPApp,
    'wukong_clip': WukongCLIP,
    'text2image_generation': TextImageGeneration,
    'image2text_generation': {
        'enable_vit': CLIPGPTImageTextGeneration, 
        'enable_vqgan': VQGANGPTImageTextGeneration,
        'others': CLIPGPTImageTextGeneration
    },
    'vqgan_image2text_generation': VQGANGPTImageTextGeneration, 
    'video2text_generation': CLIPGPTFrameTextGeneration, 
    'sequence_generation': SequenceGeneration,
    'machine_reading_comprehension': MachineReadingComprehension,
    'open_domain_dialogue': OpenDomainDialogue,
    'information_extraction': InformationExtractionModel,
    'latent_diffusion': LatentDiffusion

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
    'clip4clip': Text2VideoRetrievalEvaluator,
    'clip': CLIPEvaluator,
    'wukong_clip': WukongCLIPEvaluator,
    'text2image_generation': TextImageGenerationEvaluator,
    'image2text_generation': {
        'enable_vit': ImageTextGenerationEvaluator,
        'enable_vqgan': ImageTextGenerationEvaluator,
        'others': ImageTextGenerationEvaluator,
    },
    'video2text_generation': FrameTextGenerationEvaluator, 
    'sequence_generation': SequenceGenerationEvaluator,
    'machine_reading_comprehension': MachineReadingComprehensionEvaluator,
    'open_domain_dialogue': OpenDomainDialogueEvaluator,
    'information_extraction': InformationExtractionEvaluator,
    'latent_diffusion': LatentDiffusionModelEvaluator
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
    'clip4clip': [Text2VideoRetrievalPredictor, Text2VideoRetrieval],
    'clip': [CLIPPredictor, CLIPApp],
    'wukong_clip': [WukongCLIPPredictor, WukongCLIP],
    'text2image_generation': [TextImageGenerationPredictor, TextImageGeneration],
    'latent_diffusion': [LatentDiffusionPredictor, LatentDiffusion, StableDiffusion],
    'image2text_generation': {
        'enable_vit': [CLIPGPTImageTextGenerationPredictor, CLIPGPTImageTextGeneration],
        'enable_vqgan': [VQGANGPTImageTextGenerationPredictor, VQGANGPTImageTextGeneration],
        'others': [CLIPGPTImageTextGenerationPredictor, CLIPGPTImageTextGeneration],
    },
    'video2text_generation': [CLIPGPTFrameTextGenerationPredictor, CLIPGPTFrameTextGeneration],
    'sequence_generation': [SequenceGenerationPredictor, SequenceGeneration],
    'machine_reading_comprehension': [MachineReadingComprehensionPredictor, MachineReadingComprehension],
    'open_domain_dialogue': [OpenDomainDialoguePredictor, OpenDomainDialogue],
    'information_extraction': [InformationExtractionPredictor, InformationExtractionModel]
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
            app_parameters = {}
            if "app_parameters" in user_defined_parameters:
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
    try:
        is_mg = False
        args = get_ds_args()
        user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
        model_info = user_defined_parameters.get('pretrain_model_name_or_path', '').split('/')
        pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
        args.pretrained_model_name_or_path = pretrained_model_name_or_path or args.checkpoint_dir
        
        args.pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)
        checkpoint_files = os.listdir(args.pretrained_model_name_or_path)
        if args.mode != 'train' and os.path.exists(args.checkpoint_dir):
            checkpoint_files += os.listdir(args.checkpoint_dir)
        if 'mg' in model_info or args.mg_model or ('latest_checkpointed_iteration.txt' in checkpoint_files and 'pytorch_model.bin' not in checkpoint_files):
            args.model_name = model_info[-1]
            is_mg = True            
        if is_mg:
            torch.backends.cudnn.enabled = False
            initialize_distributed(args)
            set_random_seed(args.seed)
            main(args, user_defined_parameters)
    except Exception as e:
        print(e)
    
    if is_mg: exit()

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

        evaluator=None
        do_eval = user_defined_parameters.get('do_eval', 'True')
        if do_eval=='True':
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
