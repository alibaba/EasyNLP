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

from typing import TYPE_CHECKING
from ..modelzoo.file_utils import _BaseLazyModule
TYPE_CHECKING = True

_import_structure = {
    "sequence_classification.model": ["SequenceClassification", "SequenceMultiLabelClassification", "DistillatorySequenceClassification", "FewshotSequenceClassification", "CptFewshotSequenceClassification"],
    "sequence_labeling.model": ['SequenceLabeling'],
    "language_modeling.model": ['LanguageModeling'],
    "feature_vectorization.model": ['FeatureVectorization'],
    "text_match.model": ['TextMatch', 'TextMatchTwoTower', 'DistillatoryTextMatch', 'FewshotSingleTowerTextMatch', 'CptFewshotSingleTowerTextMatch'],
    "data_augmentation.model": ["DataAugmentation"],
    "geep_classification.model": ["GEEPClassification"],
    "text2video_retrieval.model": ["Text2VideoRetrieval"],
    "clip.model": ["CLIPApp"],
    "latent_diffusion.model": ["LatentDiffusion","StableDiffusion"],
    "wukong_clip.model": ["WukongCLIP"],
    "text2image_generation.model": ["TextImageGeneration", "TextImageGeneration_knowl"],
    "image2text_generation.model": ['VQGANGPTImageTextGeneration', 'CLIPGPTImageTextGeneration'], 
    "video2text_generation.model": ['CLIPGPTFrameTextGeneration'], 
    "sequence_generation.model": ["SequenceGeneration"], 
    "machine_reading_comprehension.model": ["MachineReadingComprehension"],
    "open_domain_dialogue.model": ["OpenDomainDialogue"],
    "information_extraction.model": ["InformationExtractionModel"], 

    "sequence_classification.evaluator": ['SequenceClassificationEvaluator', 'SequenceMultiLabelClassificationEvaluator'],
    "sequence_labeling.evaluator": ['SequenceLabelingEvaluator'],
    "language_modeling.evaluator": ['LanguageModelingEvaluator'],
    "text_match.evaluator": ['TextMatchEvaluator'],
    "geep_classification.evaluator": ['GEEPClassificationEvaluator'],
    "text2video_retrieval.evaluator": ['Text2VideoRetrievalEvaluator'],
    "clip.evaluator": ['CLIPEvaluator'],
    "wukong_clip.evaluator": ['WukongCLIPEvaluator'],
    "text2image_generation.evaluator": ["TextImageGenerationEvaluator"],
    "image2text_generation.evaluator": ["ImageTextGenerationEvaluator"], 
    "video2text_generation.evaluator": ["FrameTextGenerationEvaluator"], 
    "sequence_generation.evaluator": ["SequenceGenerationEvaluator"], 
    "machine_reading_comprehension.evaluator": ["MachineReadingComprehensionEvaluator"],
    "open_domain_dialogue.evaluator": ["OpenDomainDialogueEvaluator"],
    "information_extraction.evaluator": ["InformationExtractionEvaluator"],
    "latent_diffusion.evaluator": ["LatentDiffusionModelEvaluator"],

    "sequence_classification.predictor": ['SequenceClassificationPredictor', 'FewshotSequenceClassificationPredictor', 'CptFewshotSequenceClassificationPredictor'],
    "sequence_labeling.predictor": ['SequenceLabelingPredictor'],
    "feature_vectorization.predictor": ['FeatureVectorizationPredictor'],
    "text_match.predictor": ['TextMatchPredictor', 'TextMatchTwoTowerPredictor', 'FewshotSingleTowerTextMatchPredictor', 'CptFewshotSingleTowerTextMatchPredictor'],
    "data_augmentation.predictor": ['DataAugmentationPredictor'],
    "geep_classification.predictor": ['GEEPClassificationPredictor'],
    "text2video_retrieval.predictor": ['Text2VideoRetrievalPredictor'],
    "clip.predictor": ['CLIPPredictor'],
    "latent_diffusion.predictor": ['LatentDiffusionPredictor'],
    "wukong_clip.predictor": ['WukongCLIPPredictor'],
    "text2image_generation.predictor": ['TextImageGenerationPredictor', 'TextImageGenerationKnowlPredictor'],
    "image2text_generation.predictor": ['VQGANGPTImageTextGenerationPredictor', 'CLIPGPTImageTextGenerationPredictor'],
    "video2text_generation.predictor": ['CLIPGPTFrameTextGenerationPredictor'],
    "sequence_generation.predictor": ['SequenceGenerationPredictor'],
    "machine_reading_comprehension.predictor": ["MachineReadingComprehensionPredictor"],
    "open_domain_dialogue.predictor": ["OpenDomainDialoguePredictor"],
    "information_extraction.predictor": ["InformationExtractionPredictor"],

    "geep_classification.data": ['GEEPClassificationDataset'],
    "language_modeling.data": ['LanguageModelingDataset'],
    "text2video_retrieval.data": ['Text2VideoRetrievalDataset'],
    "clip.data": ['CLIPDataset'],
    "wukong_clip.data": ['WukongCLIPDataset'],
    "sequence_classification.data": ['ClassificationDataset', 'DistillatoryClassificationDataset', 'FewshotSequenceClassificationDataset'],
    "sequence_labeling.data": ['SequenceLabelingDataset', 'SequenceLabelingAutoDataset'],
    "text_match.data": ['TwoTowerDataset', 'SingleTowerDataset', 'DistillatorySingleTowerDataset', 'FewshotSingleTowerTextMatchDataset', 'SiameseDataset'],
    "text2image_generation.data": ['TextImageDataset', 'TextImageKnowlDataset'],
    "image2text_generation.data": ['CLIPGPTImageTextDataset', 'VQGANGPTImageTextDataset'],
    "video2text_generation.data": ['CLIPGPTFrameTextDataset'],
    "sequence_generation.data": ['SequenceGenerationDataset'],
    "machine_reading_comprehension.data": ["MachineReadingComprehensionDataset"],
    "open_domain_dialogue.data": ['OpenDomainDialogueDataset'],
    "information_extraction.data": ["InformationExtractionDataset"],
    "latent_diffusion.data": ["LdmDataset"],

    "dataset": ['BaseDataset', 'GeneralDataset', 'load_dataset', 'list_datasets'],
    "api": ['get_application_dataset', 'get_application_model', 'get_application_model_for_evaluation', 'get_application_evaluator', 'get_application_predictor'],
}

if TYPE_CHECKING:
    from .sequence_classification.model import SequenceClassification, SequenceMultiLabelClassification, DistillatorySequenceClassification, FewshotSequenceClassification, CptFewshotSequenceClassification
    from .sequence_labeling.model import SequenceLabeling
    from .language_modeling.model import LanguageModeling
    from .feature_vectorization.model import FeatureVectorization
    from .text_match.model import TextMatch, TextMatchTwoTower, DistillatoryTextMatch, FewshotSingleTowerTextMatch, CptFewshotSingleTowerTextMatch
    from .data_augmentation.model import DataAugmentation
    from .geep_classification.model import GEEPClassification
    from .text2video_retrieval.model import Text2VideoRetrieval
    from .clip.model import CLIPApp
    from .latent_diffusion.model import LatentDiffusion,StableDiffusion
    from .wukong_clip.model import WukongCLIP
    from .text2image_generation.model import TextImageGeneration, TextImageGeneration_knowl
    from .image2text_generation.model import VQGANGPTImageTextGeneration, CLIPGPTImageTextGeneration
    from .video2text_generation.model import CLIPGPTFrameTextGeneration
    from .sequence_generation.model import SequenceGeneration
    from .machine_reading_comprehension.model import MachineReadingComprehension
    from .open_domain_dialogue.model import OpenDomainDialogue
    from .information_extraction.model import InformationExtractionModel

    from .sequence_classification.evaluator import SequenceClassificationEvaluator, SequenceMultiLabelClassificationEvaluator
    from .sequence_labeling.evaluator import SequenceLabelingEvaluator
    from .language_modeling.evaluator import LanguageModelingEvaluator
    from .text_match.evaluator import TextMatchEvaluator
    from .geep_classification.evaluator import GEEPClassificationEvaluator
    from .text2video_retrieval.evaluator import Text2VideoRetrievalEvaluator
    from .clip.evaluator import CLIPEvaluator
    from .wukong_clip.evaluator import WukongCLIPEvaluator
    from .text2image_generation.evaluator import TextImageGenerationEvaluator
    from .image2text_generation.evaluator import ImageTextGenerationEvaluator
    from .video2text_generation.evaluator import FrameTextGenerationEvaluator
    from .sequence_generation.evaluator import SequenceGenerationEvaluator
    from .machine_reading_comprehension.evaluator import MachineReadingComprehensionEvaluator
    from .open_domain_dialogue.evaluator import OpenDomainDialogueEvaluator
    from .information_extraction.evaluator import InformationExtractionEvaluator
    from .latent_diffusion.evaluator import LatentDiffusionModelEvaluator

    from .sequence_classification.predictor import SequenceClassificationPredictor, FewshotSequenceClassificationPredictor, CptFewshotSequenceClassificationPredictor
    from .sequence_labeling.predictor import SequenceLabelingPredictor
    from .feature_vectorization.predictor import FeatureVectorizationPredictor
    from .text_match.predictor import TextMatchPredictor, TextMatchTwoTowerPredictor, FewshotSingleTowerTextMatchPredictor, CptFewshotSingleTowerTextMatchPredictor
    from .data_augmentation.predictor import DataAugmentationPredictor
    from .geep_classification.predictor import GEEPClassificationPredictor
    from .text2video_retrieval.predictor import Text2VideoRetrievalPredictor
    from .clip.predictor import CLIPPredictor
    from .latent_diffusion.predictor import LatentDiffusionPredictor
    from .wukong_clip.predictor import WukongCLIPPredictor
    from .text2image_generation.predictor import TextImageGenerationPredictor, TextImageGenerationKnowlPredictor
    from .image2text_generation.predictor import VQGANGPTImageTextGenerationPredictor, CLIPGPTImageTextGenerationPredictor
    from .video2text_generation.predictor import CLIPGPTFrameTextGenerationPredictor
    from .sequence_generation.predictor import SequenceGenerationPredictor
    from .machine_reading_comprehension.predictor import MachineReadingComprehensionPredictor
    from .open_domain_dialogue.predictor import OpenDomainDialoguePredictor
    from .information_extraction.predictor import InformationExtractionPredictor

    from .sequence_classification.data import ClassificationDataset, DistillatoryClassificationDataset, FewshotSequenceClassificationDataset
    from .sequence_labeling.data import SequenceLabelingDataset, SequenceLabelingAutoDataset
    from .language_modeling.data import LanguageModelingDataset
    from .text_match.data import TwoTowerDataset, SingleTowerDataset, DistillatorySingleTowerDataset, FewshotSingleTowerTextMatchDataset, SiameseDataset
    from .geep_classification.data import GEEPClassificationDataset
    from .text2video_retrieval.data import Text2VideoRetrievalDataset
    from .clip.data import CLIPDataset
    from .wukong_clip.data import WukongCLIPDataset
    from .text2image_generation.data import TextImageDataset, TextImageKnowlDataset
    from .image2text_generation.data import CLIPGPTImageTextDataset, VQGANGPTImageTextDataset
    from .video2text_generation.data import CLIPGPTFrameTextDataset
    from .sequence_generation.data import SequenceGenerationDataset
    from .machine_reading_comprehension.data import MachineReadingComprehensionDataset
    from .open_domain_dialogue.data import OpenDomainDialogueDataset
    from .information_extraction.data import InformationExtractionDataset
    from .latent_diffusion.data import LdmDataset

    from .dataset import BaseDataset, GeneralDataset
    from .dataset import load_dataset, list_datasets

    from .api import get_application_dataset, get_application_model, get_application_model_for_evaluation
    from .api import get_application_evaluator, get_application_predictor
    
    

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
