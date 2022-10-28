# -*- coding: utf-8 -*-
# @Time    : 2021/12/3 2:38 pm.
# @Author  : JianingWang
# @File    : utils.py
import copy
import pickle
import json
import unicodedata
from typing import List, Tuple

def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


class MetaExample(object):
    """The meta example of text"""

    def __init__(self,
                 guid: str,
                 text: str,
                 mention_list: List[str],
                 mention_span: List[Tuple[int, int]],
                 mention_link_entity: List[str],
                 meta_type: int = 1,  # 当文本为KG所属domain时，标记为1
                 unique_id: int = 0):
        """
        Create a new MetaExample
        :param guid: a unique textual identifier
        :param text: the sequence of text
        :param mention_list: all entity mentions extracted from the text
        :param mention_span: the start and end position span of each mentions of the text
        :param meta_type: the type id of the text (0: out-domain, 1: in-domain)
        :param unique_id: an optional numeric index
        """
        self.guid = guid
        self.text = text
        self.mention_list = mention_list
        self.mention_span = mention_span
        self.mention_link_entity = mention_link_entity
        self.meta_type = meta_type
        self.unique_id = unique_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['MetaExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['MetaExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)
