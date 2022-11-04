# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 5:41 pm.
# @Author  : JianingWang
# @File    : common.py
import sys
import logging
import datasets
import transformers


def init_logger(log_file, log_level, dist_rank):
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    datasets.utils.logging.disable_propagation()
    # transformers.utils.logging.enable_propagation()

    logger = logging.getLogger('')
    log_format = logging.Formatter(fmt='[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # transformer_logger = logging.getLogger('transformers')
    # transformer_logger.handlers = []
    # transformer_logger.propagate = True

    if dist_rank in [-1, 0]:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.getLogger('transformers').addHandler(file_handler)


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)  #
            or (0x20000 <= cp <= 0x2A6DF)  #
            or (0x2A700 <= cp <= 0x2B73F)  #
            or (0x2B740 <= cp <= 0x2B81F)  #
            or (0x2B820 <= cp <= 0x2CEAF)  #
            or (0xF900 <= cp <= 0xFAFF)
            or (0x2F800 <= cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word: str):
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not is_chinese_char(char):
            return 0
    return 1
