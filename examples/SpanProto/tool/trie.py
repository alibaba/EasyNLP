# -*- coding: utf-8 -*-
# @Time    : 2022/2/15 7:57 pm.
# @Author  : JianingWang
# @File    : trie
import logging
from typing import List
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Trie:
    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def find(self, text: str):
        states = OrderedDict()
        offsets = []
        skip = 0
        for current, current_char in enumerate(text):
            if skip and current < skip:
                continue
            to_remove = set()
            reset = False
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            break
                        elif lookstart < start:
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                break
                            next_char = text[lookahead_index]
                    offsets.append([start, end])
                    reset = True
                    break
                elif current_char in trie_pointer:
                    trie_pointer = trie_pointer[current_char]
                    states[start] = trie_pointer
                else:
                    to_remove.add(start)
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                end = len(text)
                offsets.append([start, end])
                break

        return offsets

    def split(self, text: str) -> List[str]:
        """
        Example:

        ```python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        ```
        """
        word_sets = self.find(text)
        offsets = [0]
        for w in word_sets:
            offsets.extend(w)
        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                continue
            tokens.append(text[start:end])
            start = end

        return tokens

    def __reduce__(self):
        return None


if __name__ == '__main__':
    trie = Trie()
    for word in ['低血压', '血压', '肝', '肝癌']:
        trie.add(word)
    # print(trie.find('低温、低血压，拟以“肝癌TACE术后，感染性休克”收入院。'))
    print(trie.__reduce__())
