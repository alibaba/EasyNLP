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

from ...core.predictor import Predictor
from easynlp.modelzoo import TransformerTokenizer
from ...modelzoo import AutoConfig
import logging
import os
import torch
from threading import Lock
from typing import Optional, Dict, Any, List
from ...utils import io
import sys as _sys
import math
import random

def clip_text(text, max_len):
    """
    Clip text to max length, adding ellipses.
    """
    if len(text) > max_len:
        begin_text = ' '.join(text[: math.floor(0.8 * max_len)].split(' ')[:-1])
        end_text = ' '.join(
            text[(len(text) - math.floor(0.2 * max_len)) :].split(' ')[1:]
        )
        if len(end_text) > 0:
            text = begin_text + ' ...\n' + end_text
        else:
            text = begin_text + ' ...'
    return text

def colorize(text, style):
    try:
        # if we're in ipython it's okay to use colors
        __IPYTHON__
        USE_COLORS = True
    except NameError:
        USE_COLORS = _sys.stdout.isatty()

    if not USE_COLORS:
        return text

    colorstyle = os.environ.get('PARLAI_COLORSTYLE')

    RESET = '\033[0;0m'
    if style == 'red':
        return '\033[0;31m' + text + RESET
    if style == 'yellow':
        return '\033[0;93m' + text + RESET
    if style == 'green':
        return '\033[0;32m' + text + RESET
    if style == 'blue':
        return '\033[0;34m' + text + RESET
    if style == 'brightblack':
        return '\033[0;90m' + text + RESET

    if colorstyle is None or colorstyle.lower() == 'steamroller':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY_NOBK = '\033[1m'
        LIGHT_GRAY_NOBK = '\033[0m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED_NOBK = '\033[1;31m'
        HIGHLIGHT_BLUE_NOBK = '\033[0;34m'
        if style == 'highlight':
            return HIGHLIGHT_RED_NOBK + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        elif style == 'text':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY_NOBK + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'id':
            return LIGHT_GRAY_NOBK + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE_NOBK + text + RESET
        else:
            return MAGENTA + text + RESET

    if colorstyle.lower() == 'spermwhale':
        BLUE = '\033[1;94m'
        BOLD_LIGHT_GRAY = '\033[1;37;40m'
        LIGHT_GRAY = '\033[0;37;40m'
        MAGENTA = '\033[0;95m'
        HIGHLIGHT_RED = '\033[1;37;41m'
        HIGHLIGHT_BLUE = '\033[1;37;44m'
        if style == 'highlight':
            return HIGHLIGHT_RED + text + RESET
        if style == 'highlight2':
            return HIGHLIGHT_BLUE + text + RESET
        elif style == 'text':
            return LIGHT_GRAY + text + RESET
        elif style == 'bold_text':
            return BOLD_LIGHT_GRAY + text + RESET
        elif style == 'labels' or style == 'eval_labels':
            return BLUE + text + RESET
        elif style == 'label_candidates':
            return LIGHT_GRAY + text + RESET
        elif style == 'id':
            return LIGHT_GRAY + text + RESET
        elif style == 'text2':
            return MAGENTA + text + RESET
        elif style == 'field':
            return HIGHLIGHT_BLUE + text + RESET
        else:
            return MAGENTA + text + RESET

    # No colorstyle specified/found.
    return text

def display_messages(
    msg: Dict[str, Any],
    max_len: int = 1000,
) -> Optional[str]:
    """
    Return a string describing the set of messages provided.

    If prettify is true, candidates are displayed using prettytable. add_fields provides
    a list of fields in the msgs which should be displayed if verbose is off.
    """

    def _pretty_lines(indent_space, field, value, style):
        line = '{}{} {}'.format(
            indent_space, colorize('[' + field + ']:', 'field'), colorize(value, style)
        )
        return line

    line = ''
    if msg is None:
        return None

    # Possibly indent the text (for the second speaker, if two).
    space = ''

    agent_id = msg.get('id', '[no id field]')

    # Display Text
    if msg.get('text', ''):
        value = clip_text(msg['text'], max_len)
        style = 'bold_text'
        field = agent_id
        line = _pretty_lines(
            indent_space=space, field=field, value=value, style=style
        )

    return line

class OpenDomainDialoguePredictor(Predictor):
    def __init__(self, model_dir, model_cls, data_dir, user_defined_parameters, **kwargs):
        super(OpenDomainDialoguePredictor, self).__init__(kwargs)
        self.user_defined_parameters = user_defined_parameters
        if os.path.exists(model_dir):
            local_path = model_dir
        else:
            raise FileNotFoundError('The provided model path %s does not exist, please check.' % model_dir)
        self.model_dir = local_path
        config = AutoConfig.from_pretrained(model_dir, **kwargs)
        
        vocab_file = os.path.join(local_path, 'vocab.txt')
        codecs_file = os.path.join(local_path, 'dict.codecs')

        self.tokenizer = TransformerTokenizer(
            vocab_file=vocab_file,
            codecs_file=codecs_file,
            tokenizer=config.tokenizer,
            origin_model_name=user_defined_parameters.get('pretrain_model_name_or_path', '')
        )
        self.model = model_cls(pretrained_model_name_or_path=self.model_dir, user_defined_parameters=user_defined_parameters)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # self.MUTEX = Lock()
        self.turn_cnt = 0
        self.is_finished = False
        self.display_partner_persona = True
        self.p1 = ''
        self.p2 = ''
        self.history_strings = []
        self.max_turn_num = -1
        self.text_truncate = config.text_truncate
        self.label_truncate = config.label_truncate
        self.beam_size = 10
        self.delimiter = '\n'
        self.delimiter_tok = [self.tokenizer._convert_token_to_id(self.delimiter)]
        self.END_IDX = self.model.backbone.END_IDX
        self.START_IDX = self.model.backbone.START_IDX


        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                'highlight',
            )
        )

        self.context_data = self.load_context(data_dir)
    
    def preprocess(self):
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()
            if self.p1 != '':
                context_act = {'id': 'context', 'text': self.p1}
                print(display_messages(context_act))
        try:
            reply = self.get_human_reply()
        except StopIteration:
            self.finalize_episode()
            self.reset()
            return
        
        if self.turn_cnt == 0 and self.p2 != '':
            context_act = {'id': 'context', 'text': self.p2}
            self.update_history(context_act)
        self.update_history(reply)
    
    def predict(self):
        reply = {'id': 'BlenderBot'}

        # move to GPU if necessary
        input = self.message
        if torch.cuda.is_available():
            temp = {}
            for key in input.keys():
                value = input[key]
                if torch.is_tensor(value):
                    temp[key] = value.to('cuda')
                else:
                    temp[key] = value
            input = temp

        with torch.no_grad():
            beam_preds_scores = None
            preds = None
            maxlen = self.label_truncate
            beam_preds_scores = self.model._generate(
                input, self.beam_size, maxlen
            )
            preds, _, _ = zip(*beam_preds_scores)
            text = self._v2t(preds[0].tolist()) if preds is not None else None
            reply['text'] = text
        
        return reply

    def postprocess(self, result):
        self.update_history(result)
        print(
            display_messages(result)
        )
        self.turn_cnt += 1
    
    def run(self):
        while True:
            self.preprocess()

            if self.is_finished:
                logging.info('epoch done')
                break
            if self.message is None:
                continue

            reply = self.predict()
            self.postprocess(reply)

    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        if not self.is_finished:
            print("\n[ Preparing new chat ... ]\n")
    
    def reset(self):
        self.turn_cnt = 0
        self.p1 = ''
        self.p2 = ''
        self.message = None
        self.history_strings = []
        
    def load_context(self, data_file):
        print('[ loading personas.. ]')
        print(
            "\n  [NOTE: In the BST paper both partners have a persona.\n"
            + '         You can choose to ignore yours, the model never sees it.\n'
            + '         In the Blender paper, this was not used for humans.\n'
            + '         You can also turn personas off with --include-personas False]\n'
        )

        def readlines_from_file(data_file) -> List[str]:
            print(f'****{data_file}')
            with io.open(data_file) as f:
                data_rows = f.readlines()
            return data_rows

        rows = readlines_from_file(data_file)
        contexts = []
        for r in rows:
            context1 = []
            context2 = []
            all_context = r.strip('\n').split('\t')
            context1.append('your persona: ' + all_context[0])
            context1.append('your persona: ' + all_context[1])
            context2.append('your persona: ' + all_context[2])
            context2.append('your persona: ' + all_context[3])
            c1 = '\n'.join(context1)
            c2 = '\n'.join(context2)
            contexts.append([c1, c2])
        return contexts
    
    def get_contexts(self):
        random.seed()
        p = random.choice(self.context_data)
        return p[0], p[1]
    
    def get_human_reply(self):
        reply = {'id': 'safeLocalHuman'}
        reply_text = input(colorize('Enter Your Message:', 'field') + ' ')
        reply_text = reply_text.replace('\\n', '\n')

        # check for episode done
        if '[DONE]' in reply_text:
            raise StopIteration

        # set reply text
        reply['text'] = reply_text

        # check if finished
        if '[EXIT]' in reply_text:
            self.is_finished = True
            raise StopIteration
        
        return reply
    
    def update_history(self, msg):
        self.message = msg
        if self.max_turn_num > 0:
            while len(self.history_strings) >= self.max_turn_num:
                self.history_strings.pop(0)
        text = msg['text']
        self.history_strings.append(text)

        # set the 'text_vec' field in the message
        history_string = self.get_history_str()
        if history_string is None:
            return
        self.message['full_text'] = history_string
        if history_string:
            self.message['text_vec'] = self.tokenizer(self.message['full_text'])['input_ids']
        
        # check truncation
        if self.message['text_vec'] is not None:
            truncated_vec = self._check_truncate(
                self.message['text_vec'], self.text_truncate
            )
            self.message.__setitem__('text_vec', torch.LongTensor(truncated_vec))
        return

    def get_history_str(self):
        if len(self.history_strings) > 0:
            history = self.history_strings[:]
            history = self.delimiter.join(history)
            return history
        
        return None

    def _check_truncate(self, vec, truncate):
        if truncate is None or len(vec) <= truncate:
            return vec
        else:
            return vec[-truncate:]
    
    def _v2t(self, vec):
        new_vec = []
        for i in vec:
            if i == self.END_IDX:
                break
            elif i != self.START_IDX:
                new_vec.append(i)
        return self.tokenizer.decode(new_vec)