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

from __future__ import absolute_import

import logging


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            pass

        return no_op


logger = logging.getLogger()


def init_logger(log_file=None, local_rank=-1):
    global logger
    if local_rank > 0:
        logger = NoOp()
        return
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
