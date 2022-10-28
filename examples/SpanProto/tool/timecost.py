# -*- coding: utf-8 -*-
# @Time    : 2022/3/11 3:06 pm.
# @Author  : JianingWang
# @File    : time
import time
import logging

logger = logging.getLogger(__name__)


def timecost(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed
