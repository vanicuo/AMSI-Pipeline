#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-
# @Time    : 12/18/18 5:26 PM
# @Author  : Pan
# @File    : logs.py
# @Software: PyCharm
# @Script to:

#   - 日志配置

import logging

logger = logging.getLogger("EEG-logging")
logger.propagate = False


def log(level):
    fmt = '%(asctime)-15s %(name)-12s %(levelname)-8s %(filename)s [line:%(lineno)d] %(message)s'
    level_dict = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }

    if not logger.handlers:
        handler = logging.StreamHandler()
        logging.basicConfig(format=fmt)
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level_dict[level])

    return logger
