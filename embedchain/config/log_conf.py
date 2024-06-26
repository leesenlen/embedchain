import logging
import concurrent_log


def logger_exists(name):
    return name in logging.Logger.manager.loggerDict


logger_name = 'embedchain'

if logger_exists(logger_name):
    logger = logging.getLogger(logger_name)
else:
    logger = logging