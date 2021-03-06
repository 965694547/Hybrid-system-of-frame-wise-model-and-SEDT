import logging
import sys
import logging.config
import os
import time
from utilities.distribute import is_main_process


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def create_logger(logger_name, terminal_level=logging.INFO):
    """ Create a logger.
    Args:
        logger_name: str, name of the logger
        terminal_level: int, logging level in the terminal
    """
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
    })
    logger = logging.getLogger(logger_name)
    tool_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')

    if type(terminal_level) is str:
        if terminal_level.lower() == "debug":
            res_terminal_level = logging.DEBUG
        elif terminal_level.lower() == "info":
            res_terminal_level = logging.INFO
        elif "warn" in terminal_level.lower():
            res_terminal_level = logging.WARNING
        elif terminal_level.lower() == "error":
            res_terminal_level = logging.ERROR
        elif terminal_level.lower() == "critical":
            res_terminal_level = logging.CRITICAL
        else:
            res_terminal_level = logging.NOTSET
    else:
        res_terminal_level = terminal_level

    if not is_main_process():
        res_terminal_level =  logging.ERROR

    logger.setLevel(res_terminal_level)
    # Remove the stdout handler
    logger_handlers = logger.handlers[:]
    if not len(logger_handlers):
        terminal_h = logging.StreamHandler(sys.stdout)
        terminal_h.setLevel(res_terminal_level)
        terminal_h.set_name('stdout')
        terminal_h.setFormatter(tool_formatter)
        logger.addHandler(terminal_h)
    return logger


def set_logger(info):
    log_path = './log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + info + time.strftime("-%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)
