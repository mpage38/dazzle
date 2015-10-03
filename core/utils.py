import inspect
import logging
import os
import shutil
import bcolz


def method_name():
    """Returns calling method name; useful for displaying error messages
    """
    return inspect.stack()[1][3]

def rmtree_or_file(f):
    if os.path.isdir(f):
        shutil.rmtree(f)
    else:
        os.remove(f)

def file_rough_lines_count(file_path, learn_size=1000):
    with open(file_path, 'rb') as f:
        buf = f.read(learn_size)
        line_size = len(buf) // buf.count(b'\n')

    lines_count = os.path.getsize(file_path) // line_size
    return lines_count

def eval(expr):
    return bcolz.eval(expr)

class DazzleError(Exception):
    """Generic dazzle error class
    """

class DazzleFileOrDirExistsError(DazzleError):
    pass


logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
