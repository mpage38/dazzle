import random
import string
import numpy as np
from collections import deque
import time
from dazzle.core.dazzle_expr_parser import expr, StringLiteral, Ident

class StringBuffer:
    def __init__(self, strings, starts, offset):
        self.strings = np.array(strings, dtype=np.uint8)
        self.starts = np.array(starts, dtype=np.uint64)
        self.offset = offset

    @staticmethod
    def random_string(n):
        """Generate a random string of length n, where all the characters are the same, except the first and last
            that are capitalized"""

        c = random.choice(string.ascii_letters).lower()
        return str(c.upper() + (str(c) * (n-2)) + c.upper())

    @staticmethod
    def generate(size):
        """Generate a StringBuffer with size.
            String data and index are stored onto disk into files data_path, addr_path respectively.
            An index value is an uint64 made of 40 bits indicating position in data array and 24 bits indivcating length.
            For instance 'hello\0how\0are\0you' is stored as:
            - [h e l l o \0 h o w \0 a r e \0 y o u]
        """
        chars = np.zeros([size * 10], dtype=np.uint8)
        starts = np.zeros([size], dtype=np.uint64)
        word_lengths = np.zeros([size], dtype=np.uint64)
        addrs = np.zeros([size], dtype=np.uint64)
        total_length = 0
        for i in range(size):
            letter = random.randint(97, 122)
            word_length = random.randint(2,10)
            for l in range(word_length):
                chars[total_length + l] = letter
            chars[total_length + word_length] = 0
            starts[i] = total_length
            word_lengths[i] = word_length
            total_length += (word_length + 1)

        chars1 = chars[0:total_length]
        addrs = (starts * 2 ** 24) + word_lengths
        return StringBuffer(chars1, addrs, 0)

    def __repr__(self):
        return str(self.strings.tobytes()) + '\n' + str(self.starts)


class Op:
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity


class Operand:
    def __init__(self, type, value):
        self.type = type
        self.value = value

class Expr:

    def __init__(self, op, operands):
        self.op = op
        self.operands = operands

    @staticmethod
    def build_from_parse_list(queue):
        if len(queue) > 0:
            if queue[0] in ChunkEvaluator.function_table:
                op_name = queue.popleft()
                op_dict = ChunkEvaluator.function_table[op_name]
                operands = []
                for i in range(len(op_dict['args'])):
                    operands.append(Expr.build_from_parse_list(queue))
                return Expr(op_name, operands)
            elif isinstance(queue[0], StringLiteral):
                sl = queue.popleft()
                return str.encode(sl.value)
            elif isinstance(queue[0], Ident):
                id = queue.popleft()
                sb = ChunkEvaluator.param_dict[id.name]
                return sb
            elif isinstance(queue[0], (int,  float)):
                n = queue.popleft()
                return n
            else:
                raise ValueError("Incorrect value in expr: %s" % queue[0])

    def __repr__(self):
        return self.op + "(" + ",".join([o.__repr__() for o in self.operands]) + ")"

    def eval(self):
        operands = [o.eval() if isinstance(o, Expr) else o for o in self.operands]
        if self.op == 'concat':
            return concat(operands[0], operands[1])
        elif self.op == 'upper':
            return upper(operands[0])
        elif self.op == 'lower':
            return lower(self.operands[0])
        elif self.op == 'substring':
            return substring(operands[0], operands[1], operands[2])
        else:
            raise Exception("unknown op: %s" % self.op)

class ChunkEvaluator:

    function_table = {
        "abs": {'args': ['float'], 'return': 'float'},
        "arccos": {'args': ['float'], 'return': 'float'},
        "arccosh": {'args': ['float'], 'return': 'float'},
        "arcsin": {'args': ['float'], 'return': 'float'},
        "arcsinh": {'args': ['float'], 'return': 'float'},
        "arctan": {'args': ['float'], 'return': 'float'},
        "arctan2": {'args': ['float'], 'return': 'float'},
        "arctanh": {'args': ['float'], 'return': 'float'},
        "concat": {'args': ['string', 'string'], 'return': 'string'},
        "cos": {'args': ['float'], 'return': 'float'},
        "cosh": {'args': ['float'], 'return': 'float'},
        "exp": {'args': ['float'], 'return': 'float'},
        "expm1": {'args': ['float'], 'return': 'float'},
        "log": {'args': ['float'], 'return': 'float'},
        "log10": {'args': ['float'], 'return': 'float'},
        "log1p": {'args': ['float'], 'return': 'float'},
        "lower": {'args': ['string'], 'return': 'string'},
        "sin": {'args': ['float'], 'return': 'float'},
        "sinh": {'args': ['float'], 'return': 'float'},
        "sqrt": {'args': ['float'], 'return': 'float'},
        "substring": {'args': ['string', 'int', 'int'], 'return': 'string'},
        "tan": {'args': ['float'], 'return': 'float'},
        "tanh": {'args': ['float'], 'return': 'float'},
        "upper": {'args': ['string'], 'return': 'string'},
        "where": {'args': ['bool', 'any', 'any'], 'return': 'any'},
        "+": {'args': [['float'], ['float', 'float']], 'return': 'float'},
        "-": {'args': [['float'], ['float', 'float']], 'return': 'float'},
        "*": {'args': ['float', 'float'], 'return': 'float'},
        "/": {'args': ['float', 'float'], 'return': 'float'},
        "%": {'args': ['int', 'int'], 'return': 'int'},
        "**": {'args': ['float', 'float'], 'return': 'float'},
        ">>": {'args': ['int', 'int'], 'return': 'int'},
        "<<": {'args': ['int', 'int'], 'return': 'int'},
        "~": {'args': ['bool'], 'return': 'bool'},
        "|": {'args': ['bool', 'bool'], 'return': 'bool'},
        ">": {'args': ['any', 'any'], 'return': 'bool'},
        ">=": {'args': ['any', 'any'], 'return': 'bool'},
        "<": {'args': ['any', 'any'], 'return': 'bool'},
        "<=": {'args': ['any', 'any'], 'return': 'bool'},
        "==": {'args': ['any', 'any'], 'return': 'bool'},
        "!=": {'args': ['any', 'any'], 'return': 'bool'}
    }

    param_dict = {}

    def parse(self, expr_string):
        parse_list = expr.parseString(expr_string, parseAll=True)
        return parse_list



def concat(str_arg1, str_arg2):
    if isinstance(str_arg1, StringBuffer):
        ls1 = str_arg1.strings.tobytes().split(b'\0')
        if isinstance(str_arg2, StringBuffer):
            ls2 = str_arg2.strings.tobytes().split(b'\0')
            s = b'\0'.join([(s1 + s2) for s1, s2 in zip(ls1, ls2)])
        else:
            s = (str_arg2 + b'\0').join(ls1)
    else:
        if isinstance(str_arg2, StringBuffer):
            ls2 = str_arg2.strings.tobytes().split(b'\0')
            s = (b'\0' + str_arg1).join(ls2)
        else:
            return str_arg1 + str_arg2

    result_strings = np.frombuffer(s + b'\0', dtype=np.uint8)
    # result_starts = [0]
    # result_starts.extend([(i + 1) for i, char in enumerate(s) if char == 0])
    #out_sb = StringBuffer(result_strings, np.array(result_starts, dtype=np.uint64), 0)
    result_starts = np.nonzero(result_strings == 0)
    out_sb = StringBuffer(result_strings, result_starts, 0)
    return out_sb

def upper(str_arg):
    if isinstance(str_arg, StringBuffer):
        upper_strings = np.frombuffer(str_arg.strings.tobytes().upper(), dtype=np.uint8)
        out_sb = StringBuffer(upper_strings, str_arg.starts, 0)
        return out_sb
    else:
        return str_arg.upper()

def lower(str_arg):
    if isinstance(str_arg, StringBuffer):
        lower_strings = np.frombuffer(str_arg.strings.tobytes().lower(), dtype=np.uint8)
        out_sb = StringBuffer(lower_strings, str_arg.starts, 0)
        return out_sb
    else:
        return str_arg.lower()


def substring(str_arg, from_pos, length):
    if isinstance(str_arg, StringBuffer):
        ls = str_arg.strings.tobytes().split(b'\0')
        if isinstance(from_pos, int):
            if isinstance(length, int):
                lsubst = [s[from_pos:length] for s in ls]
            else:
                lsubst = [s[from_pos:l] for s, l in zip(ls, length)]
        else:
            if isinstance(length, int):
                lsubst = [s[f:length] for s, f in zip(ls, from_pos)]
            else:
                lsubst = [s[f:l] for s, f, l in zip(ls, from_pos, length)]
    else:
        if isinstance(from_pos, int):
            if isinstance(length, int):
                return str_arg[from_pos:length]
            else:
                lsubst = [str_arg[from_pos:l] for l in length]
        else:
            if isinstance(length, int):
                lsubst = [str_arg[f:length] for f in from_pos]
            else:
                lsubst = [str_arg[f:l] for f, l in zip(from_pos, length)]

    result_strings = np.frombuffer(b'\0'.join(lsubst) + b'\0', dtype=np.uint8)
    result_starts = [0]
    result_starts.extend(np.cumsum([len(s) + 1 for s in lsubst], dtype=np.uint64))
    out_sb = StringBuffer(result_strings, result_starts, 0)
    return out_sb



if __name__ == '__main__':

    def test1():
        import bcolz
        print(bcolz.__version__)

    def test():
        sb1 = StringBuffer([115,  97, 108, 117, 116, 0,  99, 111, 109, 109, 101, 110, 116, 0,  99,  97,   0, 118,  97,   0,
                            98, 105, 101, 110,   0, 101, 116, 0, 116, 111, 105], [0, 6, 14, 17, 20, 25, 28], 0)

        sb2 = StringBuffer([111, 117, 105,  0,  99,  97,  0, 118,  97,  0,  98, 105, 101,
                           110,  0, 109, 101, 114,  99, 105,  0, 115,  97, 108, 117, 116,
                            0,  98, 121, 101], [0, 4, 7, 10, 15, 21, 27], 0)

        zeros = [0] *7
        length = [3, 1, 3, 2, 2, 5, 10]

        size = 10**6
        sb1 = StringBuffer.generate(size)
        zeros = [0] * (size)
        length = [2] * (size)

        ChunkEvaluator.param_dict = {
            'sb1': sb1,
            'zeros': zeros,
            'length': length
        }

        evaluator = ChunkEvaluator()
        test = "concat('b', upper(substring(sb1, 0, 1)))"
        parse_list = evaluator.parse(test)
        queue = deque(parse_list)
        e = Expr.build_from_parse_list(queue)

        t1 = time.time()
        e.eval()
        print(time.time() - t1)

        # Expr('upper', [Expr('substring', ['sb1', 'zeros', 'len'])]),

    test1()
