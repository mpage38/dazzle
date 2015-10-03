cimport numpy as np
from libc.stdlib cimport malloc, free
import random
import string

import time
import numpy as np
from collections import deque
from dazzle.core.dazzle_expr_parser import expr, StringLiteral, Ident

cdef extern from "string.h":
    char *strcpy(char *dest, char *src)

cdef extern from  "chunk_evaluator.h":
    ctypedef char* char_ptr
    ctypedef void* void_ptr
    ctypedef size_t* size_ptr

    void init_random_seed();

    cdef struct string_buffer:
        int type
        char_ptr strings
        int char_count
        size_ptr indices
        int string_count
    ctypedef string_buffer string_buffer_t
    ctypedef string_buffer_t* string_buffer_ptr

    string_buffer_ptr dz_string_buffer_make(char_ptr strings, int char_count, size_ptr indices, int string_count)
    string_buffer_ptr dz_string_buffer_make_random(int string_count)
    char_ptr dz_string_buffer_to_string(string_buffer_ptr sb)

    cdef enum expr_node_type:
        EXPR_NODE_TYPE_UNDEF = 0,
        EXPR_NODE_TYPE_STRING_CONST = 1,
        EXPR_NODE_TYPE_INT_CONST = 2,
        EXPR_NODE_TYPE_FLOAT_CONST = 3,
        EXPR_NODE_TYPE_STRING_BUFFER = 4,
        EXPR_NODE_TYPE_STRING_BUFFER_INDEX = 5,
        EXPR_NODE_TYPE_UPPER = 6,
        EXPR_NODE_TYPE_LOWER = 7,
        EXPR_NODE_TYPE_CONCAT = 8
    ctypedef expr_node_type expr_node_type_t

    cdef struct expr_node:
        expr_node_type_t type
        long long int_arg
        double float_arg
        char_ptr string_arg
        string_buffer_ptr string_buffer_arg
        int string_buffer_index_arg
        expr_node* args[3]  # TODO change this
    ctypedef expr_node expr_node_t
    ctypedef expr_node_t* expr_node_ptr




    expr_node_ptr dz_expr_node_int_create(long long x);
    expr_node_ptr dz_expr_node_float_create(double x);
    expr_node_ptr dz_expr_node_string_create(char_ptr s);
    expr_node_ptr dz_expr_node_string_buffer_index_create(int sb_index);
    expr_node_ptr dz_expr_node_create(expr_node_type_t type, void_ptr operand1, void_ptr operand2, void_ptr operand3);

    long long dz_expr_node_get_int(expr_node_ptr n)
    double dz_expr_node_get_float(expr_node_ptr n)
    char_ptr dz_expr_node_get_string(expr_node_ptr n)
    int dz_expr_node_get_string_buffer_index(expr_node_ptr n)
    expr_node_ptr dz_expr_node_get_arg(expr_node_ptr n, int i)
    string_buffer_ptr dz_eval(expr_node_ptr n, string_buffer_ptr string_buffers[])

cdef class StringBuffer:
    """The Python counterpart of the C string_buffer_t type
    """

    cdef np.ndarray strings

    cdef np.ndarray indices

    cdef int offset

    def __cinit__(self, strings, indices, offset):
        self.strings = strings if isinstance(strings, np.ndarray) else np.array(strings, dtype=np.uint8)
        self.indices = indices if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.uint64)
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
        index = np.zeros([size], dtype=np.uint64)
        word_lengths = np.zeros([size], dtype=np.uint64)
        addrs = np.zeros([size], dtype=np.uint64)
        total_length = 0
        for i in range(size):
            letter = random.randint(97, 122)
            word_length = random.randint(2,10)
            for l in range(word_length):
                chars[total_length + l] = letter
            chars[total_length + word_length] = 0
            index[i] = total_length
            word_lengths[i] = word_length
            total_length += (word_length + 1)

        chars1 = chars[0:total_length]
        addrs = (index * 2 ** 24) + word_lengths
        return StringBuffer(chars1, addrs, 0)

    def __repr__(self):
        return str(self.strings.tobytes()) + '\n' + str(self.indices)

class ChunkEvaluator:

    stringBuffers = []

    function_table = {                                  # TODO complete 'type'
        "abs": {'args': ['float'], 'return': 'float'},
        "arccos": {'args': ['float'], 'return': 'float'},
        "arccosh": {'args': ['float'], 'return': 'float'},
        "arcsin": {'args': ['float'], 'return': 'float'},
        "arcsinh": {'args': ['float'], 'return': 'float'},
        "arctan": {'args': ['float'], 'return': 'float'},
        "arctan2": {'args': ['float'], 'return': 'float'},
        "arctanh": {'args': ['float'], 'return': 'float'},
        "concat": {'type': EXPR_NODE_TYPE_CONCAT, 'args': ['string', 'string'], 'return': 'string'},
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
        "upper": {'type': EXPR_NODE_TYPE_UPPER, 'args': ['string'], 'return': 'string'},
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


cdef expr_node_ptr build_expr_from_parse_list(queue):
    cdef void_ptr args[3]
    cdef expr_node_ptr node
    cdef char_ptr s

    if len(queue) > 0:
        if queue[0] in ChunkEvaluator.function_table:
            op_name = queue.popleft()
            op_dict = ChunkEvaluator.function_table[op_name]
            op_type = op_dict['type']
            args = [<void_ptr>0, <void_ptr>0, <void_ptr>0]
            for i in range(len(op_dict['args'])):
                args[i] = build_expr_from_parse_list(queue)
            node = dz_expr_node_create(op_type, args[0], args[1], args[2])
        elif isinstance(queue[0], StringLiteral):
            sl = queue.popleft()
            print("sssssssssssssssssssssssssss: %s", str.encode(sl.value))
            s = <char_ptr>malloc(len(str.encode(sl.value)))
            strcpy(s, str.encode(sl.value))         # TODO change this ?
            print("sssssssssssssssssssssssssss: %s", s)
            node = dz_expr_node_string_create(s)
        elif isinstance(queue[0], Ident):
            id = queue.popleft()
            sb_index = ChunkEvaluator.stringBuffers.index(id.name)
            node = dz_expr_node_string_buffer_index_create(sb_index)
        elif isinstance(queue[0], int):
            node = dz_expr_node_int_create(queue[0])
        elif isinstance(queue[0], float):
            node = dz_expr_node_float_create(queue[0])
        else:
            raise ValueError("Incorrect value in expr: %s" % queue[0])

        return node

cdef string_buffer_ptr toC_string_buffer(StringBuffer sb):
    cdef char* strings = <char*>(sb.strings.data)
    cdef size_t* indices = <size_t*>(sb.indices.data)

    return dz_string_buffer_make(strings, <int>len(sb.strings), indices, <int>len(sb.indices))


cdef test():
    cdef string_buffer_ptr sb_list[2]

    sb1 = StringBuffer.generate(10**2)
    sb2 = StringBuffer.generate(10**2)

    # print("In python:")
    # print(sb1)
    sb_list[0] = toC_string_buffer(sb1)
    sb_list[1] = toC_string_buffer(sb2)
    evaluator = ChunkEvaluator()
    ChunkEvaluator.stringBuffers = ['sb1', 'sb2']
    test = "concat(sb1, 'ok')"
    parse_list = evaluator.parse(test)
    print(parse_list)
    queue = deque(parse_list)
    e = build_expr_from_parse_list(queue)

    t1 = time.time()
    res = dz_eval(e, sb_list)
    print(time.time() - t1)
    str = dz_string_buffer_to_string(res)
    print(str)

test()