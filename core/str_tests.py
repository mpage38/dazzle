import os
import string
import random
import bcolz
import numpy as np
import ctypes
import time

uint16_max = np.iinfo(np.uint8).max



def generate():
    import time

    t1 = time.time()
    string_count = 10**6
    element_size= 10**3
    string_size = 20

    possibilities = [random_string(19) for i in range(10000)]

    bz = bcolz.carray(np.array([],  dtype="S100000"), rootdir="/temp/dazzle/string-array1-10e9", mode='w', expectedlen=10**5)
    s = "\n".join([possibilities[i % 10000] for i in range(10**4)])
    for i in range(10**5):
        bz.append(np.array([s],  dtype="S100000"))
        if i % 10**3 == 0:
            print(i)

    bz.flush()
    print("generate")
#generate()

def generate_column():
    t = time.time()

    bz = bcolz.carray(np.array([],  dtype="uint64"), rootdir="/temp/string-col1-10e9/data", mode='w', expectedlen=10**9)
    for chunk in range(2*10**2):
        if chunk % 1000 == 0:
            print(chunk)
            for i in range(5*10**4):
                addr = chunk
                addr = (addr << 48) | ((i*20) << 24) | ((i*20)+19)
                bz.append(addr)
                print(addr)

    bz.flush()

    print(time.time() - t)

def address_to_element(addr):
    chunk_index = addr >> 48
    start_index = (addr >> 24) & 0xFFFFFF
    end_index = addr & 0xFFFFFF
    return (chunk_index, start_index, end_index)

def test_read():
    bz_string = bcolz.open("/temp/string-array1-10e9/data", mode='r')
    bz_address = bcolz.open("/temp/string-col1-10e9/data", mode='r')

    print(bz_string[0])
    # for i in range(1*10**2):
    #     chunk_index, start_index, end_index = address_to_element(bz_address[i])
    #     print("%d %d %d" % (chunk_index, start_index, end_index))
        # chunk = bz_string[chunk_index].tobytes()
        # print(chunk[start_index:end_index])

# generate_column()
#test_read()



def test3():
    count = 10**6
    s = ' ' * count
    tab = np.fromstring(s, dtype="S1")
    for i in range(count):
        tab[i] = 1


# s = [""]*10**6
# t = time.time()
# for it in range(10**2):
#     for i in range(10**6):
#         s[i]='0123456789'
#     x = "".join(s)
#
# print(time.time() - t)

class Col:
    def __init__(self):
        self.buffer_max_size = 10**7
        self.buffers = []
        self.buffer = [""] * self.buffer_max_size
        self.buffer_length = 100
        self.buffer_index = 0
        self.len = 0

    def make_col(self, count):

        t1 = time.time()
        for i in range(count):
            if i % 1000 == 0:
                print(i)
            r = random.randint(0,10)
            s = random_string(r)
            if self.len + len(s) > self.buffer_length:
                self.buffers.append(",".join(self.buffer[0:self.buffer_index]))
                self.buffer_index = 0
                self.buffer = [""] * self.buffer_max_size
                self.len = 0

            self.buffer[self.buffer_index] = s
            self.len += len(s)
            self.buffer_index += 1

        t2 = time.time()
        print(t2-t1)
# col = Col()
# col.make_col(10**3)
# print(len(col.buffers))


def upper():
    bz_string = bcolz.open("/temp/string-array1-10e9/data", mode='r')
    print(len(bz_string))
    i = 0
    for c in range(0, len(bz_string), bz_string.chunklen):
        string_chunks = bz_string[c:c + bz_string.chunklen]
        for chunk in string_chunks:
            lr = [e.upper() for e in chunk.split(",")]
            #print(lr[0:5])
            i += 1
            if i % 100 == 0:
                print(i)

# upper()









def concat():
    pos = [20 * i for i in range(10000)]
    bz1 = bcolz.open("/temp/string-array1-10e9/data")
    bz2 = bcolz.open("/temp/string-array2-10e9/data")
    for i in range(2*10**4):
        if i % 10**3 == 0:
            print(i)
        s = ctypes.create_string_buffer(1000000)
        buf1 = bz1[i].tobytes().decode().split(",")
        buf2 = bz2[i].tobytes().decode().split(",")
        # for j in range(0, 5*10**4, 20):
        #     x = buf1[j % 20]
        #     s[j:j+19] = buf1[j % 20].encode('ascii')
        x = ",".join(buf1)

#concat()
    # #elt = np.empty([string_count], dtype="S" + str(element_size))
    # block = np.empty([string_count / element_size * string_size], dtype="S1000000" + str(element_size))
    # index = np.zeros([string_count], dtype=int)
    # length = np.zeros([string_count], dtype=int)
    # chunk = np.zeros([string_count], dtype=int)
    #
    #
    # s = ""
    # chk = 0
    # for i in range(string_count):
    #     s1 = possibilities[np.random.randint(10000)]
    #     #elt[i] = s1
    #     length[i] = len(s1)
    #     if len(s) + len(s1) >= element_size:  # chunk is full
    #         block[chk] = s
    #         s = s1
    #         chk += 1
    #         index[i] = 0
    #     else:
    #         index[i] = len(s)
    #         s += s1
    #
    #     chunk[i] = chk
    #     if i % 10**5 == 0:
    #         print(i)
    #
    # bz = bcolz.carray(block, rootdir="/temp/string-array1-10e7/data", mode='w', chunklen=1)
    # #bz.append(block)
    # bcolz.ctable(columns=[chunk, index, length], names=["chunk", "index", "length"], rootdir="/temp/string-array1-10e7/index", mode='w')

    #
    # s_arr = bcolz.open("/temp/string-array1-10e7/data")
    # t = bcolz.open("/temp/string-array1-10e7/index")
    # chunk_arr = t["chunk"]
    # index_arr = t["index"]
    # length_arr = t["length"]
    # for i in range(10**7):
    #     print(i)
    #     chunk = chunk_arr[i]
    #     block = s_arr[chunk]
    #     if type(block) == np.ndarray: # TODO remove when bcolz bug corrected
    #         print(i)

    # print(time.time() - t1)

def test2():
    bz = bcolz.carray(np.array([], dtype="S500"))
    for i in range(1000):
        bz.append(np.array(["xxx"]))
        if len(bz) >= 2 and type(bz[1]) == np.ndarray:
            print("oops")


def test3():
    s_arr = bcolz.open("/temp/string-array1-10e7/data")
    t = bcolz.open("/temp/string-array1-10e7/index")
    chunk_arr = t["chunk"]
    index_arr = t["index"]
    length_arr = t["length"]
    for i in range(10**7):
        chunk = chunk_arr[i]
        print(chunk)
        block = s_arr[chunk]
        if type(block) == np.ndarray: # TODO remove when bcolz bug corrected
            block = block.tobytes()
        start=index_arr[i]
        end = index_arr[i]+length_arr[i]

        #print("%d %d %d" % (i, start, end))
        #s= block[start:end]
        # print(i)
        # print(len(block))

#test()

def make_files():
    a_list = [random_string(200) for _ in range(10**6)]
    with open('/temp/b.txt', 'w') as f:
        f.write(",".join(a_list))


# make_files()

def str_address(buffer_index, start_index, length):
    addr = buffer_index
    addr = (addr << 48) | (start_index << 24) | length
    return addr

def decode_str(addr):
    buffer_index = addr >> 48
    start_index = (addr >> 24) & 0xFFFFFF
    length = addr & 0xFFFFFF
    return (buffer_index, start_index, length)

def random_string(n):
    """Generate a random string of length n, where all the characters are the same, except the first and last
        that are capitalized"""

    c = random.choice(string.ascii_letters).lower()
    return str(c.upper() + (str(c) * (n-2)) + c.upper())

def generate_random_string_carray(string_count, buffer_size=10**5, string_size=30, path=""):
    """Generate a bcolz carray with string_count strings, stored in buffers of length buffer_size characters.
        Each string has a random length between 2 and string_size.
        The different strings inside a buffer are separated with \0.
        if path == "", resuting carray is stored in RAM, otherwise on disk using path.
    """

    # generate 100000 random strings
    elements = [random_string(random.randint(2, string_size)) for i in range(100000)]

    expected_length = string_count * ((string_size + 2) // 2)
    if path == "":
        ca_data = bcolz.carray(np.array([], dtype=np.uint8), expectedlen=expected_length)
        ca_address = bcolz.carray(np.array([],  dtype=np.int64), expectedlen=string_count)
    else:
        ca_data = bcolz.carray(np.array([], dtype=np.uint8), rootdir=os.path.join(path, "data"), mode='w', expectedlen=expected_length)
        ca_address = bcolz.carray(np.array([],  dtype=np.int64), rootdir=os.path.join(path, "address"), mode='w', expectedlen=string_count)

    current_length = 0
    current_list = []
    for i in range(string_count):
        s1 = elements[i % 100000]
        current_list.append(s1)
        if i % 10**6 == 0:
            ca_data.append(np.frombuffer(("\0".join(current_list) + '\0').encode(), dtype=np.uint8))
            current_list = []
        ca_address.append(current_length)
        current_length += len(s1) + 1
        if i % 10**6 == 0:
            print("%d M" % (i / 10**6))

    if path != "":
        ca_data.flush()
        ca_address.flush()

    return (ca_data, ca_address)

# class string_col:
#     def __init__(self, ca):
#         self.carray = ca
#         self.current_block = next(bcolz.iterblocks(ca))
#         self.block_len = len(self.current_block)
#         self.block_pointer = 0
#
#     def buffers(self):
#         for ca_segment in bcolz.iterblocks(self.carray):
#             len_ca_segment = len(ca_segment)
#             for i in range(len_ca_segment):
#                 yield ca_segment[i]
#
# def handle(buf1, buf2):
#     n1 = buf1.split('\1')
#     n2 = buf2.split('\1')
#     if n1 < n2:
#         return buf2[n2+1:]
#     elif n2 > n1:
#         return buf1[n1+1:]
#     else:
#         return None
#
# def test_iterate(ca1, ca2, result):
#     col_ca1 = string_col(ca1)
#     col_ca2 = string_col(ca2)
#     buf1 = next(col_ca1.buffers())
#     buf2 = next(col_ca2.buffers())
#
#     while buf1 is not None:
#         print(min(len(buf1), len(buf2)))
#         remain = handle(buf1, buf2)
#         if len_buf1 < len_buf2:
#             buf1 = next(col_ca1.buffers())
#             if (len(remain) < len(buf1)):
#                 buf2 = remain + next(col_ca2.buffers())
#         else:
#             buf2 = next(col_ca2.buffers())
#             if (len(remain) < len(buf2)):
#                 buf1 = remain + next(col_ca1.buffers())

# generate_random_string_carray(100, buffer_size=500, string_size=30)
#ca1_data, ca1_address = generate_random_string_carray(10**9, buffer_size=50, string_size=5, path='/temp/dazzle/test1')
#ca2_data, ca2_address =  generate_random_string_carray(10**2, buffer_size=40, string_size=7, path='/temp/dazzle/test2')


# r = bcolz.carray(np.array([]))
# test_iterate(ca1, ca2, r)

# import bcolz
# ca1 = bcolz.open('/temp/avito1/TrainSearchStream/Position')
# ca2 = bcolz.open('/temp/avito1/TrainSearchStream/IsClick')
# x = bcolz.eval('ca1+ca2')
# print(x)

class StringColumn:
    def __init__(self, data_carray, address_carray):
        self.data_carray = data_carray
        self.address_carray = address_carray

def concat_col(sc1, sc2, result):
    a1 = sc1.address_carray
    a2 = sc2.address_carray
    d1 = sc1.data_carray
    d2 = sc2.data_carray
    result_data = result.data_carray
    for ca_addr_segment in iterblocks([a1, a2]):
        len_ca_ca_addr_segment = len(ca_addr_segment)
        for i in range(len_ca_ca_addr_segment):
            start1 = ca_addr_segment[0][i]
            start2 = ca_addr_segment[1][i]
            while d1[start1] != 0:
                result_data.append([d1[start1]])
                start1 += 1
            while d2[start2] != 0:
                result_data.append([d2[start2]])
                start2 += 1


def iterblocks(col_list, start=0, stop=None):
    stop = len(col_list[0])
    ca_count = len(col_list)
    block_len = min(col_list[c].chunklen for c in range(ca_count))

    buf = [None] * ca_count
    for i in range(start, stop, block_len):
        for c in range(ca_count):
            buf[c] = np.empty(block_len, dtype=col_list[c].dtype)
            col_list[c]._getrange(i, block_len, buf[c])
            if i + block_len > stop:
                buf[c] = buf[c][:stop - i]
        yield buf

# def iterblocks(col_list, start=0, stop=None):
#     stop = len(col_list[0])
#     col_count = len(col_list)
#     block_len = min(col_list[c].chunklen if isinstance(col_list[c], bcolz.carray) else col_list[c].data_carray.chunklen for c in range(col_count))
#
#     buf = [None] * col_count
#     for i in range(start, stop, block_len):
#         for c in range(col_count):
#             if isinstance(col_list[c], bcolz.carray):
#                 buf[c] = np.empty(block_len, dtype=col_list[c].dtype)
#                 col_list[c]._getrange(i, block_len, buf[c])
#                 if i + block_len > stop:
#                     buf[c] = buf[c][:stop - i]
#             else:
#                 buf[c] = {}
#                 buf[c]['data'] = np.empty(block_len, dtype=col_list[c].dtype)
#                 buf[c]['address'] = np.empty(block_len, dtype=np.int64)
#                 col_list[c].data_carray._getrange(i, block_len, buf[c]['data'])
#                 for j in range(block_len):
#
#         yield buf


sc1 = StringColumn(bcolz.carray([100, 110, 120, 110, 140, 160, 0, 201, 221, 241, 161, 0, 132, 112, 82, 0, 63, 143, 203, 193, 0], dtype="u1", chunklen=3),
                bcolz.carray([0, 7, 12, 16], dtype="i4", chunklen=3))

sc2 = StringColumn(bcolz.carray([140, 130, 160, 100, 0, 141, 121, 181, 0, 32, 102, 92, 52, 0, 163, 243, 103, 0], dtype="u1", chunklen=2),
                bcolz.carray([0, 5, 9, 14], dtype="i4", chunklen=2))

r = StringColumn(bcolz.carray([], dtype="i4"), bcolz.carray([], dtype="i4"))

def get_string(buf, i):
    for j in range(i, 100000):
        if buf[j] == 0:
            return buf[i:j]

def concat_col3(sc1, sc2, result):
    a1 = sc1.address_carray
    a2 = sc2.address_carray
    d1 = sc1.data_carray
    d2 = sc2.data_carray

    d1_buf = None
    d2_buf = None
    result_data = result.data_carray
    a1_previous_last_address = -1
    a2_previous_last_address = -1
    for ca_addr_segment in iterblocks([a1, a2]):
        len_ca_addr_segment = len(ca_addr_segment)

        d1_range_start_index = a1_previous_last_address if a1_previous_last_address >= 0 else 0
        d2_range_start_index = a2_previous_last_address if a2_previous_last_address >= 0 else 0
        d1_range_end_index = ca_addr_segment[0][len_ca_addr_segment - 1]
        d2_range_end_index = ca_addr_segment[1][len_ca_addr_segment - 1]

        d1_buf = np.empty(d1_range_end_index - d1_range_start_index, dtype=np.uint8)
        d1._getrange(d1_range_start_index, d1_range_end_index, d1_buf)
        a1_previous_last_address = d1_range_end_index

        d2_buf = np.empty(d2_range_end_index - d2_range_start_index, dtype=np.uint8)
        d2._getrange(d2_range_start_index, d2_range_end_index, d2_buf)
        a2_previous_last_address = d2_range_end_index

        i1 = 0
        i2 = 0
        while i1 < d1_range_end_index - d1_range_start_index:
            s1 = get_string(d1_buf, i1)
            s2 = get_string(d2_buf, i2)
            i1 += len(s1) + 1
            i2 += len(s2) + 1


concat_col3(sc1, sc2, r)
