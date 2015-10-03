import numpy as np
import os
import bcolz
import re
import math

from bcolz.chunked_eval import _getvars, _eval, _eval_blocks,numexpr_functions
from dazzle.core.utils import DazzleError, method_name


class Column(object):
    """A Column is an abstract class serving as a wrapper over a bcolz carray. This carray may be either contained
       in the Column (ResultColumn), or belong to a ctable (TableColumn).
    """

    DEFAULT_BLOCK_LEN = 10**7

    def __init__(self):
        """Initialize this Column"""

        self._format = None

    @property
    def format(self):
        """Return the format of this Column
        """
        return self._format

    @format.setter
    def format(self, val):
        """Set the format of this Column
        """
        self._format = val

    @property
    def carray(self):
        """Return the carray associated to this Column. Abstract
        """
        raise DazzleError("Invalid call to Column.carray; the property is abstract")

    @property
    def values(self):
        """Return the values of the carray associated to this Column as a numpy.ndarray.
        """
        return self.carray[:]

    @property
    def dtype(self):
        """Return the dtype of this Column.
        """
        return self.carray.dtype

    def type_name(self, dtype=None):
        """Return the dtype name of this Column or of 'dtype' argument.
        """
        dt = dtype if dtype is not None else self.dtype
        return np.dtype(dt).name

    @property
    def blocks(self):
        """Block iterator allowing one to iterate over the carray's values of a Column by chunks
        """
        carr = self.carray
        for i in range(0, len(carr), carr.chunklen):
            yield carr[i:i + carr.chunklen]

    def __len__(self):
        """Return the number of values in the carray of this Column
        """
        return len(self.carray)

    def __getitem__(self, *args):
        """Array-like read-accessor in the carray of this Column
        """
        return self.carray.__getitem__(*args)

    def __setitem__(self, key, value):      # TODO check value
        """Array-like write-accessor in the carray of this Column
        """
        return self.carray.__setitem__(key, value)

    def __str__(self):
        """Return a string representation of the values in the carray of this Column
        """
        out = "["
        for block in self.blocks:
            out += ", ".join(self.str_values(block))
        out + "] - " + str(self.dtype)
        return out


    def str_values(self, head=5, tail=3, val=None, format=None):
        """Return a list containing the string representation of each value in the carray of this Column
        """
        if self.format is not None:
             f = self.format
        elif np.issubdtype(self.dtype, np.int):
            f = '%d'
        elif np.issubdtype(self.dtype, np.float):
            f = format if format is not None else '%.3f'
        else:
            f = '%s'
        nan = self.nan_value()

        arr = val if val is not None else self.carray
        arr_head = arr[0:head]
        h = np.where(arr_head != nan, np.char.mod(f, arr_head), ["nan"] * len(arr_head))

        if head < len(arr):
            tail = min(tail, len(arr) - head)
            arr_tail = self.carray[-tail:] if tail > 0 else []
            t = np.where(arr_tail != nan, np.char.mod(f, arr_tail), ["nan"] * len(arr_tail))
            out = np.concatenate((h, ["..."], t))
        else:
            out = h

        return out

    def append(self, array):        # TODO check array  data
        self.carray.append(array)

    def resize(self, nitems):
        self.carray.resize(nitems)

    def can_cast(self, dtype):   # TODO do that by chunk
        return np.all(self.carray[:].astype(dtype) == self.carray[:])

    @staticmethod
    def arange(start=None, stop=None, step=None, dtype=None):

        # Check start, stop, step values
        if (start, stop) == (None, None):
            raise ValueError("You must pass a `stop` value at least.")
        elif stop is None:
            start, stop = 0, start
        elif start is None:
            start, stop = 0, stop
        if step is None:
            step = 1

        carr = bcolz.arange(start, stop, step, dtype, expectedlen=len(range(start, stop, step)))
        return ResultColumn(carr)

    def to_numpy_array(self, val=None):
        arr = val if val is not None else self.carray[:]
        nan = self.nan_value()
        return np.where(arr != nan, arr, [np.nan] * len(arr))


    def nan_value(self, dtype=None):
        dt = dtype if dtype is not None else self.dtype
        if np.issubdtype(dt, np.int):
            return np.iinfo(dt).min
        elif np.issubdtype(dt, np.float):
            return np.nan
        else:
            return None

    def isnan(self, val=None):
        """
        Return a boolean same-sized ResultColumn indicating if the values are NaN
        """
        arr = val if val is not None else self.carray[:]

        if not isinstance(arr, list):
            return arr != arr or arr == self.nan_value()    # arr != arr <=> arr == np.nan
        elif len(arr) == 0:    # eval does not seem to work correctly on empty array
            return ResultColumn([])
        else:
            return ResultColumn(eval('arr != arr')) # see https://github.com/pydata/numexpr/issues/23

    def isnotnan(self, val=None):
        """
        Return a boolean same-sized ResultColumn indicating if the values are not NaN
        """
        arr = val if val is not None else self.carray[:]
        if len(arr) == 0:    # eval does not seem to work correctly on empty array
            return ResultColumn([])
        else:
            return ResultColumn(eval('arr == arr')) # see https://github.com/pydata/numexpr/issues/23

    def count_nan(self, val=None):
        """
        Return the number of NaN in my values
        """
        arr = val if val is not None else self.carray[:]
        return np.sum(self.isnan(arr))
    def fill_nan(self, val):
        """
        Replace NaN by val in my values
        """
        self.carray[np.isnan(np.array(self.carray))] = val

    def replace_value(self, old, new):
        """
        Replace old by new in my values
        """
        if type(new) != int and type(new) != float and type(new) != bool:
            raise ValueError("new must be int, float or bool: in %s()" % (method_name()))

        ca = self.carray
        idx = bcolz.eval('ca == ' + str(old))
        self.carray[idx] = new

    def replace_list(self, old, new):
        """
        Replace each ith value in old by each ith value in new in my values
        """
        # TODO rewrite this method: it takes very long when there are many replacements to perform
        # for instance, replacing CategoryID in userinfo in avito dataset is very sloooow

        if len([x for x in new if type(x) != int and type(x) != float and type(x) != bool]) > 0:
            raise ValueError("Values in new must be int, float or bool: in %s()" % (method_name()))

        ca = self.carray
        cond = "(" +  ") | (". join(["ca == " + str(o) for o in old]) + ")"
        mask = bcolz.eval(cond)
        rep = [i for i in mask.wheretrue()]
        pairs = dict(zip(old, new))
        if len(rep) > 0:
            self.carray[rep] = [pairs[o] for o in self.carray[rep]]

    def sum(self, dtype=np.float64, skipna=True):
        out = 0.0
        if np.issubdtype(self.carray.dtype, np.int):
            nan = self.nan_value()

            for block in self.blocks:
                mask = (block != nan)
                if not skipna and np.sum(mask) > 0:
                    return nan
                if skipna:
                    out += np.sum(block[mask], dtype=dtype)
                else:
                    out += np.sum(block, dtype=dtype)
        elif np.issubdtype(self.carray.dtype, np.float):
            for block in self.blocks:
                if skipna:
                    out += np.nansum(block, dtype=dtype)
                else:
                    out += np.sum(block, dtype=dtype)
                    if np.isnan(out):
                        return np.nan
        else:
            raise DazzleError("Unsupported type in sum(): %s" % str(self.dtype))

        return out

    def mean(self, dtype=np.float64, skipna=True):
        # check numeric
        sum = 0.0
        count = 0
        for block in self.blocks:
            if skipna:
                sum += np.nansum(block, dtype=dtype)
                count += (len(block) - np.sum(np.isnan(block)))
            else:
                sum += np.sum(block, dtype=dtype)
                if np.isnan(sum):
                    return np.nan

        if count > 0:
            return sum / count
        else:
            return np.nan

    def min(self, skipna=True):
        if np.issubdtype(self.carray.dtype, np.int):
            nan = self.nan_value()
            out = nan
            for block in self.blocks:
                mask = (block != nan)
                if not skipna and np.sum(mask) > 0:
                    return nan

                if skipna:
                    block_min = np.amin(block[mask])
                else:
                    block_min = np.amin(block)

                if out == nan or block_min < out:
                    out = block_min
        elif np.issubdtype(self.carray.dtype, np.float):
            out = np.nan
            for block in self.blocks:
                if skipna:
                    block_min = np.nanmin(block)
                else:
                    block_min = np.amin(block)
                if not skipna and np.isnan(block_min):
                    return np.nan

                if np.isnan(out) or block_min < out:
                    out = block_min
        else:
            raise DazzleError("Unsupported type in min(): %s" % str(self.dtype))

        return out

    def max(self, skipna=True):
        if np.issubdtype(self.carray.dtype, np.int):
            nan = self.nan_value()
            out = nan
            for block in self.blocks:
                mask = (block != nan)
                if not skipna and np.sum(mask) > 0:
                    return nan

                if skipna:
                    block_max = np.amax(block[mask])
                else:
                    block_max = np.amax(block)

                if out == nan or block_max > out:
                    out = block_max
        elif np.issubdtype(self.carray.dtype, np.float):
            out = np.nan
            for block in self.blocks:
                if skipna:
                    block_max = np.nanmax(block)
                else:
                    block_max = np.amax(block)
                if not skipna and np.isnan(block_max):
                    return np.nan

                if np.isnan(out) or block_max > out:
                    out = block_max
        else:
            raise DazzleError("Unsupported type in max(): %s" % str(self.dtype))

        return out

    def max_raw(self, skipna=True):
        # check numeric
        nan = self.nan_value()
        out = nan
        for block in self.blocks:
            block_max = np.nanmax(block)
            if not np.isnan(block_max) and (out == nan or block_max > out):
                out = block_max
        return out

    def max2(self, skipna=True):
        # check numeric
        nan = self.nan_value()
        out = nan
        for block in self.blocks:
            masked_block = np.ma.masked_values(block, nan)

            block_max = masked_block.max()
            if out == nan or block_max > out:
                out = block_max
        return out


    #
    #
    # def all(self):
    #     if self.bz_array is None:
    #         raise DazzleError('Invalid call to %s(): carray is undefined' % method_name())
    #
    #     if len(self.bz_array) == 0:
    #         return True # Numpy said so
    #
    #     out = True
    #     for i in range(0, len(self.bz_array), LiteralColumn.DEFAULT_BLOCK_LEN):
    #         block = self.bz_array[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
    #         out = out and np.all(block)
    #     return out
    #
    # def any(self):
    #     if self.bz_array is None:
    #         raise DazzleError('Invalid call to %s(): carray is undefined' % method_name())
    #
    #     if len(self.bz_array) == 0:
    #         return False # Numpy said so
    #
    #     out = False
    #     for i in range(0, len(self.bz_array), LiteralColumn.DEFAULT_BLOCK_LEN):
    #         block = self.bz_array[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
    #         out = out or np.any(block)
    #     return out

    # @staticmethod
    # def randint(low=0, high=None, size=None):
    #     """
    #     Similar to np.random.randint().
    #     Returns a bz_array
    #     """
    #     out = bcolz.carray([], dtype="i4", expectedlen=size)
    #     remaining = size
    #     while remaining > 0:
    #         block_size = min(remaining, out.chunklen)
    #         random_block = np.random.randint(low, high, block_size)
    #         out.append(random_block)
    #         remaining -= block_size
    #     return bz_array(out)
    #
    # @staticmethod
    # def randfloat(low=0.0, high=None, size=None):
    #     """
    #     Similar to np.random.uniform().
    #     Returns a bz_array
    #     """
    #     if low > high:
    #         raise ValueError("low should be less or equal to high in %s()" % (method_name()))
    #
    #     # TODO low compression ratio: should be kept on disk
    #     out = bcolz.carray([], dtype="f8", expectedlen=size)
    #     remaining = size
    #     while remaining > 0:
    #         block_size = min(remaining, out.chunklen)
    #         random_block = (high - low) * np.random.random(block_size) + low
    #         out.append(random_block)
    #         remaining -= block_size
    #     return bz_array(out)

class TableColumn(Column):

    def __init__(self, name, table, data=None):  # TODO add default value (when resize), and expected_length params
        super().__init__()

        if not re.match("[A-Za-z][_a-zA-Z0-9]*$", name):
            raise DazzleError("Invalid column identifier: '%s' in %s()" % (name, method_name()))

        if type(table).__name__ != "Table":
            raise DazzleError("Table parameter expected %s" % (table))

        if table.get_column(name) is not None:
            raise DazzleError("there is already a column with name %s' in table %s" % (name, table.name))

        self._name = name
        self._ref_column = None # defined here and not in RefColumn because this property is set *before* the Column is transformed into a RefColumn

        self._table = table
        table.columns.append(self)

        if data is not None:
            if len(self._table.columns) > 0 and len(self._table.columns[0].carray) != len(data):
                raise DazzleError("Column %s should have same number of values as existing columns" % (name))

            self.table.ctable.addcol(data, name, expectedlen=self.table.expected_length)

    @property
    def name(self):
        return self._name

    @property
    def ref_column(self):
        return self._ref_column

    @ref_column.setter
    def ref_column(self, col):
        self._ref_column = col

    @property
    def table(self):
        if self._table is None:
            raise DazzleError('Invalid call to %s: table is undefined' % method_name())

        return self._table

    @property
    def carray(self):
        return self._table.ctable.cols[self._name]

    @property
    def data_dir(self):
        return os.path.join(self.table.data_dir, self._name)

    @property
    def position(self):
        return self.table.columns.index(self)

    def rename(self, new_name):
        if not re.match("[A-Za-z][_a-zA-Z0-9]*$", new_name):
            raise DazzleError("Invalid column identifier: '%s' in %s" % (new_name, method_name()))

        ct = self.table.ctable

        if new_name in ct.names:
            raise DazzleError("LiteralColumn identifier already in use: '%s' in %s" % (new_name, method_name()))

        col_pos = self.position
        carray = self.carray
        ct.delcol(self._name) # TODO should use keep=True, but didn't succeed
        ct.addcol(carray, name=new_name, pos=col_pos)

        self._name = new_name

    def set_type(self, dtype):
        if self.dtype != dtype:
            self.fill_nan(self.nan_value(dtype=dtype))
            carray = bcolz.carray(self.carray, dtype= np.dtype(dtype).name)  # TODO do it chunk by chunk + check data
            ct = self._table._ctable
            col_pos = self.position
            ct.delcol(self._name)
            ct.addcol(carray, name=self._name, pos=col_pos)

class RefColumn(TableColumn):

    def __init__(self, name, table, data=None):
        super().__init__(name, table, data)

    def nan_value(self, dtype=None):
        return 0

    def sum(self, dtype=np.float64, skipna=True):
        raise DazzleError("RefColumn.%() should not be called" % method_name())

    def mean(self, dtype=np.float64, skipna=True):
        raise DazzleError("RefColumn.%() should not be called" % method_name())

    def min(self, skipna=True):
        nan = self.nan_value()
        out = nan
        for block in self.blocks:
            block_min = np.amin(block)
            if skipna and block_min == 0:
                block_min = np.amin(block[block != 0])
            if block_min < out:
                out = block_min
        return out

    def max(self, skipna=True):
        nan = self.nan_value()
        out = nan
        for block in self.blocks:
            block_max = np.amax(block)
            if not skipna and np.amin(block) == 0:
                return 0
            if block_max > out:
                out = block_max
        return out


class LiteralColumn(TableColumn):

    def __init__(self, name, table, data=None):
        super().__init__(name, table, data)

    def same(self, other):
        return self._name == other._name \
               and ((self._table is None and other._table is None) or self._table.name == other._table.name) \
               and self.dtype == other.dtype \
               and ((self.carray is None and other.carray is None)
                    or (self.carray is not None and other.carray is not None
                        and np.all(np.isclose(self.carray[:], other.carray[:], equal_nan=True))))




    #
    # def unique(self, max_value=None):
    #     """Compute the set of unique values in self.bz_array. Array is_inside works as a bitset
    #     is_inside[i] == 1 iff i belongs to self.bz_array
    #     """
    #     # TODO: this method may make the RAM blow out
    #     # a binning step should be performed as for sample_sort
    #     # and then the code here could be used
    #     # probably ~30 sec for 1 billion items on my machine
    #
    #     upper_bound = max_value if max_value is not None else len(self.bz_array)
    #     is_inside = np.zeros(upper_bound, dtype=np.uint8)
    #     is_inside[self.bz_array[:]] = 1
    #     return np.nonzero(is_inside)
    #
    # def sample_sort(self):
    #
    #     """Sort self carray, using a sample sort:
    #     https://en.wikipedia.org/wiki/Samplesort
    #     """
    #
    #     if len(self.bz_array.carray) <= LiteralColumn.DEFAULT_BLOCK_LEN:
    #         return np.sort(self.bz_array.carray[:])
    #     else:
    #         # dir for toring temp data
    #         dir = 'C:/github/kaggle-fiddling/avito-context-ad-clicks/data/'
    #
    #         bucket_count = (len(self.bz_array.carray) // LiteralColumn.DEFAULT_BLOCK_LEN) + 1
    #
    #         print ("-- sampling")
    #
    #         # build a sample of data in self.bz_array for determining the bounds used for the binning of buckets (splitters)
    #         # take 1000 elements in each block
    #
    #         sample = []
    #         block_count = (len(self.bz_array.carray) // LiteralColumn.DEFAULT_BLOCK_LEN) + 1
    #         for i in range(0, block_count):
    #             block = self.bz_array[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
    #             if i < len(self.bz_array.carray) - 1:        # last block may not be completely full
    #                 block_sample = np.random.choice(block, size=min(len(block), 1000))
    #                 sample.extend(block_sample)
    #
    #         # split sample in bucket_count of same size
    #         # and the bounds used for the binning of buckets
    #
    #         sorted_sample = np.sort(sample)
    #         splits = np.array_split(sorted_sample, bucket_count)
    #         bins = [splits[s][0] for s in range(len(splits))]
    #
    #         # fill the buckets
    #         # It would probably be more efficient to bufferize the writing in the bcolz bucket
    #
    #         bcolz_bucket = None
    #         for bu in range(0, bucket_count):
    #             bucket = [[] for i  in range (bucket_count)]
    #
    #             # create bucket on disk
    #             bcolz_bucket = [bcolz.bz_array.carray([], rootdir=dir + 'test' + str(i), mode='w', expectedlen=len(self.bz_array.carray) // bucket_count)
    #                                 for i  in range (bucket_count)]
    #             # for testing in memory
    #             # bcolz_bucket = [bcolz.bz_array([], expectedlen=len(self.bz_array) // bucket_count) for i  in range (bucket_count)]
    #
    #         print ("-- filling buckets")
    #         for i in range(0, block_count):
    #             print (i)
    #             block = self.bz_array[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
    #             inds = np.digitize(block, bins)
    #             for bu in range(0, bucket_count):
    #                 bin_index = bu + 1 # inds starts at index 1
    #                 bcolz_bucket[bu].append(block[inds == bin_index])
    #
    #         # sort each bucket and append the result to the result bcolz array
    #
    #         print ("-- sorting")
    #         bcolz_out = bcolz.bz_array.carray([], rootdir=dir + 'test', mode='w', expectedlen=len(self.bz_array.carray) )
    #         for bu in range(0, bucket_count):
    #            print (bu)
    #            bcolz_out.append(np.sort(bcolz_bucket[bu][:]))
    #
    #         return bcolz_out
    #

    #
    # def select(self, expression, var_dict):
    #     """See bcolz.eval()
    #     """
    #     vm = 'numexpr'
    #     out_flavor = "carray"
    #
    #     # Gather info about sizes and lengths
    #     typesize, vlen = 0, 1
    #     for name, var in var_dict.items():
    #         if isinstance(var, LiteralColumn):  # numpy array
    #             var = var.bz_array
    #
    #         if hasattr(var, "__len__") and not hasattr(var, "dtype"):
    #             raise ValueError("only numpy/carray sequences supported")
    #         if hasattr(var, "dtype") and not hasattr(var, "__len__"):
    #             continue
    #         if hasattr(var, "dtype"):  # numpy/carray arrays
    #             if isinstance(var, np.ndarray):  # numpy array
    #                 typesize += var.dtype.itemsize * np.prod(var.shape[1:])
    #             elif isinstance(var, bcolz.bz_array):  # carray array
    #                 typesize += var.dtype.itemsize
    #             else:
    #                 raise ValueError("only numpy/carray objects supported")
    #         if hasattr(var, "__len__"):
    #             if vlen > 1 and vlen != len(var):
    #                 raise ValueError("arrays must have the same length")
    #             vlen = len(var)
    #
    #         if typesize == 0:
    #             return [r for r in bcolz.numexpr.evaluate(expression, local_dict=var_dict).wheretrue()]
    #
    #         return [r for r in _eval_blocks(expression, var_dict, vlen, typesize, vm, out_flavor).wheretrue()]




    def test(self):
        import time
        bucket_count = (len(self.carray) // LiteralColumn.DEFAULT_BLOCK_LEN) + 1

        print ("-- sampling")

        # build a sample of data in self.carray for determining the bounds used for the binning of buckets (splitters)
        # take 1000 elements in each block

        sample = []
        block_count = (len(self.carray) // LiteralColumn.DEFAULT_BLOCK_LEN) + 1
        for i in range(0, block_count):
            block = self.carray[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
            if i < len(self.carray.carray) - 1:        # last block may not be completely full
                block_sample = np.random.choice(block, size=min(len(block), 1000))
                sample.extend(block_sample)

        # split sample in bucket_count of same size
        # and the bounds used for the binning of buckets

        sorted_sample = np.sort(sample)
        splits = np.array_split(sorted_sample, bucket_count)
        bins = [splits[s][0] for s in range(len(splits))]

        # fill the buckets
        # It would probably be more efficient to bufferize the writing in the bcolz bucket

        start_time = time.time()

        bcolz_bucket = None
        for bu in range(0, bucket_count):
            bucket = [[] for i  in range (bucket_count)]

            # create bucket on disk
            #bcolz_bucket = [bcolz.carray([], rootdir=self.data_dir + 'test' + str(i), mode='w', expectedlen=len(self.carray) // bucket_count)
            #                    for i  in range (bucket_count)]
            # for testing in memory
            bcolz_bucket = [bcolz.carray([], expectedlen=len(self.carray) // bucket_count) for i  in range (bucket_count)]

        print ("-- filling buckets")
        for i in range(0, block_count):
            print (i)
            block = self.carray[i : i + LiteralColumn.DEFAULT_BLOCK_LEN]
            inds = np.digitize(block, bins)
            for bu in range(0, bucket_count):
                bin_index = bu + 1 # inds starts at index 1
                bcolz_bucket[bu].append(block[inds == bin_index])

        print( time.time() - start_time)


class ResultColumn(Column):

    def __init__(self, val):
        super().__init__()

        if isinstance(val, bcolz.carray):
            self._carray = val
        elif isinstance(val, list) or isinstance(val, np.ndarray):
            self._carray = bcolz.carray(val, expectedlen=Column.DEFAULT_BLOCK_LEN)
        else:
            raise DazzleError("Invalid argument in ResultColumn.%s()" % method_name())

    @property
    def carray(self):
        return self._carray

if __name__ == '__main__':
    import time

    def test1():
        import numpy as np
        col = ResultColumn(np.array([1,4,3.40282e+38], dtype=np.float32))
        print(col.to_numpy_array())


    def test2():
        from dazzle.core.dataset import DataSet
        from dazzle.core.table import Table

        test_dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(test_dir, force_create=True)
        t = Table("t", ds, [("a", np.int)])
        ca = t.get_column("a")
        #t.append({'a': np.random.randint(10000000, size=5*(10**8)).astype(np.int32)})
        print(ca.__dict__)


    def test3():
        t1 = time.time()

        x = Column.arange(10**9)
        x[1:10**9:100] = x.nan_value()

        t2 = time.time(); print("build int: " + str(t2 - t1))

        print(x.max())

        t3 = time.time(); print("max with self made mask: " + str(t3 - t2))

        print(x.max2())

        t4 = time.time(); print("max with numpy.ma: " + str(t4 - t3))

        y = Column.arange(10**9, dtype=np.float64)
        y[1:10**9:100] = np.nan

        t5 = time.time(); print("build float: " + str(t5 - t4))

        print(y.max_raw())

        t6 = time.time(); print("float arrays with np.nan: " + str(t6 - t5))


    # test3()

    def test4():
        from dazzle.core.dataset import DataSet
        from dazzle.core.table import Table

        test_dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(test_dir, force_create=True)
        t = Table("t", ds, [("a", np.array([np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan]))])
        ca = t.get_column("a")
        print(ca.str_values(format="%.4f"))

    test4()

#     LiteralColumn.DEFAULT_BLOCK_LEN = (10**7)
#     c = ArrayColumn("a", data=[11,4,7,1,4,7,4,6,9,23,11,4,78,524,65,2,7,5,10,23])
#     #c = ArrayColumn("a", data=np.random.randint(100, size=  (10**8)))
#     #c.sample_sort()
#
#     # d = np.array(c)
#     # print('.')
#     #x = bcolz.eval('c < 10')
#     x = c.select('c < 10', {'c': c}) # TODO ne marche pas A RETRAVAILLER
#     print(x)
#     #print(len(x))