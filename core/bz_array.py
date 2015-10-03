"""
This class encapsulates bcolz.carray

Wrapping bcolz.carray was preferred to subclassing it because it is very hard to subclass
bcolz.carray. See https://github.com/Blosc/bcolz/issues/238
"""
from __future__ import absolute_import, print_function, division

from dazzle.core.utils import DazzleError, method_name
import bcolz
import numpy as np


class bz_array(object):

    MAX_LEN = 10**7

    @property
    def itemsize(self):
        return self.carray.dtype.itemsize

    @property
    def __len__(self):
        return len(self.carray)

    def __getitem__(self, *args):
        return self.carray.__getitem__(*args)

    def __setitem__(self, key, value):
        return self.carray.__setitem__(key, value)

    def __repr__(self):
        s = repr(self.carray)
        s = type(self).__name__ + s[6:]
        return s

    @property
    def cbytes(self):
        return self.carray.cbytes

    @property
    def nbytes(self):
        return self.carray.nbytes

    @property
    def chunklen(self):
        return self.carray.chunklen

    @property
    def size(self):
        return self.carray.size

    def __init__(self, val):
        """Initialize from val, a lsit or a a bcolz.carray
        """
        if isinstance(val, list):
            self.carray = bcolz.carray(val)
        elif isinstance(val, bcolz.carray):
            self.carray = val
        else:
            raise DazzleError("Argument must be a list or bcolz.carray instance in %s()" % (method_name()))


    def resize(self, nitems):
        self.carray.resize(nitems)

    def append(self, array):
        self.carray.append(array) # TODO check dtype compatibility

    @staticmethod
    def randint(low=0, high=None, size=None):
        """
        Similar to np.random.randint().
        Returns a bz_array
        """
        out = bcolz.carray([], dtype="i4", expectedlen=size)
        remaining = size
        while remaining > 0:
            block_size = min(remaining, out.chunklen)
            random_block = np.random.randint(low, high, block_size)
            out.append(random_block)
            remaining -= block_size
        return bz_array(out)

    @staticmethod
    def randfloat(low=0.0, high=None, size=None):
        """
        Similar to np.random.uniform().
        Returns a bz_array
        """
        if low > high:
            raise ValueError("low should be less or equal to high in %s()" % (method_name()))

        # TODO low compression ratio: should be kept on disk
        out = bcolz.carray([], dtype="f8", expectedlen=size)
        remaining = size
        while remaining > 0:
            block_size = min(remaining, out.chunklen)
            random_block = (high - low) * np.random.random(block_size) + low
            out.append(random_block)
            remaining -= block_size
        return bz_array(out)

    @property
    def blocks(self):
        """
        Generate a carray block of len chunklen
        """
        for i in range(0, len(self.carray), self.carray.chunklen):
            yield self.carray[i : i + self.carray.chunklen]

    def sample(self, size):
        """
        Generate a sample of given size from my values
        """
        if size > len(self.carray):
            raise DazzleError("Invalid sample size is greater than array size: in %s()" % (method_name()))

        if size <= self.carray.chunklen:
            sample = np.random.choice(self.carray[:], size=size)
            return bz_array(bcolz.carray(sample))
        else:   # at least one chunk
            sample = np.zeros(size, self.carray.dtype)
            block_count = (len(self.carray) // self.carray.chunklen)
            block_sample_size = (size // block_count) + 1
            i = 0
            remaining = size
            for block in self.blocks:
                if remaining >= block_sample_size:        # last block may not be completely full
                    block_sample = np.random.choice(block, size=block_sample_size)
                    sample[i * block_sample_size:(i + 1) * block_sample_size] = block_sample
                else:
                    block_sample = np.random.choice(block, size=remaining)
                    sample[i * block_sample_size:(i * block_sample_size) + remaining] = block_sample
                    break

                remaining -= len(block_sample)
                i += 1

            return bz_array(bcolz.carray(sample))

    def isnan(self):
        """
        Return a boolean same-sized bz_array indicating if the values are NaN
        """
        ca = self.carray
        if len(ca) == 0:    # eval does not seem to work correctly on empty array
            return bz_array(bcolz.carray([]))
        else:
            return bz_array(bcolz.eval('ca != ca')) # see https://github.com/pydata/numexpr/issues/23

    def isnotnan(self):
        """
        Return a boolean same-sized bz_array indicating if the values are not NaN
        """
        ca = self.carray
        if len(ca) == 0:    # eval does not seem to work correctly on empty array
            return bz_array(bcolz.carray([]))
        else:
            return bz_array(bcolz.eval('ca == ca')) # see https://github.com/pydata/numexpr/issues/23

    def count_nan(self):
        """
        Return the number of NaN in my values
        """
        return np.sum(self.isnan().carray[:])

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
        if len([x for x in new if type(x) != int and type(x) != float and type(x) != bool]) > 0:
            raise ValueError("Values in new must be int, float or bool: in %s()" % (method_name()))

        ca = self.carray
        cond = "(" +  ") | (". join(["ca == " + str(o) for o in old]) + ")"
        idx = bcolz.eval(cond)
        pairs = dict(zip(old, new))
        self.carray[idx] = [pairs[o] for o in self.carray[idx]]

    def sum(self, dtype=np.float64, skipna=True):
        # check numeric
        out = 0.0
        for block in self.blocks:
            if skipna:
                out += np.nansum(block, dtype=dtype)
            else:
                out += np.sum(block, dtype=dtype)
                if np.isnan(out):
                    return np.nan

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
        # check numeric
        out = np.nan
        for block in self.blocks:
            if skipna:
                block_min = np.nanmin(block)
            else:
                block_min = np.amin(block)
                if np.isnan(block_min):
                    return np.nan

            if not np.isnan(block_min) and (np.isnan(out) or block_min < out):
                out = block_min
        return out

    def max(self, skipna=True):
        # check numeric
        out = np.nan
        for block in self.blocks:
            if skipna:
                block_max = np.nanmax(block)
            else:
                block_max = np.amax(block)
                if np.isnan(block_max):
                    return np.nan

            if not np.isnan(block_max) and (np.isnan(out) or block_max < out):
                out = block_max
        return out

    def unique(self):

        #TODO optimize: 150 seconds for 1 billion rows. 72% time in build_buckets() and 28% in np.unique()
        #TODO handle NaN

        buckets_count = (len(self) // bz_array.MAX_LEN) + 1
        if buckets_count < 2:
            return bz_array(np.unique(self[:]))
        else:
            buckets = self.build_buckets(buckets_count)
            out = bz_array([], dtype=self.dtype, expectedlen=len(self))
            for b in range(buckets_count):
                out.append(np.unique(buckets[b][:]))
            return out

    def sort(self):
        #TODO optimize: 160 seconds for 1 billion rows.
        #TODO handle NaN

        buckets_count = (len(self) // bz_array.MAX_LEN) + 1
        if buckets_count < 2:
            return bz_array(np.unique(self[:]))
        else:
            buckets = self.build_buckets(buckets_count)
            out = bz_array([], dtype=self.dtype, expectedlen=len(self))
            for b in range(buckets_count):
                out.append(np.sort(buckets[b][:]))
            return out

    def build_buckets(self, bucket_count):

        # TODO 80 seconds for 10**9 rows. Ouch !
        # this method has to be rewritten into a Cython extension: np.digitize() and data = block[inds == bin_index] hurt alot

        pivots = self.bucket_pivots(bucket_count)
        bcolz_buckets = None
        for bu in range(0, bucket_count):
            bucket = [[] for i  in range (bucket_count)]

            # create buckets on disk
            #bcolz_buckets = [bcolz.carray([], rootdir=self.data_dir + 'test' + str(i), mode='w', expectedlen=len(self.bz_array) // bucket_count)
            #                    for i  in range (bucket_count)]

            # create buckets in memory
            bcolz_buckets = [bcolz.carray([], expectedlen=len(self) // bucket_count) for i in range (bucket_count)]

        for block in self.blocks:
            inds = np.digitize(block, pivots)
            for bu in range(0, bucket_count):
                bin_index = bu + 1 # inds starts at index 1
                data = block[inds == bin_index]
                bcolz_buckets[bu].append(data)  # TODO data should be buffered

        return bcolz_buckets

    def bucket_pivots(self, bucket_count):
        sample = self.sample(10**3)[:]
        sorted_sample = np.sort(sample)
        splits = np.array_split(sorted_sample, bucket_count)
        return [splits[s][0] for s in range(len(splits))]

if __name__ == '__main__':
    # import time
    # start_time = time.time()
    # r = bz_array.randint(0, 1000, 5 * 10**8)
    # c = r.sort()
    # print(c)
    # print( time.time() - start_time)

    a = bz_array([6, np.nan, 7, 4, 6, 9])
    print(a.isnull())


