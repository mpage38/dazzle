from __future__ import absolute_import, print_function, division
import os
import bcolz

import numpy as np
from nose.tools import *
from dazzle.core.bz_array import bz_array
from dazzle.core.utils import DazzleError
import unittest

def assert_array_equal(expect, actual):
    expect = np.asarray(expect)
    actual = np.asarray(actual)
    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)


def assert_close(x, y):
    return abs(x - y) < 0.0001


class Test_bz_array(unittest.TestCase):

    def setUp(self):
        pass

    @raises(DazzleError)
    def test_init01(self):
        return bz_array(None)

    @raises(DazzleError)
    def test_init02(self):
        return bz_array(np.array([]))

    def test_randint01(self):
        r = bz_array.randint(1, 5, 10).carray
        self.assertEqual(len(r), 10)
        self.assertTrue(np.all(r[:] >= 1))
        self.assertTrue(np.all(r[:] < 5))

    @raises(ValueError)
    def test_randint02(self):
        r = bz_array.randint(10, 5, 10)

    def test_randfloat01(self):
        r = bz_array.randfloat(1.0, 5.0, 10).carray
        self.assertEqual(len(r), 10)
        self.assertTrue(np.all(r[:] >= 1.0))
        self.assertTrue(np.all(r[:] < 5.0))

    @raises(ValueError)
    def test_randfloat02(self):
        r = bz_array.randfloat(10.0, 5.0, 10)

    def test_blocks01(self):
        r = bz_array.randfloat(5.0, 10.0, 10)
        b = 0
        block = None
        for block in r.blocks:
            b += 1
        self.assertEqual(b, 1)
        self.assertEqual(len(block), 10)

    def test_sample01(self):
        r = bz_array.sample(bz_array.randfloat(5.0, 10.0, 1000), 10)
        self.assertEqual(len(r.carray), 10)

    def test_sample02(self):
        r = bz_array.sample(bz_array.randfloat(5.0, 10.0, 200000), 100000)
        self.assertEqual(len(r.carray), 100000)

    def test_isnan01(self):
        r = bz_array(bcolz.carray([1,0,2]))
        assert_array_equal(r.isnan().carray, [False, False, False])

    def test_isnan02(self):
        r = bz_array(bcolz.carray([1,np.NaN,2]))
        assert_array_equal(r.isnan().carray, [False, True, False])

    def test_isnan03(self):
        r = bz_array(bcolz.carray([1,2,None]))
        assert_array_equal(r.isnan().carray, [False, False, False])

    def test_isnan04(self):
        r = bz_array(bcolz.carray([]))
        assert_array_equal(r.isnan().carray, [])

    def test_isnotnan01(self):
        r = bz_array(bcolz.carray([1,0,2]))
        assert_array_equal(r.isnotnan().carray, [True, True, True])

    def test_isnotnan02(self):
        r = bz_array(bcolz.carray([1,np.NaN,2]))
        assert_array_equal(r.isnotnan().carray, [True, False, True])

    def test_isnotnan04(self):
        r = bz_array(bcolz.carray([]))
        assert_array_equal(r.isnotnan().carray, [])

    def test_isnotnan03(self):
        r = bz_array(bcolz.carray([1,2,None]))
        assert_array_equal(r.isnotnan().carray, [True, True, True])

    def test_count_nan01(self):
        r = bz_array(bcolz.carray([1,np.NaN,np.NaN]))
        assert_array_equal(r.count_nan(), 2)

    def test_count_nan02(self):
        r = bz_array(bcolz.carray([1, 2]))
        assert_array_equal(r.count_nan(), 0)

    def test_count_nan03(self):
        r = bz_array(bcolz.carray([]))
        assert_array_equal(r.count_nan(), 0)

    def test_count_nan04(self):
        r = bz_array(bcolz.carray([None]))
        assert_array_equal(r.count_nan(), 0)

    def test_replace_value01(self):
        r = bz_array(bcolz.carray([1,2,2]))
        r.replace_value(2,3)
        assert_array_equal(r.carray, [1,3,3])

    @raises(ValueError)
    def test_replace_value02(self):
        r = bz_array(bcolz.carray([1,2,2]))
        r.replace_value(2,'a')

    def test_replace_list01(self):
        r = bz_array(bcolz.carray([1,2,2]))
        r.replace_list([1,2], [2,3])
        assert_array_equal(r.carray, [2,3,3])

    @raises(ValueError)
    def test_replace_list02(self):
        r = bz_array(bcolz.carray([1,2,2]))
        r.replace_list([1,2], [2,'a'])

    def test_sum01(self):
        assert_close(bz_array([]).sum(skipna=True), 0.0)

    def test_sum02(self):
        assert_close(bz_array([]).sum(skipna=False), 0.0)

    def test_sum03(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).sum(skipna=True), 36.0)

    def test_sum04(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).sum(skipna=False), 36.0)

    def test_sum05(self):
        assert_close(bz_array([6, 4, np.nan, 4, 6, 9]).sum(skipna=True), 30.0)

    def test_sum06(self):
        self.assertTrue(np.isnan(bz_array([6, 4, np.nan, 4, 6, 9]).sum(skipna=False)))

    def test_sum07(self):
        assert_close(bz_array([np.nan, np.nan]).sum(skipna=True), 0.0)

    def test_sum08(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).sum(skipna=False)))

    def test_mean01(self):
        self.assertTrue(np.isnan(bz_array([]).mean(skipna=True)))

    def test_mean02(self):
        self.assertTrue(np.isnan(bz_array([]).mean(skipna=False)))

    def test_mean03(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).mean(skipna=True), 6.0)

    def test_mean04(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).mean(skipna=False), 6.0)

    def test_mean05(self):
        assert_close(bz_array([6, 4, np.nan, 4, 6, 9]).mean(skipna=True), 5.0)

    def test_mean06(self):
        self.assertTrue(np.isnan(bz_array([6, 4, np.nan, 4, 6, 9]).mean(skipna=False)))

    def test_mean07(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).mean(skipna=True)))

    def test_mean08(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).mean(skipna=False)))

    def test_min01(self):
        self.assertTrue(np.isnan(bz_array([]).min(skipna=True)))

    def test_min02(self):
        self.assertTrue(np.isnan(bz_array([]).min(skipna=False)))

    def test_min03(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).min(skipna=True), 4)

    def test_min04(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).min(skipna=False), 4)

    def test_min05(self):
        assert_close(bz_array([6, 4, np.nan, 4, 6, 9]).min(skipna=True), 4)

    def test_min06(self):
        self.assertTrue(np.isnan(bz_array([6, 4, np.nan, 4, 6, 9]).min(skipna=False)))

    def test_min07(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).min(skipna=True)))

    def test_min08(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).min(skipna=False)))

    def test_max01(self):
        self.assertTrue(np.isnan(bz_array([]).max(skipna=True)))

    def test_max02(self):
        self.assertTrue(np.isnan(bz_array([]).max(skipna=False)))

    def test_max03(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).max(skipna=True), 9)

    def test_max04(self):
        assert_close(bz_array([6, 4, 7, 4, 6, 9]).max(skipna=False), 9)

    def test_max05(self):
        assert_close(bz_array([6, 4, np.nan, 4, 6, 9]).max(skipna=True), 9)

    def test_max06(self):
        self.assertTrue(np.isnan(bz_array([6, 4, np.nan, 4, 6, 9]).max(skipna=False)))

    def test_max07(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).max(skipna=True)))

    def test_max08(self):
        self.assertTrue(np.isnan(bz_array([np.nan, np.nan]).max(skipna=False)))




    # @raises(DazzleError)
    # def test_data_dir01(self):
    #     """no table associated"""
    #     print(LiteralColumn("a", np.int).data_dir)
    #
    # def test_data_dir02(self):
    #     self.assertEqual(self.ca.data_dir, os.path.join(self.test_dir, "t", "a"))
    #
    # @raises(DazzleError)
    # def test_bz_array01(self):
    #     """no table associated"""
    #     print(LiteralColumn("a", np.int).bz_array)
    #
    # def test_bz_array02(self):
    #     assert_array_equal(self.ca.bz_array[:], [])
    #
    # @raises(DazzleError)
    # def test_init01(self):
    #     LiteralColumn("", np.int)
    #
    # @raises(DazzleError)
    # def test_init02(self):
    #     LiteralColumn("1a", np.int)
    #
    # @raises(DazzleError)
    # def test_init03(self):
    #     LiteralColumn("_a", np.int)
    #
    # @raises(DazzleError)
    # def test_init04(self):
    #     LiteralColumn("a", "XX")
    #
    # @raises(DazzleError)
    # def test_init05(self):
    #     LiteralColumn("a", self)
    #
    # def test_init06(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     assert_array_equal(self.ca.bz_array[:], self.a)
    #
    # def test_dtype01(self):
    #     self.assertEqual(self.ca.dtype, np.int)
    #
    # def test_dtype02(self):
    #     self.t.append(self.append_dict)
    #     self.assertEqual(self.ca.dtype, np.int)
    #
    # def test_itemsize01(self):
    #     self.assertEqual(self.ca.itemsize, 4)
    #
    # def test_itemsize02(self):
    #     self.t.append(self.append_dict)
    #     self.assertEqual(self.ca.itemsize, 4)
    #
    # def test_len01(self):
    #     self.assertEqual(self.ca.len, 0)
    #
    # def test_len02(self):
    #     self.t.append(self.append_dict)
    #     self.assertEqual(self.ca.len, 2)
    #
    # def test_len03(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(len(self.ca), 6)
    #
    # def test_cbytes01(self):
    #     self.assertTrue(self.ca.cbytes > 0)
    #
    # def test_nbytes01(self):
    #     self.assertTrue(self.ca.nbytes == 0)
    #
    # def test_nbytes02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertTrue(self.ca.nbytes > 0)
    #
    # def test_size01(self):
    #     self.assertEqual(self.ca._size, 0)
    #
    # def test_size02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(self.ca._size, 6)
    #
    # def test_position01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(self.ca.position, 0)
    #
    # def test_position02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(self.t.get_column("x").position, 1)
    #
    # def test_getitem01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(self.ca[0], 6)
    #
    # def test_getitem02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(self.ca[1], 4)
    #
    # @raises(IndexError)
    # def test_getitem03(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     _ = self.ca[10]
    #
    # def test_getitem04(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     assert_array_equal(self.ca[:], self.ca.bz_array[:])
    #
    # def test_getitem05(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     assert_array_equal(self.ca[0:2], [6, 4])
    #
    # def test_setitem01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca[0] = 2
    #     self.assertEqual(self.ca[0], 2)
    #     assert_array_equal(self.ca.bz_array[:], [2, 4, 7, 4, 6, 9])
    #
    # def test_repr011(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.assertEqual(repr(self.ca)[0:6], "LiteralColumn")
    #
    # def test_append01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca._append(self.a)
    #     self.assertEqual(self.ca._size, 12)
    #     assert_array_equal(self.ca.bz_array[:], [6, 4, 7, 4, 6, 9, 6, 4, 7, 4, 6, 9])
    #
    # def test_resize01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca._resize(3)
    #     self.assertEqual(self.ca._size, 3)
    #
    # def test_resize02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca._resize(10)
    #     self.assertEqual(self.ca._size, 10)
    #     assert_array_equal(self.ca.bz_array[:], [6, 4, 7, 4, 6, 9, 0, 0, 0, 0])
    #
    # @raises(DazzleError)
    # def test_rename01(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca.rename("")
    #
    # @raises(DazzleError)
    # def test_rename02(self):
    #     self.t.append({'a': self.a, 'x': self.a})
    #     self.ca.rename("x")
    #
    # def test_rename03(self):
    #     self.t.append(self.append_dict)
    #     self.ca.rename("b")
    #     self.assertTrue(os.path.exists(os.path.join(self.test_dir, "t", "b")), "'b' dir should exist")
    #     self.assertFalse(os.path.exists(os.path.join(self.test_dir, "t", "a")), "'a' dir should not exist")
    #     self.assertEqual(self.ca.data_dir, os.path.join(self.test_dir, "t", "b"))
    #     self.assertEqual(self.ca._name, "b", "column should be named 'b'")
    #     self.assertEqual(self.t.get_column("b").position, 0, "column should be at position 0")
    #


    #
    # def test_all01(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[6,4,7,4,6,9])
    #     res = c.all()
    #     self.assertTrue(res)
    #
    # def test_all02(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[6,4,7,4,0,9])
    #     res = c.all()
    #     self.assertFalse(res)
    #
    # def test_all03(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[])
    #     res = c.all()
    #     self.assertTrue(res)
    #
    # def test_any01(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[0,0,0,0,0,0])
    #     res = c.any()
    #     self.assertFalse(res)
    #
    # def test_any02(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[6,4,7,4,0,9])
    #     res = c.any()
    #     self.assertTrue(res)
    #
    # def test_any03(self):
    #     LiteralColumn.DEFAULT_BLOCK_LEN = 3
    #     c = ArrayColumn("a", data=[])
    #     res = c.any()
    #     self.assertFalse(res)

if __name__ == '__main__':
    unittest.main()

