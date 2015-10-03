from __future__ import absolute_import, print_function, division
import os

import numpy as np
from nose.tools import *
from dazzle.core.column import *
from dazzle.core.dataset import DataSet
from dazzle.core.table import Table
from dazzle.core.utils import DazzleError
import unittest


def assert_array_equal(expect, actual):
    expect = np.asarray(expect)
    actual = np.asarray(actual)
    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)


def assert_close(x, y):
    return abs(x - y) < 0.0001


class TestColumn(unittest.TestCase):

    def setUp(self):
        self.a = [6, 4, 7, 4, 6, 9]
        self.test_dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(self.test_dir, force_create=True)
        self.t = Table("t", ds, [("a", np.array([1, 3], dtype=np.int8)), ("x", np.array([2, 4], dtype=np.float))], force_create=True)
        self.ca = self.t.get_column("a")

    @raises(DazzleError)
    def test_data_dir01(self):
        """no table associated"""
        print(LiteralColumn("a", np.array([], np.int)).data_dir)

    def test_data_dir02(self):
        self.assertEqual(self.ca.data_dir, os.path.join(self.test_dir, "t", "a"))

    @raises(DazzleError)
    def test_carray01(self):
        """no table associated"""
        print(LiteralColumn("a", np.array([], np.int)).carray)

    def test_carray02(self):
        assert_array_equal(self.ca.carray[:], [1, 3])

    @raises(DazzleError)
    def test_init01(self):
        LiteralColumn("", [])

    @raises(DazzleError)
    def test_init02(self):
        LiteralColumn("1a", [])

    @raises(DazzleError)
    def test_init03(self):
        LiteralColumn("_a", [])

    @raises(DazzleError)
    def test_init04(self):
        LiteralColumn("a", "XX")

    @raises(DazzleError)
    def test_init05(self):
        LiteralColumn("a", self)

    def test_init06(self):
        assert_array_equal(self.ca.carray[:], [1, 3])

    def test_len01(self):
        self.assertEqual(len(self.ca), 2)

    def test_position01(self):
        self.t.append({'a': self.a, 'x': self.a})
        self.assertEqual(self.ca.position, 0)

    def test_position02(self):
        self.t.append({'a': self.a, 'x': self.a})
        self.assertEqual(self.t.get_column("x").position, 1)

    def test_getitem01(self):
        self.assertEqual(self.ca[0], 1)

    def test_getitem02(self):
        self.assertEqual(self.ca[1], 3)

    @raises(IndexError)
    def test_getitem03(self):
        self.t.append({'a': self.a, 'x': self.a})
        _ = self.ca[10]

    def test_getitem04(self):
        self.t.append({'a': self.a, 'x': self.a})
        assert_array_equal(self.ca[:], self.ca.carray[:])

    def test_getitem05(self):
        self.t.append({'a': self.a, 'x': self.a})
        assert_array_equal(self.ca[0:5], [1, 3, 6, 4, 7])

    def test_setitem01(self):
        self.ca[0] = 2
        self.assertEqual(self.ca[0], 2)
        assert_array_equal(self.ca.carray[:], [2, 3])

    def test_append01(self):
        self.t.append({'a': self.a, 'x': self.a})
        self.ca.append(self.a)
        assert_array_equal(self.ca.carray[:], [1, 3, 6, 4, 7, 4, 6, 9, 6, 4, 7, 4, 6, 9])

    @raises(DazzleError)
    def test_rename01(self):
        self.t.append({'a': self.a, 'x': self.a})
        self.ca.rename("")

    @raises(DazzleError)
    def test_rename02(self):
        self.t.append({'a': self.a, 'x': self.a})
        self.ca.rename("x")

    def test_rename03(self):
        self.ca.rename("b")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "t", "b")), "'b' dir should exist")
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "t", "a")), "'a' dir should not exist")
        self.assertEqual(self.ca.data_dir, os.path.join(self.test_dir, "t", "b"))
        self.assertEqual(self.ca._name, "b", "column should be named 'b'")
        self.assertEqual(self.t.get_column("b").position, 0, "column should be at position 0")

    def test_sum01(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=True), 0.0)

    def test_sum02(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=False), 0.0)

    def test_sum03(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, 7, 4, 6, 9], np.int))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=True), 36.0)

    def test_sum04(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, 7, 4, 6, 9], np.int))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=False), 36.0)

    def test_sum05(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=True), 29.0)

    def test_sum06(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.sum(skipna=False)))

    def test_sum07(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        assert_close(ca.sum(skipna=True), 0.0)

    def test_sum08(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.sum(skipna=False)))

    def test_sum09(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([3, nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertEqual(ca.sum(skipna=True), 5)

    def test_sum10(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.sum(skipna=False)))

    def test_sum11(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, nan], np.int8))], force_create=True).get_column("a")
        self.assertEqual(ca.sum(skipna=True), 0)

    def test_mean01(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.mean(skipna=True)))

    def test_mean02(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.mean(skipna=False)))

    def test_mean03(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.mean(skipna=True), 6.0)

    def test_mean04(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds,[("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.mean(skipna=False), 6.0)

    def test_mean05(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        assert_close(ca.mean(skipna=True), 5.0)

    def test_mean06(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.mean(skipna=False)))

    def test_mean07(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.mean(skipna=True)))

    def test_mean08(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.mean(skipna=False)))

    def test_min01(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        x = ca.min()
        self.assertTrue(ca.isnan(ca.min(skipna=True)))

    def test_min02(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=False)))

    def test_min03(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.min(skipna=True), 4)

    def test_min04(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.min(skipna=False), 4)

    def test_min05(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        assert_close(ca.min(skipna=True), 4)

    def test_min06(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=False)))

    def test_min07(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=True)))

    def test_min08(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=False)))

    def test_min09(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([3, nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertEqual(ca.min(skipna=True), 2)

    def test_min10(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=False)))

    def test_min11(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, nan], np.int8))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.min(skipna=False)))

    def test_max01(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=True)))

    def test_max02(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([], np.int))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=False)))

    def test_max03(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.max(skipna=True), 9)

    def test_max04(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", [np.array([6, 4, 7, 4, 6, 9], np.int)])], force_create=True).get_column("a")
        assert_close(ca.max(skipna=False), 9)

    def test_max05(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        assert_close(ca.max(skipna=True), 9)

    def test_max06(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([6, 4, np.nan, 4, 6, 9], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=False)))

    def test_max07(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=True)))

    def test_max08(self):
        ds = DataSet(self.test_dir, force_create=True)
        ca = Table("t", ds, [("a", np.array([np.nan, np.nan], np.float))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=False)))

    def test_max09(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([3, nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertEqual(ca.max(skipna=True), 3)

    def test_max10(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, 2], np.int8))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=False)))

    def test_max11(self):
        ds = DataSet(self.test_dir, force_create=True)
        nan = np.iinfo(np.int8).min
        ca = Table("t", ds, [("a", np.array([nan, nan], np.int8))], force_create=True).get_column("a")
        self.assertTrue(ca.isnan(ca.max(skipna=False)))

    # def test_itemsize01(self):
    #     c = ArrayColumn("a", data=self.a)
    #     self.assertEqual(c.itemsize(), 4)
    #
    # @raises(DazzleError)
    # def test_itemsize01(self):
    #     c = ArrayColumn("a", data=self.a)
    #     c.carray = None
    #     self.assertEqual(c.itemsize(), 4)
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

