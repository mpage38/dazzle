from __future__ import absolute_import, print_function, division
import os
import bcolz
import pandas as pd
import numpy as np
from nose.tools import *

from dazzle.core.column import *
from dazzle.core.dataset import DataSet
from dazzle.core.table import Table
from dazzle.core.utils import DazzleError
import unittest

AVITO_DATA_DIR = "/github/kaggle-fiddling/avito-context-ad-clicks/data"

def assert_array_equal(expect, actual):

    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)

def assert_close(x, y):
    return abs(x - y) < 0.0001

def assert_equal_table(t1, t2):
    return t1.name == t2.name \
           and t1.expected_length == t2.expected_length \
           and len(t1.columns) == len(t2.columns) \
           and [sc.same(oc) for sc, oc in zip(t1.columns, t2.columns)]

class TestTable(unittest.TestCase):

    def assert_string_equal(self, s1, s2):
        return self.assertEqual(''.join(s1.split()), ''.join(s2.split()))

    def assert_table_content(self, table, to_check):
        for check, val in to_check.items():
            if check == 'data_dir':
                self.assertEqual(table.data_dir, val)
            elif check == 'len':
                self. assertEqual(len(table.ctable), val)
            elif check == 'type':
                self. assertEqual(type(table), val)
            elif check == 'columns':
                index = 0
                for col_name, attrs in val:
                    self.assert_column_content(table, col_name, index, attrs)
                    index += 1
            else:
                raise DazzleError("Invalid key: %s" % check)

    def assert_column_content(self, table, col_name, index, to_check):
        self.assertTrue(isinstance(table._columns[index], LiteralColumn))
        col = table._columns[index]
        self.assertTrue(col._table == table)
        self.assertTrue(col._name == col_name)
        self.assertTrue(table.ctable.names[index] == col_name)
        bz_col = table.ctable.cols._cols[col_name]
        self.assertEqual(col.carray, bz_col)
        self.assertTrue(isinstance(bz_col, bcolz.carray))
        for check, val in to_check.items():
            if check == 'len':
                self.assertEqual(bz_col.len, val)
            elif check == 'content':
                assert_array_equal(bz_col[:], val)
            elif check == 'type':
                self.assertEqual(col.dtype, val)
            else:
                raise DazzleError("Invalid key: %s" % check)

    def setUp(self):
        self.a = [6, 4, 7, 4, 6, 9]
        self.test_dir = os.path.join("/temp", "dazzle-test")
        self.ds = DataSet(self.test_dir, force_create=True)
        self.t = Table("t", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float))], force_create=True)
        self.u = Table("u", self.ds, [("a", np.array([1, 2], np.int)), ("b", np.array([1.1, 2.2], np.float))], force_create=True)

    def test_init01(self):
        self.assert_table_content(self.t, {
            'data_dir': os.path.join(self.test_dir, self.t._name),
            'len': 0,
            'type': Table,
            'columns': [('a', {'type': np.int, 'content': []})]})

    @raises(DazzleError)
    def test_init02(self):
        Table("_", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float))], force_create=True)

    @raises(DazzleError)
    def test_init03(self):
        Table("t", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float))], mode='open', force_create=True)

    @raises(DazzleError)
    def test_init04(self):
        Table("t", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float))], force_create=True)

    @raises(ValueError)
    def test_init05(self):
        Table("t", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float))], mode='open')

    @raises(ValueError)
    def test_init06(self):
        Table("t", self.ds, [{"a": np.array([], np.int)}], force_create=True)

    @raises(DazzleError)
    def test_init07(self):
        Table("t", self.ds, [], force_create=True)

    @raises(ValueError)
    def test_init08(self):
        Table("t", self.ds, [("a", 3)], force_create=True)

    @raises(ValueError)
    def test_init09(self):
        Table("t", self.ds, [{"a": np.array([True, False], np.bool)}], force_create=True)

    @raises(ValueError)
    def test_init10(self):
        Table("t", self.ds, ("a", np.array([], np.int)), force_create=True)

    @raises(ValueError)
    def test_init11(self):
        Table("t", self.ds, [("a", np.array([], np.int)), ("b", np.array([], np.float), 'oops')], force_create=True)

    @raises(DazzleError)
    def test_init11(self):
        Table("t", self.ds, [("a", np.array([], np.bool)), ("b", np.array([], np.float))], force_create=True)

    def test_init12(self):
        v = Table("v", self.ds,  [("a", [3])])
        self.assert_table_content(v, {
            'data_dir': os.path.join(self.test_dir, "v"),
            'len': 1,
            'type': Table,
            'columns': [('a', {'type': np.int, 'content': [3]})]})

    def test_dataset01(self):
        self.assertEqual(self.ds, self.t.dataset)

    @raises(DazzleError)
    def test_data_dir01(self):
        """no table associated"""
        print(LiteralColumn("a", None).data_dir)

    @raises(DazzleError)
    def test_copy01(self):
        Table.copy("t", self.ds, "/temp/dazzle-test")

    @raises(DazzleError)
    def test_copy02(self):
        Table.copy("t", self.ds, "/bim/bam")

    @raises(DazzleError)
    def test_copy03(self):
        test_dir = os.path.join("/temp/dazzle-test2")
        ds2 = DataSet(test_dir, force_create=True)
        Table.copy("_", ds2, "/temp/dazzle-test/t")

    def test_copy04(self):
        test_dir = os.path.join("/temp/dazzle-test2")
        ds2 = DataSet(test_dir, force_create=True)
        t = Table.copy("t", ds2, "/temp/dazzle-test/t")
        assert_equal_table(t, self.ds.get_table("t"))

    @raises(FileNotFoundError)
    def test_from_csv01(self):
        Table.from_csv("Category", self.ds, "/bim/bam/test.csv", usecols=['CategoryID', 'ParentCategoryID'], verbose=False)

    @raises(ValueError)
    def test_from_csv02(self):
        Table.from_csv("Category", self.ds, "/temp/dazzle-test/dataset.json", usecols=['CategoryID', 'ParentCategoryID'], verbose=False)

    @raises(DazzleError)
    def test_from_csv03(self):
        cat = Table.from_csv("Category", self.ds, os.path.join(AVITO_DATA_DIR, "Category.tsv"), verbose=False)

    def test_from_csv04(self):
        cat = Table.from_csv("Category", self.ds, os.path.join(AVITO_DATA_DIR, "Category.tsv"), delimiter='\t',
                                   usecols=['CategoryID', 'ParentCategoryID'], verbose=False)
        self.assertEqual(len(cat.ctable), 68)
        self.assertEqual(len(cat.columns), 2)

    def test_from_dataframe01(self):
        df = pd.DataFrame({'a': [1,2], 'b': [3., 4.]})
        v = Table.from_dataframe("v", self.ds, df)
        self.assertEqual(len(v.ctable), 2)

    def test_get_column01(self):
        self.assertTrue(self.t.get_column("x") is None)

    def test_get_column02(self):
        self.assertEqual(self.t.get_column("a").name, "a")

    @raises(ValueError)
    def test_remove_column01(self):
        self.t.remove_column("x")

    def test_remove_column02(self):
        self.t.remove_column("a")
        self.assertTrue(self.t.get_column("a") is None)
        self.assertEqual(self.t.columns[0], self.t.get_column("b"))
        self.assertEqual(self.t.ctable.names[0], "b")

    def test_to_dataframe01(self):
        self.assertEqual(len(self.u.to_dataframe()), 2)

    def test_append01(self):
        self.t.append({'a': [1,2], 'b': [3., 4.]})
        self.assert_table_content(self.t, {
            'len': 2,
            'columns': [('a', {'content': [1,2]}), ('b', {'content': [3., 4.]})]})

    def test_append02(self):
        self.t.append({'b': [3., 4.], 'a': [1,2]})
        self.assert_table_content(self.t, {
            'len': 2,
            'columns': [('a', {'content': [1,2]}), ('b', {'content': [3., 4.]})]})

    def test_append03(self):
        self.t.append({'a': [5.4, 2], 'b': [3., 4.]})

    @raises(ValueError)
    def test_append04(self):
        self.t.append({'a': ["bla", 2], 'b': [3., 4.]})

    @raises(ValueError)
    def test_append05(self):
        self.t.append({'a': [], 'b': [3., 4.]})

    @raises(ValueError)
    def test_append06(self):
        self.t.append({'a': []})

    @raises(ValueError)
    def test_append05(self):
        self.t.append([[], [3., 4.]])

    def test_get_item01(self):
        self.assertEqual(self.u[0]['a'], 1)
        self.assertEqual(self.u[0]['b'], 1.1)

    def test_get_item02(self):
        assert_array_equal(self.u['a'], np.array([1,2]))

    @raises(IndexError)
    def test_get_item03(self):
        print(self.u[0,1])

    def test_get_item04(self):
        assert_array_equal(self.u[[0,1]]['a'], np.array([1, 2]))
        assert_array_equal(self.u[[0,1]]['b'], np.array([1.1, 2.2]))

    def test_get_item05(self):
        assert_array_equal(self.u['a'][[0,1]], np.array([1, 2]))
        assert_array_equal(self.u['b'][[0,1]], np.array([1.1, 2.2]))

    def test_set_item01(self):
        self.u[0] = (10, 20.2)
        self.assertEqual(self.u[0]['a'], 10)
        self.assertEqual(self.u[0]['b'], 20.2)

    def test_set_item02(self):
        self.u[[0, 1]] = [(10, 20.2), (190, 32.4)]
        self.assertEqual(self.u[0]['b'], 20.2)
        self.assertEqual(self.u[1]['a'], 190)

    # def test_set_item03(self):
    #     self.u[[0, 1]]['a'] = 40 # makes a copy; u is not modified
    #     self.assertEqual(self.u[0]['a'], 40)

    # def test_set_item04(self):
    #     self.u[0]['a'] = 14  # makes a copy; u is not modified
    #     self.assertEqual(self.u[0]['a'], 14)

    def test_str01(self):
        s = \
            "u(a: int32, b: float64)" \
            "2 row(s) - compressed: 2.00 MB - comp. ratio: 0.00" \
            "+---+-------+" \
            "| a |     b |" \
            "+---+-------+" \
            "| 1 | 1.100 |" \
            "| 2 | 2.200 |" \
            "+---+-------+"
        self.assert_string_equal(self.u.__str__(), s)

    def test_str02(self):
        s = \
            "u(a: int32, b: float64)" \
            "2 row(s) - compressed: 2.00 MB - comp. ratio: 0.00" \
            "+---+-------+" \
            "| a |     b |" \
            "+---+-------+" \
            "| 1 | 1.100 |" \
            "| 2 | 2.200 |" \
            "+---+-------+"
        self.assert_string_equal(self.u.__str__(head=20), s)

    def test_str03(self):
        s = \
            "u(a: int32, b: float64)" \
            "2 row(s) - compressed: 2.00 MB - comp. ratio: 0.00" \
            "+---+-----+" \
            "| a |   b |" \
            "+---+-----+" \
            "| 1 | 1.1 |" \
            "| 2 | 2.2 |" \
            "+---+-----+"
        self.u.get_column("b").format = "%.1f"
        self.assert_string_equal(self.u.__str__(head=20), s)

    def test_head01(self):
        s = \
            "u(a: int32, b: float64)" \
            "2 row(s) - compressed: 2.00 MB - comp. ratio: 0.00" \
            "+-----+-------+" \
            "| a   |     b |" \
            "+-----+-------+" \
            "|   1 | 1.100 |" \
            "| ... |   ... |" \
            "+-----+-------+"
        self.assert_string_equal(self.u.head(1), s)

    def test_tail01(self):
        s = \
            "u(a: int32, b: float64)" \
            "2 row(s) - compressed: 2.00 MB - comp. ratio: 0.00" \
            "+-----+-------+" \
            "| a   |     b |" \
            "+-----+-------+" \
            "| ... |   ... |" \
            "|   2 | 2.200 |" \
            "+-----+-------+"
        self.assert_string_equal(self.u.tail(1), s)

    def test_rebuild01(self):
        cat = Table.from_csv("Category", self.ds, os.path.join(AVITO_DATA_DIR, "Category.tsv"), delimiter='\t',
                                   usecols=['CategoryID', 'ParentCategoryID', 'Level'], verbose=False)


        cat.rebuild({"CategoryID": np.int8, "Level": np.int8, "ParentCategoryID": np.int8})
        self.assertEqual(len(cat[:]), 69)
        self.assertEqual(cat['CategoryID'].dtype, np.int8)
        self.assertEqual(cat[0]['CategoryID'], -128) # int8.min
        self.assertEqual(cat[0]['Level'], -128) # int8.min
        self.assertEqual(cat[0]['ParentCategoryID'], -128) # int8.min

    @raises(DazzleError)
    def test_rebuild02(self):
        cat = Table.from_csv("Category", self.ds, os.path.join(AVITO_DATA_DIR, "Category.tsv"), delimiter='\t',
                                   usecols=['CategoryID', 'ParentCategoryID', 'Level'], verbose=False)
        cat.rebuild({"CategoryID": np.uint8, "Level": np.uint8, "ParentCategoryID": np.uint8})

    def test_add_join_column(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        t = Table("t", ds, [('a', np.array([10, 2, 3, 5, 4, 7, 1, 8, 6, 9])),
                            ('c', np.array([100, 20, 30, 50, 40, 70, 10, 80, 60, np.nan]))])

        a_ref = np.array([1, 5, 4, 5, 6, 4, 1, 1, 9, 7, 8, 4, 5, 5, 2, 2, 8, 5, 4, 20])
        u = Table("u", ds, [('a', a_ref), ("y", a_ref * 10)])

        u.get_column("a").ref_column = t.get_column("a")
        t.rebuild({'a': np.int8, 'c': np.int8})
        u.rebuild({'a': np.int8, 'y': np.int16})

        u.add_reference_column(u.get_column("a"), t.get_column("a"))
        # print(t.head(20))
        # print(u.head(30))

        u.add_join_column("result", [u.get_column("a_ref"), t.get_column("c")])
        #print(u.head(30))
        assert np.array_equal(u['result'],
                              [-128, 10, 50, 40, 50, 60, 40, 10, 10, -128, 70, 80, 40, 50, 50, 20, 20, 80, 50, 40, -128])

if __name__ == '__main__':
    unittest.main()


