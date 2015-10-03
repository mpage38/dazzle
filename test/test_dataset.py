from __future__ import absolute_import, print_function, division
import json
import numpy as np
from nose.tools import *
from dazzle.core.dataset import DataSet
from dazzle.core.table import Table
from dazzle.core.utils import *
import unittest


def assert_array_equal(expect, actual):
    expect = np.asarray(expect)
    actual = np.asarray(actual)
    assert np.array_equal(expect, actual), \
        '\nExpect:\n%r\nActual:\n%r\n' % (expect, actual)



class TestDataSet(unittest.TestCase):

    def assert_string_equal(self, s1, s2):
        return self.assertEqual(''.join(s1.split()), ''.join(s2.split()))

    def assert_close(self, x, y):
        return self.assertTrue(abs(x - y) < 0.0001)

    def assert_equal_table(self, t1, t2):
        return self.assertTrue(t1.name == t2.name \
           and t1.expected_length == t2.expected_length \
           and len(t1.columns) == len(t2.columns) \
           and [sc.same(oc) for sc, oc in zip(t1.columns, t2.columns)]  )

    def setUp(self):
        dir = os.path.join("/temp", "dazzle-test")
        self.ds = DataSet(dir, force_create=True)

        t = Table("t", self.ds, [("a", []), ("b", [])])
        u = Table("u", self.ds, [("a", []), ("y", [])])

    def test_init01(self):
        dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(dir, force_create=True)
        self.assertTrue(isinstance(ds, DataSet))

    def test_compression_params01(self):
        dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(dir, force_create=True)
        self.assertTrue(isinstance(ds.compression_params, bcolz.cparams))

    def test_data_dir01(self):
        dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(dir, force_create=True)
        self.assertEqual(ds.data_dir, dir)

    def test_tables01(self):
        self.assertEqual(len(self.ds.tables), 2)
        self.assertEqual(self.ds.tables[0].name, "t")
        self.assertEqual(self.ds.tables[1].name, "u")

    def test_compressed_bytes01(self):
        self.assertTrue(self.ds.compressed_bytes > 0)

    def test_uncompressed_bytes01(self):
        self.assertTrue(self.ds.uncompressed_bytes >= 0)

    def test_get_table01(self):
        self.assertTrue(self.ds.get_table("t").dataset == self.ds)

    def test_add_table01(self):
        v = Table("v", self.ds, [("d", [])])

        self.assertEqual(v.dataset, self.ds)
        self.assertTrue(os.path.isdir(v.data_dir))
        self.assertEqual(self.ds.get_table(v.name), v)

    @raises(DazzleError)
    def test_add_table02(self):
        "table name exists"
        t = Table("t", self.ds, [("d", [])])

    def test_remove_table01(self):
        t = self.ds.get_table("t")
        u = self.ds.get_table("u")
        self.ds.remove_table(t)
        self.assertTrue(not os.path.exists(t.data_dir))
        self.assertTrue(os.path.exists(u.data_dir))
        self.assertTrue(self.ds.get_table("t") is None)
        self.assertEqual(self.ds.get_table("u"), u)

    @raises(DazzleError)
    def test_remove_table02(self):
        t = self.ds.get_table("t")
        u = self.ds.get_table("u")
        self.ds.remove_table(t)
        self.ds.remove_table(t)

    def test_create01(self):
        dir = os.path.join("/temp", "dazzle-test1")
        ds = DataSet(dir, force_create=True)

        t = Table("t", ds, [("a", []), ("b", [])])

        self.assertEqual(t.dataset, ds)
        self.assertEqual(t.data_dir, os.path.join(dir, "t"))
        self.assertTrue(os.path.isdir(t.data_dir))
        self.assertEqual(ds.get_table(t.name), t)

        u = Table("u", ds, [("a", []), ("y", [])])

        self.assertEqual(u.dataset, ds)
        self.assertEqual(u.data_dir, os.path.join(dir, "u"))
        self.assertTrue(os.path.isdir(u.data_dir))
        self.assertEqual(ds.tables[1], u)

    @raises(DazzleError)
    def test_create02(self):
        """dir exists"""
        dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(dir)

    @raises(DazzleError)
    def test_create03(self):
        """parent dir does not exist"""
        dir = os.path.join("/bim", "bam", "boum")
        ds = DataSet(dir)

    def test_to_json01(self):
        js = self.ds.to_json()
        self.ds.save()

        with open(os.path.join(self.ds.data_dir + "/" + "dataset.json")) as json_file:
            js2 = json.load(json_file)

        self.assertEqual(json.loads(js), js2)

    def test_str01(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        t = Table("t", ds, [("a", np.array([11,2], dtype=np.int32)), ("b", [0, 1]), ("c", np.array([3,4], dtype=np.int64))])

        u = Table("u", ds, [("x", np.array([], dtype=np.int)), ("y", np.array([], dtype=np.float))])
        #print(ds.__str__())
        self.assert_string_equal(ds.__str__(), "Dir: /temp/dazzle-test" +
                                               "Compression params: cparams(clevel=5, shuffle=True, cname='blosclz') " +
                                               "Tables:	" +
                                               "t(a:int32, b:int32, c:int64) 2 row(s)-compressed:3.00MB-comp.ratio:0.00	" +
                                               "u(x:int32, y:float64)0 row(s) - compressed: 2.00 MB - comp. ratio: 0.00")

    def test_open01(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        t = Table("t", ds, [('a', [11,2]), ('b', [1, 0]), ('c', [3,4])])

        u = Table("u", ds, [("x", []), ("y", [])])
        ds.save()
        ds1 = DataSet.open("/temp/dazzle-test")
        self.assert_equal_table(ds.get_table("t"), ds1.get_table("t"))
        self.assert_equal_table(ds.get_table("u"), ds1.get_table("u"))

    @raises(DazzleError)
    def test_open02(self):
        DataSet.open("/temp/bim/bam/boum")

    def test_copy01(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        t = Table("t", ds, [('a', [11, 2]), ('b', [1, 0]), ('c', [3, 4])])

        u = Table("u", ds, [("x", []), ("y", [])])
        ds.save()

        ds_copy = ds.copy("/temp/dazzle-test-copy", force_create=True)
        self.assert_equal_table(ds.get_table("t"), ds_copy.get_table("t"))
        self.assert_equal_table(ds.get_table("u"), ds_copy.get_table("u"))

    @raises(DazzleError)
    def test_copy02(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        ds.copy("/temp/bim/bam/boum")

    @raises(DazzleFileOrDirExistsError)
    def test_copy03(self):
        ds = DataSet("/temp/dazzle-test", force_create=True)
        ds.copy("/temp/dazzle-test")

if __name__ == '__main__':
    unittest.main()
