from os.path import dirname
import json
import numpy as np
from dazzle.core.utils import *
from dazzle.core.table import Table


class DataSet(object):
    """A DataSet represents a collection of tables.

    A DataSet is associated with a directory (data_dir) and the content description of the DataSet is stored
    in the file 'dataset.json', in 'data_dir'.

    Each table data is stored in a separate directory, and each of these tables'
    directories is a subdirectory of  DataSet's 'data_dir'
    """

    def __init__(self, data_dir, force_create=False, compression_params=None, **kwargs):
        """Initialize this dataset. If data_dir exists, use force_create=True to delete existing directory
           and create DataSet

           Initialization occurs when user creates a Dataset: ds = DataSet("data dir"), when a Dataset is opened, and
           when a DataSet is converted from another format: CSV, DB, Pandas DataFrame
        """
        self._tables = []           # list of Tables that I contain
        self._data_dir = data_dir   # root directory containing dataset.json and bcolz tables subdirectories
        self._compression_params = compression_params if compression_params is not None else bcolz.cparams()  # bcol.caparams object

        if kwargs.get('mode', 'create') == 'create':

            if not force_create and os.path.exists(data_dir):
                raise DazzleFileOrDirExistsError("Specified path '%s' exists. Use force_create=True to override" % data_dir)

            if not os.path.exists(dirname(data_dir)):
                raise DazzleError("Incorrect path: '%s'. '%s' does not exist" % (data_dir, dirname(data_dir)))

            if os.path.exists(data_dir) and force_create:
                rmtree_or_file(data_dir)

            os.mkdir(data_dir)

            self.save()

    @property
    def tables(self):
        """Return list of tables contained in this dataset
        """
        return self._tables

    @property
    def data_dir(self):
        """Return directory where tables of this dataset are located
        """
        return self._data_dir

    @property
    def compression_params(self):
        """Return the default compression parameters that should be used for the tables in this dataset
        """
        return self._compression_params

    @property
    def uncompressed_bytes(self):
        """Return the original (uncompressed) size of this dataset (in bytes).
        """
        return self._get_stats()[0]

    @property
    def compressed_bytes(self):
        """Return the compressed size of this dataset (in bytes)
        """
        return self._get_stats()[1]

    def get_table(self, table_name):
        """Return table by its name, or None if no table with name table_table_name is found
        """
        return next((table for table in self._tables if table.name == table_name), None)

    def _add_table(self, table):
        """Add a table to this dataset.
        This method is protected because it is not part of the Dazzle API: this method is executed by Table(),
        Table.from_csv(), Table.from_dataframe(), Table.copy()
        """
        if self.get_table(table.name) is not None:
            raise DazzleError("Dataset already contains a table named '%s' in %s()" % (table.name, method_name()))

        table._dataset = self
        self._tables.append(table)

        self.save()

    def remove_table(self, table):
        """Remove a table from this dataset (and from disk)
        """
        # Make sure table belongs to this dataset
        if self.get_table(table.name) is None:
            raise DazzleError("Dataset contains no table named '%s' in %s()" % (table.name, method_name()))

        self._tables.remove(table)
        shutil.rmtree(table.data_dir)
        self.save()

    @staticmethod
    def exists(data_dir):
        """Return True iff a DataSet exists in 'data_dir'.
        """
        json_file = os.path.join(data_dir, "dataset.json")
        return os.path.exists(json_file)

    @staticmethod
    def open(data_dir):
        """Open and return an existing DataSet.
        Side effect: open each Table in this dataset.
        """
        json_file = os.path.join(data_dir, "dataset.json")
        if not os.path.exists(json_file):
            raise DazzleError("No 'dataset.json' file found in %s" % data_dir)

        ds = DataSet(data_dir, mode='open')

        with open(json_file, 'rb') as f:
            data = json.loads(f.read().decode('ascii'))
            params = data["compression_params"]
            ds._compression_params = bcolz.cparams(clevel=params["_clevel"], shuffle=params["_shuffle"],
                                                   cname=params["_cname"])
            for table in data["tables"]:
                table = Table(table["name"], ds, [], mode='open')
                table._ctable = bcolz.open(table.data_dir)
                table._build_columns_from_ctable()
                table._dataset._add_table(table)

        ds.save()
        return ds

    def save(self):
        """Save this dataset objects network onto disk
        """
        with open(os.path.join(self.data_dir, "dataset.json"), 'wb') as f:
            f.write(self.to_json().encode('ascii'))
            f.write(b'\n')

    def to_json(self):
        """Return this dataset JSON representation.
        It would be simpler and faster to pickle, but for debugging purpose, it is better to have a human readable form
        """
        tables = []
        for table in self._tables:
            cols = []
            for col in table.columns:
                cols.append({'name': col.name, 'dtype': np.dtype(col.dtype).name})
            tables.append({'name': table.name, 'columns': cols})

        data = {'compression_params': self.compression_params.__dict__, 'tables': tables}
        return json.dumps(data)

    def copy(self, dest_dir, force_create=False):
        """Make a copy of this dataset to dest_dir and return it
        """
        if os.path.exists(dest_dir) and not force_create:
            raise DazzleFileOrDirExistsError("Specified path '%s' exists. Use force_create=True to override" % dest_dir)

        if not os.path.exists(dirname(dest_dir)):
            raise DazzleError("Incorrect path: '%s'. '%s' does not exist" % (dest_dir, dirname(dest_dir)))

        if os.path.exists(dest_dir):
            rmtree_or_file(dest_dir)

        shutil.copytree(self.data_dir, dest_dir)
        return DataSet.open(dest_dir)

    def _get_stats(self):
        """Return some stats (uncompressed_bytes, compressed_bytes, compression_ratio) about this dataset

        Returns
        -------
        out : a (uncompressed_bytes, compressed_bytes, compression_ratio) tuple
            uncompressed_bytes is the number of uncompressed bytes in tables.
            compressed_bytes is the number of compressed bytes.
            compression_ratio is the compression ratio.

        """
        uncompressed_bytes = 0
        compressed_bytes = 0
        for table in self._tables:
            uncompressed_bytes += table.ctable.nbytes
            compressed_bytes += table.ctable.cbytes
        compression_ratio = uncompressed_bytes / float(compressed_bytes)
        out = (uncompressed_bytes, compressed_bytes, compression_ratio)
        return out

    def __str__(self):
        """String representation of this dataset
        """
        s = "Dir: " + self._data_dir
        s += "\nCompression params: " + self._compression_params.__repr__()
        s += "\nTables:\n\t" + "\n\t".join([table.short_descr() for table in self._tables])
        return s


# Dummy tests

if __name__ == '__main__':

    def test1():
        ds1 = DataSet("/temp/dazzle-test", force_create=True)
        Table("t", ds1, [('a', [11, 2]), ('b', [1, 0]), ('c', [3, 4])])
        Table("u", ds1, [("x", []), ("y", [])])
        ds1.save()
        ds2 = DataSet.open("/temp/dazzle-test")
        for table in ds2.tables:
            print(table)

    test1()
