import pandas as pd
# import bcolz
# import numpy as np
from dazzle.core.utils import *
from dazzle.core.column import *
from prettytable import PrettyTable


class Table(object):
    """A Table is mainly a wrapper over a bcolz ctable.
    It has a name, which is the same as the ctable name, and a list of columns, each of which
    is a wrapper over a bcolz carray. """

    """:type _ctable: bcolz.ctable"""   # let's make PyCharm happy

    """the default value for bcolz expectedlen parameter used for creating tables and columns"""
    EXPECTED_LENGTH = 10**8

    def __init__(self, name, dataset, columns_attrs, force_create=False, expected_length=None, compression_params=None,
                 mode='create'):
        """Initialize this Table.

        Parameters
        ----------
        name :
            name of this Table.
        dataset :
            The DataSet which this Table belongs to.
        column_attrs :
            a list of pairs (column name, value: list | np.ndarray)
        force_create:
            indicates whether Table should overwrite on existing one
        expected_length: expected number of rows of this Table; taken from bcolz. Serve to
            decide the best `chunklen` used for compression and memory I/O purposes.
        compression_params:
            bcolz compression parameters. Don't use unless you know what you are doing.
        mode:
            initialization mode: 'create' (default), 'copy', 'read_csv', 'pandas'

        """
        # Build structure

        self._name = name
        self._columns = []
        self._dataset = dataset
        self._ctable = None

        self._expected_length = expected_length if expected_length is not None else Table.EXPECTED_LENGTH
        self._compression_params = compression_params if compression_params is not None else dataset.compression_params

        # Check parameters

        if not re.match("[A-Za-z][_a-zA-Z0-9]*$", name):
            raise DazzleError("Invalid table identifier: '%s' in %s()" % (name, method_name()))

        if mode == 'open' and force_create:
            raise DazzleError("mode='open' and force_create=True cannot be used together")

        if mode != 'open' and not force_create and os.path.exists(self.data_dir):
            raise DazzleError("Specified path '%s' exists. Use force_create=True to override" % self.data_dir)

        if mode != 'create':
            if type(columns_attrs) != list or columns_attrs != []:
                raise ValueError("'columns_attrs' should be a [] for this mode")
        else:
            if type(columns_attrs) != list:
                raise ValueError("'columns_attrs' should be a list of pairs (column name, value)")

            if len(columns_attrs) == 0:
                raise DazzleError("A table should be created with at least one column in %s()" % (method_name()))

            # Build ctable from parameters

            carrays = []
            for cattr in columns_attrs:

                if not isinstance(cattr, tuple) or len(cattr) != 2 or not isinstance(cattr[0], str) \
                        or (not isinstance(cattr[1], list) and type(cattr[1]) != np.ndarray):
                    raise ValueError("'columns_attrs' should be a list of pairs (column name, list | np.ndarray)")

                npa = None
                if isinstance(cattr[1], list):
                    npa = np.array(cattr[1])
                elif isinstance(cattr[1], np.ndarray):
                    npa = cattr[1]

                if (not np.issubdtype(npa.dtype, np.int) or np.dtype(npa.dtype).kind == 'u') and not np.issubdtype(npa.dtype, np.float):
                    raise DazzleError("Type not supported in Dazzle: %s" % npa.dtype)
                else:
                    carrays.append(npa)

            col_names = [cattr[0] for cattr in columns_attrs]

            self._ctable = bcolz.ctable(columns=list(carrays), names=col_names, rootdir=self.data_dir, mode='w',
                                        expectedlen=self._expected_length, cparams=self._compression_params)

            # Build network of Dazzle objects from ctable

            self._build_columns_from_ctable()

            self._dataset._add_table(self)
            self._dataset.save()

    @property
    def name(self):
        """Return name of this Table
        """
        return self._name

    @property
    def columns(self):
        """Return list of Columns of this Table
        """
        return self._columns

    @property
    def dataset(self):
        """Return dataset of this Table
        """
        return self._dataset

    @property
    def data_dir(self):
        """Return directory where tables of this table are located
        """
        return os.path.join(self._dataset.data_dir, self._name)

    @property
    def ctable(self):
        """Return bcolz ctable of this Table
        """
        return self._ctable

    @property
    def expected_length(self):
        """Return _expected_length of this Table
        """
        return self._expected_length

    @staticmethod
    def copy(name, dataset, data_dir, force_create=False):
        """Copy an existing Table to dataset.
           Data files from are copied from 'data_dir' into dataset directory.
        """
        if not os.path.exists(data_dir):
            raise DazzleError("Specified path '%s' does not exist." % data_dir)

        dest_dir = os.path.join(dataset.data_dir, name)
        if not force_create and os.path.exists(dest_dir):
            raise DazzleError("Specified path '%s' exists. Use force_create=True to override" % dest_dir)

        table = Table(name, dataset, [], mode='copy')
        ctable = bcolz.open(data_dir)
        table._ctable = ctable.copy(rootdir=table.data_dir, expectedlen=len(ctable), cparams=dataset.compression_params)
        table._build_columns_from_ctable()

        table._dataset._add_table(table)
        table._dataset.save()

        return table

    @staticmethod
    def from_csv(table_name, dataset, csv_file, **kwargs):  # csv_params, filter=None, force_create=False):
        """Create a Table from data read in CSV file.
           Use pandas CSV reader for implementation.
           Parameters in kwargs are the same as those from pandas.read_csv() except for the following one:
                row_filter:
                    a numexpr string serving as filtering the rows that are to be imported in the Table
                    # TODO be more precise
                verbose:
                    indicates whether feedback should be given to user through logging
                force_create:
                    indicates whether Table should overwrite on existing one
                expected_length: expected number of rows of this Table; taken from bcolz. Serve to
                    decide the best `chunklen` used for compression and memory I/O purposes.
                compression_params:
                    bcolz compression parameters. Don't use unless you know what you are doing.
        """

        # Extract parameters

        usecols = kwargs.get('usecols', None)
        if usecols is None:
            raise DazzleError("You must specify 'usecols' parameter")

        kwargs['chunksize'] = kwargs.get('chunksize', 10**7)
        force_create = kwargs.pop('force_create', False)
        row_filter = kwargs.pop('filter', None)
        expected_length = kwargs.pop('expected_length', file_rough_lines_count(csv_file))
        compression_params = kwargs.pop('compression_params', dataset.compression_params)
        verbose = kwargs.pop('verbose', True)

        if verbose:
            logger.info("Reading %s from CSV file ..." % table_name)

        # Build Table from CSV file
        # Rows are built using pandas reader, by chunks so as to minimze RAM usage

        table = Table(table_name, dataset, [], expected_length=expected_length, force_create=force_create,
                      compression_params=compression_params, mode='read_csv')

        with open(csv_file, 'rb') as f:
            iter_csv = pd.read_csv(f, **kwargs)
            i = 0
            for chunk in iter_csv:
                filtered_chunk = chunk if row_filter is None else chunk.query(row_filter)
                if i == 0:
                    table._ctable = bcolz.ctable.fromdataframe(filtered_chunk, rootdir=table.data_dir, mode='w',
                                                               expectedlen=expected_length, cparams=compression_params)
                else:
                    table._ctable.append(bcolz.ctable.fromdataframe(filtered_chunk))

                if verbose:
                    logger.info("Chunk #%d processed" % (i+1))

                i += 1

            table._build_columns_from_ctable()
            table._dataset._add_table(table)
            table._dataset.save()

            # Report

            if verbose:
                logger.info("Processed %s" % table_name)
                logger.info("%s" % table.short_descr())
                if len(table._ctable) < 1000000:
                    logger.info("Processed %s - %d rows" % (csv_file, len(table._ctable)))
                else:
                    logger.info("Processed %s - %.1fM rows" % (csv_file, len(table._ctable) / 1000000))

            return table

    @staticmethod
    def from_dataframe(table_name, dataset, df, **kwargs):
        """Create a Table from pandas dataframe.
        """
        # Extract parameters

        force_create = kwargs.pop('force_create', False)
        expected_length = kwargs.pop('expected_length', len(df))
        compression_params = kwargs.pop('compression_params', dataset.compression_params)

        # Build Table from dataframe

        table = Table(table_name, dataset, [], expected_length=expected_length, force_create=force_create,
                      compression_params=compression_params, mode='pandas')

        table._ctable = bcolz.ctable.fromdataframe(df)
        table._build_columns_from_ctable()
        table._dataset._add_table(table)
        table._dataset.save()

        return table

    def _build_columns_from_ctable(self):
        """Build internals of Table from bcolz ctable
        """
        for c in range(len(self._ctable.cols)):
            col_name = self._ctable.names[c]

            # When this method is executed, we don't know if the columns will actually be Ref_Columns or Literal_Columns
            # so, we create Literal_Column so that 0 is not interpreted as nan
            LiteralColumn(col_name, self)

    def get_column(self, col_name):
        """Return Column by its name, or None if no column with name col_name is found
        """
        col = next((column for column in self._columns if column.name == col_name), None)
        return col

    def remove_column(self, col_name):
        """Remove a column from this table
        """
        column = self.get_column(col_name)
        if column is None:
            raise ValueError("There is no '%s' column in table '%s', in %s" % (col_name, self.name, method_name()))

        self._ctable.delcol(col_name)
        self._columns.remove(column)

    def to_dataframe(self):
        """Return a pandas dataframe from this table
        """
        return self._ctable.todataframe()

    def append(self, data):  # TODO check type compatibility + data
        """Append data to this Table.
         Data are specified by dict: {'col_name': list | ndarray}
        """
        if isinstance(data, dict):
            if len(data) != len(self._columns):
                raise ValueError("Dict should have %d value(s) in %s" % (len(self._columns), method_name()))

            if len(data) >= 1:
                col_names = []
                carrays = []
                for col_name, value in data.items():
                    col_names.append(col_name)
                    carrays.append(value)

                to_append = bcolz.ctable(columns=list(carrays), names=col_names)
                self._ctable.append(to_append)
        else:
            raise ValueError("Values to be appended must be specified by a dict in %s" % (method_name()))

    def __getitem__(self, *args):
        """Array-like read accessor. Delegate to ctable
        """
        return self._ctable.__getitem__(*args)      # TODO should return a Column and not a carray

    def __setitem__(self, key, value):
        """Array-like write accessor. Delegate to ctable
        """
        return self._ctable.__setitem__(key, value)     # TODO check data

    def __array__(self):
        """Array interface
        """
        return self._ctable[:]

    def __str__(self, head=5, tail=3):
        """Return the string representation (structure and data) of this Table.
        Use PrettyTable (https://code.google.com/p/prettytable/) for formatting
        """
        pt = PrettyTable()
        for col in self._columns:
            if len(col.carray) <= head + tail:
                vals = col.str_values()
            else:
                l = len(col.carray)
                vals = col.str_values(head, tail)
            pt.add_column(col.name, vals)

        pt.align = "r"
        return "\n" + self.short_descr() + "\n" + pt.get_string()

    def head(self, size=10):
        """Return the string representation of this Table: structure and first 'size' rows.
        """
        return self.__str__(head=size, tail=0)

    def tail(self, size=10):
        """Return the string representation of this Table: structure and last 'size' rows.
        """
        return self.__str__(head=0, tail=size)

    def short_descr(self):
        """Return a string summary of this Table: structure and compression stats.
        """
        out = self._name + "(" + ", ".join([col.name + ": " + col.type_name() for col in self._columns]) + ")\n" \
            + '{0:,}'.format(len(self._ctable)) + " row(s) - compressed: " + ("%.2f MB" % (self._ctable.cbytes / (1024 * 1024))) \
            + " - comp. ratio: " + ("%.2f" % (self._ctable.nbytes / self._ctable.cbytes))
        return out

    def rebuild(self, schema):
        """Re-structure a table after it has been loaded, using schema {"column name": dtype}. Only int and float
        dtypes are presently supported.

            1. Each column is set with desired dtype ;
            2. A nan row is inserted as first row: required for referencing the table (1st row corresponds to nan)
            3. Data are checked and nan values are converted to their appropriate value, for each column
        """
        i = 0
        ctable = None
        for col in self.columns:

            # dtype = np.dtype(self.appropriate_numeric_dtype(col)).name  # no longer used: 2/3 of the time spent
            # for rebuilding  is used here. Should do something about that
            dtype = schema.get(col.name, col.dtype)

            if (np.issubdtype(dtype, np.int) and np.dtype(dtype).kind != 'u') or np.issubdtype(dtype, np.float):

                # Re-create column with desired type, inserting nan as first row

                nan = col.nan_value(dtype)
                carray = bcolz.carray([nan], dtype=col.type_name(dtype), expectedlen=len(col.carray))

                # Convert existing nan values and append to re-created column

                for block in col.blocks:
                    block[np.isnan(block)] = nan
                    carray.append(block)    # TODO check data + check that former values to not contain nan
            else:
                raise DazzleError("Type not supported: in Dazzle: %s" % col.dtype)

            # Since a bcolz ctable cannot be created with an empty columns set, we create ctable after 1st column
            # has been rebuilt, and we append, for subsequent columns
            if i == 0:
                ctable = bcolz.ctable(columns=[carray], names=[col.name], rootdir=self.ctable.rootdir + "_tmp",
                                      expectedlen=len(self.columns[0].carray), mode='w')
            else:
                ctable.addcol(carray, col.name)

            i += 1
        self.ctable.flush()
        os.rename(self.ctable.rootdir, self.ctable.rootdir + "-raw")

        ctable.flush()
        os.rename(self.ctable.rootdir + "_tmp", self.ctable.rootdir)

        self._ctable = bcolz.open(self.data_dir)

    # @staticmethod
    # def appropriate_numeric_dtype(col):
    #     """Return the minimal dtype required for containing data of this Table"
    #         Note: no longer used because too slow; user has should specify desired columns' dtypes
    #     """
    #     dtype = None
    #     for block in col.blocks:
    #         block[np.isnan(block)] = 0
    #         dtype = Table._min_numeric_dtype(block, dtype)
    #     return dtype
    #
    # @staticmethod
    # def _min_numeric_dtype(data, init_dtype):
    #     """Auxiliary method for appropriate_numeric_dtype().
    #         Return the minimal dtype required for containing data of this Table"
    #         Note: no longer used
    #     """
    #     if init_dtype is None:
    #         conv = np.ndarray.astype(data, np.int64)
    #         if not np.allclose(conv, data):
    #             return Table._min_numeric_dtype(data, np.float32)
    #         else:
    #             return Table._min_numeric_dtype(data, np.int8)
    #     elif init_dtype == np.float64:
    #         return np.float64
    #     else:
    #         conv = np.ndarray.astype(data, init_dtype)
    #         if not np.allclose(conv, data):
    #             return Table._min_numeric_dtype(data, Table._next_numeric_dtype(init_dtype))
    #         else:
    #             return init_dtype
    #
    # @staticmethod
    # def _next_numeric_dtype(dtype):
    #     """Auxiliary method for _min_numeric_dtype().
    #         Return the minimal dtype required for containing data of this Table"
    #         Note: no longer used
    #     """
    #     if dtype == np.int8:
    #         return np.int16
    #     elif dtype == np.int16:
    #         return np.int32
    #     elif dtype == np.int32:
    #         return np.int64
    #     elif dtype == np.int64:
    #         return np.float32
    #     else:
    #         return np.float64

    def add_reference_column(self, foreign_col, primary_col):
        """Add a column to this Table containing the indices of rows for column foreign_col instead of IDs.
            Referenced IDs are contained in Table column primary_col.

            Presently Dazzle uses pandas indexing mechanism for building the correspondence (ID -> index).
            This code must be re-written as a Cython extension
        """
        foreign_col_df = self._ctable[[foreign_col.name]].todataframe()

        # build pandas index to have an (ID -> index) translation table

        idx = primary_col.table.ctable[[primary_col.name]].todataframe()
        idx_name = primary_col.table.name + "_idx"
        idx[idx_name] = np.arange(len(primary_col.table.ctable), dtype=np.int32)
        idx.set_index([primary_col.name], inplace=True)

        refs = foreign_col_df[foreign_col.name].map(idx[idx_name])
        nan_count = np.isnan(refs.values).sum()

        # note: str(nan_count), otherwise PyCharm issues a warning
        logger.info(str(nan_count) + " row(s) of '%s' will have a NA value in '%s': '%s' has no corresponding value in '%s'"
                    % (self.name, primary_col.name + "_ref", primary_col.name, foreign_col.table.name))

        ref_col_name = foreign_col.name + "_ref"
        ref_data = np.array(np.nan_to_num(refs.values), dtype=np.int32)
        _ = RefColumn(ref_col_name, self, data=ref_data)

    def add_join_column(self, result_col_name, join_path):
        """Perform a sequence of joins specified in join_path.
            join_path is a list of Columns Ci (i=1..n) such that:
             - C1 is a RefColumn of this Table
             - Ci (1 < i < n) is a RefColumn referencing the table of Column Ci-1
             - Cn is the target column to be added to this Table
             Example:
                T(a, b, c)
                A(aID, d, e)
                D(dID, g)

             After rebuild:
                T(a, b, c, a_ref)
                A(aID, d, e, d_ref)
                D(dID, g)

             T.add_join_column("res", [T.a_ref, A.d_ref, D.g]) will perform the equivalent of the following SQL query:
                SELECT g AS res
                FROM T, A, D
                WHERE T.a_ref = A.index
                AND A.d_ref = D.index

            In the above query, we assume that each table has an implicit 'index' field that is the positional index
            of each record
        """
        temp = join_path[0]
        for col in join_path[1:]:
            print("Joining with " + col.table.name + "." + col.name + " ...")
            x = col.carray[:]
            temp2 = ResultColumn(np.array([], dtype=col.dtype))
            for block in temp.blocks:
                y = x[block]
                temp2.append(y)
            temp = temp2

        return LiteralColumn(result_col_name, self, data=temp.carray)

# Dummy tests

if __name__ == '__main__':

    def test1():
        from dazzle.core.dataset import DataSet

        test_dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(test_dir, force_create=True)
        t = Table("t", ds, [("a", [4, 11, 2, 4, 7, 1, 156, 90, 25, -2, -292]),
                            ("b", [1.63654, 0, 7.875675, 2.9786, 4.4242, 7e-2, 1.978987, 156.65468, -90.8479684, 25.541,
                                   -2.787])])
        # t = Table("u", ds, [("a", [4, 11, 2, 4, 7]),
        #                     ("b", [False, True, False, True, False]),
        #                     ("c", [1, 7, 11, 2, 4])])
        a = t.get_column("a")
        print(t.head(20))

    def test2():
        from dazzle.core.dataset import DataSet

        avito_data_dir = "/github/kaggle-fiddling/avito-context-ad-clicks/data"

        test_dir = os.path.join("/temp", "dazzle-test")
        ds = DataSet(test_dir, force_create=True)

        t = Table.from_csv("Category", ds, os.path.join(avito_data_dir, "Category.tsv"), delimiter='\t',
                           usecols=['CategoryID', 'ParentCategoryID', 'Level'])
        print(t.head(20))

    test1()