from dataset_loader import DataSet
from dataset_loader import Table
from dataset_loader import Column
import pandas as pd
import bcolz

# -----------------------------------------------

class TrainSearchStream(Table):

    def csv_params(self):
          return {'file': "trainSearchStream.tsv",
                  'delimiter': '\t',
                  'chunksize': 10**7,
                  'dtype': {'SearchID':'u4', 'AdID':'u4', 'Position':'u1', 'ObjectType':'u1', 'HistCTR':'f4', 'IsClick':'f1'}
                  }

    def filter_chunk(self, chunk):
        return chunk[(chunk['ObjectType'] == 3) & (pd.notnull(chunk['IsClick']))]

    def process(self):
        isClick = self.column('IsClick')
        isClick.fillna(0)
        isClick.set_type("u1")

class AdsInfo(Table):

    def csv_params(self):
        return {'file': "AdsInfo.tsv",
                'delimiter': '\t',
                'chunksize': 10**7,
                'dtype': {'AdID':'u4', 'CategoryID':'f4'}
                }

    def process(self):
        cat = self.column('CategoryID')
        cat.replace_list([250001, 250002, 250003, 250004, 250005, 250006, 500001], [61, 62, 63, 64, 65, 66, 67])
        cat.fillna(0)
        cat.set_type("u1")

class SearchInfo(Table):

    def csv_params(self):
        return {'file': "SearchInfo.tsv",
                'delimiter': '\t',
                'chunksize': 10**7,
                'dtype': {'SearchID':'u4', 'IPID':'u4', 'UserID':'f4', 'IsUserLoggedOn':'f1',
                                       'LocationID':'f4', 'CategoryID':'f4'}
                }

    def process(self):
        cat = self.column('CategoryID')
        cat.replace_list([250001, 250002, 250003, 250004, 250005, 250006, 500001], [61, 62, 63, 64, 65, 66, 67])
        cat.fillna(0)
        cat.set_type("u1")

class Category(Table):

    def csv_params(self):
        return {'file': "Category.tsv",
                'delimiter': '\t',
                'chunksize': 10**7,
                'dtype': {'CategoryID':'u4', 'ParentCategoryID':'u1', 'Level': 'u1'}
                }

    def process(self):
        cat = self.column('CategoryID')
        cat.replace_list([250001, 250002, 250003, 250004, 250005, 250006, 500001], [61, 62, 63, 64, 65, 66, 67])
        cat.set_type("u1")


class UserInfo(Table):

    def csv_params(self):
        return {'file': "UserInfo.tsv",
                'delimiter': '\t',
                'chunksize': 10**7,
                'dtype': {'UserID':'u4', 'UserAgentID':'u4', 'UserAgentOSID':'u4',
                                    'UserDeviceID':'u4', 'UserAgentFamilyID':'u4'}
                }

# -----------------------------------------------

dir = "C:/github/kaggle-fiddling/avito-context-ad-clicks/data"

def create_dataset(dir):

    dataset = DataSet()
    dataset.set_csv_dir(dir)

    trainSearchStream = TrainSearchStream('trainSearchStream',
                                            [Column('SearchID','u4'),
                                             Column('AdID','u4'),
                                             Column('Position','u1'),
                                             Column('ObjectType','u1'),
                                             Column('HistCTR','f4'),
                                             Column('IsClick','u1')])

    adsInfo = AdsInfo('AdsInfo',
                            [Column('SearchID','u4'),
                            Column('AdID','u4'),
                            Column('CategoryID','u4')])

    searchInfo = SearchInfo('SearchInfo',
                                [Column('SearchID','u4'),
                                Column('IPID','u4'),
                                Column('UserID','u4'),
                                Column('IsUserLoggedOn','u1'),
                                Column('LocationID','u4'),
                                Column('CategoryID','u4')])

    category = Category('Category',
                            [Column('CategoryID','u1'),
                            Column('ParentCategoryID','u1'),
                            Column('Level','u1')])

    userInfo = UserInfo('UserInfo',
                            [Column('UserID','u1'),
                            Column('UserAgentID','u1'),
                            Column('UserAgentOSID','u1'),
                            Column('UserDeviceID','u1'),
                            Column('UserAgentFamilyID','u1')])

    dataset.set_tables([trainSearchStream, adsInfo, searchInfo, category, userInfo])

    return dataset


def check_dataset(ds):
    for t in ds._tables:
        tbz = bcolz.open(ds._bcolz_dir + "/" + t._name)
        print("%s: %s \nsize: %d rows - uncompressed: %d MB  - compressed: %d MB\n" % (t._name, tbz.dtype, tbz.size, tbz.nbytes / (1024 * 1024), tbz.cbytes / (1024 * 1024)))


def open_table(table_name, bcolz_dir):
    ds = create_dataset(dir)
    table = ds.get_table(table_name)
    table._bz_table = bcolz.open(dir + "/" + bcolz_dir + "/" + table_name)
    return table

def test1():
    ds = create_dataset(dir)

    # directory must have been created before
    ds.from_csv(dir, dir + "/avito-raw")

    check_dataset(ds)

def test2():
    c = open_table('Category', 'avito-raw')
    cat = c.column('CategoryID')
    #cat.replace_value(250001, 61)
    cat.replace_list([250001, 250002, 250003, 250004, 250005, 250006, 500001], [61, 62, 63, 64, 65, 66, 67])
    print(cat.hasna())
    cat.set_type("u1")

    c = open_table('Category', 'avito-raw')
    cat = c.column('CategoryID')
    print(cat.bz_col)

def test3():
    c = open_table('Category', 'avito-raw')
    cat = c.column('CategoryID')
    print(cat.bz_col)

if __name__ == '__main__':
    test2()
