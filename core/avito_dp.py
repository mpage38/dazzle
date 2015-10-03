import bcolz
from dazzle.core.dataset import DataSet
from dazzle.core.table import Table
import numpy as np

csv_dir = "/github/kaggle-fiddling/avito-context-ad-clicks/data"
raw_dir = "/temp/avito-raw"     # where raw dataset is created
dataset_dir = "/temp/avito1"    # where "final" dataset is created

def load_dataset():
    """'Raw'-dataset is the result of loading the CSV sources data into dazzle tables, only filtering out data
    that we don't want to further process.

    The method is programmed in a non-destructive way so as to be able to launch it several times
    before getting the job done.
    """
    import os
    from dazzle.core.dataset import DataSet



    if DataSet.exists(raw_dir):
        ds = DataSet.open(raw_dir)
    else:
        ds = DataSet(raw_dir, force_create=True)

    # Notes:
    # - many of the following attributes should be unsigned int instead of signed int, but numexpr works only on
    # signed data.
    # - Simlarly to pandas, we use the types required to contain the existing data, not the types we desire to use
    if ds.get_table("Category") is None:
        t = Table.from_csv("Category", ds, os.path.join(csv_dir, "Category.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['CategoryID', 'ParentCategoryID', 'Level'],
                           dtype={'CategoryID': 'i4', 'ParentCategoryID': 'i1', 'Level': 'i1'})
        t = None

    # Notice the filter attribute that does not exist in pandas.read_csv(). It makes it possible to skip some rows
    # based on a numexpr expression. IsClick == IsClick is true iff IsClick is not na
    if ds.get_table("TrainSearchStream") is None:
        t = Table.from_csv("TrainSearchStream", ds, os.path.join(csv_dir, "trainSearchStream.tsv"), delimiter='\t', chunksize=10**7,
                         usecols=['SearchID', 'AdID', 'Position', 'ObjectType', 'HistCTR', 'IsClick'],
                         dtype={'SearchID':'i4', 'AdID':'i4', 'Position':'i1', 'ObjectType':'i1', 'HistCTR':'f4', 'IsClick':'f1'},
                         filter='(ObjectType == 3) & (IsClick == IsClick)')
        t = None

    # We avoid to load the string fields. We will see this problem later with Don
    if ds.get_table("AdsInfo") is None:
        t = Table.from_csv("AdsInfo", ds, os.path.join(csv_dir, "AdsInfo.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['AdID', 'LocationID', 'CategoryID', 'Price', 'IsContext'],
                           dtype={'AdID':'i4', 'LocationID':'f4', 'CategoryID':'f4', 'Price': 'f4', 'IsContext': 'f1'})
        t = None

    # We avoid to load the string fields. We will see this problem later with Don
    if ds.get_table("SearchInfo") is None:
        t = Table.from_csv("SearchInfo", ds, os.path.join(csv_dir, "SearchInfo.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['SearchID', 'IPID', 'UserID', 'IsUserLoggedOn', 'LocationID', 'CategoryID'],
                           dtype={'SearchID':'i4', 'IPID':'i4', 'UserID':'f4', 'IsUserLoggedOn':'f1',
                                       'LocationID':'f4', 'CategoryID':'f4'})
        t = None

    if ds.get_table("userInfo") is None:
        t = Table.from_csv("userInfo", ds, os.path.join(csv_dir, "userInfo.tsv"), delimiter='\t', chunksize=10**7,
                            usecols=['UserID', 'UserAgentID', 'UserAgentOSID','UserDeviceID', 'UserAgentFamilyID'],
                            dtype={'UserID':'i4', 'UserAgentID':'i4', 'UserAgentOSID':'i4',
                                   'UserDeviceID':'i4', 'UserAgentFamilyID':'i4'})
        t = None

    if ds.get_table("Location") is None:
        t = Table.from_csv("Location", ds, os.path.join(csv_dir, "Location.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['LocationID', 'CityID', 'RegionID'],
                           dtype={'LocationID': 'i4', 'CityID':'f4', 'RegionID': 'f4'})
        t = None

    if ds.get_table("PhoneRequestsStream") is None:
        t = Table.from_csv("PhoneRequestsStream", ds, os.path.join(csv_dir, "PhoneRequestsStream.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['UserID', 'IPID', 'AdID', 'PhoneRequestDate'],
                           dtype={'UserID':'i4', 'IPID':'i4', 'AdID':'i4', 'PhoneRequestDate': 'object'})
        t = None

    if ds.get_table("VisitsStream") is None:
        t = Table.from_csv("VisitsStream", ds, os.path.join(csv_dir, "VisitsStream.tsv"), delimiter='\t', chunksize=10**7,
                           usecols=['UserID', 'IPID', 'AdID', 'ViewDate'],
                           dtype={'UserID':'i4', 'IPID':'i4', 'AdID':'i4', 'ViewDate': 'object'})
        t = None

    return ds

#load_dataset()


def check_raw_dataset():
    # Category(CategoryID: int32, Level: int8, ParentCategoryID: int8): 68 row(s) - compressed: 0.06 MB - comp. ratio: 0.01
    # TrainSearchStream(SearchID: int32, AdID: int32, Position: int8, ObjectType: int8, HistCTR: float32, IsClick: float32): 190,157,735 row(s) - compressed: 1479.89 MB - comp. ratio: 2.21
    # AdsInfo(AdID: int32, LocationID: float32, CategoryID: float32, Price: float32, IsContext: float32): 36,893,298 row(s) - compressed: 280.61 MB - comp. ratio: 2.51
    # SearchInfo(SearchID: int32, IPID: int32, UserID: float32, IsUserLoggedOn: float32, LocationID: float32, CategoryID: float32): 91,019,228 row(s) - compressed: 1043.73 MB - comp. ratio: 2.00
    # userInfo(UserID: int32, UserAgentID: int32, UserAgentOSID: int32, UserDeviceID: int32, UserAgentFamilyID: int32): 4,284,823 row(s) - compressed: 20.32 MB - comp. ratio: 4.02
    # Location(LocationID: int32, RegionID: float32, CityID: float32): 4,080 row(s) - compressed: 0.38 MB - comp. ratio: 0.12
    # PhoneRequestsStream(UserID: int32, IPID: int32, AdID: int32, PhoneRequestDate: bytes168): 13,717,580 row(s) - compressed: 139.27 MB - comp. ratio: 3.10
    # VisitsStream(UserID: int32, IPID: int32, AdID: int32, ViewDate: bytes168): 286,821,375 row(s) - compressed: 2548.20 MB - comp. ratio: 3.54
    ds = DataSet.open(raw_dir)
    for table in ds.tables:
        print(table.short_descr())

#check_raw_dataset()

def preprocess_dataset():
    # This step takes around 3 mins (6 mins if dataset must be copied)
    #
    # Table.add_reference_column(), which uses pandas is partly responsible for this. In addition,
    # it consumes a lot of RAM.

    # 1. Make a copy of the raw dataset, if this has not already be done: we don't want to reload the whole CSV stuff
    # if something wrong happens

    if not DataSet.exists(dataset_dir):
        print("Copying dataset ...")
        raw_ds = DataSet.open(raw_dir)
        ds = raw_ds.copy(dataset_dir)     # almost 3 mins !
    else:
        ds = DataSet.open(dataset_dir)

    # 2. Rebuild each table. This means:
    #
    # 2.1 inserting a nan row at the head of each table. This is necessary because we use index=0 in each RefColumn
    #     for indicating a null reference
    #
    # 2.2 assigning the desired dtype of each column
    #
    # 2.3 Setting data in each column using the setting dtype
    #
    # 2.4 Replace Numpy NA values by those of the corresponding column class (Ref/Literal) and dtype
    #

    Category = ds.get_table("Category")
    Location = ds.get_table("Location")
    userInfo = ds.get_table("userInfo")
    AdsInfo = ds.get_table("AdsInfo")
    SearchInfo = ds.get_table("SearchInfo")
    TrainSearchStream = ds.get_table("TrainSearchStream")

    print("Re-building tables with given dtypes ...")

    Category.rebuild({"CategoryID": np.int32, "Level": np.int8, "ParentCategoryID": np.int32})
    Location.rebuild({"LocationID": np.int16, "RegionID": np.int8, "CityID": np.int16})
    userInfo.rebuild({"UserID": np.int32, "UserAgentID": np.int32, "UserAgentOSID": np.int8,
                        "UserDeviceID": np.int16, "UserAgentFamilyID": np.int8})
    AdsInfo.rebuild({"AdID": np.int32, "LocationID": np.int16, "CategoryID": np.int32, "Price": np.float32,
                     "IsContext": np.int8})
    SearchInfo.rebuild({"SearchID": np.int32, "IPID": np.int32, "UserID": np.int32, "IsUserLoggedOn": np.int8,
                          "LocationID": np.int16, "CategoryID": np.int32})
    TrainSearchStream.rebuild({"SearchID": np.int32, "AdID": np.int32, "Position": np.int8,
                                 "ObjectType": np.int8, "HistCTR": np.float32, "IsClick": np.int8})

    # 3. Add references between columns: foreign keys (like LocationID in AdsInfo) are kept
    # but an additional column (xxx_ref) is added with the index of the row containing the referenced value
    #

    print("Building references from AdsInfo ...")

    AdsInfo.add_reference_column(AdsInfo.get_column("LocationID"), Location.get_column("LocationID"))
    AdsInfo.add_reference_column(AdsInfo.get_column("CategoryID"), Category.get_column("CategoryID"))
    print(AdsInfo)

    print("Building references from SearchInfo ...")

    SearchInfo.add_reference_column(SearchInfo.get_column("UserID"), userInfo.get_column("UserID"))
    SearchInfo.add_reference_column(SearchInfo.get_column("LocationID"), Location.get_column("LocationID"))
    SearchInfo.add_reference_column(SearchInfo.get_column("CategoryID"), Category.get_column("CategoryID"))

    print("Building references from TrainSearchStream ...")

    TrainSearchStream.add_reference_column(TrainSearchStream.get_column("SearchID"), SearchInfo.get_column("SearchID"))
    TrainSearchStream.add_reference_column(TrainSearchStream.get_column("AdID"), AdsInfo.get_column("AdID"))

    print(TrainSearchStream)
    print("Done")

# preprocess_dataset()

def compute_category_similarity():
    ds = DataSet.open(dataset_dir)

    TrainSearchStream = ds.get_table("TrainSearchStream")
    AdsInfo = ds.get_table("AdsInfo")
    Category = ds.get_table("Category")
    SearchInfo = ds.get_table("SearchInfo")

    TrainSearchStream.add_join_column("AdCategoryID", [TrainSearchStream.get_column("AdID_ref"),
                                                          AdsInfo.get_column("CategoryID_ref")])
    TrainSearchStream.add_join_column("AdCategoryLevel", [TrainSearchStream.get_column("AdID_ref"),
                                                          AdsInfo.get_column("CategoryID_ref"),
                                                          Category.get_column("Level")])
    TrainSearchStream.add_join_column("AdCategoryParentID", [TrainSearchStream.get_column("AdID_ref"),
                                                          AdsInfo.get_column("CategoryID_ref"),
                                                          Category.get_column("ParentCategoryID")])

    TrainSearchStream.add_join_column("SearchCategoryID", [TrainSearchStream.get_column("SearchID_ref"),
                                                          SearchInfo.get_column("CategoryID_ref")])
    TrainSearchStream.add_join_column("SearchCategoryLevel", [TrainSearchStream.get_column("SearchID_ref"),
                                                          SearchInfo.get_column("CategoryID_ref"),
                                                          Category.get_column("Level")])
    TrainSearchStream.add_join_column("SearchCategoryParentID", [TrainSearchStream.get_column("SearchID_ref"),
                                                          SearchInfo.get_column("CategoryID_ref"),
                                                          Category.get_column("ParentCategoryID")])
    print(TrainSearchStream)

    # unfinished : similarity evaluation

#compute_category_similarity()


def test_sum():
    import time
    ds = DataSet.open(dataset_dir)
    tss = ds.get_table("TrainSearchStream")
    p = tss.get_column("Position")

    t = time.time()
    print(p.sum())
    print(time.time() - t)
test_sum()