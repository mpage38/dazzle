import unittest
from dazzle.test.test_dataset import TestDataSet
from dazzle.test.test_table import TestTable
from dazzle.test.test_column import TestColumn

class AllTests(unittest.TestCase):
    def suite(self):
        suite1 = unittest.TestLoader().loadTestsFromTestCase(TestDataSet)
        suite2 = unittest.TestLoader().loadTestsFromTestCase(TestTable)
        # suite3 = unittest.TestLoader().loadTestsFromTestCase(TestColumn)
        return unittest.TestSuite([suite1, suite2, ])

def main():
    unittest.main()

if __name__ == '__main__':
    main()
