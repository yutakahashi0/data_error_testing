#!/usr/bin/python3

import pytest
import numpy as np

from ...src.checktable import checktable as ct

format_file  = 'dataconf/colors.yml'
rsdtype_file = 'dataconf/colors.csv'
rawdata_file = 'data/colors.tsv'

colors = ct.CheckTable(format_file, rsdtype_file, rawdata_file)

class Test_max_sizes:
    def test_value(self):
        assert colors.max_sizes() == {'color_name': 17, 'hex': 7, 'rgb': 11, 
                                      'pinteger': 8674000000, 'realnumber': None,
                                      'all_null': np.nan}

class Test_count_nan:
    def test_value(self):
        assert colors.count_nan() == {'color_name': 0, 'hex': 0, 'rgb': 1,
                                      'pinteger': 1, 'all_null': 10}

class Test_size_error:
    def test_value(self):
        assert colors.size_error() == ['color_name', 'pinteger']

class Test_not_null_constraint_error:
    def test_value(self):
        assert colors.not_null_constraint_error() == ['rgb', 'pinteger', 'all_null']

class Test_superkey_error:
    def test_value(self):
        assert colors.superkey_error() is True