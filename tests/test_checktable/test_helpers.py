#!/usr/bin/python3

import pytest
import numpy as np

from ...src.checktable import helpers as hlp

# Redshift data types: representative and its aliases (equivalent classes)
redshift_dtypes = {
    'smallint': ['smallint', 'int2'],
    'integer': ['integer', 'int', 'int4'],
    'bigint': ['bigint', 'int8'],
    'decimal': ['decimal', 'numeric'],
    'real': ['real' ,'float4'],
    'double precision': ['double precision', 'float8', 'float'],
    'boolean': ['boolean', 'bool'],
    'char': ['char', 'character', 'nchar', 'bpchar'],
    'varchar': ['varchar', 'character varying', 'nvarchar', 'text'],
    'date': ['date'],
    'timestamp': ['timestamp', 'timestamp without time zone'],
}

# mapping dtypes of numpy to those of redshift
mapping_dtypes = {
    'np.int16': ['smallint',],
    'np.int32': ['integer',],
    'np.int64': ['bigint',],
    'np.float32': ['real',],
    'np.float64': ['decimal', 'double precision'],
    'np.str_': ['boolean', 'char', 'varchar', 'date', 'timestamp']
}

# Combine redshift_dtypes and mapping_dtypes
numpy_redshift_dtypes = {npdtype: [] for npdtype in mapping_dtypes.keys()}
for npdtype, repls in mapping_dtypes.items():
    for repl, aliases in redshift_dtypes.items():
        if repl in repls:
            numpy_redshift_dtypes[npdtype].extend(aliases)

# Test Codes
class Test_split_dtype_and_args:
    def test_split_2_args(self):
        """Test for a function: split_dtype_and_args"""
        assert hlp.split_dtype_and_args('numeric(11,4)') == {'numeric': [11, 4]}
        assert hlp.split_dtype_and_args('numeric( 11, 4 )') == {'numeric': [11, 4]}
        assert hlp.split_dtype_and_args(' numeric(11, 4) ') == {'numeric': [11, 4 ]}
    def test_split_1_arg(self):
        assert hlp.split_dtype_and_args('char(64)') == {'char': [64, np.nan]}
        assert hlp.split_dtype_and_args('char( 64 )') == {'char': [64, np.nan]}
        assert hlp.split_dtype_and_args(' char(64) ') == {'char': [64, np.nan]}
    def test_split_0_args(self):
        assert hlp.split_dtype_and_args('bigint') == {'bigint': [np.nan, np.nan]}
        assert hlp.split_dtype_and_args(' bigint ') == {'bigint': [np.nan, np.nan]}
    def test_exception(self):
        with pytest.raises(TypeError):
            hlp.split_dtype_and_args('decimal(11,4,5)')

class Test_convert_to_representative:
    """Test for a function: convert_to_representative"""
    def test_values(self):
        for repl, equivs in redshift_dtypes.items():
            for equiv in equivs:
                assert hlp.convert_to_representative(equiv) == repl
    def test_exception(self):
        with pytest.raises(TypeError):
            hlp.convert_to_representative('timestamptz')

class Test_convert_to_numpy_dtypes:
    """Test for a function: convert_to_numpy_dtypes"""
    def test_values(self):
        for repl, equivs in redshift_dtypes.items():
            for equiv in equivs:
                assert hlp.convert_to_representative(equiv) == repl
    def test_exception(self):
        with pytest.raises(TypeError):
            hlp.convert_redshift_to_numpy_dtypes('timestampz')

class Test_get_2args:
    """Test for a function: get_2args"""
    def test_values(self):
        assert hlp.get_2args('NUMERIC(11, 4)') == {'numeric': [11, 4]}
        assert hlp.get_2args('CHAR(64)') == {'char': [64, np.nan]}
        assert hlp.get_2args('SMALLINT') == {'smallint': [np.nan, np.nan]}

class Test_convert_redshift_to_numpy_dtypes:
    """Test for a function: convert_redshift_to_numpy_dtypes"""
    def test_values(self):
        assert hlp.convert_redshift_to_numpy_dtypes('SMALLINT') == 'np.int16'
        assert hlp.convert_redshift_to_numpy_dtypes('INT') == 'np.int32'
        assert hlp.convert_redshift_to_numpy_dtypes('NUMERIC(11, 4)') == 'np.float64'
        assert hlp.convert_redshift_to_numpy_dtypes('CHAR(128)') == 'np.str_'
        assert hlp.convert_redshift_to_numpy_dtypes('VARCHAR(256)') == 'np.str_'

class Test_convert_redshift_to_numpy_dtypes_with_2args:
    """Test for a function: convert_redshift_to_numpy_dtypes_with_2args"""
    def test_values(self):
        assert hlp.convert_redshift_to_numpy_dtypes_with_2args('SMALLINT') == {'np.int16': [np.nan, np.nan]}
        assert hlp.convert_redshift_to_numpy_dtypes_with_2args('INT') == {'np.int32': [np.nan, np.nan]}
        assert hlp.convert_redshift_to_numpy_dtypes_with_2args('NUMERIC(11, 4)') == {'np.float64': [11, 4]}
        assert hlp.convert_redshift_to_numpy_dtypes_with_2args('CHAR(128)') == {'np.str_': [128, np.nan]}
        assert hlp.convert_redshift_to_numpy_dtypes_with_2args('VARCHAR(256)') == {'np.str_': [256, np.nan]}

class Test_convert_redshift_to_numpy_dtypes_with_range:
    """Test for a function: convert_redshift_to_numpy_dtypes_with_range"""
    def test_values(self):
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('SMALLINT') == {'np.int16': {'min': -32_768, 'max': 32_767}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('INT') == {'np.int32': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('NUMERIC(11, 4)') == {'np.float64': {}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('CHAR(128)') == {'np.str_': {'bytes': 128}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('VARCHAR(256)') == {'np.str_': {'bytes': 256}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('TIMESTAMP') == {'np.str_': {'bytes': 64}}
        assert hlp.convert_redshift_to_numpy_dtypes_with_range('BOOLEAN') == {'np.str_': {'bytes': 64}}

class Test_dtype_converter:
    """Test for a main function: dtype_converter"""
    def test_values(self):
        assert hlp.dtype_converter('SMALLINT') == {'np.int64': {'min': -32_768, 'max': 32_767}}
        assert hlp.dtype_converter('INT') == {'np.int64': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        assert hlp.dtype_converter('NUMERIC(11, 4)') == {'np.float64': {}}
        assert hlp.dtype_converter('CHAR(128)') == {'np.str_': {'bytes': 128}}
        assert hlp.dtype_converter('VARCHAR(256)') == {'np.str_': {'bytes': 256}}
        assert hlp.dtype_converter('TIMESTAMP') == {'np.str_': {'bytes': 64}}
        assert hlp.dtype_converter('BOOLEAN') == {'np.str_': {'bytes': 64}}