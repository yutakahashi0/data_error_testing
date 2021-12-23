#!/usr/bin/python3
"""Converting Redshift data types to numpy dtypes.

This module consists of helper functions for the main module `checktable.py`.
The main function in the script is the last one `dtype_converter`.

`dtype_converter` receives one argument, Redshift data type and converts it
to numpy dtype with size info. The size info form is different depending on what
kind of numpy dtype is.
If numpy dtype is 'int_type' such as np.int8, np.int16, np.int32 and np.int64,
the info form is {'min': <minimum dtype value>, 'max': <maximum dtype value>}.
For example, np.int16 is a 2-byte signed integer. So the min and max are 
'min': -32_768 and 'max': 32_767.
If numpy dtype is 'float_type' such as np.float32 and np.float64, the info form
is an empty dict {}.
If numpy dtype is 'object_type' such as np.str_ and np.object_, the info form
is {'bytes': <utf-8 byte size>}. For example, np.object_ with 128 utf-8 byte
size has {'bytes': 128}.


The Main Function
=================
dtype_converter(column_dtype)
See the function docstring in detail.


How This Module is Made: What is the Algorithm?
===============================================

1. What the Module Provides
---------------------------

Suppose that you were given a Redshift dtype table such as

{
    'column1': {'pk': 1, 'not_null_constraint': 1, 'data_type': 'CHARACTER(64)'},
    'column2': {'pk': 1, 'not_null_constraint': 1, 'data_type': 'VARCHAR(64)'},
    'column3': {'pk': 0, 'not_null_constraint': 1, 'data_type': 'INT'},
    'column4': {'pk': 0, 'not_null_constraint': 0, 'data_type': 'NUMERIC(11,4)'},
},

this module converts 'data_type' from Redshift data type to numpy dtype. The
converted result of the example is
{
    'column1': {... 'data_type': {'np.str_': {'bytes': 64}}},
    'column2': {... 'data_type': {'np.str_': {'bytes': 64}}},
    'column3': {... 'data_type': {'np.int64': {'min': -32_768, 'max': 32_767}}},
    'column4': {... 'data_type': {'np.float64': {}}}
}.

Since the Redshift data type has many aliaces, the conversion is not as easy as a
one-to-one converting problem.

2. Converting Algorithm
-----------------------

The converting algorithm uses two correspondence tables: redshift_dtypes and 
mapping_dtypes.

The converting algorithm is as follows:

1. Make characters lowercase.
   
   e.g. 'CHARACTER(64)'  -> 'character(64)'

2. Split data type and utf-8 byte size arguments.
   
   Two utf-8 byte size arguments are accepted since the most used data types 
   have two arguments at most.
   
   e.g. 'character(64)'  -> ['character', 64, np.nan]
        'numeric(11, 4)' -> ['numeric', 11, 4]

3. Create an equivalent class of Redshift data type and take a replesentative 
   for each Redshift data type, where aliases is the equivalence relation.
   
   The equivalent class is represented by the `redshift_dtypes` dict. A key
   is a representative and the values of the key are equivalent Redshift data 
   type to the representative. Not mathematically, the values are aliaces each
   other and the key is one of the elements.
   
   The data types are converted to the representatives with `redshift_dtypes`
   such as 
   e.g. 'character' -> 'char'
        'numeric'   -> 'decimal'
        'decimal'   -> 'decimal'

4. Create an correspondence between Redshift data type representatives and 
   numpy dtypes and convert them to the numpy dtype.

   The correspondence between Redshift data type representatives and a numpy
   dtypes is `mapping_dtypes`.

   The representatives are converted to the numpy dtypes with `mapping_dypes`
   such as
   e.g. 'char'    -> 'np.str_'
        'decimal' -> 'np.float64'

5. By Using 1 - 4, convert Redshift data type to numpy dtype with two arguments.

   e.g. 'INT2'           -> {'np.int16': [np.nan, np.nan]}
        'INT'            -> {'np.int32': [np.nan, np.nan]}
        'BIGINT'         -> {'np.int64': [np.nan, np.nan]}
        'NUMERIC(11, 4)' -> {'np.float32': [11, 4]}
        'FLOAT8'         -> {'np.float64': [np.nan, np.nan]}
        'CHAR(256)'      -> {'np.str_': [256, np.nan]}
        'VARCHAR(512)'   -> {'np.str_': [512, np.nan]}
        'TIMESTAMP'      -> {'np.str_': [np.nan, np.nan]}

6. Change the value of a dict and get it to have utf-8 byte size or min/max 
integer value info.
   
   e.g. 'INT2'           -> {'np.int16': {'min': -32_768, 'max': 32_767}}
        'INT'            -> {'np.int32': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        'BIGINT'         -> {'np.int64': {'min': -9_223_372_036_854_775_808, 'max': 9_223_372_036_854_775_807}}
        'NUMERIC(11, 4)' -> {'np.float32': {}}
        'FLOAT8'         -> {'np.float64': {}}
        'CHAR(256)'      -> {'np.str_': {'bytes': 256}}
        'VARCHAR(512)'   -> {'np.str_': {'bytes': 512}}
        'TIMESTAMP'      -> {'np.str_': {'bytes': 64}}

7. Convert 'int_type' and 'float_type' to greater dtypes.
   
   All 'int_type's are converted to np.int64, all 'float_type's are to
   np.float64.

   e.g. 'INT2'           -> {'np.int64': {'min': -32_768, 'max': 32_767}}
        'INT'            -> {'np.int64': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        'BIGINT'         -> {'np.int64': {'min': -9_223_372_036_854_775_808, 'max': 9_223_372_036_854_775_807}}
        'NUMERIC(11, 4)' -> {'np.float64': {}}
        'FLOAT8'         -> {'np.float64': {}}
        'CHAR(256)'      -> {'np.str_': {'bytes': 256}}
        'VARCHAR(512)'   -> {'np.str_': {'bytes': 512}}
        'TIMESTAMP'      -> {'np.str_': {'bytes': 64}}
"""
import numpy as np
import re

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

# mapping dtypes of numpy to those of Redshift
mapping_dtypes = {
    'np.int16': ['smallint',],
    'np.int32': ['integer',],
    'np.int64': ['bigint',],
    'np.float32': ['real',],
    'np.float64': ['decimal', 'double precision'],
    'np.str_': ['boolean', 'char', 'varchar', 'date', 'timestamp']
}

# dtype category
category_dtype = {
    'int_type': ['np.int8', 'np.int16', 'np.int32', 'np.int64'],
    'float_type': ['np.float32', 'np.float64'],
    'object_type': ['np.str_',]
}

# int type range
int_range = {
    'np.int8': [-128, 127],
    'np.int16': [-32_768, 32_767],
    'np.int32': [-2_147_483_648, 2_147_483_647],
    'np.int64': [-9_223_372_036_854_775_808, 9_223_372_036_854_775_807]
}

# helper to helper functions:
def split_dtype_and_args(column_dtype: str) -> dict:
    """Split Redshift dtype and its arguments.
    
    The maximum number of arguments is 2. 
    Args:
        column_dtype: Redshift dtype with two arguments at most.
    Returns:
        key: column name
        value: two arguments
               If the Redshift dtype does not have any arguments or has just one
               argument, value is [np.nan, np.nan] or [arg, np.nan].
    e.g.
        >>> d1 = 'bigint'
        >>> d2 = 'numeric(11,4)'
        >>> print(split_dtype_and_args(d1))
        {'bigint': [np.nan, np.nan]}
        >>> print(split_dtype_and_args(d2))
        {'numeric': [11, 4]}
    """
    p = re.compile(r'[\(\), ]')
    result = {}
    splitter = [i for i in p.split(column_dtype) if i]
    if   len(splitter) == 1:
        result[splitter[0]] = [np.nan, np.nan]
    elif len(splitter) == 2:
        result[splitter[0]] = [int(splitter[1]), np.nan]
    elif len(splitter) == 3:
        result[splitter[0]] = [int(splitter[1]), int(splitter[2])]
    else:
        raise TypeError(f"'{column_dtype}': \
split_dtype_and_args does not support more than two Redshift dtypes.")
    return result

def convert_to_representative(column_dtype: str) -> str:
    """Convert a column dtype into a redshift_dtypes representative of the equivalent class.
    
    Convert Redshift data type to the representative data type of the equivalent
    class: a key of `redshift_dtypes`.
    Args:
        column_dtype: redshfit data type in lower case.
    Returns:
        representative in Redshift data type: a key of Redshift_dtypes.
    Raises:
        If column_dtype is not in the values of the dict redshift_dtypes,
        exception TypeError occures. 
    e.g.
        >>> d = 'int'
        >>> print(convert_to_representative(d))
        'integer'
    """
    for repr, equivs in redshift_dtypes.items():
        if column_dtype in equivs:
            return repr
    raise TypeError(f"'{column_dtype}': \
Possibly, Redshift data type is misspelled or not suppoted \
by the redshift_dtypes dict in convert_redshift_dtypes.py.")

    
def convert_to_numpy_dtype(column_dtype: str) -> str:
    """Convert a Redshift dtype representative to a numpy dtype with the mapping_dtypes dict.
    
    Args:
        column_dtype: Redshift data type in redshift_dtypes.keys().
    Returns:
        numpy dtype
    Raises:
        If column_dtype is not in the values of the dict mapping_dtypes,
        exception TypeError occures.
    e.g.
        >>> d = 'integer'
        >>> print(convert_to_numpy_dtype(d))
        np.int32
    """
    for npdtype, rsdtypes in mapping_dtypes.items():
        if column_dtype in rsdtypes:
            return npdtype
    raise TypeError(f"'{column_dtype}': \
Possibly, Redshift data type is not supported by the mapping_dtypes dict \
in convert_redshift_dtypes.py.")

# helper functions:
def get_2args(column_dtype: str) -> dict:
    """Get two args from Redshift data type.
    
    Args:
        column_dtype: Redshift data type with two arguments.
    Returns:
        key: data type.
        value: list of two arguments.
    
    e.g.
        'NUMERIC(11, 4)' -> {'numeric': [11, 4]}
        'CHAR(64)'       -> {'char': [64, np.nan]}
        'INT'            -> {'int': [np.nan, np.nan]}
    """
    
    return split_dtype_and_args(column_dtype.lower())

def convert_redshift_to_numpy_dtypes(column_dtype: str) -> str:
    """Convert a Redshift data type to a numpy dtype.
    
    Args:
        column_dtype: Redshift data type
    Returns:
        numpy dtype
    e.g.
        'NUMERIC(11, 4)' -> 'np.float64'
        'CHAR(64)'       -> 'np.str_'
        'INT'            -> 'np.int32'
    """
    redshift_dtype = list(get_2args(column_dtype).keys())[0] # str
    return convert_to_numpy_dtype(convert_to_representative(redshift_dtype))

def convert_redshift_to_numpy_dtypes_with_2args(column_dtype: str) -> dict:
    """Convert a Redshift data type to a numpy dtype with utf-8 byte sizes.
    
    Args:
        column_dtype: Redshift data type
    Returns:
        key: numpy dtype
        value(list): two args originally the Redshift data type has
    e.g.
        'NUMERIC(11, 4)' -> {'np.float64': [11, 4]}
        'CHAR(64)'       -> {'np.str_': [64, np.nan]}
        'INT'            -> {'np.int32': [np.nan, np.nan]}
    """
    tmp_dict = get_2args(column_dtype)
    result = {}
    for rsdtype, args in tmp_dict.items():
        result[convert_redshift_to_numpy_dtypes(rsdtype)] = args
    return result

def convert_redshift_to_numpy_dtypes_with_range(column_dtype: str) -> dict:
    """Convert Redshfit data type to numpy dtype with utf-8 byte size or max/min integer.
    
    Convert Redshift data type to numpy dtype based on `mapping_dtypes` (See 
    the script of this module). The output dict value form is different due to
    the numpy dtype: 'object_type', 'int_type' or 'float_type'.
    Args:
        column_dtype: Redshift data type
    Returns:
        key: numpy dtype
        value: {'bytes': <maximum utf-8 byte size>}                 if numpy dtype is 'object type'
               {'min': <minimum integer>, 'max': <maximum integer>} if numpy dtype is 'int type'
               {}                                                   if numpy dtype if 'float type',
        where 'object type' dtypes are np.str_ and np.object_,
              'int type' dtypes are np.int8, np.int16, np.int32 and np.int64 and
              'float type' dtypes are np.float16 and np.float64.
    e.g.
        'NUMERIC(11, 4)' -> {'np.float64': {}}
        'CHAR(64)'       -> {'np.str_': {'bytes': 64}}
        'INT'            -> {'np.int32': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        'TIMESTAMP'      -> {'np.str_': {'bytes': 64}}
        'DATE'           -> {'np.str_': {'bytes': 64}}
    
    NOTES:
        The Input Redshift data type converted to np._str (See `redshift_dtypes`
        and `mapping_dtypes`) without any arguments such as 'DATE', 'TIMESTAMP'
        and 'BOOLEAN' has {'bytes': 64} in the output of value.
    """
    tmp_dict = convert_redshift_to_numpy_dtypes_with_2args(column_dtype)
    for dtype, args in tmp_dict.items():
        if dtype in category_dtype['object_type']:
            if np.isnan(tmp_dict[dtype][0]):
                tmp_dict[dtype] = {'bytes': 64}
            else:
                tmp_dict[dtype] = {'bytes': tmp_dict[dtype][0]}
        elif dtype in category_dtype['int_type']:
            tmp_dict[dtype] = {'min': int_range[dtype][0], 'max': int_range[dtype][1]}
        elif dtype in category_dtype['float_type']:
            tmp_dict[dtype] = {}
    return tmp_dict

# The main functions:
def dtype_converter(column_dtype: str) -> dict:
    """Convert Redshfit data type to numpy dtype with utf-8 byte size or max/min integer.
    
    Convert Redshift data type to numpy dtype based on `mapping_dtypes` (See 
    the script of this module). The output dict value form is different due to
    the numpy dtype: 'object_type', 'int_type' or 'float_type'.

    This function is used for checking raw data so the conversion from Redshift
    data type to numpy dtype is **not strict** about input column byte size. 
    They are converted to greater than or equal to the original byte size. 
    For example, all int data type in Redshift such as SMALLINT, INTEGER, BIGINT
    are converted to np.int64 and all float data type such as DECIMAL, REAL, 
    DOUBLE PRECISION are converted to np.float64.
    The philosophy is that this module (checktable.py) is for testing raw data 
    about its size, null-constraint, superkey etc. for every column so raw data
    needs to be imported to this module no matter how the setting of Redshift
    data type byte size is wrong.
    It is meaningless if raw data is not imported to the module.
    
    Args:
        column_dtype: Redshift data type
    Returns:
        key: numpy dtype
        value: {'bytes': <maximum utf-8 byte size>}                 if numpy dtype is 'object type'
               {'min': <minimum integer>, 'max': <maximum integer>} if numpy dtype is 'int type'
               {}                                                   if numpy dtype if 'float type',
        where 'object type' dtypes are np.str_ and np.object_,
              'int type' dtypes are np.int8, np.int16, np.int32 and np.int64 and
              'float type' dtypes are np.float16 and np.float64.
    e.g.
        INPUT ARGUMENTS     OUTPUT VALUES
        ---------------     -------------
        'NUMERIC(11, 4)' -> {'np.float64': {}}
        'CHAR(64)'       -> {'np.str_': {'bytes': 64}}
        'INT'            -> {'np.int64': {'min': -2_147_483_648, 'max': 2_147_483_647}}
        'TIMESTAMP'      -> {'np.str_': {'bytes': 64}}
        'DATE'           -> {'np.str_': {'bytes': 64}}
    
    NOTES:
        The Input Redshift data type converted to np._str (See `redshift_dtypes`
        and `mapping_dtypes`) without any arguments such as 'DATE', 'TIMESTAMP'
        and 'BOOLEAN' has {'bytes': 64} in the output of value.

        DECIMAL and its alias NUMERIC in Redshift data type are user-defined
        float data type; users can define the precision and the scale. Python 
        and Numpy do not support that case of dtype. This function converts
        them to np.float64, in other words, users cannot check the precision of
        data.
    """
    tmp_dict = convert_redshift_to_numpy_dtypes_with_range(column_dtype)
    result = {}
    for dtype, ranges in tmp_dict.items():
        if dtype in category_dtype['int_type']:
            result['np.int64'] = ranges
        elif dtype in category_dtype['float_type']:
            result['np.float64'] = ranges
        else:
            result[dtype] = ranges
    return result
