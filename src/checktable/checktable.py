#!/usr/bin/python3

import pandas as pd
import numpy as np
import csv
import yaml

from .helpers import dtype_converter

class CheckTable:
    """Contractor to make attributes from three input files.

    Inputs:
        format_file:  yml file stored in dataconf/ with raw data file format 
                      info:
                      primary key flags, to be more accurate, superkey flags,
                      not null constraint flags and Redshift data type info.
                      
                      delimiter: raw file's delimiter
                      encoding : raw file's variable-width character encoding
                      na_value : raw file's character representing NULL

                      e.g.
                      delimiter: ','
                      encoding : 'utf_8'
                      na_value : ''
                      
        rsdtype_file: csv file stored in dataconf/ with Redshift data type and
                      other info:
                      physical column name, primary key flags, to be more 
                      accurate, superkey flags, not null constraint flags 
                      and Redshift data type info.

                      NOTE: rsdtype_file has to have a header:
                      physical_column_name,pk,not_null_constraint,data_type
                      
                      physical_column_name: column's physical name
                      pk                  : superkey flag
                                            1: superkey element
                                            0: otherwise
                      not null constraint : not null constraint flag
                                            1: has not null constraint
                                            0: otherwise
                      data type:            column's data type in Redshift
                                            INT, NUMERIC(11, 4), VARCHAR(64)...

        rawdata_file: raw data file stored in data/, usually csv or tsv file.
                      NOTE: rawdata_file is not allowed to have a header.
    
    Attributes:
        self.file (dict) : raw file format information
                           'delimiter': raw file delimiter
                           'encoding' : raw file encoding
                           'na_value' : raw file na value

                           e.g.
                           >>> print(self.file)
                           {'delimiter': ',', 'encoding': 'utf_8', 'na_value': ''}
        self.dinfo (dict): data type information
                           KEY (str):
                               'physical_colum_name': physical colum name
                           VALUES (dict):
                               'pk'                 : 1: pk element
                                                      0: otherwise
                               'not_null_constraint': 1: not null constraint
                                                      0: otherwise
                               'dtype'              : numpy dtypes (written in 
                                                      str) with size info.
                           NOTE: dtype is what a column's data_type in Redshift
                                 converted into numpy dtype with the helper 
                                 module, helpers.py

                           e.g.
                           >>> print(self.dinfo)
                           {'column1': {
                               'pk': 1,
                               'not_null_constraint': 1,
                               'dtype': {'np.str_': {'bytes': 64}}
                            },
                            'column1': {
                                'pk': 0,
                                'not_null_constraint': 0,
                                'dtype': {'np.int64': {'min': -32_768, 'max': 32_767}}
                            },
                            'column2': {
                                'pk': 0,
                                'not_null_constraint': 1,
                                'dtype': {'np.float64': {}}
                            }
                           }
        self.df          : pd.DataFrame created from rawdata_file
    
    More on self.dinfo['dtype']:
        The dict value form of self.dinfo['dtype'] is set different 
        due to what kind of numpy dtype that a column data type in Redshift is 
        converted to: 'object_type', 'int_type' or 'float_type'.

        key  : numpy dtype (written in str)
               'np.int64'             
               'np.float64'           
               'np.str_' or 'np.object_'
        value: {'bytes': <maximum utf-8 byte size>}                 if numpy dtype is 'object type'
               {'min': <minimum integer>, 'max': <maximum integer>} if numpy dtype is 'int type'
               {}                                                   if numpy dtype if 'float type',
               
               where 'object type' dtypes are np.str_ and np.object_,
                     'int type' dtypes are np.int8, np.int16, np.int32 and np.int64 and
                     'float type' dtypes are np.float16 and np.float64.
    
    Main Methods:
    - For a Summary of Error Testing:
        size_error
            Returns:
                a list of colum names that have a size error
        
        not_null_constraint_error
            Returns:
                a list of colum names that have a not-null constraint error
        
        superkey_error
            Returns:
                a boolean value. 
                True : the column set supposed to be a superkey is not a superkey.
                False: the column set supposed to be a superkey is a superkey.
        
    - For Details of Error Testing:
        max_sizes
            Returns:
                a dict such that
                key: column name.
                value: the maximum size of column.
            Notes:
                float type returns None.
        
        count_nan
            Returns:
                a dict such that
                key: column name
                value: the number of NaN
    
    Notes:
        count_nan and not_null_constraint_error work only for columns which are
        supposed to have not-NULL constraint. 
    """
    def __init__(self, format_file: str, rsdtype_file: str, rawdata_file: str):

        with open(format_file, mode='r', newline='') as f:
            self.file = yaml.safe_load(f)
        
        self.dinfo = {}
        with open(rsdtype_file, mode='r', newline='') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                self.dinfo[row['physical_column_name']] \
                    = {'pk': int(row['pk']),
                       'not_null_constraint': int(row['not_null_constraint']),
                       'dtype': str(row['data_type'])
                      }
        for column, _ in self.dinfo.items():
            self.dinfo[column].update({'dtype': dtype_converter(self.dinfo[column]['dtype'])})
        
        # For a technical reason, pandas does not import data with missing values
        # (empty cell) as integer dtype. So change integer dtype into np.float64.
        dtypes = {column: eval(dtype) for column, formats in self.dinfo.items() 
                        for dtype in formats['dtype'].keys()
        }
        dtypes_cover_missing_values = {column: '' for column in dtypes.keys()}
        for column, dtype in dtypes.items():
            if dtype == np.int64:
                dtypes_cover_missing_values[column] == np.float64
            else:
                dtypes_cover_missing_values[column] == dtype

        self.df = pd.read_csv(rawdata_file,
                              names = self.dinfo.keys(),
                              dtype = dtypes_cover_missing_values,
                              sep = self.file['delimiter'],
                              encoding = self.file['encoding'],
                              na_values = self.file['na_value'],
                              engine='python')

    def max_size_of(self, column_name: str) -> int:
        """Get the maximum size of a column.
        
        Args:
            column_name: column name
        Returns:
            The maximum size of column.
            What kind of size is due to dtype of column:
            - utf-8 byte size         if dtype is np.str_ or np.object_.
            - maximum absolute value  if dtype is np.int64.
            - None                    if dtype is np.float64.
        """
        if list(self.dinfo[column_name]['dtype'].keys())[0] == 'np.int64':
            if np.isnan(self.df[column_name].abs().max()):
                return self.df[column_name].abs().max()
            else:
                return int(self.df[column_name].abs().max())
        elif list(self.dinfo[column_name]['dtype'].keys())[0] == 'np.str_':
            if np.isnan(self.df[column_name].str.encode('utf-8').str.len().max()):
                return self.df[column_name].str.encode('utf-8').str.len().max()
            else:
                return int(self.df[column_name].str.encode('utf-8').str.len().max())
        else:
            return None

    def max_sizes(self) -> dict:
        """Get the maximum sizes for every column.
        
        Returns:
            key: column name.
            value: the maximum size of column.
            What kind of size is due to dtype of column:
            - utf-8 byte size         if dtype is np.str_ or np.object_.
            - maximum absolute value  if dtype is np.int64.
            - None                    if dtype is np.float64.
        
            e.g.
                {'column1': 64, 'column2': 128, 'column3': 6, 'column4': None}
        """
        result = {column_name: '' for column_name in self.df.columns}
        for column_name in self.df.columns:
            result[column_name] = self.max_size_of(column_name)
        return result

    def size_error(self) -> list:
        """Returns a list of column names that have a size error.
        
        Returns:
            colum names that have a size error

            e.g.
                ['column1']  means column1 has a size error.
                []           means any columns do not have a size error.
        """
        result = []
        for column_name, size in self.max_sizes().items():
            if (dtype := list(self.dinfo[column_name]['dtype'].keys())[0]) == 'np.str_':
                if size > self.dinfo[column_name]['dtype'][dtype]['bytes']:
                    result.append(column_name)
            elif (dtype := list(self.dinfo[column_name]['dtype'].keys())[0]) == 'np.int64':
                if size > self.dinfo[column_name]['dtype'][dtype]['max']:
                    result.append(column_name)
        return result

    def count_nan_of(self, column_name: str) -> int:
        """Count NaN in a column.
        
        Args:
            column_name: column name
        Returns:
            The number of NaN values of the column.
        """
        return self.df[column_name].isnull().sum()

    def count_nan(self) -> dict:
        """Count NaN in columns supposed to have not-null constraint.
        
        Returns:
            key: column name
            value: the number of NaN

            e.g.
                {'column1': 0, 'column2': 0, 'column3': 2}
                In that case, column3 has 2 NaN even though it is not supposed
                to have any NaN.
        """
        result = {column_name: '' for column_name in self.df.columns
                    if self.dinfo[column_name]['not_null_constraint'] == 1
        }
        for column_name in self.df.columns:
            if self.dinfo[column_name]['not_null_constraint']:
                result[column_name] = self.count_nan_of(column_name)
        return result

    def not_null_constraint_error(self) -> list:
        """Returns a list of column names that have a not-null constraint error.
        
        Returns:
            colum names that have a not-null constraint error

            e.g.
                ['column1']  means column1 has some NaN even though colulm1 is
                             not supposed to have any.
                []           means any columns do not have a not-null constraint
                             error.
        """
        return [column_name for column_name, nans in self.count_nan().items() if nans > 0]

    def superkey_error(self) -> bool:
        """Returns if the column set supposed to be a superkey is a real superkey.
        
        Returns:
            True: the column set supposed to be a superkey is not a superkey.
            False: the column set supposed to be a superkey is a superkey.
        """
        not_nulls = []
        for column_name in self.df.columns:
            if self.dinfo[column_name]['pk']:
                not_nulls.append(column_name)
        return not len(self.df) == len(self.df[not_nulls].drop_duplicates())

if __name__ == '__main__':
    """Run this module in pandas-numpy/"""
    format_file = 'dataconf/colors.yml'
    rsdtype_file = 'dataconf/colors.csv'
    rawdata_file = 'data/colors.tsv'
    colors = CheckTable(format_file, rsdtype_file, rawdata_file)


    print(f"SIZE ERROR: {colors.size_error()}")
    print(f"NOT-NULL CONSTRAINT ERROR: {colors.not_null_constraint_error()}")
    print(f"SUPERKEY ERROR: {colors.superkey_error()}")
    print(f"MAX SIZES: {colors.max_sizes()}")
    print(f"COUNT NaN: {colors.count_nan()}")