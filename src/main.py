#!/usr/bin/python3
"""Make an error testing result file.

Usage
-----
Assume that the current working directory is the top level (mandatory).

The summary error testing result is produced by
$ python3 src/main.py -s table_name.tsv

The detail error testing result is produced by
$ python3 src/main.py -d table_name.tsv

Args
----
first arg : -s: summary data
            -d: detail data
second arg: raw data file name without a directory name

NOTES
-----
The order of arguments is not interchangeable. You should not change the order.

PREPARATION AND RULES
---------------------
Three files are required to execute this module.

FILES:
    data/table_name.csv or data/table_name.tsv: raw data
    dataconf/table_name.csv                   : Colum name and CREATE TABLE info
    dataconf/table_name.yml                   : raw data format info file

RULES:
    These three files are required to have the same name such as table_name.
    Execute this module in the top level.
"""
import sys
import numpy as np
from checktable import checktable as ct

def execute_checktable(format_file: str,
                       rsdtype_file: str,
                       rawdata_file: str,
                       output_file: str,
                       analysis_type = 'summary'):
    """Make an error testing result file named as output_file path.
    
    Args:
        format_file  : yml file stored in dataconf/ with raw data file format 
                       info: primary key flags, to be more accurate, 
                       superkey flags, not null constraint flags and Redshift
                       data type info.
        rsdtype_file : csv file stored in dataconf/ with Redshift data type and
                       other info:
                       physical column name, primary key flags, to be more 
                       accurate, superkey flags, not null constraint flags 
                       and Redshift data type info.
        rawdata_file : raw data file stored in data/, usually csv or tsv file.
        output_file  : The output file path.
        analysis_type: 'summary' executes a summary error testing.
                       'detail' executes a detail error testing.
    Procedures:
        Make an error testing result file named as output_file path.
    """
    with open(output_file, mode='w', encoding='utf_8') as f:
        testing = ct.CheckTable(format_file, rsdtype_file, rawdata_file)
        
        column_max_length = len(max(testing.df.columns, key = len))

        if analysis_type == 'summary':
            size = testing.size_error()
            not_null = testing.not_null_constraint_error()
            superkey = testing.superkey_error()
            
            if size:
                result  = f"{(se := 'SIZE ERROR')}\n{'-' * len(se)}\n"
                for column_name in size:
                    result += f"\t{column_name}\n"
                result += f"\n"
            if not_null:
                result += f"{(nl := 'NOT-NULL CONSTRAINT ERROR')}\n{'-' * len(nl)}\n"
                for column_name in not_null:
                    result += f"\t{column_name}\n"
                result += f"\n"
            if superkey:
                result += f"{(sk := 'SUPERKEY ERROR')}\n{'-' * len(sk)}\n"
                result += f"\tThe candidate set of columns is not a superkey."
            f.write(result)
        elif analysis_type == 'detail':
            size = testing.max_sizes()
            not_null = testing.count_nan()
            superkey = testing.superkey_error()

            if size:
                result = f"{(se := 'SIZE ERROR')}\n{'-' * len(se)}\n"
                result += f"\t{'COLUMN': <{column_max_length}}  MAX SIZE\n"
                for column_name, maxsize in size.items():
                    if not (maxsize is None or np.isnan(maxsize)):
                        result += f"\t{column_name: <{column_max_length}}: {maxsize}\n"
                result += f"\n"
            if not_null:
                result += f"{(nl := 'NOT-NULL CONSTRAINT ERROR')}\n{'-' * len(nl)}\n"
                result += f"\t{'COLUMN': <{column_max_length}}  NULL COUNT\n"
                for column_name, null_count in not_null.items():
                    result += f"\t{column_name: <{column_max_length}}: {null_count}\n"
                result += f"\n"
            if superkey:
                result += f"{(sk := 'SUPERKEY ERROR')}\n{'-' * len(sk)}\n"
                result += f"\tThe candidate set of columns is not a superkey."
            f.write(result)

table_name = sys.argv[2].split('.')[0]
format_file  = f"dataconf/{table_name}.yml"
rsdtype_file = f"dataconf/{table_name}.csv"
rawdata_file = f"data/{sys.argv[2]}"

if sys.argv[1] == '-s':
    output_file  = f"output/{table_name}-summary.txt"
    execute_checktable(format_file   = format_file,
                       rsdtype_file  = rsdtype_file,
                       rawdata_file  = rawdata_file,
                       output_file   = output_file,
                       analysis_type = 'summary')
elif sys.argv[1] == '-d':
    output_file  = f"output/{table_name}-detail.txt"
    execute_checktable(format_file   = format_file,
                       rsdtype_file  = rsdtype_file,
                       rawdata_file  = rawdata_file,
                       output_file   = output_file,
                       analysis_type = 'detail')