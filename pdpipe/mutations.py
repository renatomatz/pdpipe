"""Commmon mutations used in an ApplyByCols/Rows stage for single column 
transformation or multiple columns in an ColByFrameFunc

These should take in a dataframe or series as their only required argument and
return a series with the same number of rows as the original dataframe

Any mutation that takes in configurable parameters should instead be just
callable objects
"""

import pandas as pd

def to_weekday(X):
    """Convert column of pandas timestamps into their weekday from 0 to 6
    """
    return X.apply(lambda x: x.weekday)  

def to_time_of_day(X):
    """Convert column of pandas timestamps into their time of day from 0 to 23
    """
    return X.apply(lambda x: x.hour)

class diff:
    """Return the difference between two columns

    cols: tuple of two indexes
    - second will be subtracted from first
    """

    cols: tuple

    def __init__(self, cols=(0,1)):

        if len(cols) != 2:
            raise ValueError("cols argument should be of size 2")

        self.cols = cols

    def __call__(self, X, *args, **kwargs):
        
        return X[self.cols[0]] - X[self.cols[1]]

class time_diff(diff):
    """Return time difference between two columns

    units: string containing time delta units, defaults to hours
    """

    units: str

    def __init__(self, cols=(0,1), units="h"):
        super().__init__(cols)

        self.units=units

    def __call__(self, X, *args, **kwargs):
        return super()\
            .__call__(X, *args, **kwargs)\
            .astype("timedelta64[{}]".format(self.units))
