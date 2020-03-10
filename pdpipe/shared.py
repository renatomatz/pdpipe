"""Shared inner functionalities for pdpipe."""
from warnings import warn


def _interpret_columns_param(columns):
    if isinstance(columns, str):
        return [columns]
    elif hasattr(columns, '__iter__'):
        return columns
    return [columns]


def _list_str(listi):
    if listi is None:
        return None
    if isinstance(listi, (list, tuple)):
        return ', '.join([str(elem) for elem in listi])
    return listi


def _get_df_cols(df, cols):
    """Adds more flexibility in the cols parameter

    Allows for use of more uses of the column parameter, like using callables
    and filters instead of just manual naming.

    These features facilitate selecting columns in dataframes with many 
    similarly-named features, like one-hot-encoded column variables
    """
    if cols is None:
        columns_to_transform = set(cols).difference(df.columns)
    elif isinstance(cols, str):
        columns_to_transform = [cols]
    elif hasattr(cols, "__call__"):
        columns_to_transform = [*filter(cols, df.columns)]
    elif isinstance(cols, filter):
        columns_to_transform = [*cols]
    else:
        columns_to_transform = cols

    if columns_to_transform is None:
        warn("Specified column parameter did not yield any columns from df")

    return columns_to_transform
 
