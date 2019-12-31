"""Basic pdpipe PdPipelineStages."""

import types
from collections import deque

from strct.dicts import reverse_dict_partial

from pdpipe.core import PdPipelineStage
# from pdpipe.util import out_of_place_col_insert
from pdpipe.shared import (
    _interpret_columns_param,
    _list_str
)


class ColDrop(PdPipelineStage):
    """A pipeline stage that drops columns by name.

    Parameters
    ----------
    columns : str, iterable or callable
        The label, or an iterable of labels, of columns to drop. Alternatively,
        columns can be assigned a callable returning bool values for
        pandas.Series objects; if this is the case, every column for which it
        return True will be dropped.
    errors : {‘ignore’, ‘raise’}, default ‘raise’
        If ‘ignore’, suppress error and existing labels are dropped.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
        >>> pdp.ColDrop('num').apply(df)
          char
        1    a
        2    b
    """

    _DEF_COLDROP_EXC_MSG = ("ColDrop stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_COLDROP_APPLY_MSG = 'Dropping columns {}...'

    def _default_desc(self):
        if isinstance(self._columns, types.FunctionType):
            return "Drop columns by lambda."
        return "Drop column{} {}".format(
            's' if len(self._columns) > 1 else '', self._columns_str)

    def __init__(self, columns, errors=None, **kwargs):
        self._columns = columns
        self._errors = errors
        self._columns_str = _list_str(self._columns)
        if not callable(columns):
            self._columns = _interpret_columns_param(columns)
        super_kwargs = {
            'exmsg': ColDrop._DEF_COLDROP_EXC_MSG.format(self._columns_str),
            'appmsg': ColDrop._DEF_COLDROP_APPLY_MSG.format(self._columns_str),
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        if callable(self._columns):
            return True
        if self._errors != 'ignore':
            return set(self._columns).issubset(df.columns)
        return True

    def _transform(self, df, verbose):
        if callable(self._columns):
            cols_to_drop = [
                col for col in df.columns
                if self._columns(df[col])
            ]
            return df.drop(cols_to_drop, axis=1, errors=self._errors)
        return df.drop(self._columns, axis=1, errors=self._errors)


class ValDrop(PdPipelineStage):
    """A pipeline stage that drops rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to drop.
    columns : str or list-like, default None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1,4],[4,5],[18,11]], [1,2,3], ['a','b'])
        >>> pdp.ValDrop([4], 'a').apply(df)
            a   b
        1   1   4
        3  18  11
        >>> pdp.ValDrop([4]).apply(df)
            a   b
        3  18  11
    """

    _DEF_VALDROP_EXC_MSG = ("ValDrop stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_VALDROP_APPLY_MSG = "Dropping values {}..."

    def _default_desc(self):
        if self._columns:
            return "Drop values {} in column{} {}".format(
                self._values_str, 's' if len(self._columns) > 1 else '',
                self._columns_str)
        return "Drop values {}".format(self._values_str)

    def __init__(self, values, columns=None, **kwargs):
        self._values = values
        self._values_str = _list_str(self._values)
        self._columns_str = _list_str(columns)
        if columns is None:
            self._columns = None
            apply_msg = ValDrop._DEF_VALDROP_APPLY_MSG.format(
                self._values_str)
        else:
            self._columns = _interpret_columns_param(columns)
            apply_msg = ValDrop._DEF_VALDROP_APPLY_MSG.format(
                "{} in {}".format(
                    self._values_str, self._columns_str))
        super_kwargs = {
            'exmsg': ValDrop._DEF_VALDROP_EXC_MSG.format(self._columns_str),
            'appmsg': apply_msg,
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _transform(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        columns_to_check = self._columns
        if self._columns is None:
            columns_to_check = df.columns
        for col in columns_to_check:
            inter_df = inter_df[~inter_df[col].isin(self._values)]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df


class ValKeep(PdPipelineStage):
    """A pipeline stage that keeps rows by value.

    Parameters
    ----------
    values : list-like
        A list of the values to keep.
    columns : str or list-like, default None
        The name, or an iterable of names, of columns to check for the given
        values. If set to None, all columns are checked.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
        >>> pdp.ValKeep([4, 5], 'a').apply(df)
           a   b
        2  4   5
        3  5  11
        >>> pdp.ValKeep([4, 5]).apply(df)
           a  b
        2  4  5
    """

    _DEF_VALKEEP_EXC_MSG = ("ValKeep stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_VALKEEP_APPLY_MSG = "Keeping values {}..."

    def _default_desc(self):
        if self._columns:
            return "Keep values {} in column{} {}".format(
                self._values_str, 's' if len(self._columns) > 1 else '',
                self._columns_str)
        return "Keep values {}".format(self._values_str)

    def __init__(self, values, columns=None, **kwargs):
        self._values = values
        self._values_str = _list_str(self._values)
        self._columns_str = _list_str(columns)
        if columns is None:
            self._columns = None
            apply_msg = ValKeep._DEF_VALKEEP_APPLY_MSG.format(
                self._values_str)
        else:
            self._columns = _interpret_columns_param(columns)
            apply_msg = ValKeep._DEF_VALKEEP_APPLY_MSG.format(
                "{} in {}".format(
                    self._values_str, self._columns_str))
        super_kwargs = {
            'exmsg': ValKeep._DEF_VALKEEP_EXC_MSG.format(self._columns_str),
            'appmsg': apply_msg,
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _transform(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        columns_to_check = self._columns
        if self._columns is None:
            columns_to_check = df.columns
        for col in columns_to_check:
            inter_df = inter_df[inter_df[col].isin(self._values)]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df


class ColRename(PdPipelineStage):
    """A pipeline stage that renames a column or columns.

    Parameters
    ----------
    rename_map : dict
        Maps old column names to new ones.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[8,'a'],[5,'b']], [1,2], ['num', 'char'])
        >>> pdp.ColRename({'num': 'len', 'char': 'initial'}).apply(df)
           len initial
        1    8       a
        2    5       b
    """

    _DEF_COLDRENAME_EXC_MSG = ("ColRename stage failed because not all columns"
                               " {} were found in input dataframe.")
    _DEF_COLDRENAME_APP_MSG = "Renaming column{} {}..."

    def __init__(self, rename_map, **kwargs):
        self._rename_map = rename_map
        columns_str = _list_str(list(rename_map.keys()))
        suffix = 's' if len(rename_map) > 1 else ''
        super_kwargs = {
            'exmsg': ColRename._DEF_COLDRENAME_EXC_MSG.format(columns_str),
            'appmsg': ColRename._DEF_COLDRENAME_APP_MSG.format(
                suffix, columns_str),
            'desc': "Rename column{} with {}".format(suffix, self._rename_map)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._rename_map.keys()).issubset(df.columns)

    def _transform(self, df, verbose):
        return df.rename(columns=self._rename_map)


class DropNa(PdPipelineStage):
    """A pipeline stage that drops null values.

    Supports all parameter supported by pandas.dropna function.

    Parameters

    ---------

    columns: str or list-like, default None
        Here simply for more standardized package notation, passed as the 
        "subset" parameter in a dropna() call

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1,4],[4,None],[1,11]], [1,2,3], ['a','b'])
        >>> pdp.DropNa().apply(df)
           a     b
        1  1   4.0
        3  1  11.0
    """

    _DEF_DROPNA_EXC_MSG = "DropNa stage failed."
    _DEF_DROPNA_APP_MSG = "Dropping null values..."
    _DROPNA_KWARGS = ['axis', 'how', 'thresh', 'subset', 'inplace']

    def __init__(self, columns=None, **kwargs):

        common = set(kwargs.keys()).intersection(DropNa._DROPNA_KWARGS)

        self.dropna_kwargs = {key: kwargs.pop(key) for key in common}

        if columns is not None:
            self.dropna_kwargs["subset"] = _interpret_columns_param(columns)
            
        super_kwargs = {
            'exmsg': DropNa._DEF_DROPNA_EXC_MSG,
            'appmsg': DropNa._DEF_DROPNA_APP_MSG,
            'desc': "Drops null values."
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        before_count = len(df)
        ncols_before = len(df.columns)
        inter_df = df.dropna(**self.dropna_kwargs)
        if verbose:
            print("{} rows, {} columns dropeed".format(
                before_count - len(inter_df),
                ncols_before - len(inter_df.columns),
            ))
        return inter_df


class FreqDrop(PdPipelineStage):
    """A pipeline stage that drops rows by value frequency.

    Parameters
    ----------
    threshold : int
        The minimum frequency required for a value to be kept.
    column : str
        The name of the colums to check for the given value frequency.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1,4],[4,5],[1,11]], [1,2,3], ['a','b'])
        >>> pdp.FreqDrop(2, 'a').apply(df)
           a   b
        1  1   4
        3  1  11
    """

    _DEF_FREQDROP_EXC_MSG = ("FreqDrop stage failed because column {} was not"
                             " found in input dataframe.")
    _DEF_FREQDROP_APPLY_MSG = ("Dropping values with frequency < {} in column"
                               " {}...")
    _DEF_FREQDROP_DESC = "Drop values with frequency < {} in column {}."

    def __init__(self, threshold, column, **kwargs):
        self._threshold = threshold
        self._column = column
        apply_msg = FreqDrop._DEF_FREQDROP_APPLY_MSG.format(
            self._threshold, self._column)
        super_kwargs = {
            'exmsg': FreqDrop._DEF_FREQDROP_EXC_MSG.format(self._column),
            'appmsg': apply_msg,
            'desc': FreqDrop._DEF_FREQDROP_DESC.format(
                self._threshold, self._column)
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._column in df.columns

    def _transform(self, df, verbose):
        inter_df = df
        before_count = len(inter_df)
        valcount = df[self._column].value_counts()
        to_drop = valcount[valcount < self._threshold].index
        inter_df = inter_df[~inter_df[self._column].isin(to_drop)]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df


class ColReorder(PdPipelineStage):
    """A pipeline stage that reorders columns.

    Parameters
    ----------
    positions : dict
        A mapping of column names to their desired positions after reordering.
        Columns not included in the mapping will maintain their relative
        positions over the non-mapped colums.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[8,4,3,7]], columns=['a', 'b', 'c', 'd'])
        >>> pdp.ColReorder({'b': 0, 'c': 3}).apply(df)
           b  a  d  c
        0  4  8  7  3
    """

    _DEF_ORD_EXC_MSG = ("ColReorder stage failed because not all columns in {}"
                        " were found in input dataframe.")

    def __init__(self, positions, **kwargs):
        self._col_to_pos = positions
        self._pos_to_col = reverse_dict_partial(positions)
        super_kwargs = {
            'exmsg': ColReorder._DEF_ORD_EXC_MSG.format(self._col_to_pos),
            'appmsg': "Reordering columns by {}".format(self._col_to_pos),
            'desc': "Reorder columns by {}".format(self._col_to_pos),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._col_to_pos.keys()).issubset(df.columns)

    def _transform(self, df, verbose):
        cols = df.columns
        map_cols = list(self._col_to_pos.keys())
        non_map_cols = deque(x for x in cols if x not in map_cols)
        new_columns = []
        try:
            for pos in range(len(cols)):
                if pos in self._pos_to_col:
                    new_columns.append(self._pos_to_col[pos])
                else:
                    new_columns.append(non_map_cols.popleft())
            return df[new_columns]
        except (IndexError):
            raise ValueError("Bad positions mapping given: {}".format(
                new_columns))


class RowDrop(PdPipelineStage):
    """A pipeline stage that drop rows by callable conditions.

    Parameters
    ----------
    conditions : list-like or dict
        The list of conditions that make a row eligible to be dropped. Each
        condition must be a callable that take a cell value and return a bool
        value. If a list of callables is given, the conditions are checked for
        each column value of each row. If a dict mapping column labels to
        callables is given, then each condition is only checked for the column
        values of the designated column.
    reduce : 'any', 'all' or 'xor', default 'any'
        Determines how row conditions are reduced. If set to 'all', a row must
        satisfy all given conditions to be dropped. If set to 'any', rows
        satisfying at least one of the conditions are dropped. If set to 'xor',
        rows satisfying exactly one of the conditions will be dropped. Set to
        'any' by default.
    columns : str or iterable, optional
        The label, or an iterable of labels, of columns. Optional. If given,
        input conditions will be applied to the sub-dataframe made up of
        these columns to determine which rows to drop.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1,4],[4,5],[5,11]], [1,2,3], ['a','b'])
        >>> pdp.RowDrop([lambda x: x < 2]).apply(df)
           a   b
        2  4   5
        3  5  11
        >>> pdp.RowDrop({'a': lambda x: x == 4}).apply(df)
           a   b
        1  1   4
        3  5  11
    """

    _DEF_ROWDROP_EXC_MSG = ("RowDrop stage failed because not all columns {}"
                            " were found in input dataframe.")
    _DEF_ROWDROP_APPLY_MSG = "Dropping rows by conditions on columns {}..."

    _REDUCERS = {
        'all': all,
        'any': any,
        'xor': lambda x: sum(x) == 1
    }

    def _default_desc(self):
        return "Drop rows by conditions: {}".format(self._conditions)

    def _row_condition_builder(self, conditions, reduce):
        reducer = RowDrop._REDUCERS[reduce]
        if self._cond_is_dict:
            def _row_cond(row):
                res = [cond(row[lbl]) for lbl, cond in conditions.items()]
                return reducer(res)
        else:
            def _row_cond(row):
                res = [reducer(row.apply(cond)) for cond in conditions]
                return reducer(res)
        return _row_cond

    def __init__(self, conditions, reduce=None, columns=None, **kwargs):
        self._conditions = conditions
        if reduce is None:
            reduce = 'any'
        self._reduce = reduce
        self._columns = None
        if columns:
            self._columns = _interpret_columns_param(columns)
        if reduce not in RowDrop._REDUCERS.keys():
            raise ValueError((
                "{} is an unsupported argument for the 'reduce' parameter of "
                "the RowDrop constructor!").format(reduce))
        self._cond_is_dict = isinstance(conditions, dict)
        self._columns_str = ""
        if self._cond_is_dict:
            valid = all([callable(cond) for cond in conditions.values()])
            if not valid:
                raise ValueError(
                    "Condition dicts given to RowDrop must map to callables!")
            self._columns = list(conditions.keys())
            self._columns_str = _list_str(self._columns)
        else:
            valid = all([callable(cond) for cond in conditions])
            if not valid:
                raise ValueError(
                    "RowDrop condition lists can contain only callables!")
        self._row_cond = self._row_condition_builder(conditions, reduce)
        super_kwargs = {
            'exmsg': RowDrop._DEF_ROWDROP_EXC_MSG.format(self._columns_str),
            'appmsg': RowDrop._DEF_ROWDROP_APPLY_MSG.format(self._columns_str),
            'desc': self._default_desc()
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _transform(self, df, verbose):
        before_count = len(df)
        subdf = df
        if self._columns is not None:
            subdf = df[self._columns]
        drop_index = ~subdf.apply(self._row_cond, axis=1)
        inter_df = df[drop_index]
        if verbose:
            print("{} rows dropped.".format(before_count - len(inter_df)))
        return inter_df
