"""Basic pdpipe PdPipelineStages."""

import numpy as np
import pandas as pd
import sortedcontainers as sc
import tqdm

from pdpipe.core import PdPipelineStage
from pdpipe.util import out_of_place_col_insert, get_numeric_column_names
from pdpipe.sklearn_stages import Impute                

from pdpipe.shared import _interpret_columns_param, _list_str

from .exceptions import PipelineApplicationError


class Bin(PdPipelineStage):
    """A pipeline stage that adds a binned version of a column or columns.

    If drop is set to True the new columns retain the names of the source
    columns; otherwise, the resulting column gain the suffix '_bin'

    Parameters
    ----------
    bin_map : dict
        Maps column labels to bin arrays. The bin array is interpreted as
        containing start points of consecutive bins, except for the final
        point, assumed to be the end point of the last bin. Additionally, a
        bin array implicitly projects a left-most bin containing all elements
        smaller than the left-most end point and a right-most bin containing
        all elements larger that the right-most end point. For example, the
        list [0, 5, 8] is interpreted as the bins (-∞, 0),
        [0-5), [5-8) and [8, ∞).
    drop : bool, default True
        If set to True, the source columns are dropped after being binned.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[-3],[4],[5], [9]], [1,2,3, 4], ['speed'])
        >>> pdp.Bin({'speed': [5]}, drop=False).apply(df)
           speed speed_bin
        1     -3        <5
        2      4        <5
        3      5        5≤
        4      9        5≤
        >>> pdp.Bin({'speed': [0,5,8]}, drop=False).apply(df)
           speed speed_bin
        1     -3        <0
        2      4       0-5
        3      5       5-8
        4      9        8≤
    """

    _DEF_BIN_EXC_MSG = (
        "Bin stage failed because not all columns "
        "{} were found in input dataframe."
    )
    _DEF_BIN_APP_MSG = "Binning column{} {}..."

    def _default_desc(self):
        string = ""
        columns = list(self._bin_map.keys())
        col1 = columns[0]
        string += "Bin {} by {},\n".format(col1, self._bin_map[col1])
        for col in columns[1:]:
            string += "bin {} by {},\n".format(col, self._bin_map[col])
        string = string[0:-2] + "."
        return string

    def __init__(self, bin_map, drop=True, **kwargs):
        self._bin_map = bin_map
        self._drop = drop
        columns_str = _list_str(list(bin_map.keys()))
        super_kwargs = {
            "exmsg": Bin._DEF_BIN_EXC_MSG.format(columns_str),
            "appmsg": Bin._DEF_BIN_APP_MSG.format(
                "s" if len(bin_map) > 1 else "", columns_str
            ),
            "desc": self._default_desc(),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._bin_map.keys()).issubset(df.columns)

    @staticmethod
    def _get_col_binner(bin_list):
        sorted_bins = sc.SortedList(bin_list)
        last_ix = len(sorted_bins) - 1

        def _col_binner(val):
            if val in sorted_bins:
                ind = sorted_bins.bisect(val) - 1
                if ind == last_ix:
                    return "{}≤".format(sorted_bins[-1])
                return "{}-{}".format(sorted_bins[ind], sorted_bins[ind + 1])
            try:
                ind = sorted_bins.bisect(val)
                if ind == 0:
                    return "<{}".format(sorted_bins[ind])
                return "{}-{}".format(sorted_bins[ind - 1], sorted_bins[ind])
            except IndexError:
                return "{}≤".format(sorted_bins[sorted_bins.bisect(val) - 1])

        return _col_binner

    def _transform(self, df, verbose):
        inter_df = df
        colnames = list(self._bin_map.keys())
        if verbose:
            colnames = tqdm.tqdm(colnames)
        for colname in colnames:
            if verbose:
                colnames.set_description(colname)
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_bin"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.apply(
                    self._get_col_binner(self._bin_map[colname])
                ),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class OneHotEncode(PdPipelineStage):
    """A pipeline stage that one-hot-encodes categorical columns.

    By default only k-1 dummies are created fo k categorical levels, as to
    avoid perfect multicollinearity between the dummy features (also called
    the dummy variabletrap). This is done since features are usually one-hot
    encoded for use with linear models, which require this behaviour.

    Parameters
    ----------
    columns : single label or list-like, default None
        Column labels in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be converted, except
        those given in the exclude_columns parameter.
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    exclude_columns : str or list-like, default None
        Name or names of categorical columns to be excluded from encoding
        when the columns parameter is not given. If None no column is excluded.
        Ignored if the columns parameter is given.
    col_subset : bool, default False
        If set to True, and only a subset of given columns is found, they are
        encoded (if the missing columns are encoutered after the stage is
        fitted they will be ignored). Otherwise, the stage will fail on the
        precondition requiring all given columns are in input dataframes.
    drop_first : bool or single label, default True
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level. If a non bool argument matching one of the categories is
        provided, the dummy column corresponding to this value is dropped
        instead of the first level; if it matches no category the first
        category will still be dropped.
    drop : bool, default True
        If set to True, the source columns are dropped after being encoded.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([['USA'], ['UK'], ['Greece']], [1,2,3], ['Born'])
        >>> pdp.OneHotEncode().apply(df)
           Born_UK  Born_USA
        1        0         1
        2        1         0
        3        0         0
    """

    _DEF_1HENCODE_EXC_MSG = (
        "OneHotEncode stage failed because not all columns "
        "{} were found in input dataframe."
    )
    _DEF_1HENCODE_APP_MSG = "One-hot encoding {}..."

    class _FitterEncoder(object):
        def __init__(self, col_name, dummy_columns):
            self.col_name = col_name
            self.dummy_columns = dummy_columns

        def __call__(self, value):
            this_dummy = "{}_{}".format(self.col_name, value)
            return pd.Series(
                data=[
                    int(this_dummy == dummy_col)
                    for dummy_col in self.dummy_columns
                ],
                index=self.dummy_columns,
            )

    def __init__(
        self,
        columns=None,
        dummy_na=False,
        exclude_columns=None,
        col_subset=False,
        drop_first=True,
        drop=True,
        **kwargs
    ):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns)
        self._dummy_na = dummy_na
        if exclude_columns is None:
            self._exclude_columns = []
        else:
            self._exclude_columns = _interpret_columns_param(exclude_columns)
        self._col_subset = col_subset
        self._drop_first = drop_first
        self._drop = drop
        self._dummy_col_map = {}
        self._encoder_map = {}
        col_str = _list_str(self._columns)
        super_kwargs = {
            "exmsg": OneHotEncode._DEF_1HENCODE_EXC_MSG.format(col_str),
            "appmsg": OneHotEncode._DEF_1HENCODE_APP_MSG.format(
                col_str or "all columns"
            ),
            "desc": "One-hot encode {}".format(
                col_str or "all categorical columns"
            ),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        if self._col_subset:
            return True
        return set(self._columns or []).issubset(df.columns)

    def _fit_transform(self, df, verbose):
        columns_to_encode = self._columns
        if self._columns is None:
            columns_to_encode = list(
                set(
                    df.select_dtypes(include=["object", "category"]).columns
                ).difference(self._exclude_columns)
            )
        if self._col_subset:
            columns_to_encode = [
                x for x in columns_to_encode if x in df.columns
            ]
        self._cols_to_encode = columns_to_encode
        assign_map = {}
        if verbose:
            columns_to_encode = tqdm.tqdm(columns_to_encode)
        for colname in columns_to_encode:
            if verbose:
                columns_to_encode.set_description(colname)
            dummies = pd.get_dummies(
                df[colname],
                drop_first=False,
                dummy_na=self._dummy_na,
                prefix=colname,
                prefix_sep="_",
            )
            nan_col = colname + "_nan"
            if self._drop_first:
                dfirst_col = colname + "_" + str(self._drop_first)
                if dfirst_col in dummies:
                    if verbose:
                        print(
                            (
                                "Dropping {} dummy column instead of first "
                                "column when one-hot encoding {}."
                            ).format(dfirst_col, colname)
                        )
                    dummies.drop(dfirst_col, axis=1, inplace=True)
                elif nan_col in dummies:
                    dummies.drop(nan_col, axis=1, inplace=True)
                else:
                    dummies.drop(dummies.columns[0], axis=1, inplace=True)
            self._dummy_col_map[colname] = list(dummies.columns)
            self._encoder_map[colname] = OneHotEncode._FitterEncoder(
                colname, list(dummies.columns)
            )
            for column in dummies:
                assign_map[column] = dummies[column]

        inter_df = df.assign(**assign_map)
        self.is_fitted = True
        if self._drop:
            return inter_df.drop(columns_to_encode, axis=1)
        return inter_df

    def _transform(self, df, verbose):
        assign_map = {}
        for colname in self._cols_to_encode:
            col = df[colname]
            encoder = self._encoder_map[colname]
            res_cols = col.apply(encoder)
            for res_col in res_cols:
                assign_map[res_col] = res_cols[res_col]
        inter_df = df.assign(**assign_map)
        if self._drop:
            return inter_df.drop(self._cols_to_encode, axis=1)
        return inter_df


class MapColVals(PdPipelineStage):
    """A pipeline stage that replaces the values of a column by a map.

    Parameters
    ----------
    columns : single label or list-like
        Column labels in the DataFrame to be mapped.
    value_map : dict, function or pandas.Series
        A dictionary mapping existing values to new ones. Values not in the
        dictionary as keys will be converted to NaN. If a function is given, it
        is applied element-wise to given columns. If a Series is given, values
        are mapped by its index to its values.
    result_columns : single label or list-like, default None
        Labels for the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, then the label of the source column is
        used; otherwise, the label of the source column is used with the suffix
        '_map'.
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    suffix : str, default '_map'
        The suffix mapped columns gain if no new column labels are given.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> df = pd.DataFrame([[1], [3], [2]], ['UK', 'USSR', 'US'], ['Medal'])
        >>> value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
        >>> pdp.MapColVals('Medal', value_map).apply(df)
               Medal
        UK      Gold
        USSR  Bronze
        US    Silver
    """

    _DEF_MAP_COLVAL_EXC_MSG = (
        "MapColVals stage failed because column{} "
        "{} were not found in input dataframe."
    )
    _DEF_MAP_COLVAL_APP_MSG = "Mapping values of column{} {} with {}..."

    def __init__(
        self,
        columns,
        value_map,
        result_columns=None,
        drop=True,
        suffix=None,
        **kwargs
    ):
        self._columns = _interpret_columns_param(columns)
        self._value_map = value_map
        if suffix is None:
            suffix = "_map"
        self.suffix = suffix
        if result_columns is None:
            if drop:
                self._result_columns = self._columns
            else:
                self._result_columns = [
                    col + self.suffix for col in self._columns
                ]
        else:
            self._result_columns = _interpret_columns_param(result_columns)
            if len(self._result_columns) != len(self._columns):
                raise ValueError(
                    "columns and result_columns parameters must"
                    " be string lists of the same length!"
                )
        col_str = _list_str(self._columns)
        sfx = "s" if len(self._columns) > 1 else ""
        self._drop = drop
        super_kwargs = {
            "exmsg": MapColVals._DEF_MAP_COLVAL_EXC_MSG.format(sfx, col_str),
            "appmsg": MapColVals._DEF_MAP_COLVAL_APP_MSG.format(
                sfx, col_str, self._value_map
            ),
            "desc": "Map values of column{} {} with {}.".format(
                sfx, col_str, self._value_map
            ),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

    def _transform(self, df, verbose):
        inter_df = df
        for i, colname in enumerate(self._columns):
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = self._result_columns[i]
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.map(self._value_map),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


def _always_true(x):
    return True


class ApplyToRows(PdPipelineStage):
    """A pipeline stage generating columns by applying a function to each row.

    Parameters
    ----------
    func : function
        The function to be applied to each row of the processed DataFrame.
    colname : single label, default None
        The label of the new column resulting from the function application. If
        None, 'new_col' is used. Ignored if a DataFrame is generated by the
        function (i.e. each row generates a Series rather than a value), in
        which case the laebl of each column in the resulting DataFrame is used.
    follow_column : str, default None
        Resulting columns will be inserted after this column. If None, new
        columns are inserted at the end of the processed DataFrame.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.
    prec : function, default None
        A function taking a DataFrame, returning True if it this stage is
        applicable to the given DataFrame. If None is given, a function always
        returning True is used.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3, 2143], [10, 1321], [7, 1255]]
        >>> df = pd.DataFrame(data, [1,2,3], ['years', 'avg_revenue'])
        >>> total_rev = lambda row: row['years'] * row['avg_revenue']
        >>> add_total_rev = pdp.ApplyToRows(total_rev, 'total_revenue')
        >>> add_total_rev(df)
           years  avg_revenue  total_revenue
        1      3         2143           6429
        2     10         1321          13210
        3      7         1255           8785
        >>> def halfer(row):
        ...     new = {'year/2': row['years']/2, 'rev/2': row['avg_revenue']/2}
        ...     return pd.Series(new)
        >>> half_cols = pdp.ApplyToRows(halfer, follow_column='years')
        >>> half_cols(df)
           years   rev/2  year/2  avg_revenue
        1      3  1071.5     1.5         2143
        2     10   660.5     5.0         1321
        3      7   627.5     3.5         1255
    """

    _DEF_APPLYTOROWS_EXC_MSG = "Applying function {} failed."
    _DEF_APPLYTOROWS_APP_MSG = "Applying function {}..."
    _DEF_COLNAME = "new_col"

    def __init__(
        self,
        func,
        colname=None,
        follow_column=None,
        func_desc=None,
        prec=None,
        **kwargs
    ):
        if colname is None:
            colname = ApplyToRows._DEF_COLNAME
        if func_desc is None:
            func_desc = ""
        if prec is None:
            prec = _always_true
        self._func = func
        self._colname = colname
        self._follow_column = follow_column
        self._func_desc = func_desc
        self._prec_func = prec
        super_kwargs = {
            "exmsg": ApplyToRows._DEF_APPLYTOROWS_EXC_MSG.format(func_desc),
            "appmsg": ApplyToRows._DEF_APPLYTOROWS_APP_MSG.format(func_desc),
            "desc": "Generating a column with a function {}.".format(
                self._func_desc
            ),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return self._prec_func(df)

    def _transform(self, df, verbose):
        new_cols = df.apply(self._func, axis=1)
        if isinstance(new_cols, pd.Series):
            loc = len(df.columns)
            if self._follow_column:
                loc = df.columns.get_loc(self._follow_column) + 1
            return out_of_place_col_insert(
                df=df, series=new_cols, loc=loc, column_name=self._colname
            )
        elif isinstance(new_cols, pd.DataFrame):
            sorted_cols = sorted(list(new_cols.columns))
            new_cols = new_cols[sorted_cols]
            if self._follow_column:
                inter_df = df
                loc = df.columns.get_loc(self._follow_column) + 1
                for colname in new_cols.columns:
                    inter_df = out_of_place_col_insert(
                        df=inter_df,
                        series=new_cols[colname],
                        loc=loc,
                        column_name=colname,
                    )
                    loc += 1
                return inter_df
            assign_map = {
                colname: new_cols[colname] for colname in new_cols.columns
            }
            return df.assign(**assign_map)
        raise TypeError(  # pragma: no cover
            "Unexpected type generated by applying a function to a DataFrame."
            " Only Series and DataFrame are allowed."
        )


class ApplyByCols(PdPipelineStage):
    """A pipeline stage applying an element-wise function to columns.

    Parameters
    ----------
    columns : str or list-like
        Names of columns on which to apply the given function.
    func : function
        The function to be applied to each element of the given columns.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with the suffix
        '_app'.
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import math;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> round_ph = pdp.ApplyByCols("ph", math.ceil)
        >>> round_ph(df)
           ph  lbl
        1   4  acd
        2   8  alk
        3  13  alk
    """

    _BASE_STR = "Applying a function {} to column{} {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."

    def __init__(
        self,
        columns,
        func,
        result_columns=None,
        drop=True,
        func_desc=None,
        **kwargs
    ):
        self._columns = _interpret_columns_param(columns)
        self._func = func
        if result_columns is None:
            if drop:
                self._result_columns = self._columns
            else:
                self._result_columns = [col + "_app" for col in self._columns]
        else:
            self._result_columns = _interpret_columns_param(result_columns)
            if len(self._result_columns) != len(self._columns):
                raise ValueError(
                    "columns and result_columns parameters must"
                    " be string lists of the same length!"
                )
        self._drop = drop
        if func_desc is None:
            func_desc = ""
        self._func_desc = func_desc
        col_str = _list_str(self._columns)
        sfx = "s" if len(self._columns) > 1 else ""
        base_str = ApplyByCols._BASE_STR.format(self._func_desc, sfx, col_str)
        super_kwargs = {
            "exmsg": base_str + ApplyByCols._DEF_EXC_MSG_SUFFIX,
            "appmsg": base_str + ApplyByCols._DEF_APP_MSG_SUFFIX,
            "desc": base_str + ApplyByCols._DEF_DESCRIPTION_SUFFIX,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

    def _transform(self, df, verbose):
        inter_df = df
        for i, colname in enumerate(self._columns):
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = self._result_columns[i]
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.apply(self._func),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class ColByFrameFunc(PdPipelineStage):
    """A pipeline stage adding a column by applying a dataframw-wide function.

    Parameters
    ----------
    result_columns : str, default None
        The name of the resulting column.
    func : function
        The function to be applied to the input dataframe. The function should
        return a pandas.Series object.
    follow_column : str, default None
        Resulting columns will be inserted after this column. If None, new
        columns are inserted at the end of the processed DataFrame.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3, 3], [2, 4], [1, 5]]
        >>> df = pd.DataFrame(data, [1,2,3], ["A","B"])
        >>> func = lambda df: df['A'] == df['B']
        >>> add_equal = pdp.ColByFrameFunc("A==B", func)
        >>> add_equal(df)
           A  B   A==B
        1  3  3   True
        2  2  4  False
        3  1  5  False
    """

    _BASE_STR = "Applying a function{} to column {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."

    def __init__(
        self,
        func,
        result_columns=None,
        follow_column=None,
        func_desc=None,
        **kwargs
    ):
        self._func = func
        self._result_columns = result_columns if result_columns is not None \
            else None
        self._follow_column = follow_column
        if func_desc is None:
            func_desc = ""
        else:
            func_desc = " " + func_desc
        self._func_desc = func_desc
        base_str = ColByFrameFunc._BASE_STR.format(self._func_desc, result_columns)
        super_kwargs = {
            "exmsg": base_str + ColByFrameFunc._DEF_EXC_MSG_SUFFIX,
            "appmsg": base_str + ColByFrameFunc._DEF_APP_MSG_SUFFIX,
            "desc": base_str + ColByFrameFunc._DEF_DESCRIPTION_SUFFIX,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _transform(self, df, verbose):
        inter_df = df
        try:
            new_col = self._func(df)
        except Exception:
            raise PipelineApplicationError(
                "Exception raised applying function{} to dataframe.".format(
                    self._func_desc
                )
            )
        if self._follow_column:
            loc = df.columns.get_loc(self._follow_column) + 1
        else:
            loc = len(df.columns)
        inter_df = out_of_place_col_insert(
            df=inter_df, 
            series=new_col,
            loc=loc,
            column_name=self._result_columns
        )
        return inter_df


class AggByCols(PdPipelineStage):
    """A pipeline stage applying a series-wise function to columns.

    Parameters
    ----------
    columns : str or list-like
        Names of columns on which to apply the given function.
    func : function
        The function to be applied to each element of the given columns.
    result_columns : str or list-like, default None
        The names of the new columns resulting from the mapping operation. Must
        be of the same length as columns. If None, behavior depends on the
        drop parameter: If drop is True, the name of the source column is used;
        otherwise, the name of the source column is used with a defined suffix.
    drop : bool, default True
        If set to True, source columns are dropped after being mapped.
    func_desc : str, default None
        A function description of the given function; e.g. 'normalizing revenue
        by company size'. A default description is used if None is given.
    suffix : str, optional
        The suffix to add to resulting columns in case where results_columns
        is None and drop is set to False. Of not given, defaults to '_agg'.


    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp; import numpy as np;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> log_ph = pdp.ApplyByCols("ph", np.log)
        >>> log_ph(df)
                 ph  lbl
        1  1.163151  acd
        2  1.974081  alk
        3  2.493205  alk
    """

    _BASE_STR = "Applying a function {} to column{} {}"
    _DEF_EXC_MSG_SUFFIX = " failed."
    _DEF_APP_MSG_SUFFIX = "..."
    _DEF_DESCRIPTION_SUFFIX = "."
    _DEF_COLNAME_SUFFIX = "_agg"

    def __init__(
        self,
        columns,
        func,
        result_columns=None,
        drop=True,
        func_desc=None,
        suffix=None,
        **kwargs
    ):
        if suffix is None:
            suffix = AggByCols._DEF_COLNAME_SUFFIX
        self._suffix = suffix
        self._columns = _interpret_columns_param(columns)
        self._func = func
        if result_columns is None:
            if drop:
                self._result_columns = self._columns
            else:
                self._result_columns = [col + suffix for col in self._columns]
        else:
            self._result_columns = _interpret_columns_param(result_columns)
            if len(self._result_columns) != len(self._columns):
                raise ValueError(
                    "columns and result_columns parameters must"
                    " be string lists of the same length!"
                )
        self._drop = drop
        if func_desc is None:
            func_desc = ""
        self._func_desc = func_desc
        col_str = _list_str(self._columns)
        sfx = "s" if len(self._columns) > 1 else ""
        base_str = ApplyByCols._BASE_STR.format(self._func_desc, sfx, col_str)
        super_kwargs = {
            "exmsg": base_str + ApplyByCols._DEF_EXC_MSG_SUFFIX,
            "appmsg": base_str + ApplyByCols._DEF_APP_MSG_SUFFIX,
            "desc": base_str + ApplyByCols._DEF_DESCRIPTION_SUFFIX,
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns).issubset(df.columns)

    def _transform(self, df, verbose):
        inter_df = df
        for i, colname in enumerate(self._columns):
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = self._result_columns[i]
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=source_col.agg(self._func),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class Log(PdPipelineStage):
    """A pipeline stage that log-transforms numeric data.

    Parameters
    ----------
    columns : str or list-like, default None
        Column names in the DataFrame to be encoded. If columns is None then
        all the columns with a numeric dtype will be transformed, except those
        given in the exclude_columns parameter.
    exclude : str or list-like, default None
        Name or names of numeric columns to be excluded from log-transforming
        when the columns parameter is not given. If None no column is excluded.
        Ignored if the columns parameter is given.
    drop : bool, default False
        If set to True, the source columns are dropped after being encoded,
        and the resulting encoded columns retain the names of the source
        columns. Otherwise, encoded columns gain the suffix '_log'.
    non_neg : bool, default False
        If True, each transformed column is first shifted by smallest negative
        value it includes (non-negative columns are thus not shifted).
    const_shift : int, optional
        If given, each transformed column is first shifted by this constant. If
        non_neg is True then that transformation is applied first, and only
        then is the column shifted by this constant.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> log_stage = pdp.Log("ph", drop=True)
        >>> log_stage(df)
                 ph  lbl
        1  1.163151  acd
        2  1.974081  alk
        3  2.493205  alk
    """

    _DEF_LOG_EXC_MSG = (
        "Log stage failed because not all columns "
        "{} were found in input dataframe."
    )
    _DEF_LOG_APP_MSG = "Log-transforming {}..."

    def __init__(
        self,
        columns=None,
        exclude=None,
        drop=False,
        non_neg=False,
        const_shift=None,
        **kwargs
    ):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns)
        if exclude is None:
            self._exclude = []
        else:
            self._exclude = _interpret_columns_param(exclude)
        self._drop = drop
        self._non_neg = non_neg
        self._const_shift = const_shift
        self._col_to_minval = {}
        col_str = "all numeric columns"
        if self._columns:
            col_str = _list_str(self._columns)
        super_kwargs = {
            "exmsg": Log._DEF_LOG_EXC_MSG.format(col_str),
            "appmsg": Log._DEF_LOG_APP_MSG.format(col_str),
            "desc": "Log-transform {}".format(col_str),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _fit_transform(self, df, verbose):
        columns_to_transform = self._columns
        if self._columns is None:
            columns_to_transform = get_numeric_column_names(df)
        columns_to_transform = list(
            set(columns_to_transform).difference(self._exclude)
        )
        self._cols_to_transform = columns_to_transform
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        inter_df = df
        for colname in columns_to_transform:
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_log"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            new_col = source_col
            if self._non_neg:
                minval = min(new_col)
                if minval < 0:
                    new_col = new_col + abs(minval)
                    self._col_to_minval[colname] = abs(minval)
            # must check not None as neg numbers eval to False
            if self._const_shift is not None:
                new_col = new_col + self._const_shift
            new_col = np.log(new_col)
            inter_df = out_of_place_col_insert(
                df=inter_df, series=new_col, loc=loc, column_name=new_name
            )
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        inter_df = df
        columns_to_transform = self._cols_to_transform
        if verbose:
            columns_to_transform = tqdm.tqdm(columns_to_transform)
        for colname in columns_to_transform:
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_log"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            new_col = source_col
            if self._non_neg:
                if colname in self._col_to_minval:
                    absminval = self._col_to_minval[colname]
                    new_col = new_col + absminval
            # must check not None as neg numbers eval to False
            if self._const_shift is not None:
                new_col = new_col + self._const_shift
            new_col = np.log(new_col)
            inter_df = out_of_place_col_insert(
                df=inter_df, series=new_col, loc=loc, column_name=new_name
            )
        return inter_df

class PivotCols(PdPipelineStage):
    """Spread columns to make columns out of categorical values

    Parameters

    ----------

    columns: str or list-like
       names of columns to spread

    values: str or list-like
        values significant to these categories

    impute_with: Impute or param, default None
        Impute instance to be applied to newly-pivoted columns or a value
        to be passed onto the Impute constructor. If None, columns missing 
        values will remain
    """

    _DEF_PIVOTCOLS_EXC_MSG = "Pivoting columns {} with values {} failed"
    _DEF_PIVOTCOLS_APP_MSG = "Pivoting columns {} with values {} ..."


    def __init__(
        self, 
        columns, 
        values, 
        impute_with=None,
        **kwargs):

        """Initialize object

        impute_with: how to impute missing values, can be one of
        - None, in which case, missing values will remain untouched
        - a constant to imput missing values with (imputer will be created)
        - an Imputer strategy ("mean", "median", "most_frequent")
        - a pre-created imputer
        """
        self.columns = _interpret_columns_param(columns)
        self.values = _interpret_columns_param(values)

        if not isinstance(impute_with, Impute) and impute_with is not None:
            impute_with = Impute(imputer=impute_with)

        self._imputer = impute_with

        super_kwargs = {
            "exmsg": PivotCols._DEF_PIVOTCOLS_EXC_MSG.format(columns, values),
            "appmsg": PivotCols._DEF_PIVOTCOLS_APP_MSG.format(columns, values),
            "desc": "Pivot columns {} with values {}.".format(columns, values) 
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)


    def _transform(self, df, verbose):
        if self._imputer is not None:
            new_df = pd.pivot_table(df, columns=self.columns, values=self.values)
            new_cols = list(set(new_df.columns) - set(df.columns))
            return self._imputer._set_cols(new_cols).apply(new_df)
        else:
            return pd.pivot_table(df, columns=self.columns, values=self.values)

