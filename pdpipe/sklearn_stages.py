"""PdPipeline stages dependent on the scikit-learn Python library.

Please note that the scikit-learn Python package must be installed for the
stages in this module to work.

When attempting to load stages from this module, pdpipe will first attempt to
import sklearn. If it fails, it will issue a warning, will not import any of
the pipeline stages that make up this module, and continue to load other
pipeline stages.
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.impute import SimpleImputer
import tqdm
from skutil.preprocessing import scaler_by_params

from pdpipe.core import PdPipelineStage
from pdpipe.util import out_of_place_col_insert
from pdpipe.shared import _interpret_columns_param, _list_str

from .exceptions import PipelineApplicationError


class Encode(PdPipelineStage):
    """A pipeline stage that encodes categorical columns to integer values.

    The encoder for each column is saved in the attribute 'encoders', which
    is a dict mapping each encoded column name to the
    sklearn.preprocessing.LabelEncoder object used to encode it.

    Parameters
    ----------
    columns : str or list-like, default None
        Column names in the DataFrame to be encoded. If columns is None then
        all the columns with object or category dtype will be encoded, except
        those given in the exclude_columns parameter.
    exclude_columns : str or list-like, default None
        Name or names of categorical columns to be excluded from encoding
        when the columns parameter is not given. If None no column is excluded.
        Ignored if the columns parameter is given.
    drop : bool, default True
        If set to True, the source columns are dropped after being encoded,
        and the resulting encoded columns retain the names of the source
        columns. Otherwise, encoded columns gain the suffix '_enc'.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, "acd"], [7.2, "alk"], [12.1, "alk"]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","lbl"])
        >>> encode_stage = pdp.Encode("lbl")
        >>> encode_stage(df)
             ph  lbl
        1   3.2    0
        2   7.2    1
        3  12.1    1
        >>> encode_stage.encoders["lbl"].inverse_transform([0,1,1])
        array(['acd', 'alk', 'alk'], dtype=object)
    """

    _DEF_ENCODE_EXC_MSG = (
        "Encode stage failed because not all columns "
        "{} were found in input dataframe."
    )
    _DEF_ENCODE_APP_MSG = "Encoding {}..."

    def __init__(
        self, columns=None, exclude_columns=None, drop=True, **kwargs
    ):
        if columns is None:
            self._columns = None
        else:
            self._columns = _interpret_columns_param(columns)
        if exclude_columns is None:
            self._exclude_columns = []
        else:
            self._exclude_columns = _interpret_columns_param(exclude_columns)
        self._drop = drop
        self.encoders = {}
        col_str = _list_str(self._columns)
        super_kwargs = {
            "exmsg": Encode._DEF_ENCODE_EXC_MSG.format(col_str),
            "appmsg": Encode._DEF_ENCODE_APP_MSG.format(col_str),
            "desc": "Encode {}".format(col_str or "all categorical columns"),
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _fit_transform(self, df, verbose):
        self.encoders = {}
        columns_to_encode = self._columns
        if self._columns is None:
            columns_to_encode = list(
                set(
                    df.select_dtypes(include=["object", "category"]).columns
                ).difference(self._exclude_columns)
            )
        if verbose:
            columns_to_encode = tqdm.tqdm(columns_to_encode)
        inter_df = df
        for colname in columns_to_encode:
            lbl_enc = sklearn.preprocessing.LabelEncoder()
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=lbl_enc.fit_transform(source_col),
                loc=loc,
                column_name=new_name,
            )
            self.encoders[colname] = lbl_enc
        self.is_fitted = True
        return inter_df

    def _transform(self, df, verbose):
        inter_df = df
        for colname in self.encoders:
            lbl_enc = self.encoders[colname]
            source_col = df[colname]
            loc = df.columns.get_loc(colname) + 1
            new_name = colname + "_enc"
            if self._drop:
                inter_df = inter_df.drop(colname, axis=1)
                new_name = colname
                loc -= 1
            inter_df = out_of_place_col_insert(
                df=inter_df,
                series=lbl_enc.transform(source_col),
                loc=loc,
                column_name=new_name,
            )
        return inter_df


class Scale(PdPipelineStage):
    """A pipeline stage that scale"review_creation_date"s data.

    Parameters
    ----------
    scaler : str
        The type of scaler to use to scale the data. One of 'StandardScaler',
        'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'
        and 'Normalizer'.
    exclude_columns : str or list-like, default None
        Name or names of columns to be excluded from scaling. Excluded columns
        are appended to the end of the resulting dataframe.
    exclude_object_columns : bool, default True
        If set to True, all columns of dtype object are added to the list of
        columns excluded from scaling.
    **kwargs : extra keyword arguments
        All valid extra keyword arguments are forwarded to the scaler
        constructor on scaler creation (e.g. 'n_quantiles' for
        QuantileTransformer). PdPipelineStage valid keyword arguments are used
        to override Scale class defaults.

    Example
    -------
        >>> import pandas as pd; import pdpipe as pdp;
        >>> data = [[3.2, 0.3], [7.2, 0.35], [12.1, 0.29]]
        >>> df = pd.DataFrame(data, [1,2,3], ["ph","gt"])
        >>> scale_stage = pdp.Scale("StandardScaler")
        >>> scale_stage(df)
                 ph        gt
        1 -1.181449 -0.508001
        2 -0.082427  1.397001
        3  1.263876 -0.889001
    """

    _DESC_PREFIX = "Scale data"
    _DEF_SCALE_EXC_MSG = "Scale stage failed."
    _DEF_SCALE_APP_MSG = "Scaling data..."

    def __init__(
        self,
        scaler,
        exclude_columns=None,
        exclude_object_columns=True,
        **kwargs
    ):
        self.scaler = scaler
        if exclude_columns is None:
            self._exclude_columns = []
            desc_suffix = "."
        else:
            self._exclude_columns = _interpret_columns_param(exclude_columns)
            col_str = _list_str(self._exclude_columns)
            desc_suffix = " except columns {}.".format(col_str)
        self._exclude_obj_cols = exclude_object_columns
        super_kwargs = {
            "exmsg": Scale._DEF_SCALE_EXC_MSG,
            "appmsg": Scale._DEF_SCALE_APP_MSG,
            "desc": Scale._DESC_PREFIX + desc_suffix,
        }
        self._kwargs = kwargs
        valid_super_kwargs = super()._init_kwargs()
        for key in kwargs:
            if key in valid_super_kwargs:
                super_kwargs[key] = kwargs[key]
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return True

    def _fit_transform(self, df, verbose):
        cols_to_exclude = self._exclude_columns.copy()
        if self._exclude_obj_cols:
            obj_cols = list((df.dtypes[df.dtypes == object]).index)
            obj_cols = [x for x in obj_cols if x not in cols_to_exclude]
            cols_to_exclude += obj_cols
        self._col_order = list(df.columns)
        if cols_to_exclude:
            excluded = df[cols_to_exclude]
            apply_to = df[
                [col for col in df.columns if col not in cols_to_exclude]
            ]
        else:
            apply_to = df
        self._scaler = scaler_by_params(self.scaler, **self._kwargs)
        try:
            res = pd.DataFrame(
                data=self._scaler.fit_transform(apply_to),
                index=apply_to.index,
                columns=apply_to.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    apply_to.columns
                )
            )
        if cols_to_exclude:
            res = pd.concat([res, excluded], axis=1)
            res = res[self._col_order]
        self.is_fitted = True
        return res

    def _transform(self, df, verbose):
        cols_to_exclude = self._exclude_columns.copy()
        if self._exclude_obj_cols:
            obj_cols = list((df.dtypes[df.dtfeaturesypes == object]).index)
            obj_cols = [x for x in obj_cols if x not in cols_to_exclude]
            cols_to_exclude += obj_cols
        self._col_order = list(df.columns)
        if cols_to_exclude:
            excluded = df[cols_to_exclude]
            apply_to = df[
                [col for col in df.columns if col not in cols_to_exclude]
            ]
        else:
            apply_to = df
        try:
            res = pd.DataFrame(
                data=self._scaler.transform(apply_to),
                index=apply_to.index,
                columns=apply_to.columns,
            )
        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Scale applied to columns {}".format(
                    apply_to.columns
                )
            )
        if cols_to_exclude:
            res = pd.concat([res, excluded], axis=1)
            res = res[self._col_order]
            
        return res


class Impute(PdPipelineStage):
    """Impute missing values given a strategy

    Parameters

    ----------

    columns: str or list-like, default None
        Columns to be imputed. If None, all columns will be imputed.

    imputer: hashable, hasattr(imputer, "fit_transform"), dict or function, 
             default None
        If hashable, this will be imputed as a constant using an 
        sklearn..SimpleImputer. If imputer has attribute "fit_transform"
        assumes this will return imputed values. If dict, must have na_value as
        key for mapping. If function without a "fit_transform" method, 
        each column to be imputed will be passed onto it and the output is 
        expected to be a series or array of the same length as before
        If None, sklearn..SimpleImputer with default values will be used.

    na_values: comparable or list like of such, default np.nan
        Value or values to be imputed. 
    """

    _DEF_IMPUTE_EXC_MSG = (
        "Imputation failed because not all columns "
        "{} where found in the data frame."
    )

    _DEF_IMPUTE_APP_MSG = "Imputing {}"

    def __init__(
        self, columns=None, imputer=None, na_values=np.nan, **kwargs
    ):
        self._columns = None if columns is None \
            else _interpret_columns_param(columns)

        self._na_values = na_values

        if imputer is None:
            self._imputer = SimpleImputer(missing_values=self._na_values) 
        elif _ishashable(imputer):
            self._imputer = SimpleImputer(
                                missing_values=self._na_values,
                                strategy="constant",
                                fill_value=imputer
                            )
        elif isinstance(imputer, dict):
            if self._na_values in imputer.keys():
                self._imputer = imputer
            else:
                raise ValueError("{} must be a key in imputer"\
                                    .format(self._na_values))
        else:
            self._imputer = imputer

        super_kwargs = {
            "exmsg": Impute._DEF_IMPUTE_EXC_MSG.format(str(self._columns)),
            "appmsg": Impute._DEF_IMPUTE_APP_MSG.format(str(self._columns)),
            "desc": "Impute {} by filling NaN values".format(
                str(self._columns)
            )
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def set_imputer(self, imputer):
        self._imputer = imputer

    def _set_cols(self, columns):
        self._columns = _interpret_columns_param(columns)
        return self

    def set_simple_imputer(self, strategy="mean", **kwargs):
        """Set imputer to be an sklearn..SimpleImputer with given arguments
        """
        self._imputer = SimpleImputer(
                            missing_values=self._na_values,
                            strategy=strategy,
                            **kwargs
                        )

    def _fit_transform(self, df, verbose):

        columns_to_impute = self._columns

        if self._columns is None:
            columns_to_impute = df.columns

        if verbose:
            tqdm.tqdm(columns_to_impute)

        try:
            if hasattr(self._imputer, "fit_transform"):
                imputed_cols = self._imputer\
                    .fit_transform(df[columns_to_impute])
            elif isinstance(self._imputer, dict):
                imputed_cols = df[[columns_to_impute]].replace(self._imputer)
            else:
                imputed_cols = df[[columns_to_impute]]
                for _, col in enumerate(columns_to_impute):
                    imputed_cols[col] = self._imputer(columns_to_impute[col])            

            df[columns_to_impute] = imputed_cols

        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Imputation applied to columns {}".format(
                    df.columns
                )
            )

        return df

    def _transform(self, df, verbose):

        columns_to_impute = self._columns

        if self._columns is None:
            columns_to_impute = df.columns

        if verbose:
            tqdm.tqdm(columns_to_impute)

        try:
            if hasattr(self._imputer, "transform"):
                return self._imputer.transform(df[columns_to_impute])
            else:
                return self._fit_transform(df, verbose)

        except Exception:
            raise PipelineApplicationError(
                "Exception raised when Imputation applied to columns {}".format(
                    df.columns
                )
            )

        return df


class SKStage(PdPipelineStage):
    """Apply a scikit learn transformation onto a given dataframe. This should
    not be used broadly as it cuts many paths, but it's made to be as flexible 
    as possible with the given transformations.

    Parameters

    ----------

    columns: str or list-like, default None
        Columns to be imputed. If None, all columns will be transformed.

    transformer: sklearn transformer with some "fit_transform" attribute
                 default None
       Scikit learn transformer to be used on data. If transformation leads to
       more columns than there k, see description of new_col_suffix

    drop: bool, default True
        If set to True, the source columns are dropped after being transformed,
        and the resulting transformed columns retain the names of the source
        columns. Otherwise, see description of new_col_suffix

    new_col_suffix: str, default to "new_col"
        Suffix to be added to new columns if 
            a) old column is not dropped 
            b) transformations gives a different number of columns than
               originally specified
        Columns will be named new_col_suffix+"_"+str(n) where n is the column
        relative to the newly transformed columns
    """


    def __init__(
        self, 
        columns=None,
        transformer=None,
        drop=True,
        new_col_suffix="new_col",
        **kwargs
    ):
        self._columns = None if columns is None \
            else _interpret_columns_param(columns)

        if hasattr(transformer, "fit_transform") or \
           (hasattr(transformer, "fit") and hasattr(transformer, "transform")):
           self._transformer=transformer
        else:
            raise ValueError("transformer must be transformable")

        self._drop = drop
        self._new_col_name_suffix = new_col_suffix        

        super_kwargs = {
            "exmsg": None,
            "appmesg": None,
            "desc": "Apply an scikit learn transformer on {}".format(
                str(self._columns)
            )
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df):
        return set(self._columns or []).issubset(df.columns)

    def _col_name_gen(self):
        i = 0
        while(True):
            yield "{}_{}".format(self._new_col_name_suffix, i)
            i += 1

    def _get_new_col_names(self, n):
        gen = self._col_name_gen()
        return pd.Index([next(gen) for _ in range(n)])

    def set_transformer(self, transformer):
        self._transformer = transformer

    def _fit_transform(self, df, verbose):

        columns_to_transform = self._columns

        if self._columns is None:
            columns_to_transform = df.columns

        if verbose:
            tqdm.tqdm(columns_to_transform)

        try:
            if hasattr(self._transformer, "fit_transform"):
                transformed_cols = self._transformer\
                    .fit_transform(df[columns_to_transform])
            else:
                transformed_cols = self._transformer\
                    .fit(df[columns_to_transform])\
                    .transform(df[columns_to_transform])

            if self._drop: 

                if len(transformed_cols.columns) == len(columns_to_transform):
                        df[columns_to_transform] = transformed_cols
                else:

                    loc = df.columns.get_loc(columns_to_transform[0])
                    gen = self._col_name_gen()

                    df.drop(columns_to_transform)
                    
                    for i in range(transformed_cols.shape[1]):
                        df.insert(loc, next(gen), transformed_cols[:, i])

            else:


                new_col_names = self._get_new_col_names(
                    transformed_cols.shape[1]
                )

                df[new_col_names] = transformed_cols

        except Exception:
            raise PipelineApplicationError(
                "Exception raised when transformation applied to columns {}".format(
                    df.columns
                )
            )

        return df

def _ishashable(x):
    try:
        hash(x)
    except TypeError:
        return False
    return True
