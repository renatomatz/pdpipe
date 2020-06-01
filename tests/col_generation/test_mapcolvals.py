"""Testing MapColVals pipeline stages."""

import pandas as pd
import pytest

from pdpipe.col_generation import MapColVals


def _test_df():
    return pd.DataFrame([[1], [3], [2]], ['UK', 'USSR', 'US'], ['Medal'])


def test_mapcolvals():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map).apply(df)
    assert res_df['Medal']['UK'] == 'Gold'
    assert res_df['Medal']['USSR'] == 'Bronze'
    assert res_df['Medal']['US'] == 'Silver'


def test_mapcolvals_no_drop():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map, drop=False).apply(df)
    assert res_df['Medal']['UK'] == 1
    assert res_df['Medal']['USSR'] == 3
    assert res_df['Medal']['US'] == 2
    assert res_df['Medal_map']['UK'] == 'Gold'
    assert res_df['Medal_map']['USSR'] == 'Bronze'
    assert res_df['Medal_map']['US'] == 'Silver'


def test_mapcolvals_with_res_name():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    res_df = MapColVals('Medal', value_map, result_columns='Metal').apply(df)
    assert res_df['Metal']['UK'] == 'Gold'
    assert res_df['Metal']['USSR'] == 'Bronze'
    assert res_df['Metal']['US'] == 'Silver'


def test_mapcolvals_with_res_name_no_drop():
    """Testing MapColVals pipeline stages."""
    df = _test_df()
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    map_stage = MapColVals(
        'Medal', value_map, result_columns='Metal', drop=False)
    res_df = map_stage(df)
    assert res_df['Medal']['UK'] == 1
    assert res_df['Medal']['USSR'] == 3
    assert res_df['Medal']['US'] == 2
    assert res_df['Metal']['UK'] == 'Gold'
    assert res_df['Metal']['USSR'] == 'Bronze'
    assert res_df['Metal']['US'] == 'Silver'


def test_mapcolvals_bad_res_name_len():
    """Testing MapColVals pipeline stages."""
    value_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    with pytest.raises(ValueError):
        map_stage = MapColVals('Medal', value_map, result_columns=['A', 'B'])
        assert isinstance(map_stage, MapColVals)
