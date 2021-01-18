import numpy as np
import pandas as pd
from os.path import exists, join


def get_year_range(dataframe):
    height = dataframe.shape[0]
    out = set()
    if dataframe.shape[1] > 5:
        year_range = {int(i) for i in dataframe.columns.str.split("_").str[0].unique()[2:]}
        for year in year_range:
            add = True
            for i in range(5, 13):
                if dataframe.filter(regex=f"^{year}_0*{i}$").shape != (height, 1):
                    add = False
            for i in range(1, 5):
                if dataframe.filter(regex=f"^{year + 1}_0*{i}$").shape != (height, 1):
                    add = False
            if add:
                out.add(year)
        return out
    else:
        year_range = set(dataframe['year'].astype(int))
        for year in year_range:
            df_current_year = dataframe[dataframe['year'] == year]
            add = True
            for i in range(5, 13):
                if len(df_current_year[df_current_year['month'] == i]) != 1:
                    add = False
            for i in range(1, 5):
                if len(df_current_year[df_current_year['month'] == i]) != 1:
                    add = False
            if add:
                out.add(year)
        return out


def get_common_year_range(year_range, dataframe):
    return year_range.intersection(get_year_range(dataframe))


def sort_year_range(year_range):
    year_range = list(year_range)
    year_range.sort()
    return year_range


def _get_record(dataframe, feature_name, month_range):
    data = []
    for i in month_range:
        record = dataframe[dataframe["month"] == i][feature_name].values
        if len(record) == 1:
            data.append(record[0])
        else:
            raise Exception("Data not complete")
    return data


def _load_1d(dataframe, year_range):
    feature_name = dataframe.columns[-1]
    out = [] * len(year_range)
    for year in year_range:
        try:
            data = _get_record(dataframe[dataframe["year"] == year], feature_name, range(5, 13))
            data.extend(_get_record(dataframe[dataframe["year"] == year + 1], feature_name, range(1, 5)))
            out.append(data)
        except Exception as e:
            if str(e) == "Data not complete":
                continue
    return np.array(out).astype(np.float)


def _load_2d(dataframe, year_range):
    height = dataframe.shape[0]
    out = [] * len(year_range)
    for year in year_range:
        try:
            data = [] * 12
            df_current_year = dataframe.filter(regex=f"^{year}")
            for i in range(5, 13):
                df_current_month = df_current_year.filter(regex=f"_0*{i}$", axis=1)
                if len(df_current_month) != height:
                    raise Exception("Data not complete")
                data.append(df_current_month.values)
            df_next_year = dataframe.filter(regex=f"^{year + 1}", axis=1)
            for i in range(1, 5):
                df_current_month = df_next_year.filter(regex=f"_0*{i}$", axis=1)
                if len(df_current_month) != height:
                    raise Exception("Data not complete")
                data.append(df_current_month.values)
            out.append(np.concatenate(data, axis=1)[np.newaxis, :, :, np.newaxis])
        except Exception as e:
            if str(e) == "Data not complete":
                continue
    return np.concatenate(out, axis=0).astype(np.float)


def _load_data(dataframe, year_range):
    if dataframe.shape[1] < 5:
        array = _load_1d(dataframe, year_range)
    else:
        array = _load_2d(dataframe, year_range)
    if array.dtype != "float64":
        raise Exception("Check the data please, the data should not contain string or empty space")
    return array


def reset_column_name(dataframe):
    if dataframe.shape[1] < 5:
        if "year" not in dataframe.columns or "month" not in dataframe.columns:
            if "YEAR" in dataframe.columns:
                dataframe = dataframe.rename(columns={"YEAR": "year"})
            if "MONTH" in dataframe.columns:
                dataframe = dataframe.rename(columns={"MONTH": "month"})
            if dataframe.columns[1] == '0' and dataframe.columns[2] == '1':
                dataframe = dataframe.rename(columns={'0': "year", '1': "month"}, errors="raise")
    return dataframe


def _load_csv_data(path):
    dataframe = pd.read_csv(path, index_col=0, dtype=np.float64)
    dataframe = dataframe.apply(lambda x: x.fillna(x.mean()), axis=0)
    return reset_column_name(dataframe)


def load_smb(smb_df, glacier_name, year_range):
    smb_df = smb_df[smb_df["NAME"] == glacier_name]
    smb = []
    for year in year_range:
        smb.append(smb_df[f"{year + 1}.5"].values[0])
    return np.array(smb)[:, np.newaxis]


def get_smb_year_range(smb_df):
    out = set()
    for year in smb_df.filter(regex=r"\d{4}\.5").columns:
        out.add(int(year[:-2]) - 1)
    return out


def get_centroid(glacier_name, glacier_assignment):
    df = glacier_assignment[glacier_assignment['NAME'] == glacier_name]
    if len(df) > 0:
        return df["Central"].values[0]
    else:
        raise Exception(f"Central of {glacier_name} not found")


def load_data(glacier_name, logger=None, **paths):
    smb_df = pd.read_csv(paths["smb"])
    humidity_df = _load_csv_data(paths["humidity"])
    pressure_df = _load_csv_data(paths["pressure"])
    temperature_df = _load_csv_data(paths["temperature"])
    cloud_df = _load_csv_data(paths["cloud"])
    wind_df = _load_csv_data(paths["wind"])
    precipitation_df = _load_csv_data(paths["precipitation"])
    logger.debug(f"Loaded data from {glacier_name} {paths}")
    dataframes = [smb_df, cloud_df, precipitation_df, wind_df, humidity_df, pressure_df, temperature_df]
    if "ocean" in paths.keys() and paths["ocean"] is not None:
        ocean_df = _load_csv_data(paths["ocean"])
        dataframes.append(ocean_df)
    common_year_range = get_smb_year_range(smb_df)
    for a in dataframes[1:]:
        common_year_range = get_common_year_range(common_year_range, a)
    logger.info(f"Year_range: {common_year_range}")
    if len(common_year_range) < 5:
        logger.info(f"Too small year range for {glacier_name}, skip")
        raise ValueError("No enough data")
    else:
        smb_array = load_smb(smb_df, glacier_name, common_year_range)
        x_data_set = [_load_data(i, common_year_range) for i in dataframes[1:]]
        return x_data_set, smb_array


def train_test_split(x, y, test_size=7, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
        shuffle_index = np.arange(len(y))
        np.random.shuffle(shuffle_index)
        for i, x_n in enumerate(x):
            x[i] = x_n[shuffle_index]
        y = y[shuffle_index]
    x_train, x_test = [], []
    for x_n in x:
        x_train.append(x_n[:-test_size])
        x_test.append(x_n[-test_size:])
    y_train = y[:-test_size]
    y_test = y[-test_size:]
    return x_train, x_test, y_train, y_test


def determine_path(variable_name, config, glacier_name, central):
    if exists(join(config[f"{variable_name}_PATH"], f"{variable_name}_{glacier_name}.csv")):
        return join(config[f"{variable_name}_PATH"], f"{variable_name}_{glacier_name}.csv")
    elif exists(join(config[f"{variable_name}_PATH"], f"{variable_name}_{central}.csv")):
        return join(config[f"{variable_name}_PATH"], f"{variable_name}_{central}.csv")
    elif exists(join(config[f"{variable_name}_PATH"], str(central), f"{variable_name}_{central}.csv")):
        return join(config[f"{variable_name}_PATH"], str(central), f"{variable_name}_{central}.csv")
    elif exists(join(config[f"{variable_name}_PATH"], glacier_name, f"{variable_name}_{glacier_name}.csv")):
        return join(config[f"{variable_name}_PATH"], glacier_name, f"{variable_name}_{glacier_name}.csv")
    elif exists(join(config[f"{variable_name}_PATH"], config["centroid_map"][central], f"{variable_name}.csv")):
        return join(config[f"{variable_name}_PATH"], config["centroid_map"][central], f"{variable_name}.csv")
    else:
        raise FileNotFoundError(f"{variable_name} data path not exists")
