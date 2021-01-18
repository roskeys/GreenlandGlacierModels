import os
import re
import sys
import time
import yaml
import logging
import argparse
import importlib
import traceback
import pandas as pd
from utils.generateReport import generate_report_md
from utils.utils import train_model, load_all_and_plot_all, concatenate_data
from utils.data import load_data, get_centroid, train_test_split, determine_path

sys.path.insert(0, "models")
formatter = logging.Formatter('train_model %(asctime)15s %(levelname)5s: %(message)s')
logger = logging.getLogger(f"logs/GlacierModel-{time.strftime('%d-%H-%M-%S', time.localtime(time.time()))}.log")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
ch.setFormatter(formatter)

if os.name == "posix":
    default_config_path = "config.yaml"
else:
    default_config_path = "dev.yaml"

parser = argparse.ArgumentParser(description='Training function for glacier models')
parser.add_argument('-c', type=str, default=default_config_path, help='Configuration file path')
args = parser.parse_args()

with open(args.c, "rb") as f:
    config = yaml.safe_load(f)

glacier_assignment = pd.read_csv(config["GLACIER_ASSIGNMENT_PATH"])

if isinstance(config["models"], str):
    config["models"] = [config["models"]]
if config["models"][0] == "all":
    models_files = os.listdir("models")
    models_files.remove("components")
    models_files.remove("__init__.py")
    if "__pycache__" in models_files:
        models_files.remove("__pycache__")
    models = [n[:-3] for n in models_files]
else:
    models = config["models"]

first = True
x_combine_train, x_combine_test, y_combine_train, y_combine_test = None, None, None, None

if config["train"]:
    try:
        if not os.path.exists(config["saved_model_path"]):
            os.mkdir(config["saved_model_path"])
        for glacier_name in config['glaciers']:
            try:
                central = get_centroid(glacier_name, glacier_assignment)
            except Exception as e:
                if re.search(r"Central of .* not found", str(e)):
                    continue
                else:
                    traceback.print_exc()
                    sys.exit()
            try:
                path_dict = {
                    "smb": config["SMB_PATH"],
                    "humidity": determine_path("humidity", config, glacier_name, central),
                    "pressure": determine_path("pressure", config, glacier_name, central),
                    "temperature": determine_path("temperature", config, glacier_name, central),
                    "cloud": determine_path("cloud", config, glacier_name, central),
                    "wind": determine_path("wind", config, glacier_name, central),
                    "precipitation": determine_path("precipitation", config, glacier_name, central),
                    "ocean": determine_path("ocean", config, glacier_name, central) if config["ocean_PATH"] else None
                }
            except FileNotFoundError as e:
                if "data path not exists" in str(e):
                    continue
                else:
                    traceback.print_exc()
                    sys.exit()
            try:
                x_all, y_all = load_data(glacier_name, logger=logger, **path_dict)
                target_shape = 1 if len(y_all.shape) == 1 else y_all.shape[1]
                if config["combine"]:
                    test_size = int(len(y_all) / 3) % 7
                    if first:
                        (x_combine_train, x_combine_test, y_combine_train, y_combine_test) = train_test_split(x_all, y_all,
                                                                                                              test_size=test_size)
                        first = False
                    else:
                        (x_train, x_test, y_train, y_test) = train_test_split(x_all, y_all, test_size=test_size)
                        x_combine_train, y_combine_train = concatenate_data(x_combine_train, y_combine_train, x_train,
                                                                            y_train)
                        x_combine_test, y_combine_test = concatenate_data(x_combine_test, y_combine_test, x_test, y_test)
                    if glacier_name == config["glaciers"][-1]:
                        data = (x_combine_train, x_combine_test, y_combine_train, y_combine_test)
                        logger.info(f"Combined all data, train data size: {x_combine_train[0].shape[0]} "
                                    f"test data size {x_combine_test[0].shape[0]}")
                    else:
                        continue
                else:
                    if len(y_all) < 15:
                        continue
                    test_size = int(0.2 * len(y_all))
                    data = (x_train, x_test, y_train, y_test) = train_test_split(x_all, y_all, test_size=test_size)

                if config["ocean_PATH"]:
                    (cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim, ocean_dim) = [
                        x.shape[1:] for x in data[0]]
                else:
                    (cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim, temperature_dim) = [x.shape[1:] for
                                                                                                             x in data[0]]
                    ocean_dim = None
            except ValueError as e:
                if str(e) == "No enough data":
                    continue
                else:
                    traceback.print_exc()
                    sys.exit()
            for model_name in models:
                try:
                    module = importlib.import_module(model_name)
                    if "reanalysis" in config['ocean_PATH'] or "Reanalysis" in config['ocean_PATH']:
                        name = f"reanalysis.{model_name}"
                    elif "Ocean" in config["ocean_PATH"]:
                        name = f"ocean.{model_name}"
                    else:
                        name = f"basic.{model_name}"
                    if config["combine"]:
                        name = "combine" + name
                    model = module.getModel(cloud_dim, precipitation_dim, wind_dim, humidity_dim, pressure_dim,
                                            temperature_dim,
                                            ocean_dim=ocean_dim, target_shape=target_shape, other_dim=None,
                                            name=f"{name}.{glacier_name[:15]}")
                    train_model(model, config['epoch'], data=data, loss=config["loss"], config=config,
                                optimizer=config["optimizer"], saved_model_path=config["saved_model_path"],
                                save_best_only=config["save_best_only"], metrics=config["metrics"], show=config["show"],
                                verbose=config["verbose"], logger=logger)
                except:
                    logger.error(f"Failed to get model {model_name}")
                    traceback.print_exc()
    except KeyboardInterrupt:
        sys.exit()

if config["plot_all"]:
    load_all_and_plot_all(config["saved_model_path"], logger=logger)

if config["generate_report"]:
    generate_report_md(loss_evaluate_path="loss_evaluate.csv",
                       image_path=f"{config['saved_model_path']}/PredictedvsActual", top_n=5)
