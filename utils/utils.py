import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# time stamp day-hour-minuts-second when running this function
def get_time_stamp():
    return time.strftime('%d-%H-%M-%S', time.localtime(time.time()))


# load the saved checkpoint
def load_check_point(path):
    model = load_model(path)
    return model


# load the checkpoint and keep training from that model
def transfer_learning(path, epoch, data, config, loss='mse', optimizer='rmsprop', save_best_only=True, metrics=None,
                      show=False, name=""):
    model = load_check_point(path)
    model._name = "Trans_" + model.name
    if len(name) > 0:
        model._name += ("_" + name)
    train_model(model, epoch, data, config, loss=loss, optimizer=optimizer, save_best_only=save_best_only,
                metrics=metrics, show=show)


def concatenate_data(x1, y1, x2, y2):
    x_concatenated = []
    if isinstance(x1, list) or isinstance(x1, tuple):
        for x_part1, x_part2 in zip(x1, x2):
            x_concatenated.append(np.concatenate([x_part1, x_part2], axis=0))
    else:
        x_concatenated = np.concatenate([x1, x2], axis=0)
    y_concatenated = np.concatenate([y1, y2])
    return x_concatenated, y_concatenated


def train_model(model, epoch, data, config, loss='mse', optimizer='rmsprop', saved_model_path="saved_models",
                metrics=None,
                show=False, verbose=2, save_best_only=True, logger=None):
    # evaluation matrix
    model_name = model.name
    model_path = os.path.join(os.path.abspath(os.curdir), saved_model_path, model_name, get_time_stamp())
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, "saved_checkpoints"))
    # save the original dataset
    x_train, x_test, y_train, y_test = data
    train_size, test_size = len(y_train), len(y_test)
    with open(os.path.join(model_path, "data.pickle"), 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)
    # keras build in model structure visualization
    plot_model(model, to_file=os.path.join(model_path, f"{model_name}.png"))
    model.compile(loss=loss, optimizer="rmsprop", metrics=metrics)
    logger.info(
        f"Compiled model: name: {model.name} epoch: {epoch} train size: {train_size} "
        f"test size: {test_size} optimizer: {optimizer}")
    # add keras callbacks save history, tensorboard record and checkpoints
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[
        # TensorBoard(log_dir=os.path.join(model_path, "logs"), update_freq="epoch"),
        ModelCheckpoint(
            filepath=os.path.join(model_path, "saved_checkpoints",
                                  "weights-{epoch:03d}-{val_loss:.3f}-{loss:.3f}.hdf5"),
            monitor='val_loss', mode='auto', save_freq="epoch", save_best_only=save_best_only),
        EarlyStopping(monitor='loss', min_delta=config["min_delta"], patience=config["patience"], mode='auto',
                      baseline=config["threshold"])
    ], epochs=epoch, verbose=verbose)
    model.save(os.path.join(model_path, "saved_checkpoints", f"weights-{epoch}.hdf5"))
    # plot the history
    history_plot = plot_history(history.history, show=show)
    history_plot.savefig(os.path.join(model_path, f"{model_name}.loss.png"))
    history_plot.close()
    # select the last model
    selected_file = os.listdir(os.path.join(model_path, "saved_checkpoints"))
    selected_model = load_check_point(os.path.join(model_path, "saved_checkpoints", selected_file[-1]))
    # plot the predicted value with the actual value
    x_origin, y_origin = concatenate_data(x_train, y_train, x_test, y_test)
    pred, total_error, train_error, test_error = pred_and_evaluate(selected_model, x_origin, y_origin, test_size)
    predict_plot = plot_predicted(selected_model.name, pred, y_origin, test_size=test_size, show=show,
                                  text=f"MSE:{total_error:.4f} Val_mse:{test_error:.4f}")
    predict_plot.savefig(os.path.join(model_path, f"{model_name}.value.png"))
    predict_plot.close()
    with open(os.path.join(model_path, "history.pickle"), 'wb') as f:
        pickle.dump(history.history, f)
    logger.info(f"Finished training. loss:{total_error:.4f} val_loss:{test_error:.4f} train_loss: {train_error:.4f}")


def pred_and_evaluate(model, x, y, test_size):
    train_size = len(y) - test_size
    pred = model.predict(x)
    y = y[:, np.newaxis] if len(y.shape) == 1 else y
    mse = tf.keras.losses.MeanSquaredError()
    total_error, train_error, test_error = mse(y, pred).numpy(), mse(y[:train_size], pred[:train_size]).numpy(), mse(
        y[train_size:], pred[train_size:]).numpy()
    return pred[:, 0], total_error, train_error, test_error


# plot the training and validation loss history
def plot_history(history, show=False):
    plt.figure()
    plt.plot(np.log(history['loss']))
    plt.plot(np.log(history['val_loss']))
    plt.title('Loss and val_loss')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    if show:
        plt.show()
    return plt


def plot_predicted(name, pred, y, test_size=7, show=False, text=None):
    plt.figure()
    if len(y.shape) == 2:
        y = y.squeeze(-1)
    if len(pred.shape) == 2:
        pred = pred.squeeze(-1)
    plt.plot(pred)
    plt.plot(y)
    min_y, max_y = min(min(y), min(pred)), max(max(y), max(pred))
    plt.vlines(len(y) - test_size, min_y, max_y, colors="r", linestyles="dashed")
    if text:
        plt.text(0, min(min(y), min(pred)), text)
    plt.title(name)
    plt.ylabel('SMB')
    plt.xlabel('Year')
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    if show:
        plt.show()
    return plt


def load_all_and_plot_all(saved_model_base_path, last=True, show=False, logger=None):
    model_folders = os.listdir(saved_model_base_path)
    if "loss" in model_folders:
        model_folders.remove("loss")
    if not os.path.exists(os.path.join(saved_model_base_path, "loss")):
        os.makedirs(os.path.join(saved_model_base_path, "loss"))
    if "PredictedvsActual" in model_folders:
        model_folders.remove("PredictedvsActual")
    if not os.path.exists(os.path.join(saved_model_base_path, "PredictedvsActual")):
        os.makedirs(os.path.join(saved_model_base_path, "PredictedvsActual"))
    if "PredictedvsActualCSV" in model_folders:
        model_folders.remove("PredictedvsActualCSV")
    if "loss_evaluate.csv" in model_folders:
        model_folders.remove("loss_evaluate.csv")
    if "Report.md" in model_folders:
        model_folders.remove("Report.md")
    if "logs" in model_folders:
        model_folders.remove("logs")
    if not os.path.exists(os.path.join(saved_model_base_path, "PredictedvsActualCSV")):
        os.makedirs(os.path.join(saved_model_base_path, "PredictedvsActualCSV"))
    loss_evaluate = pd.DataFrame()
    for model_name in model_folders:
        for r_index, running_time in enumerate(os.listdir(os.path.join(saved_model_base_path, model_name)), 1):
            base_path = os.path.join(saved_model_base_path, model_name, running_time)
            with open(os.path.join(base_path, "data.pickle"), 'rb') as f:
                (x_train, x_test, y_train, y_test) = pickle.load(f)
            x, y = concatenate_data(x_train, y_train, x_test, y_test)
            test_size = len(y_test)
            models_list = os.listdir(os.path.join(base_path, "saved_checkpoints"))
            if len(models_list) > 0:
                if last:
                    models_list = models_list[-1:]
                for m_index, model_selected in enumerate(models_list):
                    if int(model_selected[:-5].split('-')[1]) < 30:
                        continue
                    model = load_check_point(os.path.join(base_path, "saved_checkpoints", model_selected))
                    pred, total_error, train_error, test_error = pred_and_evaluate(model, x, y, test_size)
                    if len(y.shape) > 1:
                        y = y.squeeze(-1)
                    pd.DataFrame({"Predicted": pred, "Actual": y}).to_csv(
                        os.path.join(saved_model_base_path, "PredictedvsActualCSV",
                                     f"{model.name}_{r_index}_{m_index}_pred.csv"))
                    pred_and_actual_plot = plot_predicted(f"{model.name}_{r_index}_{m_index}", pred, y,
                                                          test_size=test_size, show=show,
                                                          text=f"MSE:{total_error:.4f} Val_mse:{test_error:.4f}")
                    pred_and_actual_plot.savefig(os.path.join(saved_model_base_path, "PredictedvsActual",
                                                              f"{model.name}_{r_index}_{m_index}_value.png"))
                    pred_and_actual_plot.close()
                    df = pd.DataFrame({
                        "name": [f"{model.name}_{r_index}_{m_index}"],
                        "Total_loss": [total_error], "Test_loss": [test_error], "Training_loss": [train_error],
                        "path": [os.path.join(base_path, "saved_checkpoints", model_selected)]
                    })
                    loss_evaluate = pd.concat([loss_evaluate, df], ignore_index=True)
            if os.path.exists(os.path.join(base_path, "history.pickle")):
                with open(os.path.join(base_path, "history.pickle"), 'rb') as f:
                    history = pickle.load(f)
                history_plot = plot_history(history, show=show)
                history_plot.savefig(os.path.join(saved_model_base_path, "loss",
                                                  f"{model_name}_loss.png"))
                history_plot.close()
    loss_evaluate.to_csv(os.path.join(saved_model_base_path, "loss_evaluate.csv"))
    logger.info(f"Finished plot and evaluate all models in {saved_model_base_path}")
