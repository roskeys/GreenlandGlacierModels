import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if not os.path.exists("result"):
    os.makedirs("result")


def getCombinedOutput(datasets, glaciers, models):
    combined_output = pd.DataFrame()
    for dataset_name in datasets:
        # load the dataframe for each dataset
        loss_evaluate_df = pd.read_csv(os.path.join("saved_models", dataset_name, "loss_evaluate.csv")).assign(
            dataset=dataset_name)
        loss_evaluate_df = loss_evaluate_df[['name', 'Training_loss', 'Test_loss', "dataset"]]
        for glacier in glaciers:
            # filter out the dataframe for a glacier
            glacier_df = loss_evaluate_df[loss_evaluate_df['name'].str.contains(glacier[:10])]
            for model in models:
                # filter out the dataframe for a model
                model_df = glacier_df[glacier_df['name'].str.contains(f"\.{model}\.")]
                if len(model_df) > 0:
                    sorted_df = model_df.assign(
                        product=model_df.apply(lambda row: row['Test_loss'] * row['Training_loss'] * row['Test_loss'],
                                               axis=1))
                    top = sorted_df.sort_values("product", ascending=True).head(1)
                    combined_output = pd.concat([combined_output, top])
    combined_output.to_csv("result/result.csv")
    return combined_output


def getCSVTable(combined_output, datasets, models, glaciers):
    # for the table
    for dataset_name in datasets:
        dataset_df = combined_output[combined_output['dataset'] == dataset_name]
        dataset_output = pd.DataFrame()
        for model in models:
            dataframe = dataset_df[dataset_df['name'].str.contains(f"\.{model}\.")]
            glacier_dict = {"loss": ['Training_loss', 'Test_loss'], "model": model}
            for glacier in glaciers:
                df = dataframe[dataframe['name'].str.contains(glacier[:10])]
                if len(df) > 0:
                    glacier_dict[glacier] = [df['Training_loss'].values[0], df['Test_loss'].values[0]]
            dataset_output = pd.concat([dataset_output, pd.DataFrame(glacier_dict)])
        dataset_output.to_csv(f"result/{dataset_name}.csv")


def plot_predicted_and_actual(combined_output, datasets, models, glaciers, year_dict, colormap):
    # for the plot
    for glacier in glaciers:
        dataframe = combined_output[combined_output['name'].str.contains(f"{glacier[:10]}")]
        fig = plt.figure(figsize=(14, 10.5))
        plt.axis('off')
        axs = [ax1, ax2, ax3, ax4, ax5, ax6] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(
            323), fig.add_subplot(324), fig.add_subplot(325), fig.add_subplot(326)
        for index, dataset_name in enumerate(datasets, 0):
            selected = dataframe[dataframe['dataset'] == dataset_name]
            axs[index].set_title(f"{index + 1}: {dataset_name}")
            first = True
            for model in models:
                file_names = selected[selected['name'].str.contains(f"\.{model}\.")]['name'].values
                if len(file_names) > 0:
                    file_name = file_names[0]
                    df = pd.read_csv(
                        os.path.join("saved_models", dataset_name, "PredictedvsActualCSV", f"{file_name}_pred.csv"))
                    test_size = int(len(df) * 0.2)
                    df = df.tail(test_size)
                    if first:
                        axs[index].plot(df['Actual'].values, color='red', label='Actual', linewidth=5)
                        axs[index].set_xticks(np.arange(len(df)))
                        assert len(df) == len(year_dict[dataset_name])
                        axs[index].set_xticklabels(year_dict[dataset_name])
                        axs[index].set_ylabel('Actual DM/DT')
                        axs[index].set_xlabel('year')
                        plt.subplots_adjust(hspace=0.3)
                        first = False
                    line = 'dashed' if 'LSTM' in model else "solid"
                    axs[index].plot(df['Predicted'].values, color=colormap[model], linestyle=line, label=model,
                                    linewidth=3)
        custom_lines = [
            Line2D([0], [0], color=colormap['hcnnLSTM'], linestyle="dashed", lw=4, label='vcnnLSTM'),
            Line2D([0], [0], color=colormap['hcnn'], lw=4, label='vcnn'),
            Line2D([0], [0], color=colormap['vcnnLSTM'], linestyle="dashed", lw=4, label='hcnnLSTM'),
            Line2D([0], [0], color=colormap['vcnn'], lw=4, label='hcnn'),
            Line2D([0], [0], color=colormap['resnetLSTM'], linestyle="dashed", lw=4, label='resnetLSTM'),
            Line2D([0], [0], color=colormap['resnet'], lw=4, label='resnet'),
            Line2D([0], [0], color='red', lw=6, label='Actual DM/DT'),
        ]
        ax6.legend(handles=custom_lines, loc='center', prop={'size': 16})
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)
        ax6.spines['bottom'].set_visible(False)
        ax6.spines['left'].set_visible(False)
        ax6.axis('off')
        plt.savefig(f"result/{glacier}_PredvsActual.png")
        plt.close()


def plot_training_and_test_loss(combined_output, datasets, models, glaciers, ticks):
    glacier_map = pd.read_csv("Training_data/Glacier_map.csv")
    slr_map = {}
    for glacier in glaciers:
        slr_map[glacier] = glacier_map[glacier_map['NAME'] == glacier]['mm SLR'].values[0]
    x = np.arange(len(models))
    fig = plt.figure(figsize=(16, 12))
    plt.title("Training mse and Test mse", fontsize=25, pad=20)
    plt.axis('off')
    ax = [p1, p2, p3, p4, p5] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(323), fig.add_subplot(
        324), fig.add_subplot(325)
    for index, dataset_name in enumerate(datasets):
        dataset_df = combined_output[combined_output['dataset'] == dataset_name]
        training_loss, test_loss = [], []
        for model in models:
            model_df = dataset_df[dataset_df['name'].str.contains(f'\.{model}\.')]
            train, test = [], []
            for glacier in glaciers:
                glacier_df = model_df[model_df['name'].str.contains(glacier[:10])]
                train.append(glacier_df['Training_loss'] / slr_map[glacier])
                test.append(glacier_df['Test_loss'] / slr_map[glacier])
            training_loss.append(np.array(train).mean())
            test_loss.append(np.array(test).mean())
        ax[index].set_title(f"{index + 1}: {dataset_name}", fontsize=15)
        l0 = ax[index].bar(x=x, height=training_loss, tick_label=ticks, width=0.3)
        l1 = ax[index].bar(x=x + 0.3, height=test_loss, tick_label=ticks, width=0.3)
        ax[index].set_xticks(x + 0.15)
        ax[index].set_xticklabels(ticks)
    plt.legend((l0[0], l1[1]), ("Training MSE", "Test MSE"), bbox_to_anchor=(2.2, 3.6))
    plt.savefig(f"result/TrainingLossvsTestingLossSLR.png")
    plt.close()


combined_output = getCombinedOutput(
    datasets=['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST', 'IGRA_DMI_GCPG_TOP4'],
    glaciers=['JAKOBSHAVN_ISBRAE', 'QAJUUTTAP_SERMIA', 'STORSTROMMEN', 'HELHEIMGLETSCHER', 'DAUGAARD-JENSEN'],
    models=['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM'])

getCSVTable(combined_output,
            datasets=['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST', 'IGRA_DMI_GCPG_TOP4'],
            glaciers=['JAKOBSHAVN_ISBRAE', 'QAJUUTTAP_SERMIA', 'STORSTROMMEN', 'HELHEIMGLETSCHER', 'DAUGAARD-JENSEN'],
            models=['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM'])

plot_predicted_and_actual(
    combined_output,
    datasets=['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST', 'IGRA_DMI_GCPG_TOP4'],
    models=['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM'], year_dict={
        "ERA5_TOPZ4": [2013, 2014, 2015, 2016, 2017],
        "IGRA_DMI": [2002, 2003, 2004, 2005, 2006, 2007],
        "IGRA_DMI_GCPG": [2003, 2004, 2005, 2006, 2007],
        "IGRA_DMI_GCPG_SST": [2003, 2004, 2005, 2006, 2007],
        "IGRA_DMI_GCPG_TOP4": [2005, 2006, 2007]
    }, glaciers=['STORSTROMMEN'],
    colormap={'hcnnLSTM': 'blue', 'hcnn': 'blue', 'vcnnLSTM': 'black', 'vcnn': 'black', 'resnetLSTM': 'purple',
              'resnet': 'purple', })

plot_training_and_test_loss(combined_output, datasets=['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST',
                                                       'IGRA_DMI_GCPG_TOP4'],
                            models=['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM'],
                            glaciers=['QAJUUTTAP_SERMIA', 'STORSTROMMEN', 'HELHEIMGLETSCHER', 'DAUGAARD-JENSEN'],
                            ticks=['vcnn', 'vcnnLSTM', 'hcnn', 'hcnnLSTM', 'resnet', 'resnetLSTM'], )

# # by glacier
# x = np.arange(len(models))
# for dataset_name in datasets:
#     fig = plt.figure(figsize=(32, 24))
#     plt.title("Training mse and Test mse", fontsize=50, pad=50)
#     plt.axis('off')
#     ax = [p1, p2, p3, p4, p5] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(323), fig.add_subplot(
#         324), fig.add_subplot(325)  # , fig.add_subplot(326)
#     dataset_df = combined_output[combined_output['dataset'] == dataset_name]
#     for index, glacier in enumerate(glaciers):
#         glacier_df = dataset_df[dataset_df['name'].str.contains(glacier[:10])]
#         # glacier_df = combined_output[combined_output['name'].str.contains(glacier[:10])]
#         training_loss, test_loss = [], []
#         for model in models:
#             model_df = glacier_df[glacier_df['name'].str.contains(model)]
#             training_loss.append(model_df['Training_loss'].mean(skipna=True))
#             test_loss.append(model_df['Test_loss'].mean(skipna=True))
#         ax[index].set_title(glacier, fontsize=30)
#         l0 = ax[index].bar(x=x, height=training_loss, tick_label=models, width=0.3)
#         l1 = ax[index].bar(x=x + 0.3, height=test_loss, tick_label=models, width=0.3)
#         ax[index].set_xticks(x + 0.15)
#         ax[index].set_xticklabels(models)
#     plt.legend((l0[0], l1[1]), ("Training MSE", "Test MSE"), bbox_to_anchor=(2.2, 3.6))
#     plt.savefig(f"result/{dataset_name}_TrainingvsTesting.png")
#     plt.close()
# glaciers = ['JAKOBSHAVN_ISBRAE']
# colormap = {'hcnnLSTM': 'blue', 'hcnn': 'blue', 'vcnnLSTM': 'black', 'vcnn': 'black', 'resnetLSTM': 'purple',
#             'resnet': 'purple', }
# datasets = ['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST', 'IGRA_DMI_GCPG_TOP4']
# models = ['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM']
# year_dict={
#         "ERA5_TOPZ4": [2013, 2014, 2015, 2016, 2017],
#         "IGRA_DMI": [ 2001, 2002, 2003, 2004, 2005, 2006, 2007],
#         "IGRA_DMI_GCPG": [2003, 2004, 2005, 2006, 2007],
#         "IGRA_DMI_GCPG_SST": [2003, 2004, 2005, 2006, 2007],
#         "IGRA_DMI_GCPG_TOP4": [2005, 2006, 2007]
#     }
#
# dataframe = combined_output[combined_output['name'].str.contains('JAKOBSHA')]
# dataset_df = dataframe[dataframe['dataset'] == 'ERA5_TOPZ4']
# plt.figure()
#
#
# for glacier in glaciers:
#     dataframe = combined_output[combined_output['name'].str.contains(f"{glacier[:10]}")]
#     fig = plt.figure(figsize=(14, 10.5))
#     plt.axis('off')
#     axs = [ax1, ax2, ax3, ax4, ax5, ax6] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(
#         323), fig.add_subplot(324), fig.add_subplot(325), fig.add_subplot(326)
#     for index, dataset_name in enumerate(datasets, 0):
#         selected = dataframe[dataframe['dataset'] == dataset_name]
#         axs[index].set_title(f"{index + 1}: {dataset_name}")
#         first = True
#         for model in models:
#             file_names = selected[selected['name'].str.contains(f"\.{model}\.")]['name'].values
#             if len(file_names) > 0:
#                 file_name = file_names[0]
#                 df = pd.read_csv(
#                     os.path.join("saved_models", dataset_name, "PredictedvsActualCSV", f"{file_name}_pred.csv"))
#                 test_size = int(len(df) * 0.2)
#                 df = df.tail(test_size)
#                 if first:
#                     axs[index].plot(df['Actual'].values, color='red', label='Actual', linewidth=5)
#                     axs[index].set_xticks(np.arange(len(df)))
#                     assert len(df) == len(year_dict[dataset_name])
#                     axs[index].set_xticklabels(year_dict[dataset_name])
#                     axs[index].set_ylabel('Actual DM/DT')
#                     axs[index].set_xlabel('year')
#                     plt.subplots_adjust(hspace=0.3)
#                     first = False
#                 line = 'dashed' if 'LSTM' in model else "solid"
#                 axs[index].plot(df['Predicted'].values, color=colormap[model], linestyle=line, label=model,
#                                 linewidth=3)
#     custom_lines = [
#         Line2D([0], [0], color=colormap['hcnnLSTM'], linestyle="dashed", lw=4, label='vcnnLSTM'),
#         Line2D([0], [0], color=colormap['hcnn'], lw=4, label='vcnn'),
#         Line2D([0], [0], color=colormap['vcnnLSTM'], linestyle="dashed", lw=4, label='hcnnLSTM'),
#         Line2D([0], [0], color=colormap['vcnn'], lw=4, label='hcnn'),
#         Line2D([0], [0], color=colormap['resnetLSTM'], linestyle="dashed", lw=4, label='resnetLSTM'),
#         Line2D([0], [0], color=colormap['resnet'], lw=4, label='resnet'),
#         Line2D([0], [0], color='red', lw=6, label='Actual DM/DT'),
#     ]
#     ax6.legend(handles=custom_lines, loc='center', prop={'size': 16})
#     ax6.spines['top'].set_visible(False)
#     ax6.spines['right'].set_visible(False)
#     ax6.spines['bottom'].set_visible(False)
#     ax6.spines['left'].set_visible(False)
#     ax6.axis('off')
#     plt.savefig(f"result/{glacier}_PredvsActual.png")
#     plt.close()
