import os
import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

datasets = ['ERA5_TOPZ4', 'IGRA_DMI', 'IGRA_DMI_GCPG', 'IGRA_DMI_GCPG_SST', 'IGRA_DMI_GCPG_TOP4']
glaciers = ['JAKOBSHAVN_ISBRAE', 'QAJUUTTAP_SERMIA', 'STORSTROMMEN', 'HELHEIMGLETSCHER',
            'DAUGAARD-JENSEN']  # 'NORDENSKIOLD_GLETSCHER',
models = ['hcnn', 'hcnnLSTM', 'vcnn', 'vcnnLSTM', 'resnet', 'resnetLSTM']
colormap = {'hcnnLSTM': 'blue', 'hcnn': 'blue', 'vcnnLSTM': 'black', 'vcnn': 'black', 'resnetLSTM': 'purple',
            'resnet': 'purple', }

if not os.path.exists("result"):
    os.makedirs("result")

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

# for the plot
for glacier in glaciers:
    dataframe = combined_output[combined_output['name'].str.contains(f"{glacier[:10]}")]
    fig = plt.figure(figsize=(14, 10.5))
    plt.axis('off')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(
        323), fig.add_subplot(324), fig.add_subplot(325), fig.add_subplot(326)
    for index, dataset_name in enumerate(datasets, 0):
        selected = dataframe[dataframe['dataset'] == dataset_name]
        row, col = int(index / 2), index % 2
        axs[index].set_title(dataset_name)
        first = True
        for model in models:
            file_names = selected[selected['name'].str.contains(f"\.{model}\.")]['name'].values
            if len(file_names) > 0:
                file_name = file_names[0]
                df = pd.read_csv(os.path.join(
                    "saved_models", dataset_name, "PredictedvsActualCSV", f"{file_name}_pred.csv"))
                test_size = int(len(df) * 0.2)
                df = df.tail(test_size)
                if first:
                    axs[index].plot(df['Actual'].values, color='red', label='Actual', linewidth=5)
                    first = False
                line = 'dashed' if 'LSTM' in model else "solid"
                axs[index].plot(df['Predicted'].values, color=colormap[model], linestyle=line, label=model,
                                linewidth=3)
    custom_lines = [
        Line2D([0], [0], color=colormap['hcnnLSTM'], linestyle="dashed", lw=4, label='hcnnLSTM'),
        Line2D([0], [0], color=colormap['hcnn'], lw=4, label='hcnn'),
        Line2D([0], [0], color=colormap['vcnnLSTM'], linestyle="dashed", lw=4, label='vcnnLSTM'),
        Line2D([0], [0], color=colormap['vcnn'], lw=4, label='vcnn'),
        Line2D([0], [0], color=colormap['resnetLSTM'], linestyle="dashed", lw=4, label='resnetLSTM'),
        Line2D([0], [0], color=colormap['resnet'], lw=4, label='resnet'),
    ]
    ax6.legend(handles=custom_lines, loc='center', prop={'size': 20})
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    ax6.axis('off')
    plt.savefig(f"result/{glacier}_PredvsActual.png")
    plt.close()

# models = ['vcnn', 'hcnn', 'resnet']
x = np.arange(len(models))
for dataset_name in datasets:
    fig = plt.figure(figsize=(16, 12))
    plt.title("Training mse and Test mse", fontsize=25, pad=20)
    plt.axis('off')
    ax = [p1, p2, p3, p4, p5] = fig.add_subplot(321), fig.add_subplot(322), fig.add_subplot(323), fig.add_subplot(
        324), fig.add_subplot(325)  # , fig.add_subplot(326)
    dataset_df = combined_output[combined_output['dataset'] == dataset_name]
    for index, glacier in enumerate(glaciers):
        glacier_df = dataset_df[dataset_df['name'].str.contains(glacier[:10])]
        # glacier_df = combined_output[combined_output['name'].str.contains(glacier[:10])]
        training_loss, test_loss = [], []
        for model in models:
            model_df = glacier_df[glacier_df['name'].str.contains(model)]
            training_loss.append(model_df['Training_loss'].mean(skipna=True))
            test_loss.append(model_df['Test_loss'].mean(skipna=True))
        ax[index].set_title(glacier)
        l0 = ax[index].bar(x=x, height=training_loss, tick_label=models, width=0.3)
        l1 = ax[index].bar(x=x + 0.3, height=test_loss, tick_label=models, width=0.3)
        ax[index].set_xticks(x + 0.15)
        ax[index].set_xticklabels(models)
    plt.legend((l0[0], l1[1]), ("Training MSE", "Test MSE"), bbox_to_anchor=(2.2, 3.6))
    plt.savefig(f"result/{dataset_name}_TrainingvsTesting.png")
    plt.close()
