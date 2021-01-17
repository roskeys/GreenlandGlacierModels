import pandas as pd


# from utils.utils import get_time_stamp
def generate_report_md(loss_evaluate_path, image_path, top_n=5):
    loss_evaluate = pd.read_csv(loss_evaluate_path)
    loss_evaluate = get_best_of_each_model(loss_evaluate)
    report = "# Training Result\n\n"
    dataframes = (loss_evaluate[loss_evaluate["name"].str.contains("basic")],
                  loss_evaluate[loss_evaluate["name"].str.contains("ocean")],
                  loss_evaluate[loss_evaluate["name"].str.contains("reanalysis")])
    for df in dataframes:
        if len(df) > 0:
            report += get_model_report_for_category(df, image_path, top_n)
    report += statistics_for_each_model(loss_evaluate)
    report += get_the_best_model_for_each_glacier(loss_evaluate, image_path, top_n)
    with open("Report.md", 'w') as f:
        f.write(report)


def get_best_of_each_model(dataframe):
    max_df = pd.DataFrame()
    for name in dataframe["name"].str.split("_", expand=True)[0].unique().tolist():
        df = dataframe[dataframe["name"].str.contains(name)]
        max_df = pd.concat([max_df, df.head(1)])
    return max_df


def get_model_report_for_category(dataframe, image_path, top_n):
    name = dataframe['name'].unique()[0].split('.')[0]
    report = f"## {name} models\n\n"
    report += dataframe_report(dataframe)
    report += get_best_performance_models(dataframe, image_path, top_n)
    return report


def dataframe_report(dataframe):
    report = f"The mean of the total loss is **{dataframe['Total_loss'].mean():.4f}**, " \
             f"The variance of the total loss is **{dataframe['Total_loss'].var():.4f}**.\n\n " \
             f"The mean of the test loss is **{dataframe['Test_loss'].mean():.4f}**, " \
             f"The variance of the test loss is **{dataframe['Test_loss'].var():.4f}**.\n\n " \
             f"The max of the total loss is **{dataframe['Total_loss'].max():.4f}**, " \
             f"The min of the total loss is **{dataframe['Total_loss'].min():.4f}**.\n\n " \
             f"The max of the test loss is **{dataframe['Test_loss'].max():.4f}**, " \
             f"The min of the test loss is **{dataframe['Test_loss'].min():.4f}**.\n\n " \
             f"The number of models that gives test loss below the threshold is shown as below, " \
             f"the total count of models is **{len(dataframe)}**.\n\n"
    table = "| Threshold | count |\n | ------- | -------- |\n"
    thresh_holds = [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.5]
    for thresh_hold in thresh_holds:
        table += f'|{thresh_hold}|{len(dataframe[dataframe["Test_loss"] <= thresh_hold])}|\n'
    return report + table


def get_best_performance_models(dataframe, image_path, top_n=5):
    name = dataframe['name'].unique()[0].split('.')[0]
    top = dataframe.sort_values("Test_loss", ascending=True).head(top_n)
    report = f"### Top {top_n} models for {name} models\n\n"
    table = "| Model | Loss | Test Loss |\n | ------- | ---- | -------- |\n"
    for index, value in top.iterrows():
        report += f'<img src="{image_path}/{value["name"]}_value.png" width="432" height="288"/>\n'
        table += f"|{value['name'].split('_')[0]}|{value['Total_loss']:.4f}|{value['Test_loss']:.4f}|\n"
    return report + "\n\n" + table


def get_the_best_model_for_each_glacier(dataframe, image_path, top_n=5):
    glaciers = {a.split('.')[-1] for a in {i.split('_')[0] for i in dataframe["name"].unique()}}
    report = "# Best model for each glacier\n\n"
    table = "| Model | Loss | Test Loss |\n | ------- | ---- | -------- |\n"
    for glacier in glaciers:
        report += f"## {glacier}\n\n"
        df = dataframe[dataframe["name"].str.contains(glacier)]
        top = df.sort_values("Test_loss", ascending=True).head(top_n)
        for _, value in top.iterrows():
            report += f'<img src="{image_path}/{value["name"]}_value.png" width="432" height="288"/>\n'
            table += f"|{value['name'].split('_')[0]}|{value['Total_loss']:.4f}|{value['Test_loss']:.4f}|\n"
    return report


def statistics_for_each_model(dataframe):
    models = {a.split('.')[1] for a in {i.split('_')[0] for i in dataframe["name"].unique()}}
    report = "# Statistics for each model\n\n"
    table = "| Model | Mean Loss | Mean Test Loss |Max Loss |Max Test Loss |Min Loss |Min Test Loss |\n " \
            "| ------- | ---- | -------- | -------- | -------- |-------- | -------- |\n"
    for model in models:
        df = dataframe[dataframe["name"].str.contains('\.' + model + "\.", regex=True)]
        table += f"|{model}|{df['Total_loss'].mean():.4f}|{df['Test_loss'].mean():.4f}|" \
                 f"{df['Total_loss'].max():.4f}|{df['Test_loss'].max():.4f}|" \
                 f"{df['Total_loss'].min():.4f}|{df['Test_loss'].min():.4f}|\n"
    return report + table


if __name__ == '__main__':
    generate_report_md("../loss_evaluate.csv", "../saved_models/PredictedvsActual", top_n=2)
