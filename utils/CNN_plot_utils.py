import matplotlib.pyplot as plt
import csv
import numpy as np


def plot_results(results_file_paths: list, y_label: str):
    results_dict = {}
    model_name = ""
    results_list = []
    models_list = []
    for i in range(len(results_file_paths)):
        file_path = results_file_paths[i]
        model_name = (file_path.split('/')[-1]).split('_')[0] + '_' + (file_path.split('/')[-1]).split('_')[1]
        models_list.append(model_name)
        results_dict[model_name] = []
        results_list = []
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                row = float(row[0])
                results_list.append(row)
        results_dict[model_name] = results_list

    for i in range(len(models_list)):
        x_range = np.linspace(0, len(results_dict[models_list[i]]), len(results_dict[models_list[i]]))
        plt.plot(x_range, results_dict[models_list[i]], label=models_list[i])
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def create_results_file(model_name, loss_results_list=[], accuracy_results_list=[], results_list=[], graph=False, parameters=True):
    parameters_result_path = ''
    results_paths_list = []
    if parameters == True:
        filename = f"{model_name}_parameters_results.csv"
        parameters_result_path = f'CNN_results/parameters/{filename}'
        with open (parameters_result_path, 'w', newline='') as f:
            csvwriter = csv.writer(f)
            for value in results_list:
                csvwriter.writerow([value])
    if graph == True:
        results_dict = {"loss": loss_results_list, "accuracy":accuracy_results_list}
        for key in results_dict.keys():
            filename = f"{model_name}_{key}_graph_results.csv"
            result_path = f'CNN_results/graphs/{filename}'
            results_paths_list.append(result_path)
            with open (result_path, 'w', newline='') as f:
                csvwriter = csv.writer(f)
                for value in results_dict[key]:
                    csvwriter.writerow([value])
    return results_paths_list, parameters_result_path

