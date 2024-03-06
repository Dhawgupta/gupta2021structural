'''
This code will produce the learning curve for different agents
that are specified in the json files
Status : Complete (not completed the key based best parameter selection part)
'''
import os, sys, time, copy
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
from src.utils.json_handling import get_sorted_dict
from analysis.utils import find_best, smoothen_runs
from src.utils.formatting import create_folder, create_file_name
from src.utils.json_handling import get_param_iterable, get_param_iterable_runs
from analysis.colors import agent_colors, line_node, critic_model

import pickle as pkl
# read the arguments etc
if len(sys.argv) < 2:
    print("usage : python analysis/learning_curve.py legend(y/n) <list of json files>")
    exit()
#
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
BIGGEST_SIZE = 25

plt.rc('font', size=BIGGER_SIZE )          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('xtick', titlesize=BIGGEST_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', titlesize=BIGGEST_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


json_files = sys.argv[1:] # all the json files

json_handles = [get_sorted_dict(j) for j in json_files]


layer_colors = {
    0 : 'red',
    1 : 'blue',
    2 : 'green',
    3 : 'purple'
}
plot_type = {
    'entropy' : '-',
    'sparsity' : ':'
}

for en, js in enumerate(json_handles):
    iterables = get_param_iterable_runs(js)
    for i in iterables:
        folder, file = create_file_name(i, 'processed')
        create_folder(folder)  # make the folder before saving the file
        filename = folder + file + '.pcsd'
        # print(f'Agent_{i["agent"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_NodesTrained_{i["number_stochastic_nodes"]}')
        # laoad the run
        try:
            with open(filename, 'rb') as fil:
                data = pkl.load(fil)
        except:
            print("Skipping")
            continue
        try:
            train = data['train']['mean']
            entropy = data['entropy']['mean']
            sparsity = data['sparsity']['mean']
            # train = np.array(data['train-accuracy'])
            # entropy = np.array(data['entropy']['train'])
            # sparsity = np.array(data['sparsity']['train'])
            # print("Accuracy",train[-1])
            entropy_mean = entropy.mean(axis = 2)
            print("Entropy",entropy_mean[-1])
            entropy_std = entropy.std(axis = 2)
            print("Entropy STD", entropy_std[-1])
            layer_entropies = entropy_mean.transpose()
            layer_entropies_std = entropy_std.transpose()

            layer_sparsity = sparsity.transpose()
            fig, axs = plt.subplots(1, figsize=(6, 4))
            smoothfactor = 0.9
            for l in range(layer_entropies.shape[0]):  # no of layers
                axs.plot(smoothen_runs(layer_entropies[l], smoothfactor), color=layer_colors[l], label=f'layer {l + 1}', linestyle = '-', linewidth = 3)
                axs.fill_between(range(layer_entropies[l].shape[0]), smoothen_runs(layer_entropies[l] - layer_entropies_std[l], smoothfactor),
                                 smoothen_runs(layer_entropies[l] + layer_entropies_std[l], smoothfactor), color=layer_colors[l], alpha=0.1)
                # plot the entropies
                axs.plot(smoothen_runs(layer_sparsity[l], smoothfactor), color = layer_colors[l], linestyle = ':', linewidth = 1)

            axs.plot(smoothen_runs(train, smoothfactor), color='black', label='accuracy', linewidth = 3)
            # plt.legend()
            axs.set_ylabel('')
            axs.set_ylim([-.01, 1])
            # axs.set_xlabel('epochs')
            # axs.set_title('entropy, sparsityand accuracy of train')
            axs.spines['right'] .set_visible(False)
            axs.spines['top'].set_visible(False)
            axs.set_xlabel('Epochs')
            axs.tick_params(axis='both', which='major', labelsize=12)
            axs.tick_params(axis='both', which='minor', labelsize=12)
            axs.set_rasterized(True)
            fig.tight_layout()
            # plt.savefig(f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_NodesTrained_{i["number_stochastic_nodes"]}.png')
            # plt.savefig(f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_Normalization_{i["mode_normalization"]}_Actions_{i["num_actions"]}.png')
            # if "critic_nodes" not in i.keys():
            #    i['critic_nodes'] = 0
            # plt.savefig(f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_criticLayer_{i["critic_layer"]}_criticAlpha_{i["critic_alpha_ratio"]}_criticNodes_{i["critic_nodes"]}.png')
            # plt.savefig(f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}.png')
            # plt.show()
            # plt.show()
            # if "critic_nodes" not in i.keys():
            #     i['critic_nodes'] = 0
            # name = f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_criticLayer_{i["critic_layer"]}_criticAlpha_{i["critic_alpha_ratio"]}_criticNodes_{i["critic_nodes"]}_NodesTrained_{i["number_stochastic_nodes"]}'
            # name = f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_NodesTrained_{i["number_stochastic_nodes"]}'
            name = f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}'
            # name = f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_criticLayer_{i["critic_layer"]}_criticAlpha_{i["critic_alpha_ratio"]}_criticNodes_{i["critic_nodes"]}'
            # name = f'entropies/entropy_Agent_{i["agent"]}_Epochs_{i["epochs"]}_Optimizer_{i["optimizer"]}_Layer_{i["model_specification"]["num_layers"]}_UnitsLayer_{i["units_layer"]}_Normalization_{i["mode_normalization"]}_Actions_{i["num_actions"]}'
            plt.savefig(f'{name}.png')
            plt.savefig(f'{name}.pdf', dpi = 300)
        except:
            print("prolly experiment no complete")
            continue

