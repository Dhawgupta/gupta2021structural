# Structural Credit Assignment in Neural Network using Reinforcement Learning (Regression Code)
​
This repository contains the code for experiments in the paper which correspond to the supervised learning problem. 
​
### Setting up the environment
```
conda create -n coagents python=3.6
pip install -r requirements.txt
conda activate coagents
```
​
### How to run the code
The experiments are defined using the `json` files, and example can be seen in `experiments/debug.json`, and you can run an experiment using the following command
​
```
python src/train.py experiments/debug.json 0
```
​
#### Explaining the json to run different experiment
Here we explain an example `json` file to write an experiment description, and hence run different agents.
```
{
    "problem" : "boston_housing", 
    "agent" : "coagent_global_baseline",
    "epochs" : 30000,
    "optimizer" : ["rmsprop"],
    "model_specification" : [{"num_layers" : 1 }],
    "alpha": [0.00048828125],
    "batch_size" : 32,
    "seed" : [0],
    "units_layer" : [64],
    "eval_greedy" : true,
	"gradient_clipping": ["none"] ,
    "pretrain" : [false]
}
```
`"problem" : "boston_housing"` : The problem specification, in this case Boston Housing. Use `correlated` for the Correlated dataset, with additional `difficulty`, `means` and `segments` keys.
​​
`"agent" : "coagent_global_baseline"` The coagent algorithm type, here it is REINFORCE, use `coagent_ac` for actor critic, `coagent_ac_offpolicy` for off policy actor critic, all of the agents can be found in `src/agents/registry.py`.
  
`"epochs" : 10` : Number of epochs for experiment 
​
`"optimizer" : ["rmsprop"]` : Optimizer to be used
​
`"model_specification" : [{"num_layers" : 1 }]` : Number of layer in network
​
`"alpha": [0.00048828125]` : learning rate
​
`"batch_size" : 32` : Batch size for data
​
`"seed" : [0]` : Randoms seed
​
`"units_layer" : [64]` Number of units per layer
​
`"eval_greedy" : true` : Evalue agent for a greedy policy
​
`"pretrain" : [false]` : No pretraining of network
​
We have different experiment `json` descriptions available in the `experiments` folder. 
​
You can also specify different values in a `list` and `0` in the above python command can be used to index different specifications.
​
### Running Multiple codes together : 
Having specified multiple experiments in a `json` file you can leverage to run different parameters simultaneously on different cpus using below code.
```
python run/local.py -p src/train.py -j experiments/debug.json -c <num-cpus>
```
You would need to have `gnu-parallel for the above to work`
​
### Processing Data
Once experiment is complete for a json file (data files stored in `results` folder) you have to process the data which gets stored in `processed` folder.

#### For Boston Housing:

```
python analysis/process_data.py experiments/debug.json

```
#### For the Correlated dataset:

```
python analysis/correlated_process_data.py experiments/debug.json
```

​
### Plotting the data
There are prepared scripts to plot data , e.g. to plot a learning curve

#### For Boston Housing:

```
python analysis/learning_curve.py y auc experiments/debug.json
```
​
#### For the Correlated dataset:

```
python analysis/correlated_learning_curve.py y auc experiments/debug.json
```

Like this there are multiple codes to plot different curves
