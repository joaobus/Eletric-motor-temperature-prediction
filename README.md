# Description

Temperature prediction for a permanent magnet synchronous motor (PMSM). Archicteture and network parameters based on: https://ieeexplore.ieee.org/abstract/document/9296842. 

The original dataset consists of a csv file with six input features and four target temperatures, three of them measured from the motor's stator and one from the rotor. In the aforementioned paper, the authors train four different neural networks, with RNN and CNN models applied to the prediction of temperatures at both the stator and the rotor. The parameters for each neural network are as follows: 

![image](https://github.com/joaobus/Eletric-motor-temperature-prediction/assets/106790069/f90aee36-baac-478b-af2c-d5c8e5df8888)

The above table shows the results of each network's hyperparameter search, as per the paper. Search Ξ optimizes RNNs on stator temperatures exclusively, whereas search Φ stands for RNNs on rotor temperatures; Ψ for TCNs on stator temperatures; and Ω for TCNs on rotor targets.

Two feature explainability techniques are also explored and applied to the trained models in order to reduce the number of features.

The original dataset can be downloaded from: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature

# File structure

The main.py file contains a general purpose Pipeline class for loading model weights, fitting, and evaluating. The utils module contains functions for loading and transforming data (data_utils.py), evaluating the models (eval_utils.py) and the configuration file config.py. The modeling module contains the neural network archictetures, and the explain model includes feature explainability functionalities. Training results, predictions, explanations and pre-trained models are saved in the out folder.

# Training a model

Models can be trained and evaluated with the Pipeline class. For example:

```
N_FEATURES = 10
p = Pipeline(rnn_rotor_model(N_FEATURES), rnn_stator_cfg)
p.compile_and_fit(max_epochs=200)
p.get_model_metrics()
```

In order to load a pretrained model, we can do as follows:

```
N_FEATURES = 10
p = Pipeline(rnn_rotor_model(N_FEATURES), rnn_stator_cfg)
p.load_model_weights('out/models/RNN_rotor_10.h5')
p.get_model_metrics()
```
