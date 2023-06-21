train_cfg = {
    'batch_size':5000,
    'patience':10,
}


rnn_stator_cfg = {
    'name':'RNN_stator',
    'type':'rnn',
    'target':'stator',
    'n_out':3,
    'n_hidden':2,
    'spans': [500,2204,6000,9000],
    'n_units': 256,
    'window': 42,
    'lr': 1.4e-3,
    'reg_rate': 1e-9,
    'dropout_rate': 0.37,
    'grad_norm': 9.4,
    'grad_clip': 0.076,
    'grad_noise': 1e-9
}


rnn_rotor_cfg = {
    'name':'RNN_rotor',
    'type':'rnn',
    'target':'rotor',
    'n_out': 1,
    'n_hidden': 1,
    'spans': [1500,2000,4000,7000],
    'n_units': 4,
    'window': 128,
    'lr': 1e-2 / 10,
    'reg_rate': 0.1,
    'dropout_rate': 0.5,
    'grad_norm': 0.25,
    'grad_clip': 0.01,
    'grad_noise': 1e-2
}


tcn_stator_cfg = {
    'name':'TCN_stator',
    'type':'tcn',
    'target':'stator',
    'n_out':3,
    'n_hidden': 4,
    'spans': [620,2915,4487,8825],
    'n_units': 121,
    'lr': 1.4e-4,
    'dropout_rate': 0.29,
    'grad_norm': None,
    'grad_clip': None,
    'kernel_size': 6,
    'window': 32
}


tcn_rotor_cfg = {
    'name':'TCN_rotor',
    'type':'tcn',
    'target':'rotor',
    'n_out':1,
    'n_hidden': 2,
    'spans': [500,2161,4000,8895],
    'n_units': 126,
    'lr': 1e-4,
    'dropout_rate': 0.35,
    'grad_norm': None,
    'grad_clip': None,
    'kernel_size': 2,
    'window': 33
}