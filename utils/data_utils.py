import numpy as np
import tensorflow as tf # type:ignore
from utils.configs import train_cfg

def normalize_data(df):
    df_norm = df.copy()

    for ind, mean in enumerate(df.mean()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] - mean

    for ind, std in enumerate(df.std()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] / std
        
    return df_norm


def add_extra_features(df,spans):
   features = df.copy()

   extra_feats = {
    'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  # Current vector norm
    'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  # Voltage vector norm
    'S_el': lambda x: x['i_s']*x['u_s'],                  # Apparent power
    'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  # Effective power
    'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
    'S_x_w': lambda x: x['S_el']*x['motor_speed'],
   }
   df_features = features.assign(**extra_feats).copy()
   
   for key in df_features.keys():
       for span in spans: 
           df_features[key + '_ewma_' + str(span)] = df_features[key].ewm(span=span).mean()
           df_features[key + '_ewms_' + str(span)] = df_features[key].ewm(span=span).std()

   df_features.fillna(method='bfill',inplace=True)
   return df_features



def batch_and_split(X,y,seq_length,
                    val_ratio = 5/140, test_ratio = 7/140):

    ds = tf.keras.utils.timeseries_dataset_from_array(
        X[:-1],
        np.roll(y, -seq_length)[:-1],
        sequence_length=seq_length,
        batch_size=train_cfg['batch_size']
    )   
    
    val_batches = int(np.ceil(val_ratio*len(X)) // train_cfg['batch_size'])
    test_batches = int(np.ceil(test_ratio*len(X)) // train_cfg['batch_size'])
    train_batches = len(ds) - val_batches - test_batches
    
    train_ds = ds.take(train_batches)
    val_ds = ds.skip(train_batches).take(val_batches)
    test_ds = ds.skip(train_batches + val_batches)
    
    print(f"Batches in the training dataset: {len(train_ds)}") 
    print(f"Batches in the validation dataset: {len(val_ds)}")
    print(f"Batches in the test dataset: {len(test_ds)}\n")
    
    return train_ds, val_ds, test_ds


