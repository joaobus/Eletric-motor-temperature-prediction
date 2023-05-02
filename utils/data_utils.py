import numpy as np
import tensorflow as tf
from utils.configs import train_cfg

def normalize_data(df):
    df_norm = df.copy()

    for ind, mean in enumerate(df.mean()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] - mean

    for ind, std in enumerate(df.std()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] / std
        
    return df_norm


def ewma_ewms_features(df,spans):
   '''
   Calcutes EWMA and EWMS for a given set of spans
    df: Pandas dataframe
    spans: list of spans. EWMA and EWMS will be calculated for 
            each and added as a feature
   '''
   df_features = df.copy()
   shape = df.shape[1]
   
   for key in df_features.keys()[0:shape]:
       for span in spans: 
           df_features[key + '_ewma_' + str(span)] = df_features[key].ewm(span=span).mean()
           df_features[key + '_ewms_' + str(span)] = df_features[key].ewm(span=span).std()

   df_features.fillna(method='bfill',inplace=True)
   return df_features


def add_extra_features(df):
    df_features = df.copy()
    extra_feats = {
     'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  # Current vector norm
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  # Voltage vector norm
     'S_el': lambda x: x['i_s']*x['u_s'],                  # Apparent power
     'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  # Effective power
     'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
     'S_x_w': lambda x: x['S_el']*x['motor_speed'],
    }
    return df_features.assign(**extra_feats)


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
    
    print(f"\nTamanho do dataset de treino: {len(train_ds)}") 
    print(f"Tamanho do dataset de validação: {len(val_ds)}")
    print(f"Tamanho do dataset de teste: {len(test_ds)}\n")
    
    return train_ds, val_ds, test_ds


