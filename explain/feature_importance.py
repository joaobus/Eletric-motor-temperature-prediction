import tensorflow as tf
import numpy as np
from tqdm import tqdm

def pfi(X, y, model, seq_length, 
        batch_size = 5000):
  '''
  Permutation feature importance method
    X: Pandas Dataframe containing the features
    y: Pandas Dataframe containing the targets
    model: model to be tested. Needs to have a evaluate method implemented
    loss: loss function. Needs to take in two lists or tensors of values and return a single value
  Returns: dictionary containing feature importance values. FI is calculated for each feature by the
  formula E_perm/E_base, where E_perm stands for the error without the feature and E_base is the base error,
  with all the features.
  '''
  fi_values = {}
  ds = tf.keras.utils.timeseries_dataset_from_array(
                          X[:-1],
                          np.roll(y, -seq_length)[:-1],
                          sequence_length=seq_length,
                          batch_size=batch_size
                      )   
  base_error = model.evaluate(ds)[0]

  for key in tqdm(X.keys()):
    X_perm = X.copy()
    X_perm[key] = X_perm[key].sample(frac=1).reset_index(drop=True)
    ds_perm = tf.keras.utils.timeseries_dataset_from_array(
                          X_perm[:-1],
                          np.roll(y, -seq_length)[:-1],
                          sequence_length=seq_length,
                          batch_size=batch_size
                      )   

    perm_error = model.evaluate(ds_perm, verbose=0)[0]
    feature_importance = {key: perm_error / base_error}
    fi_values = fi_values | feature_importance
  return fi_values