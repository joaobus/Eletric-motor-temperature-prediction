import tensorflow as tf # type:ignore
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class PFIExplainer():
  '''
  Permutation feature importance
    FI is calculated for each feature by the formula E_perm/E_base, where E_perm stands for the 
    error without the feature and E_base is the base error, with all the features.
  References:
    https://arxiv.org/abs/1801.01489
    https://christophm.github.io/interpretable-ml-book/feature-importance.html#theory-3
  '''
  def __init__(self, model, cfg):
    self.model = model
    self.cfg = cfg

  def feature_importance(self, X, y, batch_size = 5000):
    '''
    Parameters:
      X: Pandas Dataframe containing the features
      y: Pandas Dataframe containing the targets
    Returns: dictionary containing feature importance values. 
    '''
    fi_values = {}
    ds = tf.keras.utils.timeseries_dataset_from_array(
                            X[:-1],
                            np.roll(y, -self.cfg['window'])[:-1],
                            sequence_length=self.cfg['window'],
                            batch_size=batch_size
                        )   
    base_error = self.model.evaluate(ds)[0]

    for key in tqdm(X.keys()):
      X_perm = X.copy().reset_index(drop=True)
      X_perm[key] = X_perm[key].sample(frac=1).reset_index(drop=True)
      ds_perm = tf.keras.utils.timeseries_dataset_from_array(
                            X_perm[:-1],
                            np.roll(y, -self.cfg['window'])[:-1],
                            sequence_length=self.cfg['window'],
                            batch_size=batch_size
                        )   

      perm_error = self.model.evaluate(ds_perm, verbose=0)[0]
      feature_importance = {key: perm_error / base_error}
      fi_values = fi_values | feature_importance
    return fi_values
  
  
  def plot_pfi(self, fi_values, out_dir, show: bool = False):

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    fi_df = pd.DataFrame(fi_values,index=['fi_values']).sort_values(by=['fi_values'],axis=1,ascending=False).transpose()
      
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8,24))
    sns.barplot(ax=ax,x=fi_df.iloc[:,0], y=fi_df.index)
    ax.axvline(x=1)
    ax.set(xlabel='PFI', ylabel='Features', title=f'Permutation Feature Importance: {self.cfg["name"]}')

    plt.savefig(os.path.join(out_dir,f'fi_{self.cfg["name"]}.png'), bbox_inches="tight")
    if show:
      plt.show()