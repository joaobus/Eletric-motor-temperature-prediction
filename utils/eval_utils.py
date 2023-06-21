import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # type:ignore
import pandas as pd
from sklearn.metrics import r2_score
from utils.configs import train_cfg


def plot_curves(history, path: str):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.title.set_text('Mean Squared Error - RNN Stator')
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'val'], loc='upper left')
    
    ax2.plot(history.history['mean_absolute_error'])
    ax2.plot(history.history['val_mean_absolute_error'])
    ax2.title.set_text('Mean Absolute Error - RNN Stator')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(['train', 'val'], loc='upper left')
    
    plt.savefig(os.path.join(path,'training_curves.png'))
    plt.close()
    
    
def get_metrics(model, ds, type: str, index: str = 'value'):

    assert type in ['stator','rotor'], 'Invalid type'
    
    y = np.concatenate([y for x, y in ds], axis=0)
    ypred = np.array(model.predict(ds, batch_size=train_cfg['batch_size']))
    
    mse = tf.keras.metrics.MeanSquaredError()(y, ypred)
    mae = tf.keras.metrics.MeanAbsoluteError()(y, ypred)
    rmse = tf.keras.metrics.RootMeanSquaredError()(y, ypred)
    
    if type == 'stator':
        results = pd.DataFrame({'stator_winding': y[:,0],'winding_pred': ypred[:,0],
                            'stator_tooth': y[:,1],'tooth_pred': ypred[:,1],
                            'stator_yoke': y[:,2],'yoke_pred': ypred[:,2]})

        winding_r2 = r2_score(results['stator_winding'], results['winding_pred'])
        tooth_r2 = r2_score(results['stator_tooth'], results['tooth_pred'])
        yoke_r2 = r2_score(results['stator_yoke'], results['yoke_pred'])

        overall_r2 = r2_score(y, ypred)
        
        metrics = pd.DataFrame({'mse':mse.numpy(),'mae':mae.numpy(),'rmse':rmse.numpy(),
                                'winding_r2':winding_r2,'tooth_r2':tooth_r2,
                                'yoke_r2':yoke_r2,'overall_r2':overall_r2},index=[index]).transpose()
        
    else:
        results = pd.DataFrame({'pm': y,'pm_pred': ypred[:,0]})
        overall_r2 = r2_score(y, ypred)  
        
        metrics = pd.DataFrame({'mse':mse.numpy(),'mae':mae.numpy(),
                                'rmse':rmse.numpy(),'r2':overall_r2},index=[index]).transpose()
    
    return results, metrics