import pandas as pd
import wandb
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam 
from wandb.keras import WandbMetricsLogger

from models import RNNRegressor, TCNRegressor
from utils.data_utils import *
from utils.configs import *
from utils.eval_utils import plot_curves, get_test_metrics


def load_dataset():
    df = pd.read_csv('data/measures_v2.csv')
    df_rotor = df.drop(['stator_winding','stator_tooth','stator_yoke'],axis=1).copy()
    df_stator = df.drop(['pm'],axis=1).copy()
    return df_rotor, df_stator


def prepare_data(X, y, cfg):
    df_norm = normalize_data(X)
    features = add_extra_features(df_norm,cfg['spans'])
    train_ds, val_ds, test_ds = batch_and_split(features,y,cfg['window'])
    return train_ds, val_ds, test_ds


def compile_and_fit(X, y, model,
                    cfg: dict,
                    max_epochs: int = 20,
                    log: bool = False,
                    resume_training: bool = False,
                    pretrained_path = None):
    
    if pretrained_path is not None:
        model.load_weights(pretrained_path)
    
    train_data, val_data, test_data = prepare_data(X, y, cfg)

    path = os.path.join('out',cfg['name'])
    if not os.path.exists(path):
        os.makedirs(path)
    
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=train_cfg['patience'])
    early = EarlyStopping(monitor='val_loss', patience=2*train_cfg['patience'], mode='auto')
    checkpoint = ModelCheckpoint(os.path.join(path,'checkpoint'), monitor='val_loss', save_best_only=True, mode='min')
    callbacks = [reduce, early, checkpoint]

    if log:
        wandb.init(
                project=f"Motor Temperature Predicition - {cfg['name']}",

                config={
                'dataset': 'electric-motor-temperature',
                'epochs': max_epochs,
                'patience':train_cfg['patience'],
                } | cfg, 

                resume=resume_training
            )
        
        logger = WandbMetricsLogger()
        callbacks.append(logger)
    
    model.build([None, cfg['window'], 87])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=cfg['lr'], 
                                 clipnorm=cfg['grad_norm'], 
                                 clipvalue=cfg['grad_clip']),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit(train_data, epochs=max_epochs,
                      validation_data=val_data,
                      callbacks=callbacks)

    if log:
        wandb.finish()
    
    with open(os.path.join(path,'history_dict'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_curves(history, path)
    get_test_metrics(model, test_data, cfg['target'], path)
    
    return model


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    df_rotor, df_stator = load_dataset()
    y_rotor = df_rotor['pm'].copy()
    y_stator = df_stator[['stator_winding','stator_tooth','stator_yoke']].copy()
    X = df_rotor.drop(['pm'],axis=1).copy()

    rotor_rnn = RNNRegressor(rnn_rotor_cfg)
    stator_rnn = RNNRegressor(rnn_stator_cfg)
    rotor_tcn = TCNRegressor(tcn_rotor_cfg)
    stator_tcn = TCNRegressor(tcn_stator_cfg)

    model = compile_and_fit(X, y_rotor,
                            rotor_rnn,
                            rnn_rotor_cfg,
                            max_epochs=1,
                            log = True)
    model = compile_and_fit(X, y_stator,
                            stator_rnn,
                            rnn_stator_cfg,
                            max_epochs=1,
                            log = True)
    model = compile_and_fit(X, y_rotor,
                            rotor_tcn,
                            tcn_rotor_cfg,
                            max_epochs=1,
                            log = True)
    model = compile_and_fit(X, y_stator,
                            stator_tcn,
                            tcn_stator_cfg,
                            max_epochs=1,
                            log = True)



if __name__ == '__main__':
    main()