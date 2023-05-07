import pandas as pd
import wandb
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam 
from wandb.keras import WandbMetricsLogger

from modeling.subclass import RNNRegressor, TCNRegressor
from modeling.functional import cnn_rotor_model, cnn_stator_model, rnn_rotor_model, rnn_stator_model
from utils.data_utils import *
from utils.configs import *
from utils.eval_utils import plot_curves, get_test_metrics


def get_data():
    df = pd.read_csv('data/measures_v2.csv')
    df_norm = normalize_data(df)
    df_rotor = df_norm.drop(['stator_winding','stator_tooth','stator_yoke'],axis=1).copy()
    df_stator = df_norm.drop(['pm'],axis=1).copy()
    y_rotor = df_rotor['pm'].copy()
    y_stator = df_stator[['stator_winding','stator_tooth','stator_yoke']].copy()
    X = df_rotor.drop(['pm'],axis=1).copy()
    return X, y_rotor, y_stator


def prepare_data(X, y, cfg):
    features = add_extra_features(X,cfg['spans'])
    train_ds, val_ds, test_ds = batch_and_split(features,y,cfg['window'])
    return train_ds, val_ds, test_ds


def compile_and_fit(X, y, model,
                    cfg: dict,
                    max_epochs: int = 20,
                    log: bool = False,
                    resume_training: bool = False):
    
    train_data, val_data, test_data = prepare_data(X, y, cfg)

    path = os.path.join('out',cfg['name'])
    if not os.path.exists(path):
        os.makedirs(path)
    
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=train_cfg['patience'])
    early = EarlyStopping(monitor='val_loss', patience=2*train_cfg['patience'], mode='auto')
    checkpoint = ModelCheckpoint(os.path.join(path,'model.h5'), monitor='val_loss', save_best_only=True, mode='min')
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
    
    # model.build([None, cfg['window'], 87])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=Adam(learning_rate=cfg['lr'], 
                                 clipnorm=cfg['grad_norm'], 
                                 clipvalue=cfg['grad_clip']),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    print(f"Training model: {cfg['name']}\n\n")
    
    history = model.fit(train_data, epochs=max_epochs,
                      validation_data=val_data,
                      callbacks=callbacks)

    if log:
        wandb.finish()
    
    with open(os.path.join(path,'history_dict'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_curves(history, path)
    model.load_weights(os.path.join(path,'model.h5'))
    get_test_metrics(model, test_data, cfg['target'], path)
    
    return model


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    X, y_rotor, y_stator = get_data()

    # rotor_rnn = RNNRegressor(rnn_rotor_cfg)
    # stator_rnn = RNNRegressor(rnn_stator_cfg)
    # rotor_tcn = TCNRegressor(tcn_rotor_cfg)
    # stator_tcn = TCNRegressor(tcn_stator_cfg)

    rotor_rnn = rnn_rotor_model()
    stator_rnn = rnn_stator_model()
    rotor_tcn = cnn_rotor_model()
    stator_tcn = cnn_stator_model()
    
    MAX_EPOCHS = 500
    LOG = False

    # model = compile_and_fit(X, y_rotor,
    #                         rotor_rnn,
    #                         rnn_rotor_cfg,
    #                         max_epochs=MAX_EPOCHS,
    #                         log = LOG)
    # model = compile_and_fit(X, y_stator,
    #                         stator_rnn,
    #                         rnn_stator_cfg,
    #                         max_epochs=MAX_EPOCHS,
    #                         log = LOG)
    # model = compile_and_fit(X, y_rotor,
    #                         rotor_tcn,
    #                         tcn_rotor_cfg,
    #                         max_epochs=MAX_EPOCHS,
    #                         log = LOG)
    # model = compile_and_fit(X, y_stator,
    #                         stator_tcn,
    #                         tcn_stator_cfg,
    #                         max_epochs=MAX_EPOCHS,
    #                         log = LOG)

    path = os.path.join('out',rnn_rotor_cfg['name'])
    train_data, val_data, test_data = prepare_data(X, y_rotor, rnn_rotor_cfg)
    model = cnn_rotor_model()
    model.load_weights('out/TCN_rotor/model.h5')
    get_test_metrics(model, test_data, rnn_rotor_cfg['target'], path)


if __name__ == '__main__':
    main()