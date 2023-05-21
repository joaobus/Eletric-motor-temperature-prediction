import pandas as pd
import wandb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger # type:ignore
from keras.optimizers import Adam  # type:ignore
from wandb.keras import WandbMetricsLogger

from modeling.subclass import RNNRegressor, TCNRegressor
from modeling.functional import cnn_rotor_model, cnn_stator_model, rnn_rotor_model, rnn_stator_model
from utils.data_utils import *
from utils.configs import *
from utils.eval_utils import plot_curves, get_metrics
from explain.feature_importance import PFIExplainer

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Limit GPU usage
total = 49152
limit = total // 5
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])


class Session:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.out_path = os.path.join('out',self.cfg['name'])
        print(f"Model: {self.cfg['name']}")
        print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}\n")

        self.features, self.targets = self.load_data()
        self.train_ds, self.val_ds, self.test_ds = batch_and_split(self.features,self.targets,self.cfg['window'])


    def load_data(self):
        df = pd.read_csv('data/measures_v2.csv')
        df_norm = normalize_data(df)
        df_rotor = df_norm.drop(['stator_winding','stator_tooth','stator_yoke'],axis=1).copy()
        df_stator = df_norm.drop(['pm'],axis=1).copy()

        y_rotor = df_rotor['pm'].copy()
        y_stator = df_stator[['stator_winding','stator_tooth','stator_yoke']].copy()
        X = df_rotor.drop(['pm'],axis=1).copy()
        X = add_extra_features(X,self.cfg['spans'])
        return [X, y_rotor] if self.cfg['target'] == 'rotor' else [X, y_stator]
    

    def load_model_weights(self, path):
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=Adam(learning_rate=self.cfg['lr'], 
                                            clipnorm=self.cfg['grad_norm'], 
                                            clipvalue=self.cfg['grad_clip']),
                            metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.model.load_weights(path)
        return self


    def compile_and_fit(self,
                        max_epochs: int = 200,
                        log: bool = False,
                        resume_training: bool = False):
            
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=train_cfg['patience'])
        early = EarlyStopping(monitor='val_loss', patience=2*train_cfg['patience'], mode='min')
        checkpoint = ModelCheckpoint(os.path.join(self.out_path,'model.h5'), monitor='val_loss', save_best_only=False, mode='min')
        csv_logger = CSVLogger(os.path.join(self.out_path,'history_log.csv'), append=resume_training)
        callbacks = [reduce, early, checkpoint, csv_logger]

        if log:
            wandb.init(
                    project=f"Motor Temperature Predicition - {self.cfg['name']}",

                    config={
                    'dataset': 'electric-motor-temperature',
                    'epochs': max_epochs,
                    'patience':train_cfg['patience'],
                    } | self.cfg, 

                    resume=resume_training
                )
            
            logger = WandbMetricsLogger()
            callbacks.append(logger)
        
        # model.build([None, cfg['window'], 87])

        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                            optimizer=Adam(learning_rate=self.cfg['lr'], 
                                            clipnorm=self.cfg['grad_norm'], 
                                            clipvalue=self.cfg['grad_clip']),
                            metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        print(f"Training model: {self.cfg['name']}\n")
        
        history = self.model.fit(self.train_ds, epochs=max_epochs,
                                validation_data=self.val_ds,
                                callbacks=callbacks)

        if log:
            wandb.finish()
        
        # with open(os.path.join(self.out_path,'history_dict'), 'wb') as file_pi:
        #     pickle.dump(history.history, file_pi)

        plot_curves(history, self.out_path)
        self.load_model_weights(os.path.join(self.out_path,'model.h5'))        
        return history
    
    
    def get_model_metrics(self, save_dir = None):

        path = save_dir if save_dir is not None else self.out_path
        if not os.path.exists(path):
           os.makedirs(path)

        print('Getting test metrics...')
        test_predictions, test_metrics = get_metrics(self.model, self.test_ds, self.cfg['target'], index='test')
        print('Getting val metrics...')
        val_predictions, val_metrics = get_metrics(self.model, self.val_ds, self.cfg['target'], index='val')
        print('Getting train metrics...')
        train_predictions, train_metrics = get_metrics(self.model, self.train_ds, self.cfg['target'], index='train')

        metrics = pd.concat([test_metrics, val_metrics, train_metrics], axis=1)

        path = save_dir if save_dir is not None else self.out_path

        test_predictions.to_csv(os.path.join(path,'test_predictions.csv'))
        val_predictions.to_csv(os.path.join(path,'val_predictions.csv'))
        train_predictions.to_csv(os.path.join(path,'train_predictions.csv'))
        metrics.to_csv(os.path.join(path,'metrics.csv'))



def train_model(model, cfg, load_path):
    MAX_EPOCHS = 500
    LOG = True
    RESUME = False
    SAMPLE = 500000

    session = Session(model, cfg)
    session.load_model_weights(load_path)    
    # session.compile_and_fit(max_epochs=MAX_EPOCHS, log=LOG, resume_training=RESUME)
    # session.get_model_metrics()
    explainer = PFIExplainer(session.model, session.cfg)
    fi = explainer.feature_importance(session.features[-SAMPLE:], session.targets[-SAMPLE:])
    explainer.plot_pfi(fi, os.path.join(session.out_path, 'pfi'))



if __name__ == '__main__':
    N_FEATURES = 135

    # train_model(rnn_rotor_model(N_FEATURES), rnn_rotor_cfg, 'out/RNN_rotor/model.h5')
    # train_model(rnn_stator_model(N_FEATURES), rnn_stator_cfg, 'out/RNN_stator/model.h5')
    # train_model(cnn_rotor_model(N_FEATURES), tcn_rotor_cfg, 'out/TCN_rotor/model.h5')
    train_model(cnn_stator_model(N_FEATURES), tcn_stator_cfg, 'out/TCN_stator/model.h5')

    # from multiprocessing import Process

    # args = [
    #     (rnn_rotor_model(N_FEATURES), rnn_rotor_cfg, 'out/RNN_rotor/model.h5'),
    #     (rnn_stator_model(N_FEATURES), rnn_stator_cfg, 'out/RNN_stator/model.h5'),
    #     (cnn_rotor_model(N_FEATURES), tcn_rotor_cfg, 'out/TCN_rotor/model.h5'),
    #     (cnn_stator_model(N_FEATURES), tcn_stator_cfg, 'out/TCN_stator/model.h5')
    # ]

    # for arg in args:
    #     p = Process(target=train_model, args=arg)
    #     p.start()


