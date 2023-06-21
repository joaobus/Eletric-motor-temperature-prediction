import os
import pandas as pd
import wandb
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger # type:ignore
from keras.optimizers import Adam  # type:ignore
from wandb.keras import WandbMetricsLogger

from modeling.subclass import RNNRegressor, TCNRegressor
from modeling.functional import cnn_rotor_model, cnn_stator_model, rnn_rotor_model, rnn_stator_model
from utils.data_utils import *
from utils.configs import *
from utils.eval_utils import plot_curves, get_metrics
from explain.feature_importance import PFIExplainer, SHAPExplainer

# Limit GPU usage
total = 49152
limit = total // 5
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])


class Pipeline:
    '''
    General purpose class for loading data, training and evaluating
    '''
    def __init__(self, model, cfg, out_dir = None, feature_names = None):
        self.model = model
        self.cfg = cfg
        self.feature_names = feature_names
        self.out_path = out_dir if out_dir is not None else os.path.join('out',self.cfg['name'])

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

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

        if self.feature_names is not None:
            X = X[self.feature_names].copy()

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
                        resume_training: bool = False,
                        model_save_dir: str = 'out/models'):
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        model_path = os.path.join(model_save_dir, f'{self.cfg["name"]}_{len(self.features.keys())}.h5')
        
        reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=train_cfg['patience'])
        early = EarlyStopping(monitor='val_loss', patience=2*train_cfg['patience'], mode='min')
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=False, mode='min')
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

        # test_predictions.to_csv(os.path.join(path,'test_predictions.csv'))
        # val_predictions.to_csv(os.path.join(path,'val_predictions.csv'))
        # train_predictions.to_csv(os.path.join(path,'train_predictions.csv'))
        metrics.to_csv(os.path.join(path,'metrics.csv'))



def train_model(model, cfg, load_path, save_dir = None):
    
    feature_names = list(pd.read_csv(f'out/{cfg["name"]}/shap/shap_features_{cfg["name"]}.csv', index_col=0).head(10).index)
    p = Pipeline(model, cfg, save_dir, feature_names)
    print(f'Number of features: {len(p.features.keys())} \nFeatures: {feature_names}\n')

    # p.load_model_weights(load_path)
    p.compile_and_fit(max_epochs=MAX_EPOCHS, log=LOG, resume_training=RESUME)
    p.get_model_metrics(save_dir)

    # pfi_explainer = PFIExplainer(p.model, p.cfg)
    # fi = pfi_explainer.feature_importance(p.features[-SAMPLE:], p.targets[-SAMPLE:])
    # pfi_explainer.plot_pfi(fi, os.path.join(p.out_path, 'pfi'))

    background = np.concatenate([x for x, _ in p.train_ds.take(1)], axis=0)
    test = np.concatenate([x for x, _ in p.test_ds.take(1)], axis=0)
    
    shap_explainer = SHAPExplainer(p.model, p.cfg, background, test)
    shap_values = shap_explainer.feature_importance()
    shap_explainer.get_most_important_features(shap_values, p.features.keys(), os.path.join(p.out_path, 'shap'))
    shap_explainer.plot_shap_values(shap_values, os.path.join(p.out_path, 'shap'), p.features.keys())
    



if __name__ == '__main__':

    N_FEATURES = 10
    MAX_EPOCHS = 500
    LOG = True
    RESUME = False
    SAMPLE = 500000


    # train_model(rnn_rotor_model(N_FEATURES), rnn_rotor_cfg, 'out/RNN_rotor/model.h5', 'out/RNN_rotor/less_features')
    # train_model(rnn_stator_model(N_FEATURES), rnn_stator_cfg, 'out/RNN_stator/model.h5', 'out/RNN_stator/less_features')
    # train_model(cnn_rotor_model(N_FEATURES), tcn_rotor_cfg, 'out/TCN_rotor/model.h5', 'out/TCN_rotor/less_features')
    train_model(cnn_stator_model(N_FEATURES), tcn_stator_cfg, 'out/TCN_stator/model.h5', 'out/TCN_stator/less_features')

    # import multiprocessing

    # multiprocessing.set_start_method('spawn')

    # args = [
    #     (rnn_rotor_model(N_FEATURES), rnn_rotor_cfg, 'out/RNN_rotor/model.h5', 'out/RNN_rotor'),
    #     (rnn_stator_model(N_FEATURES), rnn_stator_cfg, 'out/RNN_stator/model.h5', 'out/RNN_stator'),
    #     (cnn_rotor_model(N_FEATURES), tcn_rotor_cfg, 'out/TCN_rotor/model.h5', 'out/TCN_rotor'),
    #     (cnn_stator_model(N_FEATURES), tcn_stator_cfg, 'out/TCN_stator/model.h5', 'out/TCN_stator')
    # ]

    # for arg in args:
    #     p = multiprocessing.Process(target=train_model, args=arg)
    #     p.start()


