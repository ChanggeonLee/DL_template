from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Input, concatenate, Lambda, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras import Model
import tensorflow as tf

class CreateModel():
    def __init__(self, model_config):
        self.model = None
        self.channel = model_config['channel']
        self.optimizer = model_config['optimizer']
        self.loss = model_config['loss']
        
        self.build_model()
        self.model.summary()

    def save_model(self, path):
        print('Saving model...')
        self.model.save(path + '/model.h5')
        print('model saved')

    def load_model(self, path):
        print('Loading model...')
        self.model.load_weights(path + '/model.h5')
        print('model loaded')
        
    def build_model(self):
        strategy = tf.distribute.MirroredStrategy()
        print('장치의 수: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            channel = self.channel 
            
            inputs = Input((128,256,2))
            
            outputs = inputs
            
            self.model = Model(inputs=[inputs], outputs=[outputs])
            self.model.compile(optimizer=self.optimizer, loss=self.loss,  metrics=['accuracy'])