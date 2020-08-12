import tensorflow as tf
import datetime

class CreateTrainer():
    def __init__(self, model, data, hyper_paramter):
        self.model = model
        self.epochs = hyper_paramter['epochs']
        self.batch_size = hyper_paramter['batch_size']
        self.validation_split = hyper_paramter['validation_split']
             
        (self.x_train, self.y_train) = data.get_train_data()
        (self.x_test , self.y_test ) = data.get_test_data()
      
    def evaluate(self):
        self.model.evaluate(x=self.x_test, y=self.y_test, verbose=1)
    
    def train(self):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(
            x=self.x_train, 
            y=self.y_train, 
            batch_size=self.batch_size, 
            epochs=self.epochs, 
            validation_split=self.validation_split,
            verbose=1, 
            callbacks=[tensorboard_callback])