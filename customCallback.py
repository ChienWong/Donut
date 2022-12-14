import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} ".format(
                epoch, logs["loss"]))
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if epoch % lr_anneal_epochs==0:
            lr *= lr_anneal_factor