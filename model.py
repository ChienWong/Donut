import tensorflow as tf
import numpy as np

class Donut(tf.keras.Model):
    def __init__(self, x_dims, z_dims, batch_size):
        self.x_dims=x_dims
        self.z_dims=z_dims
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=10.0)
        self.epsilon=0.0001
        super(Donut, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(x_dims,),batch_size=batch_size,dtype=tf.float32),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
                tf.keras.layers.Dense(2*z_dims,dtype=tf.float32),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(z_dims,), batch_size=batch_size,dtype=tf.float32),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
                tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                           activation=tf.nn.relu),
                tf.keras.layers.Dense(2*x_dims,dtype=tf.float32),
            ]
        )
    
    def encode(self, x):
        mean, std = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)           
        log_std=tf.math.log(tf.math.softplus(std)+self.epsilon)
        return mean, log_std

    def reparameterize(self, mean, log_std, n_z=None):
        shape=mean.shape
        if n_z is not None:
            shape=mean.shape+n_z
            eps = tf.random.normal(shape=shape)
            return tf.reduce_mean(eps*tf.expand_dims(tf.exp(log_std),-1)+tf.expand_dims(mean, -1),axis=-1)
        eps = tf.random.normal(shape=shape)
        return eps*tf.exp(log_std)+mean
    
    def decode(self, z):
        mean, std = tf.split(self.decoder(z), num_or_size_splits=2, axis=1)
        log_std=tf.math.log(tf.math.softplus(std)+self.epsilon)
        return mean, log_std

    def log_normal_pdf(self, sample, mean, log_std):
        log2pi = tf.math.log(2. * np.pi)
        return -0.5*log2pi-log_std-0.5*(sample-mean)**2.0*tf.exp(-2.0 * log_std)

    def compute_loss(self, x, labels):
        """ x : 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            labels : 2-D `int32` :class:`tf.Tensor`, the windows of
                ``(label | missing)`` in a mini-batch.
        """
        z_mean, z_log_std = self.encode(x)
        z = self.reparameterize(z_mean, z_log_std)
        x_mean, x_log_std = self.decode(z)
        log_px_z=self.log_normal_pdf(x, x_mean, x_log_std)
        alpha = tf.cast(1 - labels, dtype=tf.float32)
        beta = tf.reduce_mean(alpha, axis=-1)
        log_pz = self.log_normal_pdf(z, 0., 0.)
        log_qz_x = self.log_normal_pdf(z,z_mean, z_log_std)
        loss_gauss=tf.reduce_sum(alpha*log_px_z, axis=-1)
        loss=-tf.reduce_mean((tf.expand_dims(loss_gauss, axis=-1) + tf.expand_dims(beta, axis=-1)*log_pz) - log_qz_x,axis=-1)
        return loss,loss_gauss/tf.reduce_sum(alpha,axis=-1)

    @tf.function
    def train_step(self, data):
        x=data[0]
        labels=data[1]
        with tf.GradientTape() as tape:
            loss, gauss_loss = self.compute_loss(x, labels)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'all_loss':loss,'gauss_loss':gauss_loss}
    
    @tf.function
    def test_step(self, data):
        x=data[0]
        labels=data[1]
        loss, gauss_loss= self.compute_loss(x, labels)
        return {'all_loss':loss,'gauss_loss':gauss_loss}

    def call(self, x, n_sample_z=1024):
        z_mean, z_log_std = self.encode(x)
        z = self.reparameterize(z_mean, z_log_std, n_sample_z)
        x_mean, x_log_std = self.decode(z)
        return x_mean, x_log_std


        


