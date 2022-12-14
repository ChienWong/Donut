from dataset import dataset_sliding_window
from model import Donut
import tensorflow as tf
from predict import predict_score
import numpy as np
import os

x_dims=120
z_dims=5
batch_size=256
lr_anneal_epochs=10
lr_anneal_factor=0.75
def scheduler(epoch, lr):
    if epoch % lr_anneal_epochs==0 and epoch !=0:
        return lr * lr_anneal_factor
    else:
        return lr

path="./dataset/cpu4.csv"
train_data, valid_data, test_data=dataset_sliding_window(path, x_dims, batch_size)
model=Donut(x_dims, z_dims, batch_size)
if not os.path.exists(path.replace("dataset", "modelweight").replace("csv", "h5")):
    model.compile()
    model.fit(x=train_data,
    validation_data=valid_data, 
    epochs=256, 
    callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
    workers=4,
    use_multiprocessing=True
    )
    model.save_weights(path.replace("dataset", "modelweight").replace("csv", "h5"),overwrite=True,save_format="h5")
else:
    model.build(input_shape=(batch_size,x_dims))
    model.load_weights(path.replace("dataset", "modelweight").replace("csv", "h5"))

result=[]
for data in test_data:
    result.append(predict_score(model, data[0], data[1]))
print(np.asarray(result).flatten())

