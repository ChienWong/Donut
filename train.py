from dataset import dataset_sliding_window
from model import Donut
import tensorflow as tf
from predict import predict_score
import numpy as np
import os
import json

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
train_data, valid_data, test_data=dataset_sliding_window(path, x_dims, batch_size, inject_error_rate=0.1)
if not os.path.exists(path.replace("dataset", "modelweight").replace("csv", "h5")):
    model=Donut(x_dims, z_dims, batch_size)
    model.compile()
    model.fit(x=train_data,
    validation_data=valid_data, 
    epochs=512, 
    callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)],
    workers=4,
    use_multiprocessing=True
    )
    model.save_weights(path.replace("dataset", "modelweight").replace("csv", "h5"),overwrite=True,save_format="h5")

model=Donut(x_dims, z_dims, 1)
model.build(input_shape=(1,x_dims))
model.load_weights(path.replace("dataset", "modelweight").replace("csv", "h5"))

result=[]
kpi=[]
label=[]
for data in test_data:
    result.append(predict_score(model, data[0], data[1]))
    kpi.append(data[0][0][-1])
    label.append(data[2][0][-1])

result=np.asarray(result).flatten()
kpi=np.asarray(kpi).flatten()
label=np.asarray(label).flatten()
assert len(result)==len(kpi)
assert len(result)==len(label)
print(result)
print(kpi.tolist())
print(label.tolist())
if not os.path.exists(path.replace("dataset", "result").replace("csv", "txt")):
    with open(path.replace("dataset", "result").replace("csv", "txt"),'w') as f:
        json.dump({'result':result.tolist(),'kpi':kpi.tolist(),'label':label.tolist()},f)
        f.close()

'''
save z sample to file
'''
# result=[]
# for data in test_data:
#     z=model(data[0])
#     result.append(np.asarray(z).tolist())
# if not os.path.exists(path.replace("dataset", "result").replace("csv", "txt")):
#     with open(path.replace("dataset", "result").replace("csv", "txt"),'w') as f:
#         json.dump(result,f)
#         f.close()



