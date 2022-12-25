import tensorflow as tf
import pandas as pd
import numpy as np
from preprocess import complete_timestamp, standardize_kpi
from augmentation import DataAugmentation, MissingDataInjection, inject_error_test

def dataset_sliding_window(path, window_size, batch_size, valid_portion=0.2, test_portion=0.3, missing_rate=0.01, inject_error_rate=None):
    data=pd.read_csv(path)
    timestamps=data["timestamp"]
    values=data["value"]
    labels=data["label"]
    assert len(timestamps)==len(values)
    assert len(values)==len(labels)
    timestamp, missing, (values, labels) = complete_timestamp(timestamps, (values, labels))
    test_n = int(len(values) * test_portion)
    valid_n = int(len(values) * valid_portion)

    train_values, valid_values, test_values = values[:-(test_n+valid_n)], values[-(test_n+valid_n):-test_n], values[-test_n:]
    train_labels, valid_labels, test_labels = labels[:-(test_n+valid_n)], labels[-(test_n+valid_n):-test_n], labels[-test_n:]
    train_missing, valid_missing, test_missing = missing[:-(test_n+valid_n)], missing[-(test_n+valid_n):-test_n], missing[-test_n:]

    train_valid_values, mean, std = standardize_kpi(
        tf.concat([train_values,valid_values],0), 
        excludes=np.logical_or(tf.concat([train_labels, valid_labels],0), 
                                tf.concat([train_missing, valid_missing],0))
        )
    valid_values=train_valid_values[len(train_values):]
    train_values=train_valid_values[:len(train_values)]
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)

    aug = MissingDataInjection(mean, std, missing_rate)
    train_values, train_labels, train_missing=aug.augment(train_values, train_labels, train_missing)
    valid_values, valid_labels, valid_missing=aug.augment(valid_values, valid_labels, valid_missing)
    if inject_error_rate!=None:
        test_values, test_labels=inject_error_test(test_values, test_labels, inject_error_rate)

    train_dataset=tf.data.Dataset.from_tensor_slices(
        (train_values, 
        np.logical_or(train_labels, train_missing).astype(np.int32)
        )).window(window_size,shift=1,drop_remainder=True).flat_map(lambda x,y:tf.data.Dataset.zip((x.batch(window_size),y.batch(window_size)))).batch(batch_size,drop_remainder=True).cache()
    valid_dataset=tf.data.Dataset.from_tensor_slices(
        (valid_values, 
        np.logical_or(valid_labels, valid_missing).astype(np.int32)
        )).window(window_size,shift=1,drop_remainder=True).flat_map(lambda x,y:tf.data.Dataset.zip((x.batch(window_size),y.batch(window_size)))).batch(batch_size,drop_remainder=True).cache()
    test_dataset=tf.data.Dataset.from_tensor_slices(
        (test_values,test_missing,test_labels)
        ).window(window_size, shift=1,drop_remainder=True).flat_map(lambda x,y,z:tf.data.Dataset.zip((x.batch(window_size),y.batch(window_size),z.batch(window_size)))).batch(1).cache()
    return train_dataset, valid_dataset, test_dataset

