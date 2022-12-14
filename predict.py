import tensorflow as tf
from model import Donut
from reconstruction import iterative_masked_reconstruct

def predict_score(model, x, y, n_sample_z=1024, mcmc_iteration=10, last_point_only=True):
    if y is not None and mcmc_iteration:
        x_r = iterative_masked_reconstruct(
                reconstruct=model,
                x=x,
                mask=y,
                iter_count=mcmc_iteration,
                n_sample_z=n_sample_z,
            )
    else:
        x_r = x

    x_mean, x_log_var=model(x_r, n_sample_z)
    r_prob=model.log_normal_pdf(x, x_mean, x_log_var)
    if last_point_only:
                r_prob = r_prob[:, -1]
    return r_prob
