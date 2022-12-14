import tensorflow as tf

def masked_reconstruct(reconstruct, x, mask, n_sample_z=1024):
    """
    Replace masked elements of `x` with reconstructed outputs.
    This method can be used to do missing data imputation on `x`, with
    the reconstruction outputs for `x`.
    Args:
        reconstruct ((tf.Tensor) -> tf.Tensor): Function for reconstructing `x`.
        x: The tensor to be reconstructed by `func`.
        mask: `int32` mask, must be broadcastable into the shape of `x`.
            Indicating whether or not to mask each element of `x`.
    Returns:
        tf.Tensor: `x` with masked elements replaced by reconstructed outputs.
    """
    x = tf.convert_to_tensor(x)  # type: tf.Tensor
    mask = tf.convert_to_tensor(mask, dtype=tf.int32)  # type: tf.Tensor

    # broadcast mask against x
    old_mask = mask
    try:
        _ = tf.broadcast_static_shape(x.get_shape(), mask.get_shape())
    except ValueError:
        raise ValueError('Shape of `mask` cannot broadcast '
                            'into the shape of `x` ({!r} vs {!r})'.
                            format(old_mask.get_shape(), x.get_shape()))
    mask = mask * tf.ones_like(x, dtype=mask.dtype)

    # validate the shape of mask
    x_shape = x.get_shape()
    mask_shape = mask.get_shape()
    if mask_shape.is_fully_defined() and x_shape.is_fully_defined():
        if mask_shape != x_shape:
            # the only possible situation is that mask has more
            # dimension than x, and we consider this situation invalid
            raise ValueError('Shape of `mask` cannot broadcast '
                            'into the shape of `x` ({!r} vs {!r})'.
                            format(old_mask.get_shape(), x_shape))

        # get reconstructed x
        r_x,_ = reconstruct(x, n_sample_z)

        # get masked outputs
        return tf.where(tf.cast(mask, dtype=tf.bool), r_x, x)


def iterative_masked_reconstruct(reconstruct, x, mask, iter_count, n_sample_z=1024):
    """
    Iteratively reconstruct `x` with `mask` for `iter_count` times.
    This method will call :func:`masked_reconstruct` for `iter_count` times,
    with the output from previous iteration as the input `x` for the next
    iteration.  The output of the final iteration would be returned.
    Args:
        reconstruct: Function for reconstructing `x`.
        x: The tensor to be reconstructed by `func`.
        mask: int32 mask, must be broadcastable against `x`.
            Indicating whether or not to mask each element of `x`.
        iter_count (int or tf.Tensor):
            Number of iterations(must be greater than 1).
    Returns:
        tf.Tensor: The iteratively reconstructed `x`.
    """

    # do the masked reconstructions
    x_r, _ = tf.while_loop(
        lambda x_i, i: i < iter_count,
        lambda x_i, i: (masked_reconstruct(reconstruct, x_i, mask, n_sample_z), i + 1),
        [x, tf.constant(0, dtype=tf.int32)],
        back_prop=False
    )

    return x_r