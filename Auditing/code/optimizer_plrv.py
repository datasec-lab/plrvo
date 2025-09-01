from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
import tensorflow.compat.v1 as tf
parent_code = tf.train.Optimizer.compute_gradients.__code__
GATE_OP = tf.train.Optimizer.GATE_OP  # pylint: disable=invalid-name

from absl import logging

PLRVGradientDescentOptimizer = tf.train.GradientDescentOptimizer

def make_vectorized_optimizer_class(cls):
  """Given a subclass of `tf.compat.v1.train.Optimizer`, returns a vectorized DP-SGD subclass of it.

  Args:
    cls: Class from which to derive a PLRV DP subclass. Should be a subclass of
      `tf.compat.v1.train.Optimizer`.

  Returns:
    A PLRV DP-SGD subclass of `cls`.
  """
  child_code = cls.compute_gradients.__code__
  if child_code is not parent_code:
    logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class PLRVDPOptimizerClass(cls):  # pylint: disable=empty-docstring
    __doc__ = ("""The following instrucion is not modified for PLRV mechansim so far.
               
               Vectorized DP subclass of `{base_class}` using Gaussian
       averaging.

       You can use this as a differentially private replacement for
       `{base_class}`. This optimizer implements DP-SGD using
       the standard Gaussian mechanism. It differs from `{dp_class}` in that
       it attempts to vectorize the gradient computation and clipping of
       microbatches.

       When instantiating this optimizer, you need to supply several
       DP-related arguments followed by the standard arguments for
       `{short_base_class}`.

       Examples:

       ```python
       # Create optimizer.
       opt = {dp_vectorized_class}(l2_norm_clip=1.0, noise_multiplier=0.5, num_microbatches=1,
                <standard arguments>)
       ```

       When using the optimizer, be sure to pass in the loss as a
       rank-one tensor with one entry for each example.

       ```python
       # Compute loss as a tensor. Do not call tf.reduce_mean as you
       # would with a standard optimizer.
       loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels=labels, logits=logits)

       train_op = opt.minimize(loss, global_step=global_step)
       ```
       """).format(
           base_class='tf.compat.v1.train.' + cls.__name__,
           dp_class='DP' +
           cls.__name__.replace('Optimizer', 'PLRVOptimizer'),
           short_base_class=cls.__name__,
           dp_vectorized_class='VectorizedDP' + cls.__name__)

    def generate_plrv_noise(self, params, noise_size):
        params_bs = generate_plrv_b(params=params, noise_size=noise_size)
        return np.random.laplace(0, params_bs)
    
    
    def generate_plrv_b(self, params, noise_size=1):
        us = np.random.gamma(params['k'], params['theta'], noise_size)
        return 1/us

    def __init__(
        self,
        l2_norm_clip,
        noise_multiplier,
        num_microbatches=None,
        params=None,
        *args,  # pylint: disable=keyword-arg-before-vararg, g-doc-args
        **kwargs):
      """Initialize the PLRVDPOptimizerClass.

      Args:
        l2_norm_clip: Clipping norm (max L2 norm of per microbatch gradients).
        noise_multiplier: PLRV noise epsilon threshold.
        num_microbatches: Number of microbatches into which each minibatch is
          split. If `None`, will default to the size of the minibatch, and
          per-example gradients will be computed.
        plrv_params: noise parameters of plrv noise.
        *args: These will be passed on to the base class `__init__` method.
        **kwargs: These will be passed on to the base class `__init__` method.
      """
      super(PLRVDPOptimizerClass, self).__init__(*args, **kwargs)
      self._l2_norm_clip = l2_norm_clip
      self._noise_multiplier = noise_multiplier
      self._num_microbatches = num_microbatches
      self._params = params

    def compute_gradients(self,
                          loss,
                          var_list,
                          gate_gradients=GATE_OP,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False,
                          grad_loss=None,
                          gradient_tape=None):
      """PLRV DP-SGD version of base class method."""
      if callable(loss):
        # TF is running in Eager mode
        raise NotImplementedError('Vectorized optimizer unavailable for TF2.')
      else:
        # TF is running in graph mode, check we did not receive a gradient tape.
        if gradient_tape:
          raise ValueError('When in graph mode, a tape should not be passed.')

        batch_size = tf.shape(input=loss)[0]
        if self._num_microbatches is None:
          self._num_microbatches = batch_size

        # Note: it would be closer to the correct i.i.d. sampling of records if
        # we sampled each microbatch from the appropriate binomial distribution,
        # although that still wouldn't be quite correct because it would be
        # sampling from the dataset without replacement.
        microbatch_losses = tf.reshape(loss, [self._num_microbatches, -1])

        if var_list is None:
          var_list = (
              tf.trainable_variables() + tf.get_collection(
                  tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        def process_microbatch(microbatch_loss):
          """Compute clipped grads for one microbatch."""
          microbatch_loss = tf.reduce_mean(input_tensor=microbatch_loss)
          grads, _ = zip(*super(PLRVDPOptimizerClass, self).compute_gradients(
              microbatch_loss,
              var_list,
              gate_gradients,
              aggregation_method,
              colocate_gradients_with_ops,
              grad_loss))
          grads_list = [
              g if g is not None else tf.zeros_like(v)
              for (g, v) in zip(list(grads), var_list)
          ]
          # Clip gradients to have L2 norm of l2_norm_clip.
          # Here, we use TF primitives rather than the built-in
          # tf.clip_by_global_norm() so that operations can be vectorized
          # across microbatches.
          grads_flat = tf.nest.flatten(grads_list)
          squared_l2_norms = [
              tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
          ]
          global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
          div = tf.maximum(global_norm / self._l2_norm_clip, 1.)
          clipped_flat = [g / div for g in grads_flat]
          clipped_grads = tf.nest.pack_sequence_as(grads_list, clipped_flat)
          return clipped_grads

        clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses)

        def reduce_noise_normalize_batch(stacked_grads):
          summed_grads = tf.reduce_sum(input_tensor=stacked_grads, axis=0)
          noise = self.generate_plrv_noise(params=self._params, noise_size=tf.shape(input=summed_grads))
          noise = tf.convert_to_tensor(noise, dtype=tf.float32)
          noised_grads = summed_grads + noise
          return noised_grads / tf.cast(self._num_microbatches, tf.float32)

        final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
                                            clipped_grads)

        return list(zip(final_grads, var_list))

  return PLRVDPOptimizerClass


VectorizedPLRVDPSGDOptimizer = make_vectorized_optimizer_class(
    PLRVGradientDescentOptimizer
)
VectorizedPLRVDPSGD = VectorizedPLRVDPSGDOptimizer

