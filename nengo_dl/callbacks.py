"""
Objects to be used with the Keras callback functionality.

See https://www.tensorflow.org/beta/guide/keras/custom_callback for more information
on how to use Keras callbacks.

The short answer is that these can be passed to, e.g., `.Simulator.fit` like

.. code-block:: python

    sim.fit(..., callbacks=[nengo_dl.callbacks.NengoSummaries(...)]


"""

import nengo
from nengo.exceptions import ValidationError
import tensorflow as tf
from tensorflow.python.eager import context

from nengo_dl import utils


class NengoSummaries(tf.keras.callbacks.Callback):
    """
    Logs the values of Nengo object parameters, to be displayed in TensorBoard.

    See https://www.tensorflow.org/tensorboard/r2/get_started for general instructions
    on using TensorBoard.

    Parameters
    ----------
    log_dir : str
        Directory where log file will be written.
    sim : `.Simulator`
        Simulator object which will be used to look up parameter values.
    objects : list of `nengo.Ensemble` or `nengo.ensemble.Neurons` or `nengo.Connection`
        The object whose parameter values we want to record (passing an Ensemble will
        log its encoders, Neurons will log biases, and Connection will log connection
        weights/decoders).
    """

    def __init__(self, log_dir, sim, objects):
        super().__init__()

        self.sim = sim

        # we do all the summary writing in eager mode, so that it will be executed
        # as the callback is called
        with context.eager_mode():
            self.writer = tf.summary.create_file_writer(log_dir)

        self.summaries = []
        for obj in objects:
            if isinstance(
                obj, (nengo.Ensemble, nengo.ensemble.Neurons, nengo.Connection)
            ):
                if isinstance(obj, nengo.Ensemble):
                    param = "encoders"
                    name = "Ensemble_%s" % obj.label
                elif isinstance(obj, nengo.ensemble.Neurons):
                    param = "bias"
                    name = "Ensemble.neurons_%s" % obj.ensemble.label
                elif isinstance(obj, nengo.Connection):
                    param = "weights"
                    name = "Connection_%s" % obj.label

                self.summaries.append(
                    (utils.sanitize_name("%s_%s" % (name, param)), obj, param)
                )
            else:
                raise ValidationError(
                    "Unknown summary object %s; should be an Ensemble, Neurons, or "
                    "Connection" % obj,
                    "objects",
                )

    def on_epoch_end(self, epoch, logs=None):
        """Log parameter values at the end of each epoch."""

        summary_vals = self.sim.data.get_params(
            *[(obj, attr) for _, obj, attr in self.summaries]
        )

        with context.eager_mode(), self.writer.as_default():
            for (name, _, _), val in zip(self.summaries, summary_vals):
                tf.summary.histogram(name, val, step=epoch)

    def on_train_end(self, logs=None):
        """Close summary writer at end of training."""

        with context.eager_mode():
            self.writer.close()


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """
    A version of the Keras TensorBoard callback that also profiles inference.
    """

    def on_predict_batch_end(self, *args, **kwargs):
        """Redirect to training function."""
        self.on_batch_end(*args, **kwargs)

    def on_predict_begin(self, *args, **kwargs):
        """Redirect to training function."""
        self.on_train_begin(*args, **kwargs)

    def on_predict_end(self, *args, **kwargs):
        """Redirect to training function."""
        self.on_train_end(*args, **kwargs)
