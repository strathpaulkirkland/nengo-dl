# pylint: disable=missing-docstring

import nengo
import numpy as np
import pytest


@pytest.mark.parametrize("minibatch_size", (None, 1, 3))
def test_tensorgraph_layer(Simulator, seed, minibatch_size):
    n_steps = 100

    with nengo.Network(seed=seed) as net:
        a = nengo.Node(lambda t: np.sin(20 * np.pi * t))
        b = nengo.Ensemble(10, 1)
        nengo.Connection(a, b)
        p_a = nengo.Probe(a)
        p_b = nengo.Probe(b)

    with Simulator(net, minibatch_size=minibatch_size) as sim:
        sim.run_steps(n_steps)

    with Simulator(net, minibatch_size=minibatch_size) as layer_sim:
        inputs = layer_sim.tensor_graph.build_inputs()
        outputs = layer_sim.tensor_graph(inputs)

        output_vals = layer_sim.sess.run(
            outputs, feed_dict=layer_sim._fill_feed(n_steps)
        )

        assert len(output_vals) == 2
        assert np.allclose(output_vals[0], sim.data[p_a])
        assert np.allclose(output_vals[1], sim.data[p_b])
