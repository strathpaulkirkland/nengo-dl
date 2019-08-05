# pylint: disable=missing-docstring

import nengo
import numpy as np


def test_tensorgraph_layer(Simulator):
    with nengo.Network() as net:
        a = nengo.Node([1])
        p = nengo.Probe(a)

    with Simulator(net) as sim:
        inputs = sim.tensor_graph.build_inputs()
        outputs = sim.tensor_graph(inputs)

        output_vals = sim.sess.run(outputs, feed_dict=sim._fill_feed(10))

        assert len(output_vals) == 1
        assert np.allclose(output_vals[0], np.ones((1, 10, 1)))
