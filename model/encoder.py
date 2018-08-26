import theano
import theano.tensor as T

import nn
import ops


def gru_encoder(cell, inputs, mask, initial_state=None, dtype=None):
    if not isinstance(cell, nn.rnn_cell.rnn_cell):
        raise ValueError("cell is not an instance of rnn_cell")

    if isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a tensor, not list or tuple")

    def loop_fn(inputs, mask, state):
        mask = mask[:, None]
        output, next_state = cell(inputs, state)
        next_state = (1.0 - mask) * state + mask * next_state
        return next_state

    if initial_state is None:
        batch = inputs.shape[1]
        state_size = cell.state_size
        initial_state = theano.tensor.zeros([batch, state_size], dtype=dtype)

    seq = [inputs, mask]
    states = ops.scan(loop_fn, seq, [initial_state])

    return states


class Encoder:
    def __init__(self, dim_x, dim_hid):
        self.dim_x = dim_x
        self.dim_hid = dim_hid
        self.cell = nn.rnn_cell.gru_cell([dim_x, dim_hid])

    def forward(self, x_embedded, mask, initial_state=None, dtype=None, scope=None):
        scope = scope or "encoder"
        cell = self.cell
        with ops.variable_scope(scope, dtype=dtype):
            with ops.variable_scope("forward"):
                fd_states = gru_encoder(cell, x_embedded, mask, initial_state, dtype)
            with ops.variable_scope("backward"):
                x_embedded = x_embedded[::-1]
                mask = mask[::-1]
                bd_states = gru_encoder(cell, x_embedded, mask, initial_state, dtype)
                bd_states = bd_states[::-1]

        return fd_states, bd_states
