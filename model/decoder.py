import theano
import theano.tensor as T

import nn
import ops
from bridge import attention,domain_sensitive_attention
import numpy


class Decoder:
    def __init__(self, dim_y, dim_hid, dim_key, dim_value, dnum, n_y_vocab, *args, **kwargs):
        """
        :param dim_y: 
        :param dim_hid: dimension of decoder's hidden state 
        :param dim_key: dimension of query keys
        :param dim_value: dimension of context values
        """
        self.dim_y = dim_y
        self.dim_hid = dim_hid
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dnum = dnum

    def step(self, y_prev, mask, state, keys, values, key_mask, domain_keys, domain_annot):
        """
        forward step
        return (state, context) pair
        """
        raise NotImplementedError

    def prediction(self, y_emb, state, context, keep_prob=1.0):
        raise NotImplementedError

    def build_sampling(self, src_seq, src_mask, target_embedding, target_bias, keys, values, initial_state):
        # sampling graph, this feature is optional
        raise NotImplementedError

    def build_attention(self, src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, keys, values,
                        initial_state):
        # attention graph, this feature is optional
        raise NotImplementedError

    def get_cost(self, y_seq, mask, probs, alpha):
        assert probs.ndim == 2
        idx = T.arange(y_seq.flatten().shape[0])
        ce = -T.log(probs[idx, y_seq.flatten()])
        ce = ce.reshape(y_seq.shape) * (1 + alpha**2)
        ce = T.sum(ce * mask, 0)

        cost = T.mean(ce)

        snt_cost = ce
        return cost, snt_cost

    def scan(self, y_emb, mask, keys, key_mask, values, initial_state, domain_keys, domain_annot):
        """
        build model
        :return: 
        """
        seq = [y_emb, mask]
        outputs_info = [initial_state, None]
        non_seq = [keys, values, key_mask, domain_keys, domain_annot]
        (states, contexts) = ops.scan(self.step, seq, outputs_info, non_seq)

        return states, contexts

    def forward(self, y_seq, y_emb, mask, keys, key_mask, values, initial_state, domain_keys, domain_annot, tag_seq, keep_prob=1.0):
        """
        return states,contexts,cost 
        """
        raise NotImplementedError


class DecoderGruCond(Decoder):
    """
    prediction: s1, y0 -> y1 
    recurrence: s0, y0 -> s1
    """

    def __init__(self, dim_y, dim_hid, dim_key, dim_value, dnum, dim_readout,  dim_domain, feadim, n_y_vocab, *args, **kwargs):
        """
        see `https://github.com/nyu-dl/dl4mt-tutorial/blob/master/docs/cgru.pdf`
        1. s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        2. c_j = att(H, s_j^{\prime})
        3. s_j = GRU^2(c_j, s_j^{\prime})
        """
        Decoder.__init__(self, dim_y, dim_hid, dim_key, dim_value, dnum, n_y_vocab)
        self.dim_readout = dim_readout
        self.n_y_vocab = n_y_vocab
        self.dim_domain = dim_domain
        self.feadim = feadim
        # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        self.cell1 = nn.rnn_cell.gru_cell([dim_y, dim_hid])
        # s_j = GRU^2(c_j, s_j^{\prime})
        self.cell2 = nn.rnn_cell.gru_cell([dim_value, dim_hid])
        self.tiescope = None

    def step(self, y_prev, mask, state, keys, values, key_mask, domain_keys, domain_annot):
        mask = mask[:, None]
        # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        _, state_prime = self.cell1(y_prev, state, scope="gru1")
        state_prime = (1.0 - mask) * state + mask * state_prime
        # c_j = att(H, s_j^{\prime})
        alpha = attention(state_prime, keys, key_mask, self.dim_hid, self.dim_key)
        context = T.sum(alpha[:, :, None] * values, 0)
        d_alpha = attention(state_prime, domain_keys, key_mask, self.dim_hid, self.dim_key, scope="domain_context")
        d_context = T.sum(d_alpha[:,:, None] * domain_annot, 0)

        gate = nn.feedforward([state_prime, context, d_context], [[self.dim_hid, self.dim_value, self.dim_value],
                                                                  self.dim_value], True, scope="context_gate")
        context = gate * context + (1 - gate) * d_context

        # s_j = GRU^2(c_j, s_j^{\prime})
        output, next_state = self.cell2(context, state_prime, scope="gru2")
        next_state = (1.0 - mask) * state + mask * next_state
        
        return next_state, context

    def build_sampling(self, src_seq, src_mask, target_embedding, target_bias, keys, values, initial_state):
        # sampling graph, this feature is optional
        max_len = T.iscalar()

        def sampling_loop(inputs, state, keys, values, key_mask):
            _, state_prime = self.cell1(inputs, state, scope="gru1")
            alpha = attention(state_prime, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            output, next_state = self.cell2(context, state_prime, scope="gru2")
            probs = self.prediction(inputs, next_state, context)  # p(y_j) \propto f(y_{j-1}, c_j, s_j)
            next_words = ops.random.multinomial(probs).argmax(axis=1)
            new_inputs = nn.embedding_lookup(target_embedding, next_words)
            new_inputs = new_inputs + target_bias

            return [next_words, new_inputs, next_state]

        with ops.variable_scope("decoder"):
            batch = src_seq.shape[1]
            initial_inputs = T.zeros([batch, self.dim_y], theano.config.floatX)

            outputs_info = [None, initial_inputs, initial_state]
            nonseq = [keys, values, src_mask]
            outputs, updates = theano.scan(sampling_loop, [], outputs_info,
                                           nonseq, n_steps=max_len)
            sampled_words = outputs[0]

        sampling_inputs = [src_seq, src_mask, max_len]
        sampling_outputs = sampled_words
        sample = theano.function(sampling_inputs, sampling_outputs,
                                 updates=updates)
        return sample

    def build_attention(self, src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, keys, values, initial_state):
        # attention graph, this feature is optional
        def attention_loop(inputs, mask, state, keys, values, key_mask):
            mask = mask[:, None]
            # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
            _, state_prime = self.cell1(inputs, state, scope="gru1")
            # c_j = att(H, s_j^{\prime})
            alpha = attention(state_prime, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            # s_j = GRU^2(c_j, s_j^{\prime})
            output, next_state = self.cell2(context, state_prime, scope="gru2")
            next_state = (1.0 - mask) * state + mask * next_state
            return [alpha, next_state]

        with ops.variable_scope("decoder"):
            seq = [target_inputs, tgt_mask]
            outputs_info = [None, initial_state]
            nonseq = [keys, values, src_mask]
            (alpha, state), updaptes = theano.scan(attention_loop, seq,
                                                   outputs_info, nonseq)
            attention_score = alpha

        alignment_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        alignment_outputs = attention_score
        align = theano.function(alignment_inputs, alignment_outputs)
        return align

    def prediction(self, y_emb, state, context, keep_prob=1.0):
        """
        readout -> softmax
        p(y_j) \propto f(y_{j-1}, s_{j}, c_{j})
        :param y_emb: 
        :param state: 
        :param context: 
        :param keep_prob: 
        :return: 
        """
        features = [state, y_emb, context]
        readout = nn.feedforward(features, [[self.dim_hid, self.dim_y, self.dim_value], self.dim_readout], True,
                                 activation=T.tanh,
                                 scope="readout")

        if keep_prob < 1.0:
            readout = nn.dropout(readout, keep_prob=keep_prob)

        with ops.variable_scope(self.tiescope, reuse=True):
            target_embedding = ops.get_variable("embedding",
                                                [self.n_y_vocab, self.dim_readout])
            target_embedding = target_embedding.T
            logits = T.dot(readout, target_embedding)

        # logits = nn.linear(readout, [self.dim_readout, self.n_y_vocab], True,
        #                    scope="logits")
        if logits.ndim == 3:
            new_shape = [logits.shape[0] * logits.shape[1], -1]
            logits = logits.reshape(new_shape)

        probs = T.nnet.softmax(logits)
        return probs

    def forward(self, y_seq, y_emb, mask, keys, key_mask, values, initial_state, domain_keys, domain_annot, tag_seq, keep_prob=1.0):
        # shift embedding
        y_shifted = T.zeros_like(y_emb)
        y_shifted = T.set_subtensor(y_shifted[1:], y_emb[:-1])
        y_emb = y_shifted
        # feed
        states, contexts = Decoder.scan(self, y_emb, mask, keys, key_mask, values, initial_state, domain_keys, domain_annot)

        with ops.variable_scope("DSAdec"):
            newmask = T.set_subtensor(mask[T.cast(T.sum(mask, 0) - 1, 'int32'), T.arange(mask.shape[1])], 0.0)
            # domain_alpha = domain_sensitive_attention(states, newmask, self.dim_hid, self.dim_domain)
            domain_alpha = attention(states[-1], states, newmask,
                                     self.dim_hid,
                                     self.dim_hid)
            domain_states = states * domain_alpha[:,:,None]

            # batch * (shdim * 2)
            domain_context = T.sum(domain_states, 0)
            # batch * feadim1
            feature = nn.feedforward(domain_context, [self.dim_hid, self.feadim], True,
                                      activation=T.tanh, scope="feature")

            dscores = nn.feedforward(feature, [self.feadim, self.dnum], True, activation=T.tanh, scope="score")
            # (batch, 4)
            dprobs = T.nnet.softmax(dscores)
            pred_tag = T.argmax(dprobs, 1)
            didx = T.arange(tag_seq.flatten().shape[0])
            dce = -T.log(dprobs[didx, tag_seq.flatten()])
            domaincost = T.mean(dce)

        # p(y_j) \propto f(y_{j-1}, s_{j}, c_{j})
        probs = self.prediction(y_emb, states, contexts, keep_prob)

        # compute cost
        cost, snt_cost = self.get_cost(y_seq, mask, probs, domain_alpha)
        return states, contexts, cost, domaincost, pred_tag, snt_cost


class DecoderGruSimple(Decoder):
    """
    prediction: s0, y0 -> y1 
    recurrence: s0, y1 -> s1
    """

    def __init__(self, dim_y, dim_hid, dim_key, dim_value, dim_maxout, max_part, dim_readout, n_y_vocab, *args,
                 **kwargs):
        """
        1. c_j = att(H, s_{j-1})
        2. s_j = GRU(y_{j-1}, c_j, s_{j-1})
        """
        Decoder.__init__(self, dim_y, dim_hid, dim_key, dim_value, n_y_vocab, *args, **kwargs)
        # GRU(y_{j-1}, c_j, s_{j-1})
        self.cell = nn.rnn_cell.gru_cell([[dim_y, dim_value], dim_hid])
        self.dim_maxout = dim_maxout
        self.max_part = max_part
        self.dim_readout = dim_readout
        self.n_y_vocab = n_y_vocab

        self.tiescope = None

    def step(self, y_prev, mask, state, keys, values, key_mask):
        mask = mask[:, None]
        alpha = attention(state, keys, key_mask, self.dim_hid, self.dim_key)
        context = T.sum(alpha[:, :, None] * values, 0)
        output, next_state = self.cell([y_prev, context], state)
        next_state = (1.0 - mask) * state + mask * next_state
        return next_state, context

    def build_sampling(self, src_seq, src_mask, target_embedding, target_bias, keys, values, initial_state):
        # sampling graph, this feature is optional
        max_len = T.iscalar()

        def sampling_loop(inputs, state, keys, values, key_mask):
            alpha = attention(state, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            probs = self.prediction(inputs, state, context)
            next_words = ops.random.multinomial(probs).argmax(axis=1)
            new_inputs = nn.embedding_lookup(target_embedding, next_words)
            new_inputs = new_inputs + target_bias
            output, next_state = self.cell([inputs, context], state)

            return [next_words, new_inputs, next_state]

        with ops.variable_scope("decoder"):
            batch = src_seq.shape[1]
            initial_inputs = T.zeros([batch, self.dim_y], theano.config.floatX)

            outputs_info = [None, initial_inputs, initial_state]
            nonseq = [keys, values, src_mask]
            outputs, updates = theano.scan(sampling_loop, [], outputs_info,
                                           nonseq, n_steps=max_len)
            sampled_words = outputs[0]

        sampling_inputs = [src_seq, src_mask, max_len]
        sampling_outputs = sampled_words
        sample = theano.function(sampling_inputs, sampling_outputs,
                                 updates=updates)
        return sample

    def build_attention(self, src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, keys, values, initial_state):
        # attention graph, this feature is optional
        def attention_loop(inputs, mask, state, keys, values, key_mask):
            mask = mask[:, None]
            alpha = attention(state, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            output, next_state = self.cell([inputs, context], state)
            next_state = (1.0 - mask) * state + mask * next_state

            return [alpha, next_state]

        with ops.variable_scope("decoder"):
            seq = [target_inputs, tgt_mask]
            outputs_info = [None, initial_state]
            nonseq = [keys, values, src_mask]
            (alpha, state), updaptes = theano.scan(attention_loop, seq,
                                                   outputs_info, nonseq)
            attention_score = alpha

        alignment_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        alignment_outputs = attention_score
        align = theano.function(alignment_inputs, alignment_outputs)
        return align

    def prediction(self, y_emb, state, context, keep_prob=1.0):
        """
        maxout -> readout -> softmax
        p(y_j) \propto f(y_{j-1}, s_{j-1}, c_{j})
        :param y_emb: 
        :param state: 
        :param context: 
        :param keep_prob: 
        :return: 
        """

        features = [state, y_emb, context]
        maxhid = nn.maxout(features, [[self.dim_hid, self.dim_y, self.dim_value], self.dim_maxout],
                           self.max_part, True)
        readout = nn.linear(maxhid, [self.dim_maxout, self.dim_readout], False,
                            scope="readout")

        if keep_prob < 1.0:
            readout = nn.dropout(readout, keep_prob=keep_prob)

        with ops.variable_scope(self.tiescope, reuse=True):
            target_embedding = ops.get_variable("embedding",
                                                [self.n_y_vocab, self.dim_readout])
            target_embedding = target_embedding.T
            logits = T.dot(readout, target_embedding)

        # logits = nn.linear(readout, [self.dim_readout, self.n_y_vocab], True,
        #                    scope="logits")

        if logits.ndim == 3:
            new_shape = [logits.shape[0] * logits.shape[1], -1]
            logits = logits.reshape(new_shape)

        probs = T.nnet.softmax(logits)

        return probs

    def forward(self, y_seq, y_emb, mask, keys, key_mask, values, initial_state, keep_prob=1.0):
        # (step, batch, dim)
        states, contexts = Decoder.scan(self, y_emb, mask, keys, key_mask, values, initial_state)
        # (step, batch, dim)
        shift_inputs = T.zeros_like(y_emb)
        shift_inputs = T.set_subtensor(shift_inputs[1:],
                                       y_emb[:-1])
        init_state = initial_state[None, :, :]

        all_states = T.concatenate([init_state, states], 0)
        prev_states = all_states[:-1]

        # p(y_j) \propto f(y_{j-1}, s_{j-1}, c_j)
        probs = self.prediction(shift_inputs, prev_states, contexts,
                                keep_prob=keep_prob)
        # compute cost
        cost, snt_cost = self.get_cost(y_seq, mask, probs)
        return states, contexts, cost, snt_cost
