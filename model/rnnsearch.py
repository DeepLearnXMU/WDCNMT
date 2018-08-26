# rnnsearch.py

import numpy
import theano
import theano.sandbox.rng_mrg
import theano.tensor as T

import nn
import ops
from bridge import map_key,domain_sensitive_attention
from encoder import Encoder
from decoder import DecoderGruSimple, DecoderGruCond
from search import beam, select_nbest


class rnnsearch:
    def __init__(self, **option):
        # source and target embedding dim
        sedim, tedim = option["embdim"]
        # source, target and attention hidden dim
        shdim, thdim, ahdim, domaindim, feadim = option["hidden"]
        # maxout hidden dim
        maxdim = option["maxhid"]
        # maxout part
        maxpart = option["maxpart"]
        # deepout hidden dim
        deephid = option["deephid"]
        svocab, tvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab
        # source and target vocabulary size
        svsize, tvsize = len(sid2w), len(tid2w)

        if "scope" not in option or option["scope"] is None:
            option["scope"] = "rnnsearch"

        if "initializer" not in option:
            option["initializer"] = None

        if "regularizer" not in option:
            option["regularizer"] = None

        if "keep_prob" not in option:
            option["keep_prob"] = 1.0

        dtype = theano.config.floatX
        initializer = option["initializer"]
        regularizer = option["regularizer"]
        keep_prob = option["keep_prob"] or 1.0

        scope = option["scope"]
        decoder_scope = "decoder"


        encoder = Encoder(sedim, shdim)
        decoderType = eval("Decoder{}".format(option["decoder"]))
        decoder = decoderType(tedim, thdim, ahdim, 2 * shdim, dim_maxout=maxdim, max_part=maxpart, dim_readout=deephid,
                              dim_domain=domaindim, feadim=feadim,
                              n_y_vocab=tvsize)


        # training graph
        with ops.variable_scope(scope, initializer=initializer,
                                regularizer=regularizer, dtype=dtype):
            src_seq = T.imatrix("source_sequence")
            src_mask = T.matrix("source_sequence_mask")
            tgt_seq = T.imatrix("target_sequence")
            tgt_mask = T.matrix("target_sequence_mask")
            tag_seq = T.imatrix("domain_tag")

            with ops.variable_scope("source_embedding"):
                source_embedding = ops.get_variable("embedding",
                                                    [svsize, sedim])
                source_bias = ops.get_variable("bias", [sedim])

            with ops.variable_scope("target_embedding") as tgtembscope:
                target_embedding = ops.get_variable("embedding",
                                                    [tvsize, tedim])
                # target_bias = ops.get_variable("bias", [tedim])
                decoder.tiescope = tgtembscope

            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)

            source_inputs = source_inputs + source_bias

            if keep_prob < 1.0:
                source_inputs = nn.dropout(source_inputs, keep_prob=keep_prob)
                target_inputs = nn.dropout(target_inputs, keep_prob=keep_prob)

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            # annotation = nn.dropout(annotation, keep_prob=keep_prob)

            with ops.variable_scope("Specific"):
                domain_alpha = domain_sensitive_attention(annotation, src_mask, shdim * 2, domaindim)
                domain_context = T.sum(annotation * domain_alpha[:,:,None], 0)
                dfeature = nn.feedforward(domain_context, [shdim * 2, feadim], True,
                                          activation=T.tanh, scope="feature1")

                dscores = nn.feedforward(dfeature, [feadim, 4], True, activation=T.tanh, scope="score")
                # (batch, 2)
                dprobs = T.nnet.softmax(dscores)
                dpred_tag = T.argmax(dprobs, 1)
                didx = T.arange(tag_seq.flatten().shape[0])
                dce = -T.log(dprobs[didx, tag_seq.flatten()])
                dcost = T.mean(dce)

            share_alpha = domain_sensitive_attention(annotation, src_mask, shdim * 2, domaindim)
            share_context = T.sum(annotation * share_alpha[:, :, None], 0)
            sfeature = nn.feedforward(share_context, [shdim * 2, feadim], True,
                                      activation=T.tanh, scope="feature1")

            with ops.variable_scope("Shared"):
                sscores = nn.feedforward(sfeature, [feadim, 4], True, activation=T.tanh, scope="score")
                # (batch, 2)
                sprobs = T.nnet.softmax(sscores)
                spred_tag = T.argmax(sprobs, 1)
                sidx = T.arange(tag_seq.flatten().shape[0])
                sce = -T.log(sprobs[sidx, tag_seq.flatten()])
                scost = T.mean(sce)
                adv_sce = - sprobs[sidx, tag_seq.flatten()] * T.log(sprobs[sidx, tag_seq.flatten()])
                adv_scost = T.mean(adv_sce)

            domain_gate = nn.feedforward([dfeature,annotation], [[feadim, shdim * 2],shdim*2], True,
                                         scope="domain_gate")
            domain_annotation = annotation * domain_gate
            domain_annotation = nn.dropout(domain_annotation, keep_prob=keep_prob)
            share_gate = nn.feedforward([sfeature,annotation], [[feadim, shdim * 2],shdim*2], True,
                                        scope="share_gate")
            annotation = annotation * share_gate
            annotation = nn.dropout(annotation, keep_prob=keep_prob)

            # compute initial state for decoder
            # first state of backward encoder
            # batch * shdim
            final_state = T.concatenate([annotation[0,:,annotation.shape[-1]/2 :],
                                         domain_annotation[0, :, annotation.shape[-1]/2:]],-1)
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim * 2, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)
                # keys for query
                mapped_keys = map_key(annotation, 2 * shdim, ahdim, "semantic")
                mapped_domain_keys = map_key(domain_annotation, 2 * shdim, ahdim, "domain")

                _, _, cost, tgtdcost, tpred_tag, _  = decoder.forward(tgt_seq, target_inputs, tgt_mask, mapped_keys, src_mask,
                                                    annotation, initial_state, mapped_domain_keys, domain_annotation, tag_seq, keep_prob)

        lamb = theano.shared(numpy.asarray(option["lambda"], dtype), "lambda")
        # cwscost *= lamb
        final_cost = cost + dcost + tgtdcost - lamb * adv_scost

        tag_inputs = [src_seq, src_mask]
        tag_outputs = [dpred_tag,spred_tag]
        tag_predict = theano.function(tag_inputs, tag_outputs)
        self.tag_predict = tag_predict

        tgt_tag_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        tgt_tag_outputs = [tpred_tag]
        tgt_tag_predict = theano.function(tgt_tag_inputs, tgt_tag_outputs)
        self.tgt_tag_predict = tgt_tag_predict

        training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask, tag_seq]
        training_outputs = [cost, dcost, adv_scost, tgtdcost]

        self.cost_cla = scost
        self.inputs_cla = [src_seq, src_mask, tag_seq]
        self.outputs_cla = [scost]

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = T.ivector("prev_words")

            # disable dropout
            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            source_inputs = source_inputs + source_bias

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            with ops.variable_scope("Specific"):
                domain_alpha = domain_sensitive_attention(annotation, src_mask, shdim * 2, domaindim)
                domain_context = T.sum(annotation * domain_alpha[:,:,None], 0)
                dfeature = nn.feedforward(domain_context, [shdim * 2, feadim], True,
                                          activation=T.tanh, scope="feature1")


            share_alpha = domain_sensitive_attention(annotation, src_mask, shdim * 2, domaindim)
            share_context = T.sum(annotation * share_alpha[:,:,None], 0)
            sfeature = nn.feedforward(share_context, [shdim * 2, feadim], True,
                                          activation=T.tanh, scope="feature1")

            domain_gate = nn.feedforward([dfeature, annotation], [[feadim, shdim * 2], shdim * 2], True,
                                         scope="domain_gate")
            domain_annotation = annotation * domain_gate
            share_gate = nn.feedforward([sfeature, annotation], [[feadim, shdim * 2], shdim * 2], True,
                                        scope="share_gate")
            annotation = annotation * share_gate

            # decoder
            final_state = T.concatenate([annotation[0, :, annotation.shape[-1] / 2:],
                                         domain_annotation[0, :, annotation.shape[-1] / 2:]], -1)
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim * 2, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)
                mapped_keys = map_key(annotation, 2 * shdim, ahdim, "semantic")
                mapped_domain_keys = map_key(domain_annotation, 2 * shdim, ahdim, "domain")

            prev_inputs = nn.embedding_lookup(target_embedding, prev_words)
            # prev_inputs = prev_inputs + target_bias

            cond = T.neq(prev_words, 0)
            # zeros out embedding if y is 0, which indicates <s>
            prev_inputs = prev_inputs * cond[:, None]

            with ops.variable_scope(decoder_scope):
                mask = T.ones_like(prev_words, dtype=dtype)
                next_state, context, d_context = decoder.step(prev_inputs, mask, initial_state, mapped_keys, annotation, src_mask,
                                                   mapped_domain_keys,domain_annotation)
                if option["decoder"] == "GruSimple":
                    probs = decoder.prediction(prev_inputs, initial_state, context, d_context)
                elif option["decoder"] == "GruCond":
                    probs = decoder.prediction(prev_inputs, next_state, context, d_context)

        # encoding
        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [annotation, initial_state, mapped_keys, mapped_domain_keys, domain_annotation]
        encode = theano.function(encoding_inputs, encoding_outputs)

        if option["decoder"] == "GruSimple":
            prediction_inputs = [prev_words, initial_state, annotation,
                                 mapped_keys, src_mask]
            prediction_outputs = [probs, context, d_context]
            predict = theano.function(prediction_inputs, prediction_outputs)

            generation_inputs = [prev_words, initial_state, context]
            generation_outputs = next_state
            generate = theano.function(generation_inputs, generation_outputs)

            self.predict = predict
            self.generate = generate
        elif option["decoder"] == "GruCond":
            prediction_inputs = [prev_words, initial_state, annotation,
                                 mapped_keys, src_mask, mapped_domain_keys, domain_annotation]
            prediction_outputs = [probs, next_state]
            predict = theano.function(prediction_inputs, prediction_outputs)
            self.predict = predict

        self.cost = final_cost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        # self.align = align
        # self.sample = sample
        self.encode = encode
        # self.get_snt_cost = get_snt_cost
        self.option = option



# TODO: add batched decoding
def beamsearch(models, seq, mask=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None, suppress_unk=False):
    dtype = dtype or theano.config.floatX

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)

    # get vocabulary from the first model
    option = models[0].option
    vocab = option["vocabulary"][1][1]
    eosid = option["eosid"]
    bosid = option["bosid"]
    unk_sym = models[0].option["unk"]
    unk_id = option["vocabulary"][1][0][unk_sym]

    if maxlen is None:
        maxlen = seq.shape[0] * 3

    if minlen is None:
        minlen = seq.shape[0] / 2

    # encoding source
    if mask is None:
        mask = numpy.ones(seq.shape, dtype)

    outputs = [model.encode(seq, mask) for model in models]
    annotations = [item[0] for item in outputs]
    states = [item[1] for item in outputs]
    mapped_annots = [item[2] for item in outputs]
    mapped_domain_annots = [item[3] for item in outputs]
    domain_annotations = [item[4] for item in outputs]

    initial_beam = beam(beamsize)
    size = beamsize
    # bosid must be 0
    initial_beam.candidates = [[bosid]]
    initial_beam.scores = numpy.zeros([1], dtype)

    hypo_list = []
    beam_list = [initial_beam]
    done_predicate = lambda x: x[-1] == eosid

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidates = prev_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda cand: cand[-1], candidates), "int32")

        # compute context first, then compute word distribution
        batch_mask = numpy.repeat(mask, num, 1)
        batch_annots = map(numpy.repeat, annotations, [num] * num_models,
                           [1] * num_models)
        batch_mannots = map(numpy.repeat, mapped_annots, [num] * num_models,
                            [1] * num_models)
        batch_domain_annots = map(numpy.repeat, domain_annotations, [num] * num_models,
                            [1] * num_models)
        batch_domain_mannots = map(numpy.repeat, mapped_domain_annots, [num] * num_models,
                            [1] * num_models)

        # predict returns [probs, context, alpha]
        outputs = [model.predict(last_words, state, annot, mannot, batch_mask, mdannot, dannot)
                   for model, state, annot, mannot, mdannot, dannot in
                   zip(models, states, batch_annots,
                       batch_mannots, batch_domain_mannots, batch_domain_annots)]
        prob_dists = [item[0] for item in outputs]

        # search nbest given word distribution
        if arithmetic:
            logprobs = numpy.log(sum(prob_dists) / num_models)
        else:
            # geometric mean
            logprobs = sum(numpy.log(prob_dists)) / num_models

        if suppress_unk:
            logprobs[:, unk_id] = -numpy.inf

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf  # make sure eos won't be selected

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob  # make sure eos will be selected

        next_beam = beam(size)
        finished, remain_beam_indices = next_beam.prune(logprobs, done_predicate, prev_beam)

        hypo_list.extend(finished)  # completed translation
        size -= len(finished)

        if size == 0:  # reach k completed translation before maxlen
            break

        # generate next state
        candidates = next_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda t: t[-1], candidates), "int32")

        if option["decoder"] == "GruSimple":
            contexts = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model
            contexts = select_nbest(contexts, remain_beam_indices)

            states = [model.generate(last_words, state, context)
                      for model, state, context in zip(models, states, contexts)]
        elif option["decoder"] == "GruCond":
            states = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model

        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = numpy.array(hypo_list)[numpy.argsort(score_list)]
    score_list = numpy.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output


def batchsample(model, seq, mask, maxlen=None):
    sampler = model.sample

    vocabulary = model.option["vocabulary"]
    eosid = model.option["eosid"]
    vocab = vocabulary[1][1]

    if maxlen is None:
        maxlen = int(len(seq) * 1.5)

    words = sampler(seq, mask, maxlen)
    trans = words.astype("int32")

    samples = []

    for i in range(trans.shape[1]):
        example = trans[:, i]
        # remove eos symbol
        index = -1

        for i in range(len(example)):
            if example[i] == eosid:
                index = i
                break

        if index >= 0:
            example = example[:index]

        example = map(lambda x: vocab[x], example)

        samples.append(example)

    return samples


# used for analysis
def evaluate_model(model, xseq, xmask, yseq, ymask, alignment=None,
                   verbose=False):
    t = yseq.shape[0]
    batch = yseq.shape[1]

    vocab = model.option["vocabulary"][1][1]

    annotation, states, mapped_annot = model.encode(xseq, xmask)

    last_words = numpy.zeros([batch], "int32")
    costs = numpy.zeros([batch], "float32")
    indices = numpy.arange(batch, dtype="int32")

    for i in range(t):
        outputs = model.predict(last_words, states, annotation, mapped_annot,
                                xmask)
        # probs: batch * vocab
        # contexts: batch * hdim
        # alpha: batch * srclen
        probs, contexts, alpha = outputs

        if alignment is not None:
            # alignment tgt * src * batch
            contexts = numpy.sum(alignment[i][:, :, None] * annotation, 0)

        max_prob = probs.argmax(1)
        order = numpy.argsort(-probs)
        label = yseq[i]
        mask = ymask[i]

        if verbose:
            for i, (pred, gold, msk) in enumerate(zip(max_prob, label, mask)):
                if msk and pred != gold:
                    gold_order = None

                    for j in range(len(order[i])):
                        if order[i][j] == gold:
                            gold_order = j
                            break

                    ent = -numpy.sum(probs[i] * numpy.log(probs[i]))
                    pp = probs[i, pred]
                    gp = probs[i, gold]
                    pred = vocab[pred]
                    gold = vocab[gold]
                    print "%d: predication error, %s vs %s" % (i, pred, gold)
                    print "prob: %f vs %f, entropy: %f" % (pp, gp, ent)
                    print "gold is %d-th best" % (gold_order + 1)

        costs -= numpy.log(probs[indices, label]) * mask

        last_words = label
        states = model.generate(last_words, states, contexts)

    return costs
