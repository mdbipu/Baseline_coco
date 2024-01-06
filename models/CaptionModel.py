from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import reduce
import torch
import torch.nn as nn
import sys
sys.path.insert(3,'/ghome/mbipu/Baseline_coco/misc')
import util as utils    #import misc.utils as utils

class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)

    def beam_search(self, beam_size, init_state, init_logprobs, *args, **kwargs):
        """
        inputs are:
        init_state: ([num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size])
        init_logprobs: [beam_size, vocab_size+1]
        tmp_fc_feats: [beam_size, rnn_size]
        tmp_att_feats_mem: [beam_size, 36, rnn_size]
        tmp_p_att_feats_mem: [beam_size, 36, hidden_size]

        this function returns a list of beam_size dictionaries with the following keys:
        'seq': [16], 'logps': [16], 'unaug_p': value, 'p': value
        """

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf



        # does one step of classical beam search
        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):




            ys,ix = torch.sort(logprobsf,1,True)   # [beam_size, vocab_size+1],  [beam_size, vocab_size+1]
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1




            for c in range(cols):   # for each column (word, essentially)
                for q in range(rows):   # for each beam expansion
                    local_logprob = ys[q,c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            new_state = [_.clone() for _ in state]   # [[2*3*1000],[2*3*1000]]




            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                




            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step       
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
           



            state = new_state

            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates


        # Start diverse_beam_search
        group_size = 1
        diversity_lambda = 0.5
        decoding_constraint = 1
        max_ppl = 0
        bdash = beam_size // group_size    # beam per group


        # tables initializations
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]   # [[16, bdash]]  all zeros
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]   # [[16, bdash]]   all zeros
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]   # [[bdash]]   all zeros
        done_beams_table = [[] for _ in range(group_size)]   # list of list
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]   # [[[num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size]]]

        # init_state: ([num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size])
        # torch.stack(init_state): [2, num_layers, beam_size, rnn_size]
        # torch.stack(init_state).chunk(group_size, 2): ([2, num_layers, beam_size, rnn_size],)
        # torch.unbind(torch.stack(init_state).chunk(group_size, 2)[0]): ([num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size])
        # list(torch.unbind(torch.stack(init_state).chunk(group_size, 2)[0])): [[num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size]]
        # state_table: [[[num_layers, beam_size, rnn_size],[num_layers, beam_size, rnn_size]]]

        logprobs_table = list(init_logprobs.chunk(group_size, 0))   #  [[beam_size, vocab_size+1]]

        # Chunk elements in the args
        args = list(args)   # [[beam_size, rnn_size], [beam_size, 36, rnn_size], [beam_size, 36, hidden_size]]
        args = [_.chunk(group_size) if _ is not None else None for _ in args]   # [([beam_size, rnn_size],), ([beam_size, 36, rnn_size],), ([beam_size, 36, hidden_size],)]
        args = [[args[i][j] if args[i] is not None else None for i in range(len(args))] for j in range(group_size)]   # [[[beam_size, rnn_size], [beam_size, 36, rnn_size], [beam_size, 36, hidden_size]]]


        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobsf = logprobs_table[divm].data.float()   # [beam_size, vocab_size+1]
                    # suppress previous word
                    if decoding_constraint and t-divm > 0:
                        logprobsf.scatter_(1, beam_seq_table[divm][t-divm-1].unsqueeze(1).cuda(), float('-inf'))
                    # suppress UNK tokens in the decoding
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000



                    unaug_logprobsf = add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash)   # [beam_size, vocab_size+1] ; since divm = 0, then unaug_logprobsf is a copy of logprobsf
                    # infer new beams
                    beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm], candidates_divm = beam_step(logprobsf, unaug_logprobsf, bdash, t-divm, beam_seq_table[divm], beam_seq_logprobs_table[divm], beam_logprobs_sum_table[divm], state_table[divm])



                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().item(),
                                'p': beam_logprobs_sum_table[divm][vix].item()
                            }

                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)

                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000



                    # move the current group one step forward in time
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it.cuda(), *(args[divm] + [state_table[divm]]))


        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)

        return done_beams
