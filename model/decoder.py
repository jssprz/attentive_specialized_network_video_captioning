import torch
import torch.nn as nn


def get_init_weights(shape):
    W = torch.rand(shape)
    nn.init.xavier_normal_(W.data)
    return nn.Parameter(W)


class SCNAttnDecoder(nn.Module):
    def __init__(self, in_seq_length, out_seq_length, n_feats, n_tags, embedding_size, hidden_size, rnn_in_size, rnn_hidden_size, vocab, device, encoder_num_layers, encoder_bidirectional, 
                 pretrained_embedding=None, rnn_cell='gru', num_layers=1, dropout_p=0.5, beam_size=10, temperature=1.0, train_sample_max=False, test_sample_max=True, beam_search_logic='bfs'):
        super(SCNAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1

        # Components
        self.in_v_drop = nn.Dropout(dropout_p)
        self.in_s_drop = nn.Dropout(dropout_p)
        
        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)    
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.embedd_drop = nn.Dropout(dropout_p)

        self.Wa_i = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_f = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_o = get_init_weights((rnn_in_size, rnn_hidden_size))
        self.Wa_c = get_init_weights((rnn_in_size, rnn_hidden_size))

        self.Wb_i = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_f = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_o = get_init_weights((n_tags, rnn_hidden_size))
        self.Wb_c = get_init_weights((n_tags, rnn_hidden_size))

        self.Wc_i = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_f = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_o = get_init_weights((rnn_hidden_size, hidden_size))
        self.Wc_c = get_init_weights((rnn_hidden_size, hidden_size))

        self.Ua_i = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_f = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_o = get_init_weights((hidden_size, rnn_hidden_size))
        self.Ua_c = get_init_weights((hidden_size, rnn_hidden_size))

        self.Ub_i = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_f = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_o = get_init_weights((n_tags, rnn_hidden_size))
        self.Ub_c = get_init_weights((n_tags, rnn_hidden_size))

        self.Uc_i = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_f = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_o = get_init_weights((rnn_hidden_size, hidden_size))
        self.Uc_c = get_init_weights((rnn_hidden_size, hidden_size))

        self.b_i = Parameter(torch.zeros(hidden_size))
        self.b_f = Parameter(torch.zeros(hidden_size))
        self.b_o = Parameter(torch.zeros(hidden_size))
        self.b_c = Parameter(torch.zeros(hidden_size))

        self.out = nn.Linear(self.hidden_size * self.num_directions, self.output_size)

        self.__init_layers()


    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)


    def precompute_mats(self, v, s):
        # (batch_size x rnn_hidden_size)
        self.temp2_i = s @ self.Wb_i
        self.temp2_f = s @ self.Wb_f
        self.temp2_o = s @ self.Wb_o
        self.temp2_c = s @ self.Wb_c

        # (batch_size x rnn_hidden_size)
        self.temp5_i = s @ self.Ub_i
        self.temp5_f = s @ self.Ub_f
        self.temp5_o = s @ self.Ub_o
        self.temp5_c = s @ self.Ub_c


    def __compute_gate(self, activation, temp1, temp2, temp5, temp6, Wc, Uc, b):
        x = (temp1 * temp2) @ Wc
        h = (temp5 * temp6) @ Uc
        return activation(x + h + b)


    def step(self, rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs):
        # (batch_size x rnn_hidden_size)
        temp1_i = decoder_input @ self.Wa_i
        temp1_f = decoder_input @ self.Wa_f
        temp1_o = decoder_input @ self.Wa_o
        temp1_c = decoder_input @ self.Wa_c
            
        # (batch_size x rnn_hidden_size)
        temp6_i = rnn_h @ self.Ua_i
        temp6_f = rnn_h @ self.Ua_f
        temp6_o = rnn_h @ self.Ua_o
        temp6_c = rnn_h @ self.Ua_c

        # (batch_size x hidden_size)
        i = self.__compute_gate(torch.sigmoid, temp1_i, self.temp2_i, self.temp5_i, temp6_i, self.Wc_i, self.Uc_i, self.b_i)
        f = self.__compute_gate(torch.sigmoid, temp1_f, self.temp2_f, self.temp5_f, temp6_f, self.Wc_f, self.Uc_f, self.b_f)
        o = self.__compute_gate(torch.sigmoid, temp1_o, self.temp2_o, self.temp5_o, temp6_o, self.Wc_o, self.Uc_o, self.b_o)
        c = self.__compute_gate(torch.tanh, temp1_c, self.temp2_c, self.temp5_c, temp6_c, self.Wc_c, self.Uc_c, self.b_c)

        # (batch_size x hidden_size)
        rnn_c = f * rnn_c + i * c
        rnn_h = o * torch.tanh(rnn_c)

        return rnn_h, rnn_c 


    def forward_fn(self, video_pool, encoder_tags, encoder_hidden, encoder_outputs, captions, teacher_forcing_ratio=0.5):
        # Determine whether it is an inferred mode based on whether it is passed into caption
        infer = captions is None

        batch_size = encoder_outputs.size(0)

        # (batch_size x embedding_size)
        decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(self.device)

        if type(encoder_hidden) is tuple:
            # (encoder_n_layers * encoder_num_directions x batch_size x hidden_size) -> (encoder_n_layers x encoder_num_directions x batch_size x hidden_size) 
            rnn_h = encoder_hidden[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)
            rnn_c = encoder_hidden[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)

        rnn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        rnn_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        outputs = []

        s = self.in_s_drop(encoder_tags)
        v = self.in_v_drop(video_pool)

        self.precompute_mats(v, s)

        if infer:
            next_nodes = [([decoder_input], [], .0, rnn_h)]
            for step in range(self.out_seq_length):
                temp = []
                for tokens_seqs, word_logits_seqs, log_probs, rnn_h in next_nodes:
                    decoder_input = tokens_seqs[-1]
                    
                    if step:
                        # (batch_size x 1) -> (batch_size x embedding_size)
                        decoder_input = self.embedding(decoder_input).squeeze(1)
                        decoder_input = self.embedd_drop(decoder_input)
                        
                    rnn_h, rnn_c = self.step(rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs)

                    # compute word_logits
                    # (batch_size x output_size)
                    word_logits = self.out(rnn_h)

                    # compute word probs
                    word_log_probs = torch.log_softmax(word_logits, dim=1)

                    if self.test_sample_max:
                        # sample max probailities
                        # (batch_size x beam_size), (batch_size x beam_size)
                        sample_log_probs, sample_ids = word_log_probs.topk(k=self.beam_size, dim=1)
                    else:
                        # sample from distribution
                        word_probs = torch.exp(torch.div(word_log_probs, self.temperature))
                        # (batch_size x beam_size)
                        sample_ids = torch.multinomial(word_probs, self.beam_size).to(self.device)
                        sample_log_probs = word_log_probs.gather(dim=1, index=sample_ids)

                    for j in range(self.beam_size):
                        temp.append((tokens_seqs + [sample_ids[:,j].unsqueeze(1)],
                                     word_logits_seqs + [word_logits],
                                     log_probs + torch.mean(sample_log_probs[:,j]).item() / self.out_seq_length, 
                                     rnn_h.clone()))

                next_nodes = sorted(temp, reverse=True, key=lambda x: x[2])[:self.beam_size]

            best_seqs, best_word_logits_seq, max_avg_prob, _ = next_nodes[0]
            return torch.cat([o.unsqueeze(1) for o in best_word_logits_seq], dim=1).contiguous(), torch.cat([t for t in best_seqs[1:]], dim=1).contiguous()
        else:
            for seq_pos in range(self.out_seq_length):
                rnn_h, rnn_c = self.step(rnn_h, rnn_c, decoder_input, encoder_hidden, encoder_outputs)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio or seq_pos == 0 else False
                if use_teacher_forcing:
                    decoder_input = captions[:, seq_pos]  # use the correct words, (batch_size x 1)
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size x 1)
                    decoder_input = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size x 1)
                    decoder_input = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(decoder_input).squeeze(1)
                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)

            # (batch_size x out_seq_length x output_size), none
            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), None


    def forward(self, videos_encodes, captions, teacher_forcing_ratio=0.5):
        return self.forward_fn(video_pool=videos_encodes[3], 
                               encoder_tags=videos_encodes[2], 
                               encoder_hidden=videos_encodes[1], 
                               encoder_outputs=videos_encodes[0], 
                               captions=captions, 
                               teacher_forcing_ratio=teacher_forcing_ratio)

    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_ratio=0.0)


class VSCNAttnDecoder(nn.Module):
    def __init__(self, in_seq_length, out_seq_length, n_feats, n_tags, embedding_size, hidden_size, rnn_in_size, rnn_hidden_size, vocab, device, encoder_num_layers, encoder_bidirectional, 
                 pretrained_embedding=None, rnn_cell='gru', num_layers=1, dropout_p=0.5, beam_size=10, temperature=1.0, train_sample_max=False, test_sample_max=True, beam_search_logic='bfs'):
        super(VSCNAttnDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = len(vocab)
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.temperature = temperature
        self.train_sample_max = train_sample_max
        self.test_sample_max = test_sample_max
        self.beam_search_logic = beam_search_logic

        self.num_layers = num_layers
        self.encoder_num_layers = encoder_num_layers
        self.encoder_num_directions = 2 if encoder_bidirectional else 1
        self.num_directions = 1  # beause this decoder is not bidirectional

        # Components
        self.in_v_drop = nn.Dropout(dropout_p)
        self.in_s_drop = nn.Dropout(dropout_p)

        if pretrained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)    
        else:
            self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.embedd_drop = nn.Dropout(dropout_p)
        
        self.semantic_layer = SCNAttnDecoder(in_seq_length, out_seq_length, n_feats, n_tags, embedding_size, hidden_size,
                                             rnn_in_size, rnn_hidden_size, vocab, device, encoder_num_layers, 
                                             encoder_bidirectional, pretrained_embedding, rnn_cell, num_layers, dropout_p, 
                                             beam_size, temperature, train_sample_max, test_sample_max, beam_search_logic)

        # lo unico que hice fue cambiar el n_feats y el n_tags
        self.visual_layer = SCNAttnDecoder(in_seq_length, out_seq_length, n_tags, n_feats, embedding_size, hidden_size,
                                             rnn_in_size, rnn_hidden_size, vocab, device, encoder_num_layers, 
                                             encoder_bidirectional, pretrained_embedding, rnn_cell, num_layers, dropout_p, 
                                             beam_size, temperature, train_sample_max, test_sample_max, beam_search_logic)

        self.attn1 = Attention(self.in_seq_length, self.embedding_size, self.hidden_size, self.num_layers, self.num_directions, mode='soft')
        self.attn2 = Attention(self.in_seq_length, self.embedding_size, self.hidden_size, self.num_layers, self.num_directions, mode='soft')

        self.v_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.s_fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.merge1 = nn.Linear(self.hidden_size + 6144, self.hidden_size)
        self.merge2 = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.__init_layers()


    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)


    def __dropout(self, x, keep_prob, mask_for):
        if not self.training or keep_prob >= 1.:
            return x

        if mask_for in self.dropM:
            mask = self.dropM[mask_for]
        else:
            mask = x.new_empty(x.size(), requires_grad=False).bernoulli_(keep_prob)
            self.dropM[mask_for] = mask

        return x.masked_fill(mask==0, 0) * (1.0 / keep_prob)


    def __adaptive_attn(self, v, s, v_attn, rnn_h, semantic_h, visual_h):
        rnn_h = self.__dropout(rnn_h, .8, 'rnn_h')
        v_attn = self.__dropout(v_attn, .5, 'v_attn')
        beta = torch.sigmoid(self.merge1(torch.cat((rnn_h, v_attn), dim=1)))

        visual_h = self.__dropout(visual_h, .8, 'visual_h')
        semantic_h = self.__dropout(semantic_h, .8, 'semantic_h')

        v_h = (torch.relu(self.v_fc(visual_h)) * semantic_h) + (beta * visual_h)
        s_h = (torch.relu(self.s_fc(semantic_h)) * visual_h) + ((1-beta) * semantic_h)

        h = torch.cat((v_h, s_h), dim=1)
        return torch.relu(self.merge2(h))


    def forward_fn(self, video_pool, encoder_tags, encoder_hidden, encoder_outputs, captions, teacher_forcing_ratio=0.5):
        # Determine whether it is an inferred mode based on whether it is passed into caption
        self.training = captions is not None

        batch_size = encoder_outputs.size(0)

        # (batch_size x embedding_size)
        decoder_input = Variable(torch.Tensor(batch_size, self.embedding_size).fill_(0)).to(self.device)

        if type(encoder_hidden) is tuple:
            # (encoder_n_layers * encoder_num_directions x batch_size x hidden_size) -> (encoder_n_layers x encoder_num_directions x batch_size x hidden_size) 
            rnn_h = encoder_hidden[0].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)
            rnn_c = encoder_hidden[1].view(self.encoder_num_layers, self.encoder_num_directions, batch_size, self.hidden_size)

        semantic_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        semantic_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        visual_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)
        visual_c = Variable(torch.cat([rnn_c[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        rnn_h = Variable(torch.cat([rnn_h[-i,0,:,:] for i in range(self.num_layers, 0, -1)], dim=0)).to(self.device)

        outputs = []

        s = self.in_s_drop(encoder_tags)
        v = self.in_v_drop(video_pool)

        self.semantic_layer.precompute_mats(v, s)
        self.visual_layer.precompute_mats(s, v)

        self.dropM = {}

        if not self.training:
            words = []
            for step in range(self.out_seq_length):
                visual_h, visual_c = self.visual_layer.step(visual_h, visual_c, decoder_input, encoder_hidden, encoder_outputs)
                semantic_h, semantic_c = self.semantic_layer.step(semantic_h, semantic_c, decoder_input, encoder_hidden, encoder_outputs)
                visual_attn1 = self.attn1(encoder_outputs, visual_h)
                visual_attn2 = self.attn2(encoder_outputs, semantic_h)
                rnn_h = self.__adaptive_attn(v, s, (visual_attn1 + visual_attn2)/2, rnn_h, semantic_h, visual_h)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                # compute word probs
                if self.test_sample_max:
                    # sample max probailities
                    # (batch_size x 1), (batch_size x 1)
                    word_id = word_logits.max(dim=1)[1]
                else:
                    # sample from distribution
                    # (batch_size x 1)
                    word_id = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(word_id).squeeze(1)
                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)
                words.append(word_id)

            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), torch.cat([w.unsqueeze(1) for w in words], dim=1).contiguous()
        else:
            for seq_pos in range(self.out_seq_length):
                visual_h, visual_c = self.visual_layer.step(visual_h, visual_c, decoder_input, encoder_hidden, encoder_outputs)
                semantic_h, semantic_c = self.semantic_layer.step(semantic_h, semantic_c, decoder_input, encoder_hidden, encoder_outputs)
                visual_attn1 = self.attn1(encoder_outputs, visual_h)
                visual_attn2 = self.attn2(encoder_outputs, semantic_h)
                rnn_h = self.__adaptive_attn(v, s, (visual_attn1 + visual_attn2)/2, rnn_h, semantic_h, visual_h)

                # compute word_logits
                # (batch_size x output_size)
                word_logits = self.out(rnn_h)

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio or seq_pos == 0 else False
                if use_teacher_forcing:
                    decoder_input = captions[:, seq_pos]  # use the correct words, (batch_size x 1)
                elif self.train_sample_max:
                    # select the words ids with the max probability,
                    # (batch_size x 1)
                    decoder_input = word_logits.max(1)[1]
                else:
                    # sample words from probability distribution
                    # (batch_size x 1)
                    decoder_input = torch.multinomial(torch.softmax(word_logits, dim=1), 1)

                # (batch_size x 1) -> (batch_size x embedding_size)
                decoder_input = self.embedding(decoder_input).squeeze(1)
                decoder_input = self.embedd_drop(decoder_input)

                outputs.append(word_logits)

            # (batch_size x out_seq_length x output_size), none
            return torch.cat([o.unsqueeze(1) for o in outputs], dim=1).contiguous(), None


    def forward(self, videos_encodes, captions, teacher_forcing_ratio=0.5):
        return self.forward_fn(video_pool=videos_encodes[4], 
                               encoder_tags=videos_encodes[3], 
                               encoder_hidden=videos_encodes[2], 
                               encoder_outputs=videos_encodes[0], 
                               captions=captions, 
                               teacher_forcing_ratio=teacher_forcing_ratio)  


    def sample(self, videos_encodes):
        return self.forward(videos_encodes, None, teacher_forcing_ratio=0.0)