class SCNEncoder(nn.Module):
    def __init__(self, cnn_feature_size, c3d_feature_size, i3d_feature_size, n_tags, hidden_size, global_tagger_hidden_size, specific_tagger_hidden_size, input_dropout_p=0.2, 
                 rnn_dropout_p=0.5, n_layers=1, bidirectional=False, rnn_cell='gru', device='gpu'):
        super(SCNEncoder, self).__init__()
        
        self.cnn_feature_size = cnn_feature_size
        self.c3d_feature_size = c3d_feature_size
        self.i3d_feature_size = i3d_feature_size
        self.n_tags = n_tags
        self.hidden_size = hidden_size
        self.global_tagger_hidden_size = global_tagger_hidden_size
        self.specific_tagger_hidden_size = specific_tagger_hidden_size
        self.device = device
        
        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
                
    def forward_fn(self, v_feats, s_feats, cnn_globals, v_globals, s_globals):
        batch_size, seq_len, feats_size = v_feats.size()

        h = Variable(torch.zeros(2*2, batch_size, self.hidden_size)).to(self.device)
        c = Variable(torch.zeros(2*2, batch_size, self.hidden_size)).to(self.device)
        
        v_globals = torch.cat((v_globals, cnn_globals), dim=1)
        
        return v_feats, s_feats, (h,c), s_globals, v_globals  #pool
        
    def forward(self, cnn_feats, c3d_feats, i3d_feats, eco_feats, eco_sem_feats, tsm_sem_feats, cnn_globals, cnn_sem_globals, tags_globals, res_eco_globals):
        batch_size = cnn_feats.size(0)

        # (batch_size x max_frames x feature_size) -> (batch_size*max_frames x feature_size)
        cnn_feats = cnn_feats.view(-1, self.cnn_feature_size)
        c3d_feats = c3d_feats.view(-1, self.c3d_feature_size)
        v_concat = torch.cat((cnn_feats, c3d_feats), dim=1)
        v_concat = v_concat.view(batch_size, -1, self.cnn_feature_size + self.c3d_feature_size)
        
        return self.forward_fn(v_concat, tsm_sem_feats, cnn_globals, res_eco_globals, tags_globals)
