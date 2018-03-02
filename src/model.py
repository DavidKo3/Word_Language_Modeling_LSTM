import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class NestedLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, ntoken, ninp, nhid, dropout, nlayers):
        super(NestedLSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        """
        # lstm weights
        self.weight_fxh = nn.Linear(nhid+ninp, nhid)
        self.weight_ixh = nn.Linear(nhid+ninp, nhid)
        self.weight_cxh = nn.Linear(nhid+ninp, nhid)
        self.weight_oxh = nn.Linear(nhid+ninp, nhid)
         
        """
        self.weight_fh = nn.Linear(nhid, nhid)
        self.weight_ih = nn.Linear(nhid, nhid)
        self.weight_ch = nn.Linear(nhid, nhid)
        self.weight_oh = nn.Linear(nhid, nhid)
        
        self.weight_fx = nn.Linear(ninp, nhid)
        self.weight_ix = nn.Linear(ninp, nhid)
        self.weight_cx = nn.Linear(ninp, nhid)
        self.weight_ox = nn.Linear(ninp, nhid)
        
        self.weight_tild_fh = nn.Linear(nhid, nhid)
        self.weight_tild_ih = nn.Linear(nhid, nhid)
        self.weight_tild_ch = nn.Linear(nhid, nhid)
        self.weight_tild_oh = nn.Linear(nhid, nhid)
        
        self.weight_tild_fx = nn.Linear(ninp, nhid)
        self.weight_tild_ix = nn.Linear(ninp, nhid)
        self.weight_tild_cx = nn.Linear(ninp, nhid)
        self.weight_tild_ox = nn.Linear(ninp, nhid)
        
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        #self.rnn_type = rnn_type
        self.nhid = nhid
        self.ninp = ninp
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, input, hidden):
    def forward(self, input, h_0, c_0, tild_c_0):
        # eoncode the input characters
        emb = self.drop(self.encoder(input))
        #emb = self.encoder(input)
        if(emb.size(0) != h_0.size(0)):
           # h_0 ,c_0 = h_0[:emb.size(0)], c_0[:emb.size(0)]
            h_0 ,c_0 ,tild_c_0 = h_0[:emb.size(0)], c_0[:emb.size(0)], tild_c_0[:emb.size(0)]
        """
        buff_h0= Variable(torch.zeros(emb.size(0), emb.size(1), emb.size(2)))

        if(emb.size(0) != h_0.size(0)): 
            for i in range(emb.size(0)):
                buff_h0[i] = h_0[:]
        
        input_combined = torch.cat((emb, buff_h0), 2)   
       
        f_g = F.sigmoid(self.weight_fxh(input_combined)) # [35, 20, 200]
        i_g = F.sigmoid(self.weight_ixh(input_combined))  # [35, 20, 200]
        o_g = F.sigmoid(self.weight_oxh(input_combined)) # [35, 20, 200]
        c_int = F.tanh(self.weight_cxh(input_combined)) # [35, 20, 200]
        
        """
        f_g = F.sigmoid(self.weight_fx(emb) + self.weight_fh(h_0)) # [35, 20, 200]
        i_g = F.sigmoid(self.weight_ix(emb) + self.weight_ih(h_0)) # [35, 20, 200]
        o_g = F.sigmoid(self.weight_ox(emb) + self.weight_oh(h_0)) # [35, 20, 200]
        # intermediate cel state
        
        #c_int = F.tanh(self.weight_cx(emb) + self.weight_ch(h_0)) # [35, 20, 200]
        c_int = self.weight_cx(emb) + self.weight_ch(h_0) # [35, 20, 200]
          
        
        tild_h =  f_g*c_0 
        tild_x = i_g*c_int
        #c_x = tild_h + tild_x# [35, 20, 200]
        # c_x = f_g*c_0 + i_g*c_int# [35, 20, 200]
        #h_x = o_g*(F.tanh(c_x)) # [35, 20, 200]
        
        
        #c_x = f_g*c_0 + i_g*c_int # [35, 20, 200]
        #h_x = o_g*(F.sigmoid(c_int)) # [35, 20, 200]
        
        tild_f_g = F.sigmoid(self.weight_tild_fx(tild_x) + self.weight_tild_fh(tild_h)) # [35, 20, 200]
        tild_i_g = F.sigmoid(self.weight_tild_ix(tild_x) + self.weight_tild_ih(tild_h)) # [35, 20, 200]
        tild_o_g = F.sigmoid(self.weight_tild_ox(tild_x) + self.weight_tild_oh(tild_h)) # [35, 20, 200]
        # intermediate cel state
        
        tild_c_int = F.tanh(self.weight_tild_cx(tild_x) + self.weight_tild_ch(tild_h)) # [35, 20, 200]
        tild_c_x = tild_f_g*tild_c_0 + tild_i_g*tild_c_int
        
        tild_h_x = tild_o_g*(F.tanh(tild_c_x)) # [35, 20, 200]
        c_x = tild_h_x
        h_x = o_g*(F.tanh(c_x)) # [35, 20, 200]
        h_x = self.drop(h_x) # [35, 20, 200]
        decoded = self.decoder(h_x.view(h_x.size(0)*h_x.size(1), h_x.size(2)))
        #print("decoded size ", decoded.size())
        return decoded.view(h_x.size(0), h_x.size(1), decoded.size(1)), h_x, c_x ,tild_c_x

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
    
        h_0 = Variable(weight.new(1, bsz, self.nhid).zero_())
        c_0 = Variable(weight.new(1, bsz, self.nhid).zero_())
        tild_c_0 = Variable(weight.new(1, bsz, self.nhid).zero_())
        #h_0 = Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        #c_0 = Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        return (h_0, c_0, tild_c_0)
        #return (h_0, c_0)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input)) # [35, 20, 200]
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) # [35, 20, 200]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) # [700, 200]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden # [35, 20, 33278] , 2x20x200

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())