import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class NestedLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(NestedLSTM, self).__init__()
       #  self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
       
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
        """
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        #self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, input, hidden):
    def forward(self, input, h_0, c_0):
        # eoncode the input characters
        emb = self.encoder(input)
        #print("inputs size :", input.size())
        #print("enn size")
        #print(emb.size()) # [35x20x200]
        #emb = self.drop(self.encoder(input))
        if(emb.size(0) != h_0.size(0)):
            h_0 ,c_0 = h_0[:emb.size(0)], c_0[:emb.size(0)]
        """
        print("input type : ", input.size())
        print("emb type : ", emb.size())
        print("h_0 type : ", h_0.size())
        """
        buff_h0= Variable(torch.zeros(emb.size(0), emb.size(1), emb.size(2)))
       # buff_h0= h_0.clone()
        if(emb.size(0) != h_0.size(0)): 
            for i in range(emb.size(0)):
                buff_h0[i] = h_0[:]
        #var_ho = Variable(buff_h0)
        input_combined = torch.cat((emb, buff_h0), 2)   
        #print("input combined ", input_combined.size())
        f_g = F.sigmoid(self.weight_fxh(input_combined)) # [35, 20, 200]
        i_g = F.sigmoid(self.weight_ixh(input_combined))  # [35, 20, 200]
        o_g = F.sigmoid(self.weight_oxh(input_combined)) # [35, 20, 200]

    
        #print("weight_fxh.grad.data", self.weight_fxh.grad.data)
        #f_g = F.sigmoid(self.weight_fx(emb) + self.weight_fh(h_0)) # [35, 20, 200]
        #i_g = F.sigmoid(self.weight_ix(emb) + self.weight_ih(h_0)) # [35, 20, 200]
        #o_g = F.sigmoid(self.weight_ox(emb) + self.weight_oh(h_0)) # [35, 20, 200]
        #print("emb size :", emb.size()) # [35, 20, 200]
        #print("weight_fx size :", self.weight_fx) # [200 -> 200]
        #print("weight_fh size :", self.weight_fh) # [200 -> 200]]
        # intermediate cel state
        c_int = F.sigmoid(self.weight_cxh(input_combined)) # [35, 20, 200]
        #c_int = F.tanh(self.weight_cx(emb) + self.weight_ch(h_0)) # [35, 20, 200]
          
       
        c_x = f_g*c_0 + i_g*c_int# [35, 20, 200]
        #h_x = o_g*c_0 # [35, 20, 200]
        h_x = o_g*(F.sigmoid(c_x)) # [35, 20, 200]
        
        #c_x = f_g*c_0 + i_g*c_int # [35, 20, 200]
        #h_x = o_g*(F.sigmoid(c_int)) # [35, 20, 200]
   
        decoded = self.decoder(h_x.view(h_x.size(0)*h_x.size(1), h_x.size(2)))
        #print("decoded size ", decoded.size())
        return decoded.view(h_x.size(0), h_x.size(1), decoded.size(1)), h_x, c_x

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
    
        #h_0 = Variable(weight.new(1, bsz, self.nhid+self.nhip).zero_())
        #c_0 = Variable(weight.new(1, bsz, self.nhid+self.nhip).zero_())
        h_0 = Variable(weight.new(1, bsz, self.nhid).zero_())
        c_0 = Variable(weight.new(1, bsz, self.nhid).zero_())
        
        return (h_0, c_0)
    """
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    """

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
        #print("5. decoded size :", decoded.size()) # [700, 33278]
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden # [35, 20, 33278] , 2x20x200

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())