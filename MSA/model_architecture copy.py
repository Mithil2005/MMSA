import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class MSAModel(nn.Module):
    def __init__(self, visual_dim, acoustic_dim, language_dim, hidden_dim, pred_hidden_dim, dropout_value, output_dim):
        super(MSAModel,self).__init__()
        self.d_v = visual_dim
        self.d_a = acoustic_dim
        self.d_l = language_dim
        self.Hid = hidden_dim
        self.P_h = pred_hidden_dim
        self.Drop = dropout_value
        self.Output = output_dim
        self.BERT = RobertaModel.from_pretrained("roberta-base")

        # Tokenizer for language input
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer.add_tokens(['<SP>'])

        # MLP blocks for audio, video, and linguistic data
        self.mlp_a = self.sLSTM_MLP(self.d_a, self.Hid)
        self.mlp_v = self.sLSTM_MLP(self.d_v, self.Hid)
        self.mlp_t = self.BERT_MLP(self.d_l, self.Hid)

        # Private and Shared Encoder for all modalities embedding from respective MLP
        self.E_m_p = self.Private_Encoder(self.Hid, self.Hid)
        self.E_m_c = self.Shared_Encoder(self.Hid, self.Hid)

        # Decoder for all modalities
        self.D_m = self.Decoder(self.Hid, self.Hid)

        # HGFN 
        self.M_m = self.MAN(self.Hid)
        self.M_m1_m2 = self.MLF(self.Hid)

        # Predictor
        self.P = self.Prediction(self.Hid, self.P_h, self.Output, self.Drop)

    def Private_Encoder(self, input_dim, hidden_dim):
        ''' Private Encoder Block '''
        return nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Sigmoid()
        )
    
    def Shared_Encoder(self, input_dim, hidden_dim):
        ''' Shared Encoder Block'''
        return nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Sigmoid()
        )
    
    def Decoder(self, input_dim, hidden_dim):
        ''' Decoder Block '''
        return nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Sigmoid()
        )

    def sLSTM_MLP(self, input_dim, hidden_dim):
        ''' MLP block for audio and video '''
        return nn.Sequential(
            nn.LSTM(input_dim,input_dim),
            nn.LayerNorm(input_dim),
            nn.LSTM(input_dim,input_dim),
            nn.Linear(input_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
    
    def BERT_MLP(self, input_dim, hidden_dim):
        ''' MLP block for language '''
        return nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def MAN(self, hidden_dim):
        ''' Attention Block '''
        return nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Sigmoid()
        )
    
    def MLF(self, hidden_dim):
        ''' Multilayer Fusion '''
        return nn.Sequential(
            nn.Linear(2*hidden_dim,64),
            nn.LeakyReLU(),
            nn.Linear(64,hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh()
        )
    
    def Prediction(self, hidden_dim, pred_hid_dim, output_dim, dropout_val):
        ''' Prediction Block '''
        return nn.Sequential(
            nn.LayerNorm(3*hidden_dim),
            nn.Dropout(dropout_val),
            nn.Linear(3*hidden_dim,pred_hid_dim),
            nn.Tanh(),
            nn.Linear(pred_hid_dim,output_dim),
            nn.Tanh(),
            nn.Linear(output_dim,output_dim)
        )
    
    def forward(self, data_v, data_a, words):
        # Tokenization for language input
        verbal_input = self.tokenizer(words, return_tensors='pt', padding=True, truncation=True, 
                                      add_special_tokens=True)
        verbal_input_ids = verbal_input['input_ids']
        verbal_attention_mask = verbal_input['attention_mask']

        # Passing through MLP blocks for audio and visual data
        O_a = self.mlp_a(data_a)
        O_v = self.mlp_v(data_v)

        # Passing through BERT model for language input
        F_l = self.BERT(input_ids=verbal_input_ids, attention_mask=verbal_attention_mask).last_hidden_state
        O_l = self.BERT_MLP(F_l)

        # Joint Representation for audio
        h_a_c = self.E_m_c(O_a)
        h_a_p = self.E_m_p(O_a)
        h_a = h_a_c + h_a_p

        # Joint Representation for visual
        h_v_c = self.E_m_c(O_v)
        h_v_p = self.E_m_p(O_v)
        h_v = h_v_c + h_v_p

        # Joint Representation for linguistics
        h_l_c = self.E_m_c(O_l)
        h_l_p = self.E_m_p(O_l)
        h_l = h_l_c + h_l_p

        # Prediction from decoders
        h_a_1 = self.D_m(h_a)
        h_v_1 = self.D_m(h_v)
        h_l_1 = self.D_m(h_l)

        # HGFN
        M_a = self.M_m(h_a)
        M_v = self.M_m(h_v)
        M_l = self.M_m(h_l)

        M_al = self.M_m1_m2(torch.cat([M_a, M_l], dim=-1))
        M_av = self.M_m1_m2(torch.cat([M_a, M_v], dim=-1))
        M_lv = self.M_m1_m2(torch.cat([M_l, M_v], dim=-1))

        M_alav = self.M_m1_m2(torch.cat([M_al, M_av], dim=-1))
        M_alv = self.M_m1_m2(torch.cat([M_al, M_v], dim=-1))
        M_allv = self.M_m1_m2(torch.cat([M_al, M_lv], dim=-1))
        M_avl = self.M_m1_m2(torch.cat([M_av, M_l], dim=-1))
        M_lva = self.M_m1_m2(torch.cat([M_lv, M_a], dim=-1))
        M_avlv = self.M_m1_m2(torch.cat([M_av, M_lv], dim=-1))

        M_uni = M_a + M_l + M_v
        M_bi = M_al + M_av + M_lv
        M_tri = M_alav + M_alv + M_allv + M_avl + M_lva + M_avlv

        Fusion = torch.cat([M_uni, M_bi, M_tri], dim=-1)
        Pred = self.P(Fusion)

        H_m = (h_a, h_a_1, h_v, h_v_1, h_l, h_l_1)
        return H_m, Pred
