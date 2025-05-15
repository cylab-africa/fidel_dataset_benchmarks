import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision import models
from types import SimpleNamespace
from .vit import ViT, CNNViT

class TrOCRMyDecoder(torch.nn.Module):

    def __init__(self, input_dim, dec_num_layers, dec_num_heads,
                    d_model, d_ff, target_vocab_size, eos_token, sos_token,
                    pad_token, enc_dropout, dec_dropout, max_seq_length=512, pre_train=False):

        super(TrOCRMyDecoder, self).__init__()

        # self.encoder    = CNN_LSTM_Encoder(input_dim, 256, enc_dropout)
        # # Load TrOCR model (encoder only)
        # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        # self.encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").encoder
        # self.encoder = ViT(image_size = (256, 1024), num_classes= None, mlp_dim=1024, patch_size=32, dim=512, depth= 4, heads = 4 )
        # self.encoder = CNNViT(
        #     image_size=(256, 1024),    # same as before
        #     patch_size=32,             # same as before
        #     num_classes=None,          # same as before
        #     dim=512,                   # transformer embedding dim
        #     depth=4,                   # transformer depth
        #     heads=4,                   # transformer heads
        #     mlp_dim=1024,              # transformer MLP hidden dim
        #     pool='cls',                # or 'mean'
        #     channels=3,                # # of input image channels
        #     dim_head=64,               # per‑head dimension
        #     dropout=0.,                # transformer dropout
        #     emb_dropout=0.,            # embedding dropout
        #     cnn_hidden_channels=64,    # # channels produced by each conv layer
        #     cnn_layers=3               # # of conv→BN→ReLU layers
        # )
        self.encoder = CNNEncoder(d_model,input_channels=1)
        # Extract `d_model` from TrOCR encoder
        
        enc_d_model = self.encoder.config.hidden_size
        self.proj       = torch.nn.Linear(enc_d_model, d_model)

        self.layernorm  = torch.nn.LayerNorm(d_model)
        self.pre_train = pre_train
        self.decoder    = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff,
                dec_dropout, target_vocab_size, max_seq_length, eos_token, sos_token, pad_token, pre_train=pre_train)

        # You can experiment with different weight initialization schemes or no initialization here
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target, target_lengths):
        padded_input = padded_input.squeeze(1)
        if not self.pre_train:
        
            encoder_output = self.encoder(padded_input)["last_hidden_state"]  # Shape: (batch_size, num_patches+1, d_model)

            encoder_lens = torch.tensor([encoder_output.shape[1]] * encoder_output.shape[0] )


            encoder_output = self.proj(encoder_output)

            encoder_output = self.layernorm(encoder_output)
        else:
            encoder_output = None
            encoder_lens = None
        # passing Encoder output and Attention masks through Decoder
        output, attention_weights = self.decoder(padded_target, encoder_output, encoder_lens)

        return output, attention_weights

    def recognize(self, inp, inp_len):
        """ sequence-to-sequence greedy search -- decoding one utterence at a time """


        inp = inp.squeeze(1)

        with torch.inference_mode():
          encoder_output = self.encoder(inp)["last_hidden_state"]  # Shape: (batch_size, num_patches+1, d_model)

        encoder_lens = torch.tensor([encoder_output.shape[1]] * encoder_output.shape[0] )
        encoder_output                = self.proj(encoder_output)
        # out                            = self.decoder.recognize_beam_search(encoder_output, encoder_lens,beam_width=10)
        out                            = self.decoder.recognize_greedy_search(encoder_output, encoder_lens)


        return out




# class TrOCRMyDecoder(torch.nn.Module):

#     def __init__(self, input_dim, dec_num_layers, dec_num_heads,
#                     d_model, d_ff, target_vocab_size, eos_token, sos_token,
#                     pad_token, enc_dropout, dec_dropout, max_seq_length=512):

#         super(TrOCRMyDecoder, self).__init__()

#         # self.encoder    = CNN_LSTM_Encoder(input_dim, 256, enc_dropout)
#         # # Load TrOCR model (encoder only)
#         # self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#         # self.encoder = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").encoder
#         # self.encoder = ViT(image_size = (256, 1024), num_classes= None, mlp_dim=1024, patch_size=32, dim=512, depth= 4, heads = 4 )
#         # self.encoder = CNNViT(
#         #     image_size=(256, 1024),    # same as before
#         #     patch_size=32,             # same as before
#         #     num_classes=None,          # same as before
#         #     dim=512,                   # transformer embedding dim
#         #     depth=4,                   # transformer depth
#         #     heads=4,                   # transformer heads
#         #     mlp_dim=1024,              # transformer MLP hidden dim
#         #     pool='cls',                # or 'mean'
#         #     channels=3,                # # of input image channels
#         #     dim_head=64,               # per‑head dimension
#         #     dropout=0.,                # transformer dropout
#         #     emb_dropout=0.,            # embedding dropout
#         #     cnn_hidden_channels=64,    # # channels produced by each conv layer
#         #     cnn_layers=3               # # of conv→BN→ReLU layers
#         # )
#         self.encoder = CNNEncoder(d_model,input_channels=1)
#         # Extract `d_model` from TrOCR encoder
        
#         enc_d_model = self.encoder.config.hidden_size
#         self.proj       = torch.nn.Linear(enc_d_model, d_model)

#         self.layernorm  = torch.nn.LayerNorm(d_model)

#         self.decoder    = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff,
#                 dec_dropout, target_vocab_size, max_seq_length, eos_token, sos_token, pad_token)

#         # You can experiment with different weight initialization schemes or no initialization here
#         for p in self.parameters():
#             if p.dim() > 1:
#                 torch.nn.init.xavier_uniform_(p)

#     def forward(self, padded_input, input_lengths, padded_target, target_lengths):
#         padded_input = padded_input.squeeze(1)

#         encoder_output = self.encoder(padded_input)["last_hidden_state"]  # Shape: (batch_size, num_patches+1, d_model)

#         encoder_lens = torch.tensor([encoder_output.shape[1]] * encoder_output.shape[0] )


#         encoder_output = self.proj(encoder_output)

#         encoder_output = self.layernorm(encoder_output)
#         # passing Encoder output and Attention masks through Decoder
#         output, attention_weights = self.decoder(padded_target, encoder_output, encoder_lens)

#         return output, attention_weights

#     def recognize(self, inp, inp_len):
#         """ sequence-to-sequence greedy search -- decoding one utterence at a time """


#         inp = inp.squeeze(1)

#         with torch.inference_mode():
#           encoder_output = self.encoder(inp)["last_hidden_state"]  # Shape: (batch_size, num_patches+1, d_model)

#         encoder_lens = torch.tensor([encoder_output.shape[1]] * encoder_output.shape[0] )
#         encoder_output                = self.proj(encoder_output)
#         # out                            = self.decoder.recognize_beam_search(encoder_output, encoder_lens,beam_width=10)
#         out                            = self.decoder.recognize_greedy_search(encoder_output, encoder_lens)


#         return out






class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout,
            target_vocab_size, max_seq_length, eos_token, sos_token, pad_token, pre_train=False):
        super().__init__()

        self.EOS_TOKEN      = eos_token
        self.SOS_TOKEN      = sos_token
        self.PAD_TOKEN      = pad_token

        self.max_seq_length = max_seq_length
        self.num_layers     = num_layers
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.pre_train = pre_train

        self.target_embedding       = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding    = PositionalEncoding(d_model, max_len=max_seq_length)
        self.final_linear           = nn.Linear(d_model, target_vocab_size)
        self.dropout                = nn.Dropout(p=dropout)


    def forward(self, padded_targets, enc_output=None, enc_input_lengths=None):

        pad_mask = create_mask_1(padded_targets,  pad_idx= self.PAD_TOKEN)
        look_ahead_mask = create_mask_2(padded_targets,  pad_idx= self.PAD_TOKEN) ##
        if not self.pre_train:
            dec_enc_attn_mask = create_mask_3(enc_output, enc_input_lengths,padded_targets.shape[1])
        else:
            dec_enc_attn_mask = None

        target_embedded = self.target_embedding(padded_targets)
        target_embedded = self.positional_encoding(target_embedded)

        target_embedded = self.dropout(target_embedded)

        running_attn = {}


        for i, layer in enumerate(self.dec_layers):
            target_embedded, attn1, attn = layer(target_embedded, enc_output, enc_input_lengths, dec_enc_attn_mask, pad_mask, look_ahead_mask,pre_train= self.pre_train)
            running_attn['layer{}_dec_self'.format(i + 1)] = attn1
            running_attn['layer{}_dec_self'.format(i + 1)] =  attn
        out = self.final_linear(target_embedded)


        return out, running_attn

    def recognize_greedy_search(self, enc_outputs, enc_input_lengths):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network

            @NOTE: You do not need to make changes to this method.
        '''

        batch_size = enc_outputs.size(0)

        # start with the <SOS> token for each sequence in the batch
        target_seq = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long).to(enc_outputs.device)

        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_outputs.device)

        for _ in range(self.max_seq_length):

            # preparing attention masks
            # filled with ones becaues we want to attend to all the elements in the sequence
            pad_mask = torch.ones_like(target_seq).float().unsqueeze(-1)  # (batch_size x i x 1)
            slf_attn_mask_subseq = create_mask_2(target_seq)

            x = self.positional_encoding(self.target_embedding(target_seq))

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](
                    x, enc_outputs, enc_input_lengths, None, pad_mask, slf_attn_mask_subseq)

            seq_out = self.final_linear(x[:, -1])
            logits = torch.nn.functional.log_softmax(seq_out, dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            next_token = logits.argmax(dim=-1).unsqueeze(1)

            # appending the token to the sequence
            target_seq = torch.cat([target_seq, next_token], dim=-1)

            # checking if <EOS> token is generated
            eos_mask = next_token.squeeze(-1) == self.EOS_TOKEN
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            finished |= eos_mask

            # end if all sequences have generated the EOS token
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,
            (0, self.max_seq_length - max_length), value=self.PAD_TOKEN)

        return target_seq

    def recognize_beam_search(self, enc_outputs, enc_input_lengths, beam_width=5):
        '''
        Beam search decoding for seq2seq model with transformer decoder.
        Args:
            enc_outputs: Tensor of shape (batch_size, src_len, d_model)
            enc_input_lengths: lengths of each encoder input sequence
            beam_width: number of beams
        Returns:
            Tensor of shape (batch_size, max_seq_length) with decoded token ids
        '''
        batch_size = enc_outputs.size(0)
        device = enc_outputs.device

        # Initialize beams: each beam holds (sequence, score, finished_flag)
        # For batch processing, we flatten beams into batch_size * beam_width
        # Start with SOS token
        sos = self.SOS_TOKEN
        eos = self.EOS_TOKEN
        pad = self.PAD_TOKEN
        max_len = self.max_seq_length

        # sequences: (batch, beam, seq_len)
        sequences = torch.full((batch_size, beam_width, 1), sos, dtype=torch.long, device=device)
        # scores: (batch, beam)
        scores = torch.zeros(batch_size, beam_width, device=device)
        # finished flags
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

        # Expand encoder outputs for beams
        enc_out = enc_outputs.unsqueeze(1).expand(-1, beam_width, -1, -1)
        enc_out = enc_out.contiguous().view(batch_size * beam_width, -1, enc_outputs.size(-1))
        enc_lens = enc_input_lengths.unsqueeze(1).expand(-1, beam_width).contiguous().view(-1)

        for t in range(1, max_len + 1):
            # Prepare current target sequences for decoder input
            curr_seq = sequences.view(batch_size * beam_width, -1)
            pad_mask = torch.ones_like(curr_seq).float().unsqueeze(-1)
            slf_attn_mask_subseq = create_mask_2(curr_seq)

            x = self.positional_encoding(self.target_embedding(curr_seq))
            for layer in self.dec_layers:
                x, _, _ = layer(x, enc_out, enc_lens, None, pad_mask, slf_attn_mask_subseq)

            # get logits for last time step: shape (batch*beam, vocab)
            logits = F.log_softmax(self.final_linear(x[:, -1]), dim=-1)

            # reshape to (batch, beam, vocab)
            logits = logits.view(batch_size, beam_width, -1)

            # accumulate scores: broadcast existing scores
            total_scores = scores.unsqueeze(-1) + logits  # (batch, beam, vocab)

            # flatten beams and vocab into candidates: (batch, beam*vocab)
            flat_scores = total_scores.view(batch_size, -1)

            # select top-k candidates
            topk_scores, topk_indices = torch.topk(flat_scores, beam_width, dim=-1)

            # compute new beam and token indices
            beam_indices = topk_indices // logits.size(-1)
            token_indices = topk_indices % logits.size(-1)

            # update sequences, scores, finished flags
            new_sequences = []
            new_finished = []
            for i in range(batch_size):
                seqs = []
                fins = []
                for b in range(beam_width):
                    prev_beam = beam_indices[i, b]
                    token = token_indices[i, b]
                    seq = torch.cat([sequences[i, prev_beam], token.view(1)], dim=0)
                    seqs.append(seq)
                    # mark finished if EOS generated or was already finished
                    fins.append(finished[i, prev_beam] | (token == eos))
                # pad sequences to same length
                seqs = [F.pad(s, (0, max_len - s.size(0)), value=pad) for s in seqs]
                new_sequences.append(torch.stack(seqs))
                new_finished.append(torch.tensor(fins, device=device))

            sequences = torch.stack(new_sequences)
            scores = topk_scores
            finished = torch.stack(new_finished)

            # stop early if all finished
            if finished.all():
                break

        # choose best beam (highest score) per batch
        best_beam = scores.argmax(dim=-1)
        # gather best sequences
        final_seqs = sequences[torch.arange(batch_size), best_beam]

        return final_seqs

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.mha1       = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.mha2       = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.ffn        = FeedForward(d_model, d_ff)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout1   = nn.Dropout(p=dropout)
        self.dropout2   = nn.Dropout(p=dropout)
        self.dropout3   = nn.Dropout(p=dropout)


    def forward(self, padded_targets, enc_output, enc_input_lengths, dec_enc_attn_mask, pad_mask, slf_attn_mask, pre_train= True):
        if not pre_train:
            output1, attn_weights1 = self.mha1(padded_targets, padded_targets, padded_targets, slf_attn_mask )
            output1 = self.dropout1(output1)
            output1 = self.layernorm1(output1 +  padded_targets)
            output2, attn_weights2 = self.mha2(output1, enc_output, enc_output, dec_enc_attn_mask )
            output2 = self.dropout2(output2)

            output2 = self.layernorm2(output2 + output1)
            ffn_out = self.ffn(output2)

            ffn_out = self.dropout3(ffn_out)
            output3 = self.layernorm3(ffn_out + output2)
            return output3, attn_weights1, attn_weights2

        else:
            output1, attn_weights1 = self.mha1(padded_targets, padded_targets, padded_targets, slf_attn_mask )
            output1 = self.dropout1(output1)
            output1 = self.layernorm1(output1 +  padded_targets)
            ffn_out = self.ffn(output1)
            ffn_out = self.dropout3(ffn_out)
            output3 = self.layernorm3(ffn_out + output1)
            return output3, attn_weights1, 0
        

        # return output3, attn_weights1, attn_weights2


    def recognize_greedy_search(self, enc_outputs, enc_input_lengths):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network

            @NOTE: You do not need to make changes to this method.
        '''

        batch_size = enc_outputs.size(0)
        target_seq = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long).to(enc_outputs.device)

        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_outputs.device)

        for _ in range(self.max_seq_length):

            pad_mask = torch.ones_like(target_seq).float().unsqueeze(-1)  # (batch_size x i x 1)
            slf_attn_mask_subseq = create_mask_2(target_seq)

            x = self.positional_encoding(self.target_embedding(target_seq))

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](
                    x, enc_outputs, enc_input_lengths, None, pad_mask, slf_attn_mask_subseq)

            seq_out = self.final_linear(x[:, -1])
            logits = torch.nn.functional.log_softmax(seq_out, dim=1)
            next_token = logits.argmax(dim=-1).unsqueeze(1)

            target_seq = torch.cat([target_seq, next_token], dim=-1)

            eos_mask = next_token.squeeze(-1) == self.EOS_TOKEN
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            finished |= eos_mask

            # end if all sequences have generated the EOS token
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,
            (0, self.max_seq_length - max_length), value=self.PAD_TOKEN)

        return target_seq



def create_mask_1(padded_input, input_lengths=None, pad_idx=None):
    """ Create a mask to identify non-padding positions.

    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: Optional, the actual lengths of each sequence before padding, shape (N,).
        pad_idx: Optional, the index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T, 1), where non-padding positions are marked with 1 and padding positions are marked with 0.
    """

    assert input_lengths is not None or pad_idx is not None

    # Create a mask based on input_lengths
    if input_lengths is not None:
        N = padded_input.size(0)        # padded_input : (N x T x ...)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # (N x T)

        # Set the mask to 0 for padding positions
        for i in range(N):
          non_pad_mask[i, input_lengths[i]:] = 0

    if pad_idx is not None:             # padded_input : N x T

        assert padded_input.dim() == 2

        # Create a mask where non-padding positions are marked with 1 and padding positions are marked with 0
        non_pad_mask = padded_input.ne(pad_idx).float()

    return non_pad_mask.unsqueeze(-1)   # unsqueeze(-1) for broadcasting

def create_mask_2(seq, pad_idx=None):
    """ Create a mask to prevent positions from attending to subsequent positions.

    Args:
        seq: The input sequence tensor, shape (batch_size, sequence_length).

    Returns:
        A mask tensor with shape (batch_size, sequence_length, sequence_length),
            where positions are allowed to attend to previous positions but not to subsequent positions.
    """

    sz_b, len_s = seq.size()

    # Create an upper triangular matrix with zeros on the diagonal and below (indicating allowed positions)
    #   and ones above the diagonal (indicating disallowed positions)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)

    # Expand the mask to match the batch size, resulting in a mask for each sequence in the batch.
    mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls


    ''' Create a mask to ignore padding positions in the key sequence during attention calculation. '''

    # Expanding to fit the shape of key query attention matrix.
    if pad_idx != None:
      len_q = seq.size(1)

      # Create a mask where padding positions in the key sequence are marked with 1.
      padding_mask  = seq.eq(pad_idx)

      # Expand the mask to match the dimensions of the key-query attention matrix.
      padding_mask  = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
      mask          = (padding_mask + mask).gt(0)

    else:
      mask = mask.gt(0)

    return mask

def create_mask_3(padded_input, input_lengths, expand_length):
    """ Create an attention mask to ignore padding positions in the input sequence during attention calculation.

    Args:
        padded_input: The input tensor with padding, shape (N, Ti, ...).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
        expand_length: The length to which the attention mask should be expanded,
            usually equal to the length of the sequence that the attention scores will be applied to.

    Returns:
        An attention mask tensor with shape (N, expand_length, Ti),
            where padding positions in the input sequence are marked with 1 and other positions are marked with 0.
    """

    # Create a mask to identify non-padding positions, shape (N, Ti, 1)
    # (N x Ti x 1)
    non_pad_mask    = create_mask_1(padded_input, input_lengths=input_lengths)

    # Invert the mask to identify padding positions, shape (N, Ti)
    # N x Ti, lt(1) like-not operation
    pad_mask        = non_pad_mask.squeeze(-1).lt(1)


    # Expand the mask to match the dimensions of the attention matrix, shape (N, expand_length, Ti)
    attn_mask       = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)

    return attn_mask




class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature    = temperature                       # Scaling factor for the dot product
        self.dropout        = torch.nn.Dropout(attn_dropout)    # Dropout layer for attention weights
        self.softmax        = torch.nn.Softmax(dim=2)           # Softmax layer along the attention dimension

    def forward(self, q, k, v, mask=None):

        # Calculate the dot product between queries and keys.
        attn = torch.bmm(q, k.transpose(1, 2))

        # Scale the dot product by the temperature.
        attn = attn / self.temperature

        if mask is not None:
            # Apply the mask by setting masked positions to a large negative value.
            # This ensures they have a softmax score close to zero.
            mask_value = -1e+30 if attn.dtype == torch.float32 else -1e+4
            attn = attn.masked_fill(mask, mask_value)

        # Apply softmax to obtain attention weights.
        attn    = self.softmax(attn)

        # Apply dropout to the attention weights.
        attn    = self.dropout(attn)

        # Compute the weighted sum of values based on the attention weights.
        output  = torch.bmm(attn, v)

        return output, attn # Return the attention output and the attention weights.
def save_attention_plot(attention_weights, epoch=0):
    ''' function for saving attention weights plot to a file

        @NOTE: default starter code set to save cross attention
    '''

    plt.clf()  # Clear the current figure
    sns.heatmap(attention_weights, cmap="GnBu")  # Create heatmap

    # Save the plot to a file. Specify the directory if needed.
    plt.savefig(f"cross_attention-epoch{epoch}.png")

class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention Module '''

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head # Number of attention heads
        self.d_k    = d_model // n_head
        self.d_v    = d_model // n_head


        # Linear layers for projecting the input query, key, and value to multiple heads
        self.w_qs   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_ks   = torch.nn.Linear(d_model, n_head * self.d_k)
        self.w_vs   = torch.nn.Linear(d_model, n_head * self.d_v)

        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        # Initialize the weights of the linear layers
        self.attention = ScaledDotProductAttention(
            temperature=np.power(self.d_k, 0.5), attn_dropout=dropout)

        # Final linear layer to project the concatenated outputs of the attention heads back to the model dimension
        self.fc = torch.nn.Linear(n_head * self.d_v, d_model)

        torch.nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        # following key, value, query standard computation
        d_k, d_v, n_head    = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _      = q.size()
        sz_b, len_k, _      = k.size()
        sz_b, len_v, _      = v.size()

        # Project the input query, key, and value to multiple heads
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Rearrange the dimensions to group the heads together for parallel processing
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # Repeat the mask for each attention head if a mask is provided
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        # Apply scaled dot-product attention to the projected query, key, and value
        output, attn    = self.attention(q, k, v, mask=mask)

        # Rearrange the output back to the original order and concatenate the heads
        output          = output.view(n_head, sz_b, len_q, d_v)
        output          = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output          = self.dropout(self.fc(output))

        return output, attn
class PositionalEncoding(torch.nn.Module):
    ''' Position Encoding from Attention Is All You Need Paper '''

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Initialize a tensor to hold the positional encodings
        pe          = torch.zeros(max_len, d_model)

        # Create a tensor representing the positions (0 to max_len-1)
        position    = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term for the sine and cosine functions
        # This term creates a series of values that decrease geometrically, used to generate varying frequencies for positional encodings
        div_term    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encodings tensor and make it a buffer
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

      return x + self.pe[:, :x.size(1)]
class FeedForward(torch.nn.Module):
    ''' Projection Layer (Fully Connected Layers) '''

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1   = torch.nn.Linear(d_model, d_ff)
        self.dropout    = torch.nn.Dropout(dropout)
        self.linear_2   = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):

        # Apply the first linear layer, GeLU activation, and then dropout
        x = self.dropout(torch.nn.functional.gelu(self.linear_1(x)))

         # Apply the second linear layer to project the dimension back to d_model
        x = self.linear_2(x)

        return x
    


class CNNLSTM(nn.Module):
    def __init__(self, cnn_output_dim, lstm_hidden_dim, lstm_num_layers=2, num_classes=306):
        super(CNNLSTM, self).__init__()

        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # LSTM part
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, lstm_num_layers, batch_first=True)


        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

        # Embedding layer for input_texts
        self.embedding = nn.Embedding(num_classes, lstm_hidden_dim)

        self.cnn_to_hidden = nn.Linear(cnn_output_dim, lstm_hidden_dim * lstm_num_layers)
        self.cnn_to_cell = nn.Linear(cnn_output_dim, lstm_hidden_dim * lstm_num_layers)

    def forward(self, images, input_texts):
        # CNN forward pass
        cnn_out = self.cnn(images)

        # print("cnn_out", cnn_out.shape, self.cnn_to_hidden(cnn_out).reshape(self.lstm_num_layers, 4, self.lstm_hidden_dim).shape)

        h0 = self.cnn_to_hidden(cnn_out).reshape(self.lstm_num_layers, images.shape[0], self.lstm_hidden_dim)
        c0 = self.cnn_to_cell(cnn_out).reshape(self.lstm_num_layers, images.shape[0], self.lstm_hidden_dim)


        # print("input_texts",input_texts.max())
        embedded_inputs = self.embedding(input_texts)  # Embed input_texts
        # print("embedded_inputs",embedded_inputs.shape)



        lstm_out, _ = self.lstm(embedded_inputs, (h0, c0))

        # Fully connected layer
        out = self.fc(lstm_out)


        return out
    




class CNNEncoder(nn.Module):
    def __init__(self, d_model, input_channels=3, dropout=0.1):
        super(CNNEncoder, self).__init__()
        

        self.cnn = models.resnet18(pretrained=False)
        
        # Modify the first conv layer if grayscale
        if input_channels == 1:
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove last two layers (avgpool and fc)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        # self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        
  
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))  # Keep width flexible, reduce height to 1
        

        self.proj = nn.Linear(512, d_model)  # ResNet18 has 512 output channels
        
        self.config = SimpleNamespace(hidden_size=d_model)

        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.result = {}
        
    def forward(self, x):
        """
        Args:
            x: input images (batch, channels, height, width)
        Returns:
            encoder_output: (batch, seq_len, d_model)
            where seq_len depends on the width after CNN processing
        """
        # CNN feature extraction
        features = self.cnn(x)  # (batch, 512, H', W')
        B, Hidd, W,H = features.shape
        # Adaptive pooling to get (batch, 512, W'', 1)
        # features = self.adaptive_pool(features)
        features = features.reshape(B,Hidd, -1)
        features = features.permute(0, 2,1)
        # Squeeze height dimension and permute to (batch, W'', 512)
        # features = features.squeeze(-1).permute(0, 2, 1)
        
        # Project to d_model dimension
        features = self.proj(features)
        
        # Add normalization and dropout
        features = self.layernorm(features)
        features = self.dropout(features)
        self.result["last_hidden_state"] = features
        return self.result



class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: (batch, hidden_dim)
        # encoder_outputs: (batch, seq_len, hidden_dim)
        
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        
        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch, seq_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch, seq_len)
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
            
        return F.softmax(attention, dim=1)  # (batch, seq_len)

class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim, d_model, hidden_dim, num_layers, 
                 target_vocab_size, eos_token, sos_token, pad_token, dropout=0.1):
        super(CNNLSTMAttention, self).__init__()
        
        # CNN Encoder (your existing implementation)
        self.encoder = CNNEncoder(d_model, dropout)
        
        # LSTM Decoder
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=pad_token)
        self.lstm = nn.LSTM(
            input_size=d_model * 2,  # d_model (CNN features) + d_model (embedding)
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention
        self.attention = Attention(hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, target_vocab_size)
        
        # Tokens
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.pad_token = pad_token
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, images, transcripts, image_lengths, transcript_lengths):
        # Encoder forward
        images = images.squeeze(1)

        encoder_output = self.encoder(images)["last_hidden_state"]  # (batch, seq_len, d_model)
        transcripts  = encoder_output.to(images.device)
        # Embed target tokens
        embedded = self.embedding(transcripts)  # (batch, trg_seq_len, d_model)
        
        # Initialize hidden state
        batch_size = images.size(0)
        hidden = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        cell = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        
        # Prepare outputs
        trg_len = transcripts.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.fc.out_features).to(images.device)
        
        # Decode step-by-step
        for t in range(trg_len):
            # Attention
            attn_weights = self.attention(hidden.squeeze(0), encoder_output)  # (batch, seq_len)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_output)  # (batch, 1, d_model)
            
            # LSTM input
            lstm_input = torch.cat([embedded[:, t, :].unsqueeze(1), context], dim=2)  # (batch, 1, d_model*2)
            
            # LSTM step
            output, (hidden, cell) = self.lstm(lstm_input.transpose(0, 1), (hidden, cell))
            output = output.transpose(0, 1)  # (batch, 1, hidden_dim)
            
            # Predict next token
            output = self.fc(output.squeeze(1))  # (batch, vocab_size)
            outputs[:, t, :] = output
        
        return outputs, None  # None for compatibility with your existing code
    
    def recognize(self, images, max_len=100):
        """Greedy search decoding"""
        self.eval()
        batch_size = images.size(0)
        
        # Encode images
        encoder_output = self.encoder(images)["last_hidden_state"]  # (batch, seq_len, d_model)
        
        # Initialize with SOS token
        transcripts = torch.full((batch_size, 1), self.sos_token, dtype=torch.long).to(images.device)
        
        # Initialize hidden state
        hidden = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        cell = torch.zeros(1, batch_size, self.lstm.hidden_size).to(images.device)
        
        for _ in range(max_len - 1):
            # Embed last token
            embedded = self.embedding(transcripts[:, -1:])  # (batch, 1, d_model)
            
            # Attention
            attn_weights = self.attention(hidden.squeeze(0), encoder_output)  # (batch, seq_len)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_output)  # (batch, 1, d_model)
            
            # LSTM input
            lstm_input = torch.cat([embedded, context], dim=2)  # (batch, 1, d_model*2)
            
            # LSTM step
            output, (hidden, cell) = self.lstm(lstm_input.transpose(0, 1), (hidden, cell))
            output = output.transpose(0, 1)  # (batch, 1, hidden_dim)
            
            # Predict next token
            logits = self.fc(output.squeeze(1))  # (batch, vocab_size)
            next_token = logits.argmax(-1).unsqueeze(1)
            
            # Append to transcripts
            transcripts = torch.cat([transcripts, next_token], dim=1)
            
            # Stop if all sequences predicted EOS token
            if (next_token == self.eos_token).all():
                break
                
        return transcripts