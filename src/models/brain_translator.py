import torch 
import torch.nn as nn
from transformers import T5ForConditionalGeneration

from models.model_components.conv_block import conv_block

class BrainTranslator(nn.Module):

    '''
    Methods:
        - initialization 
        - freeze_pretrained_bart
        - freeze_pretrained_brain
        - forward
    '''

    def __init__(self, 
                 channels=208,
                 out_channels = 64,
                 in_feature = 768,
                 additional_encoder_nhead=8, 
                 endoder_num_layers=6,
                 additional_encoder_dim_feedforward = 2048,
                 patches_number = 20):
        super(BrainTranslator, self).__init__()
        
        
        # Brain transformer encoder
        self.patch_embedding = conv_block(in_channels = channels,
                                          out_channels = out_channels,
                                          in_feature = in_feature)
        self.pos_embedding = nn.Parameter(torch.randn(1, patches_number, in_feature)) 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, dropout=0.1, activation="gelu", batch_first=True)
        self.layernorm_embedding = nn.LayerNorm(in_feature, eps=1e-05)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, endoder_num_layers, self.layernorm_embedding)
        
        # Language modeling
        self.T5_decoder = T5ForConditionalGeneration.from_pretrained("t5-base")
        del self.T5_decoder.encoder
        
        self.freeze_pretrained()


    def freeze_pretrained(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if ('T5' in name):
                param.requires_grad = False

    def forward(self, meg, input_masks_invert, decoder_input_ids, decoder_input_ids_mask):
        'The brain'
        # Encode
        encoded_embedding = self.patch_embedding(meg) # shape -> 8, patches_number, in_feature 
        brain_embedding = encoded_embedding + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask=input_masks_invert)

        #brain_embedding = self.brain_projection(brain_embedding)

        # Decode
        decoder_outputs = self.T5_decoder.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_input_ids_mask,
            encoder_hidden_states=brain_embedding,
            return_dict=True
        )
        sequence_output = decoder_outputs[0]
        lm_logits = self.T5_decoder.lm_head(sequence_output)
        
        return lm_logits

    @torch.no_grad()
    def generate(self, meg, input_masks_invert):

        '''from transformers import T5ForConditionalGeneration
        import torch.nn as nn
        T5 = T5ForConditionalGeneration.from_pretrained("t5-base")
        model.BrainTranslator.T5_decoder.encoder = T5.encoder
        model.BrainTranslator.T5_decoder.encoder.block = model.BrainTranslator.encoder
        model.BrainTranslator.T5_decoder.encoder.block = model.BrainTranslator.encoder.layers.append(model.BrainTranslator.encoder.norm)
        model.BrainTranslator.T5_decoder.base_model.encoder = model.BrainTranslator.T5_decoder.encoder.block
        model.BrainTranslator.T5_decoder.encoder.main_input_name="inputs_embeds"
        model.BrainTranslator.T5_decoder.encoder.config = T5.encoder.config'''


        encoded_embedding = self.patch_embedding(meg) # shape -> 8, patches_number, in_feature 
        encoded_embedding = encoded_embedding + self.pos_embedding
        encoder_outputs = self.encoder(encoded_embedding, 
                                       src_key_padding_mask=input_masks_invert)
        encoder_outputs.last_hidden_state=encoder_outputs

        #add 
        #self.T5_decoder.encoder.main_input_name = "inputs_embeds"
        del self.T5_decoder.generation_config.encoder_outputs
        
        output = self.T5_decoder.generate(
            encoder_outputs=encoder_outputs,
            #inputs_embeds=encoded_embedding,
            #attention_mask=input_masks_invert,
            
            num_beams=5,
            do_sample=False,
            repetition_penalty=5.0,
            
        )
        
        return output