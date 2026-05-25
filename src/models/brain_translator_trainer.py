import lightning as L
from torch import nn
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer
import numpy as np

from models.brain_translator import BrainTranslator

from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


class BrainTranslator_trainer(L.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = config["lr"]
        self.BrainTranslator = BrainTranslator(
                                channels=config["channels"],
                                out_channels = config["out_channels"],
                                in_feature = config["in_feature"],
                                additional_encoder_nhead=config["additional_encoder_nhead"],
                                endoder_num_layers=config["endoder_num_layers"],
                                additional_encoder_dim_feedforward = config["additional_encoder_dim_feedforward"],
                                patches_number = config["patches_number"])
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base", legacy=True)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.predictions = []
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), 
                                 lr=self.lr)
    
    def forward(self, meg, input_masks_invert, decoder_input_ids, decoder_input_ids_mask):
        lm_logits = self.BrainTranslator(meg, input_masks_invert, decoder_input_ids, decoder_input_ids_mask)
        return lm_logits
    
    def _shared_eval(self, meg, input_masks_invert, sentences):
        tmp = self.tokenizer(sentences, padding="max_length", max_length=600, return_tensors="pt")
        decoder_input_ids = tmp.input_ids.to(meg.device)
        decoder_input_ids_mask = tmp.attention_mask.to(meg.device)
        lm_logits = self.forward(meg, input_masks_invert, decoder_input_ids, decoder_input_ids_mask)

        
        decoder_input_ids[decoder_input_ids == self.tokenizer.pad_token_id] = -100
        loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), decoder_input_ids.view(-1))

        return loss, lm_logits
        
    def training_step(self, batch, batch_idx):
        sentences, meg, subjects, input_masks_invert = batch
        loss, lm_logits = self._shared_eval(meg, input_masks_invert, sentences)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sentences, meg, subjects, input_masks_invert = batch
        loss, lm_logits = self._shared_eval(meg, input_masks_invert, sentences)

        self.log("val_loss", loss)
        self.logger.experiment.add_scalars('loss', {'val': loss},self.global_step)

        return loss
    
    def test_end(self, outputs):
        results = {
            'rouge-1-r': np.array([x['rouge-1-r'] for x in outputs]).mean(),
            'rouge-1-p': np.array([x['rouge-1-p'] for x in outputs]).mean(),
            'rouge-1-f': np.array([x['rouge-1-f'] for x in outputs]).mean(),
            'bleu-1': np.array([x['bleu-1'] for x in outputs]).mean(),
            'bleu-2': np.array([x['bleu-2'] for x in outputs]).mean(),
            'bleu-3': np.array([x['bleu-3'] for x in outputs]).mean(),
            'bleu-4': np.array([x['bleu-4'] for x in outputs]).mean(),
        }
        
        return results
                
    def test_step(self, batch, batch_idx):
        sentences, meg, subjects, input_masks_invert = batch

        #logits = self.BrainTranslator(meg, input_masks_invert)
        loss, lm_logits = self._shared_eval(meg, input_masks_invert, sentences)

        # string 
        probs = lm_logits.softmax(dim = 1)
        values, predictions = probs.topk(1)
        predictions = torch.squeeze(predictions)
        predicted_string = self.tokenizer.decode(predictions).split(self.tokenizer.pad_token)[0].replace('<s>','')

        """ calculate rouge score """
        rouge = Rouge()
        rouge_scores = rouge.get_scores(predicted_string,sentences[0], avg = True)

        # tokens
        predictions = predictions.tolist()
        truncated_prediction = []
        for t in predictions:
            if t != self.tokenizer.eos_token_id:
                truncated_prediction.append(t)
            else:
                break
        pred_tokens = self.tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)

        decoder_input_ids = self.tokenizer(sentences, padding="max_length", max_length=600, return_tensors="pt").input_ids
        target_tokens = self.tokenizer.convert_ids_to_tokens(decoder_input_ids[0].tolist(), skip_special_tokens = True)

        """ calculate corpus bleu score """
        weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
        bleu = {}
        for weight in weights_list:
            corpus_bleu_score = corpus_bleu([target_tokens], [pred_tokens], weights = weight)
            bleu[str(len(list(weight)))] = corpus_bleu_score
        
        output = dict({
            'rouge-1-r': rouge_scores['rouge-1']['r'],
            'rouge-1-p': rouge_scores['rouge-1']['p'],
            'rouge-1-f': rouge_scores['rouge-1']['f'],
            'bleu-1': bleu['1'],
            'bleu-2': bleu['2'],
            'bleu-3': bleu['3'],
            'bleu-4': bleu['4'],
        })

        self.predictions.append(output)

        return output