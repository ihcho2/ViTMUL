from functools import partial
from models_ViT.vit import VisionTransformer
from transformers.models.bert.configuration_bert import BertConfig
from models_ViT.visual_transformers import initialize_clip_ER

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np

class ViT_unlearning_ER(nn.Module):
    def __init__(self,                 
                 tokenizer = None,
                 config = None,
                 model_type = None,
                 DASH_layer = None,
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
#         self.module_setting(config)
        self.visual_encoder, _ = initialize_clip_ER(config, model_type, DASH_layer = DASH_layer)
        self.fc = nn.Linear(768, 8) if model_type == 'ViT-B-16' else nn.Linear(1024, 8)
        
        self.use_checkpoint = False
        
        
    def forward(self, image):
        image = image.to(dtype=next(self.parameters()).dtype)
        
        # image.size() = torch.Size([64, 3, 128, 128])
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        
        return self.fc(image_embeds[:,0])
    
    def forward_ER(self, image):
        image = image.to(dtype=next(self.parameters()).dtype)
        
        # image.size() = torch.Size([64, 3, 128, 128])
        image_embeds = self.visual_encoder.visual.forward_ER(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        
        return self.fc(image_embeds[:,0])


    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True

    def beam_search(self, image, question, answer=None, train=True, out_size=5):
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size) 

        return topk_ids, topk_probs
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

