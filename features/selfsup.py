import torch
import torch.nn as nn

from transformers import HubertModel, AutoModelForPreTraining, AutoFeatureExtractor



class Wav2VecExtractor(nn.Module):
    # https://huggingface.co/facebook/wav2vec2-base
    
    def __init__(self, model_scale:str='base', device='cpu'):
        super().__init__()
        assert model_scale in ['base','large']
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained(f"facebook/wav2vec2-{model_scale}") # Contains tokenizer & encoder(convs)
        self.model = AutoModelForPreTraining.from_pretrained(f"facebook/wav2vec2-{model_scale}").to(device)

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs['input_values'] = inputs['input_values'].to(self.device)
        
        # Calculate hidden feats
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        outputs = torch.cat(outputs.hidden_states, dim=0).permute(1,0,2) # Length, Nlayer, Dim
        
        # Calculate VQ-index
        conv_features = self._conv_feature(inputs['input_values'][0])
        quantized_idx = self._quantize(conv_features) # Length, group_size

        return outputs, quantized_idx
    
    def extract_hidden_feats(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states

    def extract_quantized(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats
    
    
class HuBERTExtractor(nn.Module):
    # https://huggingface.co/facebook/hubert-large-ll60k
    def __init__(self, model_scale:str='base', device='cpu'):
        super().__init__()
        assert model_scale in ['base','large']
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-large-ll60k") # Contains tokenizer & encoder(convs)
        self.model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)

    def extract(self, inputs:torch.Tensor, sr:int=16000):
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs['input_values'] = inputs['input_values'].to(self.device)
        
        # Calculate hidden feats
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        outputs = torch.cat(outputs.hidden_states, dim=0).permute(1,0,2) # Length, Nlayer, Dim
        
        # Calculate VQ-index
        # conv_features = self._conv_feature(inputs['input_values'][0])
        # quantized_idx = self._quantize(conv_features) # Length, group_size

        return outputs #, quantized_idx
    
    def extract_hidden_feats(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs['input_values'] = inputs['input_values'].to(device)
        outputs = self.model(inputs['input_values'][0], return_dict=True, output_hidden_states=True)
        return outputs.hidden_states

    def extract_quantized(self, inputs:torch.Tensor, sr:int=16000):
        device = inputs.device
        inputs = self.processor(inputs, sampling_rate=sr, return_tensors='pt')
        inputs = inputs['input_values'].to(device)

        conv_features = self._conv_feature(inputs[0])
        quantized_features = self._quantize(conv_features)

        return quantized_features

    def _quantize(self, hidden_states):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        num_groups = self.model.quantizer.num_groups

        hidden_states = self.model.quantizer.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * num_groups, -1)
        codevector_idx = hidden_states.argmax(dim=-1)

        return codevector_idx.reshape(-1, num_groups)

    def _conv_feature(self, sig):
        feats = self.model.wav2vec2.feature_extractor(sig)
        feats = feats.transpose(1, 2)
        _, feats = self.model.wav2vec2.feature_projection(feats)
        return feats