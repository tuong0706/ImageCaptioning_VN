import os
os.environ["USE_TF"] = "0"  # Bắt buộc không dùng TensorFlow
os.environ["USE_TORCH"] = "1"  # Chỉ dùng PyTorch

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import Counter
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from transformers import AutoTokenizer, BertModel, BertConfig
import os
os.environ["USE_TF"] = "0"

class BertTextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-multilingual-cased", freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

class ImageCaptionModel(nn.Module):
    def __init__(self, feature_size, vocab_size, max_seq_length):
        super().__init__()
        self.feature_size = feature_size
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.image_encoder = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )

        self.text_encoder = BertTextEncoder()

        self.fusion = nn.Sequential(
            nn.Linear(256 + 768, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )

        self.output_layer = nn.Linear(512, vocab_size)

    def forward(self, image_features, input_ids, attention_mask):
        img_emb = self.image_encoder(image_features)

        img_emb = img_emb.squeeze()
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
        img_emb = img_emb.unsqueeze(1)

        img_emb = img_emb.expand(-1, self.max_seq_length, -1)

        text_emb = self.text_encoder(input_ids, attention_mask)

        combined = torch.cat([img_emb, text_emb], dim=-1)
        fused = self.fusion(combined)

        logits = self.output_layer(fused)
        return logits

class ImageFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self._build_model()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model = nn.Sequential(*list(model.children())[:-2]) 

        for param in model.parameters():
            param.requires_grad_(False)

        model = model.to(self.device)
        model.eval()

        print("ResNet50 feature extractor loaded successfully for prediction")
        return model

    def extract_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(img_tensor)

            features = nn.AdaptiveAvgPool2d((1, 1))(features)
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
            return features

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

class TextProcessor:
    def __init__(self, model_name="bert-base-multilingual-cased", max_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        print(f"BERT tokenizer loaded for prediction with vocab size: {self.vocab_size}")

    def preprocess_caption(self, caption):
        text = "[CLS] " + caption + " [SEP]"
        return text

    def tokenize_caption(self, caption):
        text = self.preprocess_caption(caption)

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

def load_processed_data(data_dir):
    processed_data_path = os.path.join(data_dir, 'processed_data.pkl')
    try:
        with open(processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
        return processed_data['word2idx'], processed_data['idx2word'], processed_data['vocab_size']
    except FileNotFoundError:
        print(f"Error: {processed_data_path} not found. Please run image_caption.py first.")
        sys.exit()

def load_best_model(data_dir, feature_size, vocab_size, max_seq_length, device):
    model_path = os.path.join(data_dir, 'best_model.pth')
    model = ImageCaptionModel(feature_size, vocab_size, max_seq_length).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Loaded best model for prediction")
        return model
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        sys.exit()

def predict_caption_beam_search(image_path, model, feature_extractor, text_processor, max_seq_length, device, beam_width=3, temperature=1.0, repetition_penalty=1.0):

    image_features = feature_extractor.extract_features(image_path)
    if image_features is None:
        return "Error: Could not extract image features."

    image_features_tensor = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0).to(device)

    start_token_id = text_processor.tokenizer.cls_token_id
    sep_token_id = text_processor.tokenizer.sep_token_id
    pad_token_id = text_processor.tokenizer.pad_token_id

    beam_candidates = [
        (0.0,
         torch.tensor([[start_token_id] + [pad_token_id] * (max_seq_length - 1)]).to(device),
         torch.tensor([[1] + [0] * (max_seq_length - 1)]).to(device)
        )
    ]
    
    completed_sequences = []

    with torch.no_grad():
        for i in range(1, max_seq_length):
            new_beam_candidates = []
            current_active_beams = []

            for score, current_input_ids, current_attention_mask in beam_candidates:
                if current_input_ids[0, i-1].item() == sep_token_id:
                    completed_sequences.append((score, current_input_ids, current_attention_mask))
                else:
                    current_active_beams.append((score, current_input_ids, current_attention_mask))
            
            if not current_active_beams:
                break

            for score, current_input_ids, current_attention_mask in current_active_beams:
                outputs = model(image_features_tensor, current_input_ids, current_attention_mask)
                logits = outputs[:, i-1, :]

                logits = logits / temperature
                current_tokens = current_input_ids[0, :i].tolist()

                seen_tokens = set([t for t in current_tokens if t not in [start_token_id, sep_token_id, pad_token_id]])

                for token_id_seen in seen_tokens:
                    if logits[0, token_id_seen] < 0:
                        logits[0, token_id_seen] *= repetition_penalty
                    else:
                        logits[0, token_id_seen] /= repetition_penalty

                log_probs = torch.log_softmax(logits, dim=-1)

                top_log_probs, top_indices = log_probs.topk(beam_width, dim=-1)

                for k in range(beam_width):
                    token_log_prob = top_log_probs[0, k].item()
                    token_id = top_indices[0, k].item()

                    new_score = score + (-token_log_prob)

                    next_input_ids = current_input_ids.clone()
                    next_attention_mask = current_attention_mask.clone()

                    if i < max_seq_length:
                        next_input_ids[0, i] = token_id
                        next_attention_mask[0, i] = 1
                    else:
                        continue 

                    new_beam_candidates.append((new_score, next_input_ids, next_attention_mask))
            
            new_beam_candidates.sort(key=lambda x: x[0])
            beam_candidates = new_beam_candidates[:beam_width]

        completed_sequences.extend(beam_candidates)
        completed_sequences.sort(key=lambda x: x[0])

        if not completed_sequences:
            return "Could not generate a coherent caption."

        best_score, best_sequence_ids, _ = completed_sequences[0]
        
        try:
            sep_idx = (best_sequence_ids == sep_token_id).nonzero(as_tuple=True)[1][0].item()
            best_sequence_ids = best_sequence_ids[0, :sep_idx] 
        except IndexError: 
            last_valid_idx = (best_sequence_ids[0] != pad_token_id).nonzero(as_tuple=True)[0].max().item()
            best_sequence_ids = best_sequence_ids[0, :last_valid_idx + 1]

        predicted_tokens = text_processor.tokenizer.decode(best_sequence_ids.tolist(), skip_special_tokens=True)
        return predicted_tokens
    

def generate_caption(image_path):
    config = {
        'data_dir': './data',
        'max_seq_length': 64,
        'feature_size': 2048
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global model, feature_extractor, text_processor
    try:
        model
    except NameError:
        print("Loading model and processors for first use...")
        word2idx, idx2word, vocab_size = load_processed_data(config['data_dir'])
        feature_extractor = ImageFeatureExtractor(device=device)
        text_processor = TextProcessor(max_length=config['max_seq_length'])
        model = load_best_model(config['data_dir'], config['feature_size'], text_processor.vocab_size, config['max_seq_length'], device)

    caption = predict_caption_beam_search(
        image_path=image_path,
        model=model,
        feature_extractor=feature_extractor,
        text_processor=text_processor,
        max_seq_length=config['max_seq_length'],
        device=device,
        beam_width=5,
        temperature=1.2,
        repetition_penalty=2.0
    )

    return caption

if __name__ == "__main__":
    config = {
        'data_dir': './data',
        'max_seq_length': 64,
        'feature_size': 2048
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for prediction")

    word2idx, idx2word, vocab_size = load_processed_data(config['data_dir'])

    feature_extractor = ImageFeatureExtractor(device=device)

    text_processor = TextProcessor(max_length=config['max_seq_length'])

    model = load_best_model(
        config['data_dir'],
        config['feature_size'],
        text_processor.vocab_size,
        config['max_seq_length'],
        device
    )

    # image_file = input("Enter the path to the image file: ")
    image_file = "OIP.jpg"

    if os.path.exists(image_file):
        start_time = time.time()
        predicted_caption = predict_caption_beam_search(
            image_file,
            model,
            feature_extractor,
            text_processor,
            config['max_seq_length'],
            device,
            beam_width=5,
            temperature=1.2,
            repetition_penalty=2.0
        )
        end_time = time.time()
        print(f"Predicted caption: {predicted_caption}")
        print(f"Prediction time: {end_time - start_time:.4f} seconds")
    else:
        print(f"Error: Image file not found at {image_file}")

# python310 predict.py

# 365584746_681f33fa46.jpg