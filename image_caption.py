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


from underthesea import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.max_memory_allocated(0)/1024**2:.2f} MB allocated")

from transformers import AutoTokenizer, BertModel, BertConfig
print("Transformers imported successfully")


from sklearn.model_selection import train_test_split


torch.manual_seed(42)
np.random.seed(42)


class ImageCaptionDataset:
    def __init__(self, root_dir, captions_file, train_file, test_file):
        self.root_dir = root_dir
        self.captions_file = os.path.join(root_dir, captions_file)
        self.train_file = os.path.join(root_dir, train_file)
        self.test_file = os.path.join(root_dir, test_file)
        
        self.load_data()
        self.clean_data()
        self.analyze_data()
        
    def load_data(self):
        self.captions_df = pd.read_csv(
            self.captions_file, 
            sep='\t', 
            header=None, 
            names=['image', 'caption']
        )
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            self.train_images = [line.strip() for line in f.readlines()]
            
        with open(self.test_file, 'r', encoding='utf-8') as f:
            self.test_images = [line.strip() for line in f.readlines()]
            
        self.captions_df['dataset_type'] = self.captions_df['image'].apply(
            lambda x: 'train' if x in self.train_images else 'test' if x in self.test_images else 'unknown'
        )
    
    def clean_vietnamese_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text.lower())
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    
    def clean_data(self):
        print("\nBefore cleaning:")
        print(self.captions_df['caption'].head(3))
        
        self.captions_df['cleaned_caption'] = self.captions_df['caption'].apply(self.clean_vietnamese_text)
        
        print("\nAfter cleaning:")
        print(self.captions_df['cleaned_caption'].head(3))
    
    def analyze_data(self):
        print("\nData Analysis:")
        print(f"Total captions: {len(self.captions_df)}")
        print(f"Unique images: {self.captions_df['image'].nunique()}")
        print(f"Train images: {len(self.train_images)}")
        print(f"Test images: {len(self.test_images)}")
        print("\nCaption length distribution:")
        print(self.captions_df['cleaned_caption'].str.split().apply(len).describe())
    
    def build_vocabulary(self, min_word_count=5):
        word_counts = {}
        for caption in self.captions_df['cleaned_caption']:
            for word in caption.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        vocab = [word for word, count in word_counts.items() if count >= min_word_count]
        vocab = ['<pad>', '<start>', '<end>', '<unk>'] + vocab
        
        self.vocab = vocab
        self.word_counts = word_counts
        self.vocab_size = len(vocab)
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
        print(f"\nVocabulary size: {self.vocab_size}")
        print("Sample words:", vocab[:20])
    
    def save_processed_data(self, output_file):
        processed_data = {
            'captions_df': self.captions_df,
            'train_images': self.train_images,
            'test_images': self.test_images,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\nProcessed data saved to {output_file}")


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
        
        print("ResNet50 feature extractor loaded successfully")
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
    
    def batch_extract_features(self, image_paths, batch_size=32):
        features_dict = {}
        batch_images = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="Extracting features"):
            img = self.extract_features(img_path)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(os.path.basename(img_path))
        
        for img_name, feature in zip(valid_paths, batch_images):
            features_dict[img_name] = feature
        
        return features_dict

class TextProcessor:
    def __init__(self, model_name="bert-base-multilingual-cased", max_length=64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        print(f"BERT tokenizer loaded with vocab size: {self.vocab_size}")
    
    def preprocess_caption(self, caption):
        text = "[CLS] " + caption + " [SEP]"
        return text
    
    def tokenize_captions(self, captions):
        input_ids = []
        attention_masks = []
        
        for caption in tqdm(captions, desc="Tokenizing captions"):
            text = self.preprocess_caption(caption)
            
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=False,  # We've added them manually
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        
        return input_ids, attention_masks
    
    def save_processed_text(self, output_file, input_ids, attention_masks):
        processed_text = {
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'tokenizer_config': self.tokenizer.init_kwargs,
            'max_length': self.max_length
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(processed_text, f)
        
        print(f"Processed text data saved to {output_file}")


class CaptionDataset(Dataset):
    def __init__(self, image_features, input_ids, attention_masks, targets):
        self.image_features = image_features
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.targets = targets
        
    def __len__(self):
        return len(self.image_features)
    
    def __getitem__(self, idx):
        return {
            'image_features': torch.tensor(self.image_features[idx], dtype=torch.float32),
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }, self.targets[idx]


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


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ImageCaptionDataset(
        root_dir=config['data_dir'],
        captions_file=config['captions_file'],
        train_file=config['train_file'],
        test_file=config['test_file']
    )
    
    dataset.build_vocabulary()
    dataset.save_processed_data(os.path.join(config['data_dir'], 'processed_data.pkl'))
    
    if not os.path.exists(os.path.join(config['data_dir'], 'image_features.pkl')):
        feature_extractor = ImageFeatureExtractor(device=device)
        
        all_images = dataset.captions_df['image'].unique()
        image_paths = [os.path.join(config['data_dir'], 'images', img) for img in all_images]
        
        image_features = feature_extractor.batch_extract_features(image_paths)
        
        with open(os.path.join(config['data_dir'], 'image_features.pkl'), 'wb') as f:
            pickle.dump(image_features, f)
    else:
        print("Loading precomputed image features...")
        with open(os.path.join(config['data_dir'], 'image_features.pkl'), 'rb') as f:
            image_features = pickle.load(f)
    
    text_processor = TextProcessor(max_length=config['max_seq_length'])
    
    if not os.path.exists(os.path.join(config['data_dir'], 'processed_text_data.pkl')):
        input_ids, attention_masks = text_processor.tokenize_captions(dataset.captions_df['cleaned_caption'])
        text_processor.save_processed_text(
            os.path.join(config['data_dir'], 'processed_text_data.pkl'),
            input_ids,
            attention_masks
        )
    else:
        print("Loading preprocessed text data...")
        with open(os.path.join(config['data_dir'], 'processed_text_data.pkl'), 'rb') as f:
            text_data = pickle.load(f)
            input_ids = text_data['input_ids']
            attention_masks = text_data['attention_masks']
    
    train_image_names = list(set(dataset.captions_df[dataset.captions_df['dataset_type'] == 'train']['image']))
    train_indices = dataset.captions_df[dataset.captions_df['image'].isin(train_image_names)].index
    test_indices = dataset.captions_df[~dataset.captions_df['image'].isin(train_image_names)].index
    
    X_train_images = np.array([image_features[img_name] for img_name in dataset.captions_df.loc[train_indices, 'image']])
    X_test_images = np.array([image_features[img_name] for img_name in dataset.captions_df.loc[test_indices, 'image']])
    
    X_train_text = input_ids[train_indices]
    X_train_mask = attention_masks[train_indices]
    X_test_text = input_ids[test_indices]
    X_test_mask = attention_masks[test_indices]
    
    y_train = torch.roll(X_train_text, shifts=-1, dims=1)
    y_test = torch.roll(X_test_text, shifts=-1, dims=1)
    

    y_train[:, -1] = -100
    y_test[:, -1] = -100
    

    train_dataset = CaptionDataset(X_train_images, X_train_text, X_train_mask, y_train)
    val_dataset = CaptionDataset(X_test_images, X_test_text, X_test_mask, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    

    model = ImageCaptionModel(
        feature_size=2048,
        vocab_size=text_processor.vocab_size,
        max_seq_length=config['max_seq_length']
    ).to(device)
    

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.2,
        patience=3
        # verbose=True
    )
    

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            inputs, targets = batch
            
            image_features = inputs['image_features'].to(device)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(image_features, input_ids, attention_mask)
            
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                
                image_features = inputs['image_features'].to(device)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                targets = targets.to(device)
                
                outputs = model(image_features, input_ids, attention_mask)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config['data_dir'], 'best_model.pth'))
            print("Saved best model!")
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    with open(os.path.join(config['data_dir'], 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(config['data_dir'], 'training_curve.png'))
    plt.show()
    
    print("Training completed!")


if __name__ == "__main__":
    config = {
        'data_dir': './data',
        'captions_file': 'captions_vi.txt',
        'train_file': 'trainImages.txt',
        'test_file': 'testImages.txt',
        'max_seq_length': 64,
        'batch_size': 32,
        'learning_rate': 3e-5,
        'epochs': 50
    }
    
    os.makedirs(config['data_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['data_dir'], 'images'), exist_ok=True)
    
    train_model(config)