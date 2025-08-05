import os
import json
import subprocess
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    EvalPrediction,
    get_linear_schedule_with_warmup,
    set_seed
)
from transformers.modeling_outputs import TokenClassifierOutput

from torchcrf import CRF
from datasets import Dataset as HFDataset, concatenate_datasets
from seqeval.metrics import classification_report, f1_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Ensemble Discourse Detector with Dependency Features"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../sharedtask2025/data/",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="advanced-ensemble-discourse-detector-with-deps",
        help="Output directory for model and predictions"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=[
            "google/rembert",
            "FacebookAI/xlm-roberta-large", 
            "microsoft/mdeberta-v3-base",
            # "google-bert/bert-base-multilingual-cased"
        ],
        help="List of pretrained model names to use in ensemble"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--pos_embedding_dim",
        type=int,
        default=50,
        help="Dimension of POS tag embeddings"
    )
    parser.add_argument(
        "--dep_embedding_dim",
        type=int,
        default=50,
        help="Dimension of dependency relation embeddings"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="weighted",
        choices=["concat", "weighted", "attention"],
        help="Ensemble fusion method"
    )
    parser.add_argument(
        "--use_adversarial_training",
        action="store_true",
        help="Enable adversarial training"
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss for class imbalance"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0 to disable)"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Optimization
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision training"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision training (recommended for H200)"
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 on Ampere GPUs"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    
    # Evaluation
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Number of steps between evaluations"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Number of steps between checkpoints"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=25,
        help="Number of steps between logging"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep"
    )
    
    # Other options
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Disable CUDA even if available"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def set_random_seeds(seed: int):
    """Set seeds for reproducibility across all libraries."""
    logger.info(f"Setting random seed to {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Transformers
    set_seed(seed)
    
    # Additional PyTorch settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for better reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention for fusing multiple encoder outputs"""
    def __init__(self, hidden_sizes, num_heads=8):
        super().__init__()
        self.total_hidden = sum(hidden_sizes)
        self.attention = nn.MultiheadAttention(self.total_hidden, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.total_hidden)
        
    def forward(self, encoder_outputs):
        concatenated = torch.cat(encoder_outputs, dim=-1)
        attended, _ = self.attention(concatenated, concatenated, concatenated)
        return self.layer_norm(concatenated + attended)


class DependencyAwareEnsembleModelWithCRF(nn.Module):
    """
    Advanced ensemble model with POS and dependency features
    """
    def __init__(
        self,
        model_names: List[str],
        num_labels: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        num_pos_tags: int,
        num_dep_rels: int,
        pos_embedding_dim: int = 50,
        dep_embedding_dim: int = 50,
        ensemble_method: str = "weighted",
        use_focal_loss: bool = True,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        # Load multiple encoders
        self.encoders = nn.ModuleList([
            AutoModel.from_pretrained(name) for name in model_names
        ])
        
        # Get hidden sizes
        self.hidden_sizes = [enc.config.hidden_size for enc in self.encoders]
        self.ensemble_method = ensemble_method
        
        # Dropout layers
        self.dropout = nn.Dropout(0.3)
        self.feature_dropout = nn.Dropout(0.2)
        
        # POS and dependency embeddings
        self.pos_embedding = nn.Embedding(num_pos_tags, pos_embedding_dim)
        self.dep_embedding = nn.Embedding(num_dep_rels, dep_embedding_dim)
        
        # Additional linguistic feature processing
        # Combine POS + dep info
        self.linguistic_projection = nn.Linear(
            pos_embedding_dim + dep_embedding_dim, 
            pos_embedding_dim + dep_embedding_dim
        )
        
        # Fusion layers based on ensemble method
        if ensemble_method == "concat":
            fusion_size = sum(self.hidden_sizes)
        elif ensemble_method == "weighted":
            fusion_size = self.hidden_sizes[0]
            self.projection_layers = nn.ModuleList([
                nn.Linear(size, fusion_size) for size in self.hidden_sizes
            ])
            self.encoder_weights = nn.Parameter(torch.ones(len(self.encoders)) / len(self.encoders))
        elif ensemble_method == "attention":
            fusion_size = sum(self.hidden_sizes)
            self.attention_fusion = MultiHeadAttentionFusion(self.hidden_sizes)
        
        # Feature size including POS and dependency
        linguistic_features_size = pos_embedding_dim + dep_embedding_dim
        
        # Multi-layer classifier
        self.pre_classifier = nn.Linear(fusion_size + linguistic_features_size, fusion_size)
        self.classifier_norm = nn.LayerNorm(fusion_size)
        self.classifier_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(fusion_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
        # Loss functions
        self.use_focal_loss = use_focal_loss
        if use_focal_loss and class_weights is not None:
            self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        
        self.label_smoothing = label_smoothing
        
        # Store configurations
        self.num_labels = num_labels
        self.label2id = label2id
        self.id2label = id2label
        self.config = self.encoders[0].config
        self.config.num_labels = num_labels
        self.config.label2id = label2id
        self.config.id2label = id2label

    def forward(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: List[torch.Tensor],
        pos_tag_ids: Optional[torch.Tensor] = None,
        dep_rel_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        adversarial_training: bool = False,
        **kwargs
    ):
        # Get encoder outputs
        encoder_outputs = []
        for i, (encoder, input_ids, attention_mask) in enumerate(
            zip(self.encoders, input_ids_list, attention_mask_list)
        ):
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            encoder_outputs.append(outputs.last_hidden_state)
        
        # Apply fusion strategy
        if self.ensemble_method == "concat":
            fused_output = torch.cat(encoder_outputs, dim=-1)
        elif self.ensemble_method == "weighted":
            projected_outputs = [
                proj(output) for proj, output in zip(self.projection_layers, encoder_outputs)
            ]
            weights = F.softmax(self.encoder_weights, dim=0)
            fused_output = sum(w * output for w, output in zip(weights, projected_outputs))
        elif self.ensemble_method == "attention":
            fused_output = self.attention_fusion(encoder_outputs)
        
        # Add adversarial perturbation if training
        if adversarial_training and self.training:
            fused_output = fused_output + 0.01 * torch.randn_like(fused_output)
        
        # Process linguistic features
        # Handle POS embeddings
        safe_pos_tag_ids = pos_tag_ids.clone()
        safe_pos_tag_ids[safe_pos_tag_ids == -100] = 0
        pos_embeds = self.pos_embedding(safe_pos_tag_ids)
        
        # Handle dependency embeddings
        safe_dep_rel_ids = dep_rel_ids.clone()
        safe_dep_rel_ids[safe_dep_rel_ids == -100] = 0
        dep_embeds = self.dep_embedding(safe_dep_rel_ids)
        
        # Combine linguistic features
        linguistic_features = torch.cat([pos_embeds, dep_embeds], dim=-1)
        linguistic_features = self.feature_dropout(linguistic_features)
        linguistic_features = self.linguistic_projection(linguistic_features)
        linguistic_features = F.relu(linguistic_features)
        
        # Concatenate all features
        concatenated_output = torch.cat([fused_output, linguistic_features], dim=-1)
        
        # Apply classifier layers
        sequence_output = self.dropout(concatenated_output)
        pre_logits = self.pre_classifier(sequence_output)
        pre_logits = F.gelu(pre_logits)  # GELU activation
        pre_logits = self.classifier_norm(pre_logits)
        pre_logits = self.classifier_dropout(pre_logits)
        
        # Residual connection if dimensions match
        if pre_logits.shape[-1] == fused_output.shape[-1]:
            pre_logits = pre_logits + fused_output
        
        logits = self.classifier(pre_logits)
        
        loss = None
        if labels is not None:
            # Prepare CRF labels (where -100 is temporarily 0 for the CRF layer)
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0  # torchcrf expects non-negative labels
            crf_mask = attention_mask_list[0].bool() if attention_mask_list[0] is not None else torch.ones_like(labels, dtype=torch.bool)
            
            # CRF loss
            crf_loss = -self.crf(logits, crf_labels, mask=crf_mask, reduction='mean')
            
            # Create a mask that filters out BOTH padding and ignored sub-word tokens.
            active_loss_mask = labels.view(-1) != -100

            # Additional focal loss on logits if enabled
            if self.use_focal_loss and hasattr(self, 'focal_loss'):
                # Apply the mask to get only the logits and labels we should calculate loss on.
                active_logits = logits.view(-1, self.num_labels)[active_loss_mask]
                active_labels = labels.view(-1)[active_loss_mask]
                
                # Only compute loss if there are any active labels to prevent errors on empty tensors
                if active_logits.shape[0] > 0:
                    focal_loss = self.focal_loss(active_logits, active_labels)
                    loss = 0.7 * crf_loss + 0.3 * focal_loss
                else:
                    loss = crf_loss # Fallback to just crf_loss if no active labels
            else:
                loss = crf_loss
            
            # Label smoothing
            if self.label_smoothing > 0 and self.training:
                active_logits_smooth = logits.view(-1, self.num_labels)[active_loss_mask]
                active_labels_smooth = labels.view(-1)[active_loss_mask]

                if active_logits_smooth.shape[0] > 0:
                    smooth_loss = F.cross_entropy(
                        active_logits_smooth,
                        active_labels_smooth,
                        reduction='mean',
                        label_smoothing=self.label_smoothing
                    )
                    # Use the existing loss (which could be crf or crf+focal)
                    current_loss = loss if loss is not None else crf_loss
                    loss = 0.9 * current_loss + 0.1 * smooth_loss

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
    
    def predict(self, input_ids_list: List[torch.Tensor], 
                attention_mask_list: List[torch.Tensor],
                pos_tag_ids: Optional[torch.Tensor] = None,
                dep_rel_ids: Optional[torch.Tensor] = None):
        """Get predictions using Viterbi decoding"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids_list=input_ids_list,
                attention_mask_list=attention_mask_list,
                pos_tag_ids=pos_tag_ids,
                dep_rel_ids=dep_rel_ids
            )
            logits = outputs.logits
            mask = attention_mask_list[0].bool() if attention_mask_list[0] is not None else torch.ones_like(input_ids_list[0], dtype=torch.bool)
            predictions = self.crf.decode(logits, mask=mask)
        return predictions


def parse_conllu_file(filepath: str) -> List[Dict[str, List[str]]]:
    """Parse a CoNLL-U file and extract tokens, POS, dependency, and discourse connective tags."""
    examples = []
    current_tokens = []
    current_bio_tags = []
    current_pos_tags = []
    current_dep_rels = []
    current_heads = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line = end of sentence
            if not line:
                if current_tokens:
                    examples.append({
                        "tokens": current_tokens,
                        "bio_tags": current_bio_tags,
                        "pos_tags": current_pos_tags,
                        "dep_rels": current_dep_rels,
                        "heads": current_heads
                    })
                    current_tokens = []
                    current_bio_tags = []
                    current_pos_tags = []
                    current_dep_rels = []
                    current_heads = []
                continue
            
            # Skip comments
            if line.startswith('#'):
                continue
            
            # Parse CoNLL-U columns
            parts = line.split('\t')
            if len(parts) >= 10:
                # Skip multi-word tokens (e.g., "1-2")
                if '-' in parts[0]:
                    continue
                
                token = parts[1]  # Word form
                pos_tag = parts[3]  # UPOS tag
                head = parts[6]  # Head of current word
                dep_rel = parts[7]  # Dependency relation
                
                # Extract discourse connective tag from MISC field (column 10)
                misc_field = parts[9]
                bio_tag = "O"  # Default: Outside
                
                if misc_field != "_":
                    for feat in misc_field.split("|"):
                        if feat.startswith("Conn="):
                            bio_tag = feat.split("=")[1]
                            break
                
                current_tokens.append(token)
                current_bio_tags.append(bio_tag)
                current_pos_tags.append(pos_tag)
                current_dep_rels.append(dep_rel)
                current_heads.append(head)
    
    # Don't forget the last sentence
    if current_tokens:
        examples.append({
            "tokens": current_tokens,
            "bio_tags": current_bio_tags,
            "pos_tags": current_pos_tags,
            "dep_rels": current_dep_rels,
            "heads": current_heads
        })
    
    return examples


def load_data(data_dir: str) -> Tuple[List[Dict], Dict[str, List[Dict]], Dict[str, List[str]], List[str], List[str], List[str]]:
    """Load train, dev, and test data from CoNLL-U files."""
    logger.info(f"Loading data from {data_dir}")
    
    all_files = list(Path(data_dir).rglob("*.conllu"))
    pdtb_files = [f for f in all_files if '.pdtb' in str(f) or '.iso' in str(f)]
    
    logger.info(f"Found {len(pdtb_files)} CoNLL-U files")
    
    train_examples = []
    dev_examples_by_corpus = defaultdict(list)
    test_files_by_corpus = defaultdict(list)
    
    for file_path in pdtb_files:
        corpus_name = file_path.parent.name
        
        if "train" in file_path.name:
            logger.info(f"Loading train: {file_path}")
            train_examples.extend(parse_conllu_file(str(file_path)))
        elif "dev" in file_path.name:
            logger.info(f"Loading dev: {file_path}")
            examples = parse_conllu_file(str(file_path))
            dev_examples_by_corpus[corpus_name].extend(examples)
        elif "test" in file_path.name:
            test_files_by_corpus[corpus_name].append(str(file_path))
    
    # Extract unique labels, POS tags, and dependency relations
    all_labels = set()
    all_pos_tags = set()
    all_dep_rels = set()
    
    for example in train_examples:
        all_labels.update(example["bio_tags"])
        all_pos_tags.update(example["pos_tags"])
        all_dep_rels.update(example["dep_rels"])
    
    for examples in dev_examples_by_corpus.values():
        for example in examples:
            all_labels.update(example["bio_tags"])
            all_pos_tags.update(example["pos_tags"])
            all_dep_rels.update(example["dep_rels"])

    label_list = sorted(list(all_labels))
    pos_list = sorted(list(all_pos_tags))
    dep_list = sorted(list(all_dep_rels))

    logger.info(f"Found {len(label_list)} unique labels: {label_list}")
    logger.info(f"Found {len(pos_list)} unique POS tags")
    logger.info(f"Found {len(dep_list)} unique dependency relations")

    return train_examples, dev_examples_by_corpus, test_files_by_corpus, label_list, pos_list, dep_list


def tokenize_and_align_labels_multi(
    examples: Dict[str, List],
    tokenizers: List[PreTrainedTokenizerBase],
    label2id: Dict[str, int],
    pos2id: Dict[str, int],
    dep2id: Dict[str, int],
    max_length: int = 512
) -> Dict[str, List]:
    """Tokenize inputs for multiple models and align labels with linguistic features"""
    tokenized_inputs_list = []
    
    # Tokenize with each tokenizer
    for tokenizer in tokenizers:
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding=False
        )
        tokenized_inputs_list.append(tokenized)
    
    labels_list = []
    pos_ids_list = []
    dep_ids_list = []
    
    # Align labels using the first tokenizer
    for i, (label_seq, pos_seq, dep_seq) in enumerate(
        zip(examples["bio_tags"], examples["pos_tags"], examples["dep_rels"])
    ):
        word_ids = tokenized_inputs_list[0].word_ids(batch_index=i)
        label_ids = []
        pos_ids = []
        dep_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                pos_ids.append(-100)
                dep_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_seq[word_idx]])
                pos_ids.append(pos2id.get(pos_seq[word_idx], pos2id.get("UNK", 0)))
                dep_ids.append(dep2id.get(dep_seq[word_idx], dep2id.get("UNK", 0)))
            else:
                label_ids.append(-100)
                pos_ids.append(-100)
                dep_ids.append(-100)
            previous_word_idx = word_idx
        
        labels_list.append(label_ids)
        pos_ids_list.append(pos_ids)
        dep_ids_list.append(dep_ids)
    
    # Build result dictionary
    result = {
        "labels": labels_list,
        "pos_tag_ids": pos_ids_list,
        "dep_rel_ids": dep_ids_list,
    }
    
    # Add tokenized inputs for each model
    for i, tokenized in enumerate(tokenized_inputs_list):
        result[f"input_ids_{i}"] = tokenized["input_ids"]
        result[f"attention_mask_{i}"] = tokenized["attention_mask"]
    
    return result


def compute_class_weights(train_examples: List[Dict], label2id: Dict[str, int]) -> torch.Tensor:
    """Compute class weights for handling imbalanced data"""
    label_counts = Counter()
    for example in train_examples:
        label_counts.update(example["bio_tags"])
    
    total_count = sum(label_counts.values())
    weights = torch.zeros(len(label2id))
    for label, idx in label2id.items():
        count = label_counts.get(label, 1)
        weights[idx] = total_count / (len(label2id) * count)
    
    weights = weights / weights.sum() * len(weights)
    return weights


class AdvancedEnsembleDataCollator(DataCollatorForTokenClassification):
    """Data collator for multiple tokenizers with linguistic features"""
    def __init__(self, tokenizers: List[PreTrainedTokenizerBase], *args, **kwargs):
        # We call the parent init with the first tokenizer for some defaults,
        # but we will override the padding behavior.
        super().__init__(tokenizers[0], *args, **kwargs)
        self.tokenizers = tokenizers
    
    def torch_call(self, features: List[Dict[str, any]]) -> Dict[str, any]:
        # Extract features for each tokenizer from the input list of dicts
        input_ids_lists = [[] for _ in self.tokenizers]
        attention_mask_lists = [[] for _ in self.tokenizers]
        
        for i in range(len(self.tokenizers)):
            for f in features:
                input_ids_lists[i].append(f[f"input_ids_{i}"])
                attention_mask_lists[i].append(f[f"attention_mask_{i}"])
        
        labels = [f["labels"] for f in features]
        pos_tag_ids = [f["pos_tag_ids"] for f in features]
        dep_rel_ids = [f["dep_rel_ids"] for f in features]

        # Find the single maximum sequence length across ALL tokenizations in the batch.
        unified_max_length = max(len(l) for l in labels)
        for id_list in input_ids_lists:
            for ids in id_list:
                unified_max_length = max(unified_max_length, len(ids))

        # Apply pad_to_multiple_of to the unified max length
        if self.pad_to_multiple_of is not None:
            unified_max_length = (
                (unified_max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of

        padded_inputs = []
        padded_masks = []
        
        # Pad all model inputs to the same unified_max_length
        for i, tokenizer in enumerate(self.tokenizers):
            batch = tokenizer.pad(
                {"input_ids": input_ids_lists[i], "attention_mask": attention_mask_lists[i]},
                padding="max_length",  # Force padding to the specified max_length
                max_length=unified_max_length,
                return_tensors="pt"
            )
            padded_inputs.append(batch["input_ids"])
            padded_masks.append(batch["attention_mask"])
        
        # Pad labels and linguistic features to the same unified_max_length
        padded_labels = []
        padded_pos_ids = []
        padded_dep_ids = []
        
        for l, p, d in zip(labels, pos_tag_ids, dep_rel_ids):
            padding_length = unified_max_length - len(l)
            padded_labels.append(l + [-100] * padding_length)
            padded_pos_ids.append(p + [-100] * padding_length)
            padded_dep_ids.append(d + [-100] * padding_length)

        batch = {
            "input_ids_list": padded_inputs,
            "attention_mask_list": padded_masks,
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "pos_tag_ids": torch.tensor(padded_pos_ids, dtype=torch.long),
            "dep_rel_ids": torch.tensor(padded_dep_ids, dtype=torch.long)
        }
        
        return batch


class AdvancedCRFTrainer(Trainer):
    """Custom trainer with advanced training techniques"""
    
    def __init__(self, use_adversarial_training=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adversarial_training = use_adversarial_training
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to add adversarial training"""
        inputs["adversarial_training"] = self.use_adversarial_training
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to use CRF decoding"""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Use CRF decoding
            predictions = model.predict(
                input_ids_list=inputs["input_ids_list"],
                attention_mask_list=inputs["attention_mask_list"],
                pos_tag_ids=inputs.get("pos_tag_ids"),
                dep_rel_ids=inputs.get("dep_rel_ids")
            )
            
            # Convert predictions to tensor
            batch_size = inputs["input_ids_list"][0].size(0)
            seq_len = inputs["input_ids_list"][0].size(1)
            predictions_tensor = torch.full((batch_size, seq_len), -100, dtype=torch.long)
            
            for i, pred_seq in enumerate(predictions):
                predictions_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, dtype=torch.long)
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, predictions_tensor, inputs.get("labels"))


def compute_metrics(eval_preds: EvalPrediction, id2label: Dict[int, str]) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = eval_preds.predictions, eval_preds.label_ids
    
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    
    true_predictions = [
        [id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    report = classification_report(
        true_labels,
        true_predictions,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }
    
    # Add per-class F1 scores
    for label in ["B-conn", "I-conn", "O"]:
        if label in report:
            metrics[f"f1_{label}"] = report[label]["f1-score"]
    
    return metrics


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Set up environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Determine device
    if args.no_cuda:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load tokenizers
    logger.info("Loading tokenizers...")
    tokenizers = [AutoTokenizer.from_pretrained(name) for name in args.model_names]
    
    # Load data
    train_examples, dev_examples_by_corpus, test_files_by_corpus, label_list, pos_list, dep_list = load_data(args.data_dir)
    
    # Create mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    pos2id = {"UNK": 0}
    pos2id.update({pos: i+1 for i, pos in enumerate(pos_list)})
    id2pos = {i: pos for pos, i in pos2id.items()}
    
    dep2id = {"UNK": 0}
    dep2id.update({dep: i+1 for i, dep in enumerate(dep_list)})
    id2dep = {i: dep for dep, i in dep2id.items()}
    
    # Compute class weights
    class_weights = compute_class_weights(train_examples, label2id)
    logger.info(f"Class weights: {class_weights}")
    
    # Create datasets
    train_dataset = HFDataset.from_list(train_examples)
    dev_datasets = {
        corpus: HFDataset.from_list(examples)
        for corpus, examples in dev_examples_by_corpus.items()
    }
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels_multi(x, tokenizers, label2id, pos2id, dep2id, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    dev_datasets_tokenized = {}
    for corpus, dataset in dev_datasets.items():
        dev_datasets_tokenized[corpus] = dataset.map(
            lambda x: tokenize_and_align_labels_multi(x, tokenizers, label2id, pos2id, dep2id, args.max_length),
            batched=True,
            remove_columns=dataset.column_names
        )
    
    combined_dev = concatenate_datasets(list(dev_datasets_tokenized.values()))
    
    # Initialize model
    logger.info(f"Initializing dependency-aware ensemble model with {len(args.model_names)} models")
    model = DependencyAwareEnsembleModelWithCRF(
        model_names=args.model_names,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
        num_pos_tags=len(pos2id),
        num_dep_rels=len(dep2id),
        pos_embedding_dim=args.pos_embedding_dim,
        dep_embedding_dim=args.dep_embedding_dim,
        ensemble_method=args.ensemble_method,
        use_focal_loss=args.use_focal_loss,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights.to(device) if device != "cpu" else class_weights
    )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        optim="adamw_torch_fused" if device != "cpu" else "adamw_torch",
        report_to="tensorboard",
        push_to_hub=False,
        fp16_full_eval=False,
        prediction_loss_only=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        local_rank=args.local_rank,
    )
    
    # Data collator
    data_collator = AdvancedEnsembleDataCollator(
        tokenizers=tokenizers,
        pad_to_multiple_of=16
    )
    
    # Initialize trainer
    trainer = AdvancedCRFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=combined_dev,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        use_adversarial_training=args.use_adversarial_training,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on individual dev sets
    logger.info("\nEvaluating on individual development sets...")
    best_f1 = 0
    results_summary = {}
    
    for corpus, dataset in dev_datasets_tokenized.items():
        logger.info(f"\nEvaluating {corpus}...")
        metrics = trainer.evaluate(eval_dataset=dataset)
        logger.info(f"{corpus} - F1: {metrics['eval_f1']:.4f}")
        # logger.info(f"  - B-conn F1: {metrics.get('eval_f1_B-conn', 0):.4f}")
        # logger.info(f"  - I-conn F1: {metrics.get('eval_f1_I-conn', 0):.4f}")
        
        results_summary[corpus] = metrics
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
    
    logger.info(f"\nBest Dev F1: {best_f1:.4f}")
    
    # Save model
    logger.info(f"\nSaving model to {args.output_dir}/final_model")
    os.makedirs(f"{args.output_dir}/final_model", exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), f"{args.output_dir}/final_model/model.pt")
    
    # Save tokenizers
    for i, tokenizer in enumerate(tokenizers):
        tokenizer.save_pretrained(f"{args.output_dir}/final_model/tokenizer_{i}")
    
    # Save config
    config = {
        "model_names": args.model_names,
        "num_labels": len(label_list),
        "label2id": label2id,
        "id2label": id2label,
        "pos2id": pos2id,
        "id2pos": id2pos,
        "dep2id": dep2id,
        "id2dep": id2dep,
        "max_length": args.max_length,
        "ensemble_method": args.ensemble_method,
        "best_dev_f1": best_f1,
        "results_summary": results_summary,
        "args": vars(args),  # Save all arguments
        "seed": args.seed,
    }
    
    with open(f"{args.output_dir}/final_model/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Test evaluation
    logger.info("\nRunning test set evaluation...")
    os.makedirs(f"{args.output_dir}/predictions", exist_ok=True)
    
    for corpus, test_files in test_files_by_corpus.items():
        for test_file in test_files:
            logger.info(f"\nProcessing {corpus} test set: {Path(test_file).name}")
            pred_file = f"{args.output_dir}/predictions/{Path(test_file).name}"
            
            generate_test_predictions(
                model,
                tokenizers,
                test_file,
                pred_file,
                pos2id=pos2id,
                dep2id=dep2id,
                max_length=args.max_length,
                device=device
            )
            
            # Run evaluation if gold file exists
            if "test" in test_file and os.path.exists(test_file):
                run_evaluation(pred_file, test_file, corpus)
    
    logger.info("\nTraining and evaluation complete!")


def generate_test_predictions(
    model: DependencyAwareEnsembleModelWithCRF,
    tokenizers: List[PreTrainedTokenizerBase],
    test_file: str,
    output_file: str,
    pos2id: Dict[str, int],
    dep2id: Dict[str, int],
    max_length: int = 512,
    device: str = "cuda"
) -> None:
    """Generate predictions for test file in CoNLL-U format."""
    model.to(device)
    model.eval()
    
    test_examples = parse_conllu_file(test_file)
    id2label = model.id2label
    all_predictions = []
    
    # Process in batches for efficiency
    batch_size = 32
    for batch_start in range(0, len(test_examples), batch_size):
        batch_examples = test_examples[batch_start:batch_start + batch_size]
        batch_predictions = []
        
        for example in batch_examples:
            # Tokenize with all tokenizers
            tokenized_list = []
            for tokenizer in tokenizers:
                tokenized = tokenizer(
                    example["tokens"],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
                tokenized_list.append(tokenized)
            
            # Align POS and dependency tags using first tokenizer
            word_ids = tokenized_list[0].word_ids()
            pos_ids = []
            dep_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None or word_idx == previous_word_idx:
                    pos_ids.append(-100)
                    dep_ids.append(-100)
                else:
                    pos_ids.append(pos2id.get(example["pos_tags"][word_idx], pos2id.get("UNK", 0)))
                    dep_ids.append(dep2id.get(example["dep_rels"][word_idx], dep2id.get("UNK", 0)))
                previous_word_idx = word_idx
            
            # Move to device
            input_ids_list = [t["input_ids"].to(device) for t in tokenized_list]
            attention_mask_list = [t["attention_mask"].to(device) for t in tokenized_list]
            pos_tag_ids = torch.tensor([pos_ids]).to(device)
            dep_rel_ids = torch.tensor([dep_ids]).to(device)
            
            # Get predictions
            pred_ids = model.predict(input_ids_list, attention_mask_list, pos_tag_ids, dep_rel_ids)[0]
            
            # Align predictions to original tokens
            aligned_predictions = []
            previous_word_idx = None
            
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx != previous_word_idx:
                    if idx < len(pred_ids):
                        aligned_predictions.append(id2label[pred_ids[idx]])
                    previous_word_idx = word_idx
            
            batch_predictions.append(aligned_predictions)
        
        all_predictions.extend(batch_predictions)
    
    # Debug: Print prediction distribution
    all_preds_flat = [pred for preds in all_predictions for pred in preds]
    pred_counts = Counter(all_preds_flat)
    logger.info(f"Prediction distribution: {dict(pred_counts)}")
    logger.info(f"Total predictions: {len(all_preds_flat)}")
    
    # Write predictions
    write_predictions_conllu(test_file, output_file, all_predictions)


def write_predictions_conllu(
    input_file: str,
    output_file: str,
    predictions: List[List[str]]
) -> None:
    """Write predictions to CoNLL-U file."""
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        sent_idx = 0
        token_idx = 0
        
        for line in fin:
            line = line.strip()
            
            if not line or line.startswith('#'):
                fout.write(line + '\n')
                if not line and token_idx > 0:
                    sent_idx += 1
                    token_idx = 0
                continue
            
            parts = line.split('\t')
            if len(parts) >= 10 and '-' not in parts[0]:
                if sent_idx < len(predictions) and token_idx < len(predictions[sent_idx]):
                    pred = predictions[sent_idx][token_idx]
                    if pred == "B-conn":
                        parts[9] = "Conn=B-conn"
                    elif pred == "I-conn":
                        parts[9] = "Conn=I-conn"
                    else:
                        parts[9] = "_"
                    token_idx += 1
                else:
                    parts[9] = "_"
                
                fout.write('\t'.join(parts) + '\n')
            else:
                fout.write(line + '\n')


def run_evaluation(pred_file: str, gold_file: str, corpus_name: str) -> Optional[Dict]:
    """Run official evaluation script."""
    cmd = [
        "python", "../sharedtask2025/utils/disrpt_eval_2024.py",
        "-g", gold_file,
        "-p", pred_file,
        "-t", "C"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = json.loads(result.stdout)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluation Results for {corpus_name}")
        logger.info(f"{'='*50}")
        logger.info(f"Precision: {output['precision']:.4f}")
        logger.info(f"Recall: {output['recall']:.4f}")
        logger.info(f"F-Score: {output['f_score']:.4f}")
        
        return output
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed for {corpus_name}: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation output: {e}")
        return None


if __name__ == "__main__":
    main()