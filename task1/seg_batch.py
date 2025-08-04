# -*- coding: utf-8 -*-


import logging
import torch
import pandas as pd
import numpy as np
import random
import argparse
import json
from datasets import Dataset, DatasetDict
from functools import partial
from utils import *
import os
from collections import Counter
from torch.nn import CrossEntropyLoss

# Transformers, PEFT, bitsandbytes
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training

from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_datasets", nargs='+', required=True)
    parser.add_argument("--test_datasets", nargs='+', required=True)

    parser.add_argument("--output_model_path", required=True, type=str)
    # parser.add_argument("--output_prediction_path", required=True, type=str)
    parser.add_argument("--model_choice", default="bigscience/bloom-3b", type=str)

    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    # training hyper
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--learning_rate", default=2e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    # LoRA
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)

    parser.add_argument("--lora_target_modules", nargs='+', required=True,
                        help="Lora target: 'query_key_value' (BLOOM) or 'q_proj' 'v_proj' (Llama/Gemma)")

    parser.add_argument("--use_weighted_loss", action="store_true", help="weighted loss")
    parser.add_argument("--use_fgm", action="store_true", help="FGM for adv")
    parser.add_argument("--fgm_epsilon", type=float, default=1.0)

    return parser


def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def chunk_sentences(sentences, labels, max_length=640):
    final_sentences = []
    final_labels = []

    for original_sent, original_label in zip(sentences, labels):
        chunks_to_process_for_this_sent = [(original_sent, original_label)]

        i = 0
        while i < len(chunks_to_process_for_this_sent):
            current_chunk, current_chunk_label = chunks_to_process_for_this_sent[i]

            if len(current_chunk) > max_length:
                sep_point = len(current_chunk) // 2

                chunk_1 = current_chunk[:sep_point]
                label_1 = current_chunk_label[:sep_point]

                chunk_2 = current_chunk[sep_point:]
                label_2 = current_chunk_label[sep_point:]

                chunks_to_process_for_this_sent[i] = (chunk_1, label_1)
                chunks_to_process_for_this_sent.insert(i + 1, (chunk_2, label_2))

            else:
                i += 1

        for final_chunk, final_chunk_label in chunks_to_process_for_this_sent:
            final_sentences.append(final_chunk)
            final_labels.append(final_chunk_label)

    return final_sentences, final_labels


def get_raw_data(file_path):
    sentences, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_labels"]
                doc_sents = sample["doc_sents"]
                for i in range(len(doc_sent_token_labels)):
                    assert len(doc_sents[i]) == len(doc_sent_token_labels[i])

                    original_labels = doc_sent_token_labels[i]

                    corrected_labels = []
                    for label in original_labels:
                        if label == "O":
                            corrected_labels.append("Seg=O")
                        elif label.startswith("B-") or label.startswith("I-"):
                            corrected_labels.append(f"Seg={label}")
                        else:
                            corrected_labels.append(label)

                    sentences.append(doc_sents[i])
                    labels.append(corrected_labels)

    if "fra.sdrt.summre" in file_path:
        new_sentences, new_labels = chunk_sentences(sentences, labels)
        return new_sentences, new_labels

    return sentences, labels


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


class PeftTrainer(Trainer):
    def __init__(self, *args, custom_args=None, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args
        if class_weights is not None:
            logger.info("Weighted loss is activate!!!!!")
            self.loss_fct = CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fct = CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        loss = self.loss_fct(logits.to(torch.float32).view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not self.custom_args.use_fgm:
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)

        fgm = FGM(model)
        emb_name = "embeddings"
        fgm.attack(epsilon=self.custom_args.fgm_epsilon, emb_name=emb_name)

        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss_adv)

        fgm.restore(emb_name=emb_name)

        return loss.detach() / self.args.gradient_accumulation_steps


def prepare_features(examples, tokenizer, label2id, max_seq_length):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,
                                 max_length=max_seq_length)
    all_label_ids = []
    for i, label_list in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None;
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label_list[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_label_ids.append(label_ids)
    tokenized_inputs["labels"] = all_label_ids
    return tokenized_inputs


def train_model(args, train_dataset, eval_dataset, label_maps, compute_metrics_fn, class_weights=None):
    logger.info("\n--- Training... ---")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForTokenClassification.from_pretrained(args.model_choice, num_labels=len(label_maps["id2label"]),
                                                            id2label=label_maps["id2label"],
                                                            label2id=label_maps["label2id"],
                                                            quantization_config=quantization_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    logger.info(f"LoRA Config: r={args.lora_r}, alpha={args.lora_alpha}, target_modules={args.lora_target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
        target_modules=args.lora_target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_choice, add_prefix_space=True)
    tokenizer.padding_side = 'right'
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir=f"{args.output_model_path}/checkpoints",
                                      num_train_epochs=args.num_train_epochs,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                                      warmup_ratio=args.warmup_ratio, max_grad_norm=args.max_grad_norm,
                                      logging_dir=f"{args.output_model_path}/logs", logging_strategy="epoch",
                                      eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                                      metric_for_best_model="overall_f1", greater_is_better=True)

    trainer = PeftTrainer(
        model=model,
        args=training_args,
        custom_args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
    )

    logger.info("Fine-tuning started...")
    trainer.train()
    logger.info(f"Training done! Saving model in -> {args.output_model_path}")
    trainer.save_model(args.output_model_path)


def test_model(args, test_dataset, compute_metrics_fn, tok_file_name, label_maps, output_prediction_path):
    logger.info("\n--- Test and Save ---")
    logger.info(f"\n--- Test on -> {args.current_test_dataset} ---")

    logger.info(f"Loading {args.model_choice} ...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_choice, num_labels=len(label_maps["id2label"]),
                                                            id2label=label_maps["id2label"],
                                                            label2id=label_maps["label2id"],
                                                            quantization_config=quantization_config, device_map="auto")
    logger.info(f"Loading {args.output_model_path} ...")
    model = PeftModel.from_pretrained(model, args.output_model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_choice, add_prefix_space=True)
    tokenizer.padding_side = 'right'
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = PeftTrainer(model=model, compute_metrics=compute_metrics_fn, tokenizer=tokenizer,
                          data_collator=data_collator)

    logger.info("Evaluation on test set...")
    predictions, labels, metrics = trainer.predict(test_dataset)

    logger.info("\n--- test performance ---")
    for key, value in metrics.items():
        metric_name = key.replace('_', ' ').replace('test', 'Test').title()
        logger.info(f"{metric_name}: {value:.4f}")

    preds_list = np.argmax(predictions, axis=2)
    id2label_map = label_maps['id2label']

    true_tokens_list = test_dataset['tokens']
    final_all_tokens = []
    final_all_labels = []

    for i in range(len(labels)):
        current_tokens = true_tokens_list[i]
        aligned_preds = []
        for pred_id, true_id in zip(preds_list[i], labels[i]):
            if true_id != -100:
                aligned_preds.append(id2label_map[pred_id])

        if len(current_tokens) == len(aligned_preds):
            final_all_tokens.extend(current_tokens)
            final_all_labels.extend(aligned_preds)
        else:
            logger.warning(
                f"Sample tokens:({len(current_tokens)}) and aligned tokens: ({len(aligned_preds)}) not matched! ")

    seg_preds_to_file(final_all_tokens, final_all_labels, tok_file_name, args.output_model_path)
    temp_tok_name = tok_file_name.split("/")[-1]
    pred_file = f"{args.output_model_path}/{temp_tok_name}".replace(".tok", "_pred.tok")
    my_eval = SegmentationEvaluation(args.current_test_dataset, tok_file_name, pred_file, False, False)
    my_eval.compute_scores()
    my_eval.print_results()

    # save step
    results_output = my_eval.output
    with open(output_prediction_path, 'w') as f:
        f.write(f"Results for dataset: {args.current_test_dataset}\n")
        f.write(f"Model: {args.model_choice}\n")
        f.write("=" * 30 + "\n")
        for key, value in results_output.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Save report -> {output_prediction_path}")


def main():
    args = get_argparse().parse_args()
    set_seed(args.seed);
    logger.info("Training/evaluation parameters %s", args)
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    all_train_sentences, all_train_labels = [], []
    all_dev_sentences, all_dev_labels = [], []

    for dataset_name in args.train_datasets:
        logger.info(f"Loading the dataset: {dataset_name}...")
        data_dir_path = f"../data/{dataset_name}"
        train_file = f"{data_dir_path}/{dataset_name}_train.json"
        dev_file = f"{data_dir_path}/{dataset_name}_dev.json"

        train_sents, train_lbls = get_raw_data(train_file)
        dev_sents, dev_lbls = get_raw_data(dev_file)

        all_train_sentences.extend(train_sents)
        all_train_labels.extend(train_lbls)
        all_dev_sentences.extend(dev_sents)
        all_dev_labels.extend(dev_lbls)

    logger.info("--- Data analysis ---")
    for i, (tokens, labels) in enumerate(zip(all_train_sentences, all_train_labels)):
        if len(tokens) != len(labels): raise ValueError(f"Data length {i} not matched")

    full_labels_set = set(label for sublist in all_train_labels + all_dev_labels for label in sublist)
    all_labels_flat = sorted(list(full_labels_set))
    label2id = {label: i for i, label in enumerate(all_labels_flat)}
    id2label = {i: label for i, label in enumerate(all_labels_flat)}
    label_maps = {"label2id": label2id, "id2label": id2label}

    class_weights_tensor = None
    if args.use_weighted_loss:
        flat_train_labels = [label for sublist in all_train_labels for label in sublist]
        label_counts = Counter(flat_train_labels)
        total_samples = len(flat_train_labels)

        num_classes = len(label_maps['id2label'])
        id2label_map = label_maps['id2label']

        class_weights = []
        for i in range(num_classes):
            label_name = id2label_map[i]
            count = label_counts.get(label_name, 1)
            weight = total_samples / (num_classes * count)
            class_weights.append(weight)

        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(args.device)

        logger.info(f"Class weight computed: {class_weights_tensor}")

    train_dataset = Dataset.from_dict({"tokens": all_train_sentences, "labels": all_train_labels}).filter(
        lambda ex: len(ex['tokens']) > 0)
    dev_dataset = Dataset.from_dict({"tokens": all_dev_sentences, "labels": all_dev_labels}).filter(
        lambda ex: len(ex['tokens']) > 0)

    split_datasets_for_tokenization = DatasetDict({'train': train_dataset, 'validation': dev_dataset})

    tokenizer = AutoTokenizer.from_pretrained(args.model_choice, add_prefix_space=True)
    tokenizer.padding_side = 'right'
    processing_func = partial(prepare_features, tokenizer=tokenizer, label2id=label2id,
                              max_seq_length=args.max_seq_length)
    tokenized_datasets = split_datasets_for_tokenization.map(processing_func, batched=True)

    def compute_metrics(p):
        predictions, labels = p;
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in
                            zip(predictions, labels)]
        true_labels = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in
                       zip(predictions, labels)]
        return {"accuracy_score": accuracy_score(true_labels, true_predictions),
                "precision": precision_score(true_labels, true_predictions, average="micro", zero_division=0),
                "recall": recall_score(true_labels, true_predictions, average="micro", zero_division=0),
                "overall_f1": f1_score(true_labels, true_predictions, average="micro", zero_division=0)}

    if args.do_train:
        train_model(
            args,
            tokenized_datasets["train"],
            tokenized_datasets["validation"],
            label_maps,
            compute_metrics,
            class_weights=class_weights_tensor
        )

    if args.do_test:
        logger.info("\n--- Test on indiv dataset ---")
        for test_dataset_name in args.test_datasets:
            args.current_test_dataset = test_dataset_name

            # load test data
            test_data_dir_path = f"../data/{test_dataset_name}"
            test_file = f"{test_data_dir_path}/{test_dataset_name}_test.json"
            test_tok_file = f"{test_data_dir_path}/{test_dataset_name}_test.tok"
            test_sentences, test_labels = get_raw_data(test_file)

            test_dataset = Dataset.from_dict({"tokens": test_sentences, "labels": test_labels}).filter(
                lambda ex: len(ex['tokens']) > 0)
            tokenized_test_dataset = test_dataset.map(processing_func, batched=True)

            current_prediction_path = os.path.join(args.output_model_path, f"scores_{test_dataset_name}.txt")

            test_model(
                args,
                tokenized_test_dataset,
                compute_metrics,
                test_tok_file,
                label_maps,
                output_prediction_path=current_prediction_path
            )

    logger.info("\n======= Pipeline finished! =======")


if __name__ == "__main__":
    main()