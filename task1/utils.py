import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
# import fasttext
# import fasttext.util
import io, os, sys, argparse
import json
from sklearn.metrics import accuracy_score, classification_report

from seg_eval import get_scores


def token_labels_from_file(file_name):
    labels = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_labels"]
                for sent_token_labels in doc_sent_token_labels:
                    for l in sent_token_labels:
                        # labels.add(l.lower())
                        labels.add(l)
    labels = list(labels)
    labels = sorted(labels)
    print(" Total label number: %d\n" % (len(labels)))
    label_dict = {l: idx for idx, l in enumerate(labels)}
    # label_id_dict = {idx: l for idx, l in enumerate(labels)}
    return label_dict, labels


def token_pos_from_file(file_name):
    tok_pos_1 = set()
    tok_pos_2 = set()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                doc_sent_token_labels = sample["doc_sent_token_features"]
                for sent_token_labels in doc_sent_token_labels:
                    for feat in sent_token_labels:
                        tok_pos_1.add(feat[1])
                        tok_pos_2.add(feat[2])
    tok_pos_1 = list(tok_pos_1)
    tok_pos_2 = list(tok_pos_2)
    tok_pos_1 = sorted(tok_pos_1)
    tok_pos_2 = sorted(tok_pos_2)
    tok_pos_1_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_1)}
    tok_pos_2_dict = {t: idx + 1 for idx, t in enumerate(tok_pos_2)}
    tok_pos_1_dict["SEPCIAL_TOKEN"] = 0
    tok_pos_2_dict["SEPCIAL_TOKEN"] = 0
    return tok_pos_1, tok_pos_2, tok_pos_1_dict, tok_pos_2_dict


def unify_rel_labels(label, corpus_name):
    """
    Here we convert original label into an unified label
    Args:
        label: original label
        corpus_name:
    """
    if corpus_name == "eng.dep.covdtb":
        return label.lower()
    elif corpus_name == "eng.sdrt.stac":
        return label.lower()
    else:
        return label


def rel_label_to_original(label, corpus_name):
    """
    We remap the rel label to original one. Doing so, we can recall rel_eval
    Args:
        label:
        corpus_name:
    """
    if corpus_name == "eng.dep.covdtb":
        return label.upper()
    elif corpus_name == "eng.sdrt.stac":
        if label == "q_elab":
            return "Q_Elab"
        else:
            return label.capitalize()
    else:
        return label


def rel_map_for_zeroshot(label, dname):
    """
    Some zeroshot corpora contain totally different label set to
    corpora with the similar annotated theory. For example, eng.dep.covdtb
    has very different labels to eng.dep.scidtb and zho.dep.scidtb. So here
    we design a mapping function for such zero-shot corpora.

    Note, mapping only works for corpora without training set
    """
    if dname == "eng.dep.covdtb":
        mapping_dict = {
            'ATTRIBUTION': 'ATTRIBUTION', 'BG-COMPARE': 'BACKGROUND', 'BG-GENERAL': 'BACKGROUND',
            'BG-GOAL': 'BACKGROUND', 'CAUSE': 'CAUSE-RESULT', 'COMPARISON': 'COMPARISON',
            'CONDITION': 'CONDITION', 'CONTRAST': 'CONTRAST', 'ELAB-ADDITION': 'ELABORATION',
            'ELAB-ASPECT': 'ELABORATION', 'ELAB-DEFINITION': 'ELABORATION', 'ELAB-ENUMEMBER': 'ELABORATION',
            'ELAB-EXAMPLE': 'ELABORATION', 'ELAB-PROCESS_STEP': 'ELABORATION', 'ENABLEMENT': 'ENABLEMENT',
            'EVALUATION': 'EVALUATION', 'EXP-EVIDENCE': 'CAUSE-RESULT', 'EXP-REASON': 'CAUSE-RESULT',
            'JOINT': 'JOINT', 'MANNER-MEANS': 'MANNER-MEANS', 'PROGRESSION': 'PROGRESSION',
            'RESULT': 'CAUSE-RESULT', 'ROOT': 'ROOT', 'SUMMARY': 'SUMMARY', 'TEMPORAL': 'TEMPORAL'
        }
        return mapping_dict[label]
    elif dname in ["por.pdtb.tedm", "eng.pdtb.tedm", "tur.pdtb.tedm"]:
        mapping_dict = {
            "QAP.Hypophora": "Hypophora", "QAP": "Hypophora", "Expansion.Level": "Expansion.Level-of-detail",
            "Comparison": "Comparison.Concession", "Temporal": "Temporal.Synchronous"
        }
        if label in mapping_dict:
            return mapping_dict[label]
        else:
            return label


def rel_labels_from_file(file_name):
    label_frequency = defaultdict(int)
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                dname = sample["dname"]
                doc_unit_labels = sample["doc_unit_labels"]
                for label_pair in doc_unit_labels:
                    l = label_pair[0]
                    label_frequency[unify_rel_labels(l, dname)] += 1
    labels = []
    for key in label_frequency:
        if label_frequency[key] >= 0:
            labels.append(key)
    labels = sorted(labels, key=lambda x: x.upper())
    label_dict = {l: idx for idx, l in enumerate(labels)}
    # print(labels)
    # print(label_dict)
    # print(" Total label number: %d\n"%(len(labels)))

    return label_dict, labels


def process_fra_files(conllu_file, tok_file):
    sentence_lengths = []
    current_sentence_length = 0
    with open(conllu_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# sent_id'):
                if current_sentence_length > 0:
                    sentence_lengths.append(current_sentence_length)
                current_sentence_length = 0
            elif re.match(r'^\d+\t', line):
                current_sentence_length += 1
    if current_sentence_length > 0:
        sentence_lengths.append(current_sentence_length)

    all_tokens = []
    all_labels = []
    with open(tok_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split('\t')
            all_tokens.append(parts[1])

            seg_part = parts[-1]
            label = seg_part.split('=')[-1]
            all_labels.append(label)

    sentences_tokens = []
    sentences_labels = []
    current_pos = 0
    for length in sentence_lengths:
        sentences_tokens.append(all_tokens[current_pos: current_pos + length])
        sentences_labels.append(all_labels[current_pos: current_pos + length])
        current_pos += length

    return sentences_tokens, sentences_labels




def seg_preds_to_file(og_tokens, pred_labels, gold_file, save_dir):
    """
    new version of writing a result tok file
    convert prediction ids to labels, and save the results into a file with the same format as gold_file
    Args:
        all_input_ids: predicted tokens' id list
        all_label_ids: predicted labels' id list
        all_attention_mask: attention mask of the pre-trained LM
        label_id_dict: the dictionary map the labels' id to the original string label
        gold_file: the original .tok file
    """
    all_doc_data = []
    new_doc_data = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            all_doc_data.append(line)

    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            # if "newdoc_id" in line.lower():
            if "newdoc id" in line.lower() or "newdoc_id" in line.lower() or "newdoc" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0] or "." in items[0]:  # ignore such as 16-17
                    continue

                if items[1] != og_tokens[pointer]:
                    print(og_tokens[pointer])
                    print(items)
                    print(og_tokens[pointer - 5: pointer])
                    print(og_tokens[pointer: pointer + 5])
                    raise RuntimeError("111")
                items[-1] = pred_labels[pointer]
                items[-2] = og_tokens[pointer]

                new_doc_data.append("\t".join(items))
                pointer += 1

        else:
            new_doc_data.append('\n')

    gold_file_name = gold_file.split("/")[-1]
    pred_file = gold_file_name.replace(".tok", "_pred.tok")
    pred_file_path = f"{save_dir}/{pred_file}"
    with open(pred_file_path, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    return pred_file


def merge4bag(data_path, gold_tok_file):
    file_names = os.listdir(data_path)
    bagging_files = [data_path + "/" + file_name for file_name in file_names if "test_pred_bag" in file_name]
    result_list = []
    # read all pred files
    for bagging_file in bagging_files:
        with open(bagging_file, "r", encoding="utf-8") as f:
            print(bagging_file)
            lines = f.readlines()
            temp = []
            for line in lines:
                if line != '\n':
                    if "newdoc_id" in line.lower() or "newdoc" in line.lower():
                        continue
                    else:
                        items = line.split("\t")
                        if "-" in items[0]:  # ignore such as 16-17
                            continue
                        temp.append(items[-1])
        print(len(temp))
        print("===============================================================")
        result_list.append(temp)

    merge_res = []
    # select the most voted label
    for i in range(len(result_list[0])):
        temp = []
        for j in range(len(result_list)):
            temp.append(result_list[j][i])
        merge_res.append(temp)
    final_res = []
    for k in range(len(merge_res)):
        most_voted = max(merge_res[k], key=merge_res[k].count)
        final_res.append(most_voted)
    all_doc_data = []
    new_doc_data = []
    with open(gold_tok_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            all_doc_data.append(line)

    pointer = 0
    for line in all_doc_data:
        if line != '\n':
            # if "newdoc_id" in line.lower():
            if "newdoc id" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue

                items[-1] = final_res[pointer]
                # print(items)
                new_doc_data.append("\t".join(items))
                pointer += 1

        else:
            new_doc_data.append('\n')

    pred_file = gold_tok_file.replace(".tok", "_pred_bg.tok")
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    return pred_file


def merge_result(gold_tok_file):
    file_names = os.listdir(gold_tok_file)
    bagging_files = [gold_tok_file + "/" + file_name for file_name in file_names if "test_pred_bag" in file_name]
    result_list = []
    for bagging_file in bagging_files:
        with open(bagging_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            temp = []
            for line in lines:
                if line != '\n':
                    if "newdoc_id" in line.lower():
                        continue
                    else:
                        items = line.split("\t")
                        if "-" in items[0]:  # ignore such as 16-17
                            continue
                        temp.append(items)
        result_list.append(temp)

    all_preds = [[] for i in range(len(result_list[0]))]
    for sub_res in result_list:
        for i in range(len(sub_res)):
            all_preds[i].append(sub_res[i][-1])
    final_res = []
    for pred in all_preds:
        most_pred = max(set(pred), key=pred.count)
        final_res.append(most_pred)

    new_doc_data = []
    pointer = 0
    for line in result_list[0]:
        line = " ".join(line)
        if line != '\n':
            # if "newdoc_id" in line.lower():
            if "newdoc" in line.lower() or "newdoc_id" in line.lower():
                new_doc_data.append(line)
            else:
                items = line.split("\t")
                if "-" in items[0]:  # ignore such as 16-17
                    continue
                items[-1] = final_res[pointer]
                # items[-2] = og_tokens[pointer]
                new_doc_data.append("\t".join(items))
                pointer += 1
        else:
            new_doc_data.append('\n')

    pred_file = bagging_files[0].replace("bag1.tok", "bag_final.tok")
    gold_tok = "/hits/basement/nlp/yif/disrpt2023-main/data/dataset/deu.rst.pcc/deu.rst.pcc_test.tok"
    with open(pred_file, "w") as f:
        for line in new_doc_data:
            if line[-1:] != "\n":
                f.write(line + "\n")
            else:
                f.write(line)
    print(gold_tok)
    print(pred_file)
    score_dict = get_scores(gold_tok, pred_file)
    return score_dict


def generate_ft_dict(train_file_path, dev_file_path, test_file_path, output_path, ft_model_path, ft_lang):
    all_files = [train_file_path, dev_file_path, test_file_path]
    # all_files = [dev_file_path, test_file_path]
    # fasttext.util.download_model(ft_lang, if_exists='ignore') 
    # ft = fasttext.load_model(ft_model_path)
    ft = fasttext.load_model(ft_model_path)
    all_texts = []
    token_list = []
    ft_dict = {}
    for path in all_files:
        with open(path, 'r') as f:
            for line in f.readlines():
                line_content = json.loads(line)
                all_texts.append(line_content)
        for doc in all_texts:
            doc_token_list = doc["doc_sents"]
            for i in range(len(doc_token_list)):
                for j in range(len(doc_token_list[i])):
                    token_list.append(doc_token_list[i][j])
    for i in range(len(token_list)):
        ft_dict[token_list[i]] = ft.get_word_vector(token_list[i])
    np.save(output_path, ft_dict)
    print(" Finish filtering the unrelated embedding from {}.".format(ft_model_path))
    return ft_dict


def rel_preds_to_file(pred_ids, label_list, gold_file):
    dname = gold_file.split("/")[-1].split("_")[0].strip()
    pred_labels = [label_list[idx] for idx in pred_ids]
    pred_labels = [rel_label_to_original(label, dname) for label in pred_labels]
    if dname in ["eng.dep.covdtb", "por.pdtb.tedm", "eng.pdtb.tedm", "tur.pdtb.tedm"]:
        pred_labels = [rel_map_for_zeroshot(label, dname) for label in pred_labels]
    valid_lines = []
    with open(gold_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        title_line = lines[0]
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                valid_lines.append(line)

    assert len(pred_labels) == len(valid_lines), (len(pred_labels), len(valid_lines))

    pred_contents = []
    for pred, line in zip(pred_labels, valid_lines):
        items = line.split("\t")
        new_items = items[:-1]
        new_items.append(pred)
        pred_contents.append("\t".join(new_items))

    pred_file = gold_file.replace(".rels", "_pred.rels")
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("%s\n" % (title_line.strip()))
        for text in pred_contents:
            f.write("%s\n" % (text))

    return pred_file


def fix_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model


def merge_datasets(discourse_type="rst"):
    """
    merge a group of corpus for training
    Args:
        dataset_list: corpus list
        mode:
    """
    out_dir = os.path.join("data/dataset", "super.{}".format(discourse_type))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "super.{}_train.json".format(discourse_type))

    if os.path.exists(out_file):
        return out_file

    if discourse_type == "rst":
        # we remove the zho.rst.gcdt because this corpus has no overlaping labels with other corpus
        dataset_list = [
            "deu.rst.pcc", "eng.rst.gum", "eng.rst.rstdt",
            "eus.rst.ert", "fas.rst.prstc", "nld.rst.nldt",
            "por.rst.cstn", "rus.rst.rrt", "spa.rst.rststb",
            "spa.rst.sctb", "zho.rst.sctb"
        ]
    elif discourse_type == "pdtb":
        # we remove the zho.pdtb.cdtb because this corpus has no common labels with other corpus
        dataset_list = [
            "ita.pdtb.luna", "tur.pdtb.tdb", "tha.pdtb.tdtb",
            "eng.pdtb.pdtb", "por.pdtb.crpc"
        ]
    elif discourse_type == "dep":
        dataset_list = ["eng.dep.scidtb", "zho.dep.scidtb"]
    elif discourse_type == "sdrt":
        dataset_list = ["eng.sdrt.stac", "fra.sdrt.annodis"]
    all_merged_texts = []
    for dataset in dataset_list:
        data_dir = os.path.join("data/dataset", dataset)
        data_file = os.path.join(data_dir, "{}_train.json".format(dataset))
        with open(data_file, "r", encoding="utf-8") as f:
            all_texts = f.readlines()
            for text in all_texts:
                text = text.strip()
                if text:
                    sample = json.loads(text)
                    doc_units = sample["doc_units"]
                    doc_unit_labels = sample["doc_unit_labels"]
                    corpus_name = dataset
                    new_doc_unit_labels = []
                    for label in doc_unit_labels:
                        new_doc_unit_labels.append(label)

                    new_sample = {}
                    new_sample["dname"] = corpus_name
                    new_sample["doc_units"] = doc_units
                    new_sample["doc_unit_labels"] = new_doc_unit_labels
                    all_merged_texts.append(new_sample)

    with open(out_file, "w", encoding="utf-8") as f:
        for text in all_merged_texts:
            f.write("%s\n" % (json.dumps(text, ensure_ascii=False)))

    return out_file


class Evaluation:
    """
    Generic class for evaluation between 2 files.
    :load data, basic check, basic metrics, print results.
    """

    def __init__(self, name: str) -> None:
        self.output = dict()
        self.name = name
        self.report = ""
        self.fill_output('doc_name', self.name)

    def get_data(self, infile: str, str_i=False) -> str:
        """
        Stock data from file or stream.
        """
        if str_i == False:
            data = io.open(infile, encoding="utf-8").read().strip().replace("\r", "")
        else:
            data = infile.strip()
        return data

    def fill_output(self, key: str, value) -> None:
        """
        Fill results dict that will be printed.
        """
        self.output[key] = value

    def check_tokens_number(self, g: list, p: list) -> None:
        """
        Check same number of tokens/labels in both compared files.
        """
        if len(g) != len(p):
            self.report += "\nFATAL: different number of tokens detected in gold and pred:\n"
            self.report += ">>>  In " + self.name + ": " + str(len(g)) + " gold tokens but " + str(
                len(p)) + " predicted tokens\n\n"
            sys.stderr.write(self.report)
            sys.exit(0)

    def check_identical_tokens(self, g: list, p: list) -> None:
        """
        Check tokens/features are identical.
        """
        for i, tok in enumerate(g):
            if tok != p[i]:
                self.report += "\nWARN: token strings do not match in gold and pred:\n"
                self.report += ">>> First instance in " + self.name + " token " + str(i) + "\n"
                self.report += "Gold: " + tok + " but Pred: " + p[i] + "\n\n"
                sys.stderr.write(self.report)
                break

    def compute_PRF_metrics(self, tp: int, fp: int, fn: int) -> None:
        """
        Compute Precision, Recall, F-score from True Positive, False Positive and False Negative counts.
        Save result in dict.
        """
        try:
            precision = tp / (float(tp) + fp)
        except Exception as e:
            precision = 0

        try:
            recall = tp / (float(tp) + fn)
        except Exception as e:
            recall = 0

        try:
            f_score = 2 * (precision * recall) / (precision + recall)
        except:
            f_score = 0

        self.fill_output("gold_count", tp + fn)
        self.fill_output("pred_count", tp + fp)
        self.fill_output("precision", precision)
        self.fill_output("recall", recall)
        self.fill_output("f_score", f_score)

    def compute_accuracy(self, g: list, p: list, k: str) -> None:
        """
        Compute accuracy of predictions list of items, against gold list of items.
        :g: gold list
        :p: predicted list
        :k: name detail of accuracy
        """
        self.fill_output(f"{k}_accuracy", accuracy_score(g, p))
        self.fill_output(f"{k}_gold_count", len(g))
        self.fill_output(f"{k}_pred_count", len(p))

    def classif_report(self, g: list, p: list, key: str) -> None:
        """
        Compute Precision, Recall and f-score for each instances of gold list.
        """
        stats_dict = classification_report(g, p, labels=sorted(set(g)), zero_division=0.0, output_dict=True)
        self.fill_output(f'{key}_classification_report', stats_dict)

    def print_results(self) -> None:
        """
        Print dict of saved results.
        """
        # for k in self.output.keys():
        # print(f">> {k} : {self.output[k]}")

        print(json.dumps(self.output, indent=4))


class SegmentationEvaluation(Evaluation):
    """
    Specific evaluation class for EDUs segmentation.
    :parse conllu-style data
    :eval upon first token identification
    """
    LAB_SEG_B = "Seg=B-seg"  # "BeginSeg=Yes"
    LAB_SEG_I = "Seg=O"  # "_"

    def __init__(self, name: str, gold_path: str, pred_path: str, str_i=False, no_b=False) -> None:
        super().__init__(name)
        """
        :param gold_file: Gold shared task file
        :param pred_file: File with predictions
        :param string_input: If True, files are replaced by strings with file contents (for import inside other scripts)
        """
        self.mode = "edu"
        self.seg_type = "EDUs"
        self.g_path = gold_path
        self.p_path = pred_path
        self.opt_str_i = str_i
        self.no_b = True if "conllu" in gold_path.split(os.sep)[
            -1] and no_b == True else False  # relevant only in conllu

        self.fill_output('seg_type', self.seg_type)
        self.fill_output("options", {"s": self.opt_str_i})

    def compute_scores(self) -> None:
        """
        Get lists of data to compare, compute metrics.
        """
        gold_tokens, gold_labels, gold_spans = self.parse_edu_data(self.g_path, self.opt_str_i, self.no_b)
        pred_tokens, pred_labels, pred_spans = self.parse_edu_data(self.p_path, self.opt_str_i, self.no_b)

        self.output['tok_count'] = len(gold_tokens)

        self.check_tokens_number(gold_tokens, pred_tokens)
        self.check_identical_tokens(gold_tokens, pred_tokens)
        tp, fp, fn = self.compare_labels(gold_labels, pred_labels)
        self.compute_PRF_metrics(tp, fp, fn)

    def compare_labels(self, gold_labels: list, pred_labels: list) -> tuple[int, int, int]:
        """

        """
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for i, gold_label in enumerate(gold_labels):  # not verified
            pred_label = pred_labels[i]
            if gold_label == pred_label:
                if gold_label == "_":
                    continue
                else:
                    true_positive += 1
            else:
                if pred_label == "_":
                    false_negative += 1
                else:
                    if gold_label == "_":
                        false_positive += 1
                    else:  # I-Conn/B-Conn mismatch ?
                        false_positive += 1

        return true_positive, false_positive, false_negative

    def parse_edu_data(self, path: str, str_i: bool, no_b: bool) -> tuple[list, list, list]:
        """
        LABEL = in last column
        """
        data = self.get_data(path, str_i)
        tokens = []
        labels = []
        spans = []
        counter = 0
        span_start = -1
        span_end = -1
        for line in data.split("\n"):  # this loop is same than version 1
            if line.startswith("#") or line == "":
                continue
            else:
                fields = line.split("\t")  # Token
                label = fields[-1]
                if "-" in fields[0] or "." in fields[
                    0]:  # Multi-Word Expression or Ellipsis : No pred shall be there....
                    continue
                elif no_b == True and fields[0] == "1":
                    label = "_"
                elif self.LAB_SEG_B in label:
                    label = self.LAB_SEG_B
                else:
                    label = "_"  # 🚩
                    if span_start > -1:  # Add span
                        if span_end == -1:
                            span_end = span_start
                        spans.append((span_start, span_end))
                        span_start = -1
                        span_end = -1

                tokens.append(fields[1])
                labels.append(label)
                counter += 1

        if span_start > -1 and span_end > -1:  # Add last span
            spans.append((span_start, span_end))

        if not self.LAB_SEG_B in labels:
            exit(f"Unrecognized labels. Expecting: {self.LAB_SEG_B}, {self.LAB_SEG_I}...")

        return tokens, labels, spans