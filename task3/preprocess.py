import os
import json

def tok_reader(file_name):
    """
    Args:
        file_name: data path
    """
    all_doc_data = {}
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc = []
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower() or "newdoc id" in line.lower():
                    # print(line)
                    tmp_doc_id = line.split("=")[1].strip()
                else:
                    items = line.split("\t")  # check if is \t
                    if "-" in items[0] or '.' in items[0]:  # ignore such as 16-17 or 17.1(??)
                        continue
                    token_id = int(items[0].strip())
                    token = items[1].strip()
                    token_label = items[-1].strip()
                    tmp_doc.append((token_id, token, token_label))
            else:
                if len(tmp_doc) > 0 and tmp_doc_id is not None:
                    # all_doc_data.append((tmp_doc_id, tmp_doc))
                    all_doc_data[tmp_doc_id] = tmp_doc
                tmp_doc_id = None
                tmp_doc = []

    # in case the last one
    if len(tmp_doc) > 0 and tmp_doc_id is not None:
        # all_doc_data.append((tmp_doc_id, tmp_doc))
        all_doc_data[tmp_doc_id] = tmp_doc
    return all_doc_data


def conll_reader(file_name):
    all_conll_data = {}
    # errors accumulator for tok id
    tok_id_acc = 0
    # accumulator for token id
    acc_4_id = 0
    # accumulator for the previous tokens in sentences
    acc_4_sent = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        tmp_doc_id = None
        tmp_doc_info = []
        tmp_sent_id = None
        tmp_sent_type = None
        tmp_sent_info = []
        cur_id = 0
        for line in lines:
            line = line.strip()
            if line:
                if "newdoc_id" in line.lower() or "newdoc id" in line.lower():
                    if len(tmp_doc_info) > 0 and tmp_doc_id is not None:
                        # all_conll_data.append((tmp_doc_id, tmp_doc_info))
                        all_conll_data[tmp_doc_id] = tmp_doc_info
                    # reset
                    tmp_doc_id = line.split("=")[1].strip()
                    tmp_doc_info = []
                    acc_4_id = 0
                    acc_4_sent = 0
                else:
                    if "sent_id" in line.lower() in line.lower() or "newutterance_id" in line.lower() in line.lower():
                        tmp_sent_id = line.split("=")[1].strip()
                        # reset the accumulator for the wrong labeld tok id
                        acc_4_sent += acc_4_id
                        acc_4_id = 0
                        tok_id_acc = 0
                    elif "s_type" in line.lower():
                        tmp_sent_type = line.split("=")[1].strip()
                    elif line.lower()[0].isdigit():
                        # you can read now word information here
                        items = line.split("\t")  # check if is \t
                        token_id = items[0]
                        if "-" in token_id:  # ignore invalid
                            continue
                        # eng.rst.gum_train.conllu has this issue for wrong labeled tok id(the word denoted)
                        elif "." in token_id:
                            tok_id_acc += 1
                        # wrong labeling issue in ./data/ita.pdtb.luna/ita.pdtb.luna_dev.conllu the token id is 14si
                        # the token's id concatenates the token's string...

                        # this if condition is for spa. file, because there's no sent_id!
                        elif (
                                "spa." in file_name or "zho.rst.sctb" in file_name or "zho.pdtb.cdtb" in file_name or "eng.rst.rstdt" in file_name) and cur_id > int(
                                float(token_id)):
                            acc_4_sent += acc_4_id
                            acc_4_id = 0
                            tok_id_acc = 0
                        else:
                            real_id = ""
                            real_string = ""
                            for i in token_id:
                                if i.isdigit():
                                    real_id += i
                                else:
                                    real_string += i
                            token_id = real_id
                            items[0] = token_id
                            items.insert(1, real_string)
                        POS1 = items[3]
                        POS2 = items[4]
                        POS3 = items[5]
                        POS4 = items[6]
                        POS5 = items[7]
                        POS6 = items[8]

                        acc_4_id += 1

                        tmp_sent_info.append(
                            (int(float(token_id)) + tok_id_acc + acc_4_sent, POS1, POS2, POS3, POS4, POS5, POS6))
                        cur_id = int(float(token_id))
                    else:
                        continue
            else:
                if tmp_doc_info is not None:
                    tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
                else:
                    raise Exception("The tmp_doc_info should not be None!!!")
                # reset
                tmp_sent_id = None
                tmp_sent_type = None
                tmp_sent_info = []
        # here, shouldn't check the tmp_sent_id is None or not, because in the file zho.pdtb.cdtb_train.collu, there's no information
        # about the sentence id!
        # if tmp_sent_id is not None:
        if tmp_sent_info is not None:
            if tmp_doc_info is not None:
                tmp_doc_info.append((tmp_sent_id, tmp_sent_type, tmp_sent_info))
            else:
                raise Exception("The tmp_doc_info should not be None!!!")
        if len(tmp_doc_info) > 0:
            all_conll_data[tmp_doc_id] = tmp_doc_info
    return all_conll_data

def rel_reader(file_name):
    all_relation_data = [] #{}
    index_id = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                items = line.split("\t")
                doc_id = items[0]
                unit1_toks = items[1]
                unit1_txt = items[3]
                unit2_toks = items[2]
                unit2_txt = items[4]
                # label = items[11]
                label = items[-1]
                # if doc_id in all_relation_data:
                all_relation_data.append((doc_id, unit1_toks, unit1_txt, unit2_toks, unit2_txt, label))
                #else:
                #    all_relation_data[doc_id] = [(unit1_toks, unit1_txt, unit2_toks, unit2_txt, label, index_id)]
                #index_id += 1

    return all_relation_data

def rel_reader1(file_name):
    all_relation_data = [] #{}
    index_id = 0
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            line = line.strip()
            if line:
                items = line.split("\t")
                doc_id = items[0]
                unit1_toks = items[1]
                unit1_txt = items[3]
                unit2_toks = items[2]
                unit2_txt = items[4]
                unit1_sent = items[-6]
                unit2_sent = items[-5]
                # label = items[11]
                label = items[-1]
                dir = items[-4]
                # if doc_id in all_relation_data:
                all_relation_data.append((doc_id, unit1_toks, unit1_txt, unit2_toks, unit2_txt, unit1_sent, unit2_sent, dir, label))
                #else:
                #    all_relation_data[doc_id] = [(unit1_toks, unit1_txt, unit2_toks, unit2_txt, label, index_id)]
                #index_id += 1

    return all_relation_data

def preprocessing(tok_file, conllu_file, rel_file, output_file):
    """
    Args:
        tok_file: raw text
        conllu_file: parsed results of the text
        rel_file: relation annotation file
    """

    all_doc_data = tok_reader(tok_file)
    all_conll_data = conll_reader(conllu_file)
    all_relation_data = rel_reader(rel_file)
    dname = tok_file.split("/")[-1].split("_")[0].strip()

    assert len(all_doc_data) == len(all_conll_data), (len(all_doc_data), len(all_conll_data))
    # assert len(all_doc_data) == len(all_relation_data), (len(all_doc_data), len(all_relation_data))

    all_samples = []
    print_id = 5

    for doc_id in all_doc_data:
        doc_tokens = all_doc_data[doc_id]
        assert doc_id in all_conll_data
        doc_conll_info = all_conll_data[doc_id]
        flat_doc_conll_info = []

        ## for segmentation and connective detection
        doc_sent_tokens = []
        doc_sent_token_features = []
        doc_sent_token_labels = []
        for sent in doc_conll_info:
            sent_tokens = []
            sent_features = []
            sent_labels = []
            for token_info in sent[2]:
                token_id = int(token_info[0])  # start from 1
                POS1 = token_info[1]
                POS2 = token_info[2]
                POS3 = token_info[3]
                POS4 = token_info[4]
                POS5 = token_info[5]
                POS6 = token_info[6]
                token = doc_tokens[token_id - 1]
                assert token[0] == token_id, (token[0], token_id)
                sent_tokens.append(token[1])
                sent_features.append((POS1, POS2, POS3, POS4, POS5, POS6))
                sent_labels.append(token[2])
            doc_sent_tokens.append(sent_tokens)
            doc_sent_token_features.append(sent_features)
            doc_sent_token_labels.append(sent_labels)
            flat_doc_conll_info.extend(sent[2])
        assert len(doc_tokens) == len(flat_doc_conll_info), (len(doc_tokens), len(flat_doc_conll_info))
        ## for relation classification
        doc_unit_tokens = []
        doc_unit_token_features = []
        doc_unit_labels = []

        if doc_id in all_relation_data:
            doc_rel = all_relation_data[doc_id]
        else:
            doc_rel = []

        for unit_pair in doc_rel:
            unit1_ids = unit_pair[0]
            unit2_ids = unit_pair[1]
            rel = unit_pair[2]
            index_id = unit_pair[3]
            doc_unit_labels.append((rel, index_id))

            unit1_tokens = []
            unit2_tokens = []
            unit1_features = []
            unit2_features = []
            group_unit1_ids = unit1_ids.split(",")
            group_unit2_ids = unit2_ids.split(",")
            for span in group_unit1_ids:
                if "-" in span:  # a range
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit1_tokens.append(doc_tokens[idx - 1][1])
                        unit1_features.append(flat_doc_conll_info[idx - 1][1:])
                # bug in ./data/ita.pdtb.luna/ita.pdtb.luna_train.rels
                # you can look at line 9, the number of unit_toks is "_"
                elif "_" in span:  # means no unit_toks
                    unit1_tokens.append("_")
                    unit1_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit1_tokens.append(doc_tokens[span_pos - 1][1])
                    unit1_features.append(flat_doc_conll_info[span_pos - 1][1:])

            for span in group_unit2_ids:
                # a range
                if "-" in span:
                    span_start = int(span.split("-")[0])
                    span_end = int(span.split("-")[1])
                    for idx in range(span_start, span_end + 1):
                        unit2_tokens.append(doc_tokens[idx - 1][1])
                        unit2_features.append(flat_doc_conll_info[idx - 1][1:])
                elif "_" in span:  # means no unit_toks
                    unit2_tokens.append("_")
                    unit2_features.append(("_", "_"))
                else:  # a number
                    span_pos = int(span)
                    unit2_tokens.append(doc_tokens[span_pos - 1][1])
                    unit2_features.append(flat_doc_conll_info[span_pos - 1][1:])
            # print(unit1_tokens)
            # print(unit2_tokens)
            # print()
            # print_id -= 1
            # if print_id < 0:
            #     return
            doc_unit_tokens.append((unit1_tokens, unit2_tokens))
            doc_unit_token_features.append((unit1_features, unit2_features))

        # save info json
        sample = {}
        sample["doc_id"] = doc_id
        sample["dname"] = dname
        sample["doc_sents"] = doc_sent_tokens
        sample["doc_sent_token_features"] = doc_sent_token_features
        sample["doc_sent_token_labels"] = doc_sent_token_labels
        sample["doc_units"] = doc_unit_tokens
        sample["doc_unit_token_features"] = doc_unit_token_features
        sample["doc_unit_labels"] = doc_unit_labels

        all_samples.append(json.dumps(sample, ensure_ascii=False))