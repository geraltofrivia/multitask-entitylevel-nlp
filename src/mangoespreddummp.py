#!/usr/bin/env python
# coding: utf-8

# # Running Mangoes predictions locally

# In[ ]:


# In house mangoes pred (using mangoes as a lib)
import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.path.abspath(os.curdir)).parent))

# In[ ]:


import numpy as np
import glob
import torch
from collections import Counter
from mangoes.modeling import BERTForCoreferenceResolution, MangoesCoreferenceDataset
from transformers import BertTokenizerFast
from tqdm.auto import trange
from pathlib import Path
import pickle
from scipy.optimize import linear_sum_assignment as linear_assignment


# In[ ]:


from config import ROOT_LOC


# In[ ]:


def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?" or word == "/-":
        return word[1:]
    else:
        return word


print('here')
def parse_document(path, language="english"):
    """
    returns:
        words: Lists of Lists of Lists of strings. list of sentences. One sentence is a list of words.
        cluster_ids: List of Lists of List(int, int).
        speaker_ids: Lists of Lists of Lists of ints.
    """
    doc_keys = []
    doc_sents = []
    doc_speaker_ids = []
    doc_clusters = []
    sentences = []
    sentence_speaker_ids = []
    clusters = {}
    cur_sentence_words = []
    cur_sentence_speaker_ids = []
    docs = 0
    with open(path, "r") as input_file:
        for line in input_file:
            if line.startswith("#begin document"):
                doc_key = line.split()[2][:-1]
                doc_keys.append(doc_key[1:-1])
                num_words = 0
                docs += 1
            elif line.startswith("#end document"):
                assert len(sentences) == len(sentence_speaker_ids)
                assert cur_sentence_words == []
                doc_sents.append(sentences)
                merged_clusters = []
                for c1 in clusters.values():
                    existing = None
                    for m in c1:
                        for c2 in merged_clusters:
                            if tuple(m) in c2:
                                existing = c2
                                break
                        if existing is not None:
                            break
                    if existing is not None:
                        print("Merging clusters (shouldn't happen very often.)")
                        existing.update([tuple(x) for x in c1])
                    else:
                        merged_clusters.append(set([tuple(x) for x in c1]))
                merged_clusters = [list(c) for c in merged_clusters]
                doc_clusters.append(merged_clusters)
                doc_speaker_ids.append(sentence_speaker_ids)
                sentences = []
                sentence_speaker_ids = []
                clusters = {}
            else:
                data = line.split()
                sentence_end = len(data) == 0
                if sentence_end:
                    sentences.append(cur_sentence_words)
                    sentence_speaker_ids.append(cur_sentence_speaker_ids)
                    cur_sentence_words = []
                    cur_sentence_speaker_ids = []
                else:
                    cur_sentence_words.append(normalize_word(data[3], language))
                    cur_sentence_speaker_ids.append(data[9])
                    # get cluster id
                    raw_cluster_id = data[-1]
                    for part in raw_cluster_id.split("|"):
                        if "(" in part:
                            clus_id = part[1:-1] if ")" in part else part[1:]
                            if clus_id not in clusters:
                                clusters[clus_id] = []
                            clusters[clus_id].append([num_words])
                    for part in raw_cluster_id.split("|"):
                        if ")" in part:
                            clus_id = part[1:-1] if "(" in part else part[:-1]
                            for i in range(len(clusters[clus_id]) - 1, -1, -1):
                                if len(clusters[clus_id][i]) == 1:
                                    clusters[clus_id][i].append(num_words)
                                    break
                    num_words += 1
        assert len(doc_sents) == docs
        for i in range(docs):
            doc_keys[i] += f"_{i}"
        return doc_sents, doc_clusters, doc_speaker_ids, doc_keys


def parse_dataset(path):
    dataset_sents = []
    dataset_clusters_words = []
    dataset_speakers = []
    dataset_genres = []
    dataset_doc_keys = []
    for path in glob.iglob(path):

        genre = path.split("/")[-4]
        doc_sents, doc_clusters, doc_speaker_ids, doc_keys = parse_document(path)
        assert len(doc_sents) == len(doc_speaker_ids) == len(doc_keys) == len(doc_clusters)
        dataset_sents += doc_sents
        dataset_genres += [genre] * len(doc_sents)
        dataset_doc_keys += doc_keys
        dataset_clusters_words += doc_clusters
        for d in range(len(doc_speaker_ids)):
            speakers = doc_speaker_ids[d]
            speakers_to_ids = {speaker: i for i, speaker in
                               enumerate(list(set([item for sublist in speakers for item in sublist])))}
            for i in range(len(speakers)):
                for j in range(len(speakers[i])):
                    speakers[i][j] = speakers_to_ids[speakers[i][j]] + 1
            dataset_speakers.append(speakers)
    gen_to_id = {g: i for i, g in enumerate(set(dataset_genres))}
    return dataset_sents, dataset_clusters_words, dataset_speakers, dataset_genres, dataset_doc_keys, gen_to_id


# In[ ]:


model = BERTForCoreferenceResolution.load("bert-base-cased", "SpanBERT/spanbert-base-cased",
                                          max_top_antecendents=50, max_training_segments=3, top_span_ratio=0.4,
                                          ffnn_hidden_size=3000, coref_dropout=0.3, coref_depth=2,
                                          use_metadata=False)

tok = BertTokenizerFast.from_pretrained("bert-base-cased")


# In[ ]:


# In[ ]:


train_path = ROOT_LOC / 'data/raw/ontonotes/conll-2012/v5/data/train/data/english/annotations/*/*/*/*gold_conll'
valid_path = ROOT_LOC / 'data/raw/ontonotes/conll-2012/v5/data/development/data/english/annotations/*/*/*/*gold_conll'

# change this path to point to where the data is
train_sents, train_clusters, train_speakers, train_genres, train_doc_keys, _ = parse_dataset(str(train_path))

# Filter out useless ones
valid_indices = [i for i, x in enumerate(train_clusters) if x]


# In[ ]:


train_sents = [train_sents[i] for i in valid_indices]
train_clusters = [train_clusters[i] for i in valid_indices]
train_speakers = [train_speakers[i] for i in valid_indices]
train_genres = [train_genres[i] for i in valid_indices]
train_doc_keys = [train_doc_keys[i] for i in valid_indices]


# In[ ]:


train_dataset = MangoesCoreferenceDataset(tok, use_metadata=False, max_segment_len=384, max_segments=3,
                                          documents=train_sents, cluster_ids=train_clusters,
                                          speaker_ids=train_speakers, genres=train_genres,
                                          genre_to_id=model.model.genres)

print(len(train_dataset))

# In[ ]:


eval_sents, eval_clusters, eval_speakers, eval_genres, eval_doc_keys, _ = parse_dataset(str(valid_path))

eval_dataset = MangoesCoreferenceDataset(tok, use_metadata=False, max_segment_len=384, max_segments=3,
                                         documents=eval_sents, cluster_ids=eval_clusters,
                                         speaker_ids=eval_speakers, genres=eval_genres,
                                         genre_to_id=model.model.genres)

print(len(eval_dataset))

# In[ ]:


writefl = Path(ROOT_LOC / 'temp/mangoespred')
writefl.mkdir(exist_ok=True, parents=True)
print(writefl.absolute())


# In[ ]:


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[0], matching[1]])

    # similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


# In[ ]:

coref_evaluator = CorefEvaluator()

# Dumping outputs
for example_index in trange(len(eval_dataset)):
    with torch.no_grad():
        example = eval_dataset[example_index]
        example = {name: tensor.to(model.model_device) for name, tensor in example.items()}
        outputs = model.model.forward(example["input_ids"],
                                      example["attention_mask"],
                                      example["sentence_map"],
                                      #                               speaker_ids=example["speaker_ids"],
                                      #                               genre=example["genre"],
                                      gold_starts=example["gold_starts"],
                                      gold_ends=example["gold_ends"],
                                      cluster_ids=example["cluster_ids"],
                                      return_dict=True)

        gold_clusters = {}
        for i in range(len(example["cluster_ids"])):
            assert len(example["cluster_ids"]) == len(
                example["gold_starts"]) == len(example["gold_ends"])
            cid = example["cluster_ids"][i].item()
            if cid in gold_clusters:
                gold_clusters[cid].append((example["gold_starts"][i].item(),
                                           example["gold_ends"][i].item()))
            else:
                gold_clusters[cid] = [(example["gold_starts"][i].item(),
                                       example["gold_ends"][i].item())]

        gold_clusters = [tuple(v) for v in gold_clusters.values()]
        mention_to_gold = {}
        for c in gold_clusters:
            for mention in c:
                mention_to_gold[mention] = c

        top_indices = torch.argmax(outputs["top_antecedent_scores"], dim=-1, keepdim=False)
        ids = outputs["flattened_ids"]
        top_span_starts = outputs["top_span_starts"]
        top_span_ends = outputs["top_span_ends"]
        top_antecedents = outputs["top_antecedents"]
        mention_indices = []
        antecedent_indices = []
        predicted_antecedents = []
        for i in range(len(outputs["top_span_ends"])):
            if top_indices[i] > 0:
                mention_indices.append(i)
                antecedent_indices.append(top_antecedents[i][top_indices[i] - 1].item())
                predicted_antecedents.append(top_indices[i] - 1)

        cluster_sets = []
        for i in range(len(mention_indices)):
            new_cluster = True
            for j in range(len(cluster_sets)):
                if mention_indices[i] in cluster_sets[j] or antecedent_indices[i] in cluster_sets[j]:
                    cluster_sets[j].add(mention_indices[i])
                    cluster_sets[j].add(antecedent_indices[i])
                    new_cluster = False
                    break
            if new_cluster:
                cluster_sets.append({mention_indices[i], antecedent_indices[i]})

        cluster_dicts = []
        clusters = []
        for i in range(len(cluster_sets)):
            cluster_mentions = sorted(list(cluster_sets[i]))
            current_ids = []
            current_start_end = []
            for mention_index in cluster_mentions:
                current_ids.append(ids[top_span_starts[mention_index]:top_span_ends[mention_index] + 1])
                current_start_end.append((top_span_starts[mention_index].item(), top_span_ends[mention_index].item()))
            cluster_dicts.append({"cluster_ids": current_ids})
            clusters.append(tuple(current_start_end))

        mention_to_predicted = {}
        for c in clusters:
            for mention in c:
                mention_to_predicted[mention] = c

        coref_evaluator.update(clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        outputs['_clusters'] = clusters
        outputs['_gold_clusters'] = gold_clusters
        outputs['_mention_to_predicted'] = mention_to_predicted
        outputs['_mention_to_gold'] = mention_to_gold

        # dump to disk
        with (writefl / (str(example_index) + '.pkl')).open('wb+') as f:
            pickle.dump({'input': example, 'output': outputs}, f)

        del example
        del outputs

summary_dict = {}
p, r, f = coref_evaluator.get_prf()
summary_dict["Average F1 (py)"] = f
print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(eval_dataset)))
summary_dict["Average precision (py)"] = p
print("Average precision (py): {:.2f}%".format(p * 100))
summary_dict["Average recall (py)"] = r
print("Average recall (py): {:.2f}%".format(r * 100))
