#!/usr/bin/env python
# coding: utf-8

# # Running Mangoes predictions locally

# In[ ]:


# In house mangoes pred (using mangoes as a lib)
import os, sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(os.curdir)).parent))


# In[ ]:


import os
import glob
import torch
from mangoes.modeling import BERTForCoreferenceResolution, MangoesCoreferenceDataset
from transformers import BertTokenizerFast
from tqdm.auto import tqdm, trange
from pathlib import Path
import pickle


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


# Dumping outputs
for example_index in trange(len(train_dataset)):
    with torch.no_grad():
        example = train_dataset[example_index]
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

        # dump to disk
        with (writefl / (str(example_index) + '.pkl')).open('wb+') as f:
            pickle.dump({'input': example, 'output': outputs}, f)

        del example
        del outputs

# In[ ]:
