from sentence_transformers import SentenceTransformer, CrossEncoder, util, InputExample, losses
from torch.utils.data import DataLoader
from pyvi.ViTokenizer import tokenize
import os
import torch
import time
import json
import gdown
import numpy as np

def download_files(url="https://drive.google.com/drive/folders/1dcfQWXHcu0tAvUd3ngIBs4qwKyfiInaF"):
    gdown.download_folder(url, quiet=False, use_cookies=False)

def load_data(passages_file='passages.json',
                min_words = 50,
                max_words = 500,
                data_folder='data/'):
    '''
    Load json data from json file that export from Elastic Search
    '''
    passages = []
    passages_path = data_folder + passages_file
    if os.path.exists(passages_path):
        print('Load passages from ', passages_path)
        with open(passages_path, 'r') as json_file:
            passages = json.load(json_file)
    return passages

def save_data(passages, passages_file='passages.json', data_folder='data/'):
    passages_path = data_folder + passages_file
    with open(passages_path, 'w') as json_file:
        json.dump(passages, json_file)
        print('Store passages data at ', passages_path)

def load_tokenized_data(tokenized_data='tokenized-es-exported-index.json', 
                        data_raw="es-exported-index.json", 
                        data_folder='data/', 
                        min_sentence=5):
    '''
    Load pre-tokeinzed data
    '''
    inputs = []
    tokenized_path = data_folder + tokenized_data
    if os.path.exists(tokenized_path):
        print('Load tokenized input data ', tokenized_path)
        with open(tokenized_path, 'r') as json_file:
            inputs = json.load(json_file)
    else:
        print('Tokenizing input data')
        data_path = data_folder + data_raw
        with open(data_path, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                data = json.loads(line.strip())
                if (len(data['_source']['content'].split('.')) >= min_sentence):
                    inputs.append([data['_source']['book_name'], tokenize(data['_source']['content'])])
        
        # store tokenized data
        with open(tokenized_path, 'w') as json_file:
            json.dump(inputs, json_file)
            print('Store tokenized input data at ', tokenized_path)
    return inputs

def load_model(model_name='keepitreal/vietnamese-sbert', data_folder='data/'):
    '''
    Load pre-trained model if it exists, 
    otherwise train from scratch then backup it for reusing next time
    '''
    model_path = data_folder + model_name.replace("/", "_")
    if os.path.exists(model_path):
        print("Load trained model from ", model_path)
        bi_encoder = SentenceTransformer(model_path, cache_folder=data_folder)
    else:
        print("Download the pre-train model ", model_name)
        bi_encoder = SentenceTransformer(model_name, cache_folder=data_folder)
    return bi_encoder

def load_corpus_embedding(bi_encoder, passages, model_name='keepitreal/vietnamese-sbert', data_folder='data/'):
    # Load corpus embedding from file or encode the new one
    embeddings_filepath = data_folder + model_name.replace("/", "_") + "-corpus.pt"
    if os.path.exists(embeddings_filepath):
        print('Load corpus embedding from ', embeddings_filepath)
        corpus_embeddings = torch.load(embeddings_filepath, map_location=torch.device('cpu'))
        corpus_embeddings = corpus_embeddings.float()
        if torch.cuda.is_available():
            corpus_embeddings = corpus_embeddings.to('cuda')
    else:
        corpus_embeddings = encode_corpus(bi_encoder, passages)
        save_corpus_embedding(corpus_embeddings, embeddings_filepath)
    return corpus_embeddings

def encode_corpus(model, passages):
    print('Encode corpus embedding')
    return model.encode(passages, convert_to_tensor=True, show_progress_bar=True)

def append_new_corpus(corpus, model, new_passages, model_name, data_folder='data/'):
    new_embeddings = encode_corpus(model, new_passages)
    updated_corpus_embeddings = torch.cat((corpus, new_embeddings), dim=0)
    embeddings_filepath = data_folder + model_name.replace("/", "_") + "-corpus.pt"
    save_corpus_embedding(updated_corpus_embeddings, embeddings_filepath)
    return updated_corpus_embeddings

def update_new_corpus(passages, model, model_name, data_folder='data/'):
    new_embeddings = encode_corpus(model, passages)
    embeddings_filepath = data_folder + model_name.replace("/", "_") + "-corpus.pt"
    save_corpus_embedding(new_embeddings, embeddings_filepath)
    return updated_corpus_embeddings

def save_corpus_embedding(corpus_embeddings, corpus_path):
    torch.save(corpus_embeddings, corpus_path)
    print('Save corpus embedding at ', corpus_path)

def search(bi_encoder, corpus_embeddings, 
            query, passages, top_k=10, is_tokenize=False):
    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()

    if is_tokenize:
        question_embedding = bi_encoder.encode(tokenize(query), convert_to_tensor=True)
    else:
        question_embedding = bi_encoder.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query
    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))
    results = []
    for hit in hits:
        # print("\t{:.3f}\t{}".format(hit['score'], passages[hit['corpus_id']]))
        results.append(passages[hit['corpus_id']])
    return results, hits

def ranking(hits, query, passages, top_k=10, 
    cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):

    start_time = time.time()
    cross_encoder = CrossEncoder(cross_encoder_name)
    cross_inp = [[query, passages[hit['corpus_id']][1]] for hit in hits]
    cross_scores = []
    for input in cross_inp:
        cross_scores.append(cross_encoder.predict(input))
    predict_time = time.time()

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    sort_time = time.time()

    for hit in hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['cross-score'], passages[hit['corpus_id']]))
        results.append(passages[hit['corpus_id']])
    
    print("Predict time=", (predict_time-start_time), ", sort time = ", (sort_time-predict_time))
    return results, hits

def train(model, model_name, dataset, 
        batch_size=16, epochs=10, 
        warmup_steps=100, data_folder="data/"):
    print("Start training the model")
    train_examples = build_training_data(dataset)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], 
        epochs=epochs, warmup_steps=warmup_steps,
        show_progress_bar=True)

    model_path = data_folder + model_name.replace("/", "_")
    model.save(model_path)
    print("Save trained model at ", model_path)
    return model

def build_training_data(dataset):
    '''
    Example dataset:
    [
        ["question 1?", "answer 1", 0.99],
        ["question 1?", "answer 2", 0.98],
    ]
    '''
    return [InputExample(texts=[item[0], item[1]], label=item[2]) for item in dataset]