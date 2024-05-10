import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer,util


def load_dataset(path) -> pd.DataFrame:

    dataset = pd.read_csv(path)

    # data contains a column with sentences in it


    return dataset


def load_model(model_path) -> SentenceTransformer:

    # load the sentence transformer model

    model = SentenceTransformer(model_path)

    return model

def setup_multi_processing_pool() -> List:

    print("Setting Up Multi Processing Pool")
    device_count = torch.cuda.device_count()
    pool_list = [f"cuda:{device}" for device in range(device_count)]
    return pool_list

def generate_distributed_sentence_embeddings(all_sentences : List[str], batch_size : int, model_path : str) -> Dict:

    print("Computing Embeddings Using Multi-Process Pool")    
    torch.cuda.empty_cache()

    model = load_model(model_path)

    pool_list = setup_multi_processing_pool()

    pool = model.start_multi_process_pool(pool_list)

    embeddings = model.encode_multi_process(sentences = all_sentences, pool = pool, batch_size = batch_size)

    model.start_multi_process_pool(pool)

    sentence_embedding_dictionary = {}

    for sentence, embedding in zip(all_sentences, embeddings):
        sentence_embedding_dictionary[sentence] = embedding

    print("Setence:Embedding Mapping Created")
    return sentence_embedding_dictionary

