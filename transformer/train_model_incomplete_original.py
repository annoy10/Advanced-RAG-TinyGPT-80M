
import ast
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformer_blocks import Block


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Torch version:", torch.__version__)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# CONFIG
# =========================================================

NUM_SAMPLES = 200000

block_size = 128
embedding_dim = 256
n_heads = 8
n_layers = 6
dropout = 0.1

batch_size = 16
learning_rate = 3e-4

TRAIN_STEPS = 8000
PATIENCE = 1500

CHECKPOINT_PATH = "tinygpt.pt"


# =========================================================
# ANSWER CLEANING
# =========================================================

def extract_answer(answer):

    if answer is None:
        return ""

    # String
    if isinstance(answer, str):

        answer = answer.strip()

        # Sometimes answer is a stringified dict/list
        if answer.startswith("{") or answer.startswith("["):

            try:
                parsed = ast.literal_eval(answer)

                return extract_answer(parsed)

            except Exception:
                return answer

        return answer

    # Dictionary
    if isinstance(answer, dict):

        text = answer.get("text", "")

        if isinstance(text, list):

            if len(text) > 0:
                return str(text[0]).strip()

            return ""

        if isinstance(text, str):
            return text.strip()

        return ""

    # List
    if isinstance(answer, list):

        if len(answer) > 0:
            return extract_answer(answer[0])

        return ""

    return str(answer).strip()


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = " ".join(
        text.split()
    )

    return text


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading QASports2...")

dataset = load_dataset(
    "leomaurodesenv/QASports2"
)

train_data = dataset["train"]

print(
    "Dataset rows:",
    len(train_data)
)


# =========================================================
# CREATE QA PAIRS
# =========================================================

qa_pairs = []

for row in train_data:

    question = row.get(
        "question",
        ""
    )

    answer = row.get(
        "answer",
        row.get(
            "answers",
            ""
        )
    )

    answer = extract_answer(answer)

    if not isinstance(question, str):
        continue

    if not isinstance(answer, str):
        continue

    question = normalize_text(question)
    answer = normalize_text(answer)

    # Remove invalid answers
    if (
        "offset" in answer
        or answer.startswith("{")
        or answer.startswith("[")
    ):
        continue

    if question and answer:

        qa_pairs.append(
            (
                question,
                answer
            )
        )

    if len(qa_pairs) >= NUM_SAMPLES:
        break


print(
    "Usable QA pairs:",
    len(qa_pairs)
)


# =========================================================
# SHOW SAMPLE DATA
# =========================================================

print("\nSample QA:")

for question, answer in qa_pairs[:10]:

    print("Question:", question)
    print("Answer:", answer)
    print("-" * 60)


# =========================================================
# CREATE CORPUS
# =========================================================

corpus_parts = []

for question, answer in qa_pairs:

    corpus_parts.append(
        "question "
        + question
        + " answer "
        + answer
        + " <END>"
    )


corpus = " ".join(corpus_parts)


# =========================================================
# VOCABULARY
# =========================================================

words = corpus.split()

special_tokens = [
    "<PAD>",
    "<UNK>",
    "<END>"
]

vocabulary = sorted(
    set(words)
)

for token in special_tokens:

    if token not in vocabulary:
        vocabulary.append(token)


print(
    "\nVocabulary size:",
    len(vocabulary)
)


# =========================================================
# TOKENIZER
# =========================================================

word2idx = {
    word: i
    for i, word in enumerate(vocabulary)
}

idx2word = {
    i: word
    for word, i in word2idx.items()
}


PAD_TOKEN_ID = word2idx["<PAD>"]
UNK_TOKEN_ID = word2idx["<UNK>"]
END_TOKEN_ID = word2idx["<END>"]


# =========================================================
# CONVERT CORPUS TO TOKEN IDS
# =========================================================

token_ids = [
    word2idx.get(
        word,
        UNK_TOKEN_ID
    )
    for word in words
]


data = torch.tensor(
    token_ids,
    dtype=torch.long,
    device=device
)


print(
    "Total tokens:",
    len(data)
)


# =========================================================
# CHECK DATA SIZE
# =========================================================

if len(data) <= block_size + 1:

    raise ValueError(
        "Dataset is too small for the selected block_size."
    )


# =========================================================
# BATCH FUNCTION
# =========================================================

def get_batch():

    max_start = (
        len(data)
        - block_size
        - 1
    )

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        device=device
    )

    x = torch.stack(
        [
            data[
                start:
                start + block_size
            ]
            for start in starts
        ]
    )

    y = torch.stack(
        [
            data[
                start + 1:
                start + block_size + 1
            ]
            for start in starts
        ]
    )

    return x, y


# =========================================================
# TINYGPT MODEL
# ============================================

