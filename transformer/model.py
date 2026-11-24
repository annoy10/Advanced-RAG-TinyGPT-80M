
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .transformer_blocks import Block
except ImportError:
    from transformer_blocks import Block


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("TinyGPT device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CHECKPOINT PATH
# ============================================================

_checkpoint_dir = __import__("pathlib").Path(__file__).resolve().parent / "checkpoint"
_80m_checkpoint = _checkpoint_dir / "tinygpt_80m.pt"
_original_checkpoint = _checkpoint_dir / "tinygpt.pt"
CHECKPOINT_PATH = str(_80m_checkpoint if _80m_checkpoint.exists() else _original_checkpoint)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading TinyGPT checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)


# ============================================================
# LOAD VOCABULARY
# ============================================================

word2idx = checkpoint["word2idx"]

idx2word = checkpoint["idx2word"]

vocabulary = checkpoint["vocabulary"]


# ============================================================
# LOAD SPECIAL TOKEN IDS
# ============================================================

PAD_TOKEN_ID = checkpoint["PAD_TOKEN_ID"]

UNK_TOKEN_ID = checkpoint["UNK_TOKEN_ID"]

END_TOKEN_ID = checkpoint["END_TOKEN_ID"]


# ============================================================
# LOAD MODEL CONFIG
# ============================================================

block_size = checkpoint["block_size"]

embedding_dim = checkpoint["embedding_dim"]

n_heads = checkpoint["n_heads"]

n_layers = checkpoint["n_layers"]

dropout = checkpoint["dropout"]


# ============================================================
# TINY GPT
# ============================================================

class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(

            *[
                Block(
                    embedding_dim,
                    block_size,
                    n_heads,
                    dropout
                )

                for _ in range(
                    n_layers
                )
            ]
        )

        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        self.ln = nn.LayerNorm(
            embedding_dim
        )

        # ----------------------------------------------------
        # Output Head
        # ----------------------------------------------------

        self.head = nn.Linear(
            embedding_dim,
            vocab_size
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        targets=None
    ):

        B, T = x.shape

        # Token embedding
        token_emb = self.token_embedding(
            x
        )

        # Position IDs
        positions = torch.arange(
            T,
            device=x.device
        )

        # Position embedding
        position_emb = self.position_embedding(
            positions
        )

        # Add embeddings
        x = (
            token_emb
            + position_emb
        )

        # Transformer
        x = self.blocks(
            x
        )

        # LayerNorm
        x = self.ln(
            x
        )

        # Vocabulary logits
        logits = self.head(
            x
        )

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(
                    B * T,
                    C
                ),
                targets.reshape(
                    B * T
                )
            )

        return logits, loss


# ============================================================
# CREATE TINYGPT
# ============================================================

model = TinyGPT(
    len(vocabulary)
).to(device)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


print(
    "TinyGPT checkpoint loaded successfully."
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(
        text
    ).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(text):

    text = normalize_text(
        text
    )

    words = text.split()

    ids = []

    for word in words:

        token_id = word2idx.get(
            word,
            UNK_TOKEN_ID
        )

        ids.append(
            token_id
        )

    return ids


# ============================================================
# DECODE TOKENS
# ============================================================

def decode_tokens(ids):

    words = []

    for idx in ids:

        word = idx2word.get(
            int(idx),
            ""
        )

        # Stop at END
        if word == "<END>":

            break

        # Ignore special tokens
        if word in [
            "<PAD>",
            "<UNK>"
        ]:

            continue

        words.append(
            word
        )

    return " ".join(
        words
    )


# ============================================================
# CLEAN GENERATED ANSWER
# ============================================================

def clean_answer(answer):

    answer = answer.strip()

    # Remove "answer"
    answer = re.sub(
        r"^answer\s*:?\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    # Stop at another question
    answer = re.split(
        r"\bquestion\s*:?\s*",
        answer,
        flags=re.IGNORECASE
    )[0]

    # Stop at another answer
    answer = re.split(
        r"\banswer\s*:?\s*",
        answer,
        flags=re.IGNORECASE
    )[0]

    answer = answer.strip()

    return answer


# ============================================================
# GENERATE ANSWER
# ============================================================

@torch.no_grad()
def generate_answer(
    question,
    max_new_tokens=20,
    temperature=0.7
):

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Normalize question
    # --------------------------------------------------------

    question = normalize_text(
        question
    )

    if not question:

        return (
            "Please ask a sports question."
        )

    # --------------------------------------------------------
    # Create GPT prompt
    #
    # Training format:
    #
    # question ... answer ... <END>
    #
    # --------------------------------------------------------

    prompt = (
        "question "
        + question
        + " answer"
    )

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    prompt_ids = encode_text(
        prompt
    )

    if not prompt_ids:

        return (
            "Sorry, I don't know the answer."
        )

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    prompt_length = idx.shape[1]


    # ========================================================
    # GENERATION LOOP
    # ========================================================

    for _ in range(
        max_new_tokens
    ):

        # ----------------------------------------------------
        # Keep sequence inside block size
        # ----------------------------------------------------

        idx_cond = idx[
            :,
            -block_size:
        ]

        # ----------------------------------------------------
        # TinyGPT forward
        # ----------------------------------------------------

        logits, _ = model(
            idx_cond
        )

        # ----------------------------------------------------
        # Last token prediction
        # ----------------------------------------------------

        logits = logits[
            :,
            -1,
            :
        ]

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        logits = (
            logits
            / temperature
        )

        # ----------------------------------------------------
        # Disable PAD and UNK
        # ----------------------------------------------------

        logits[
            :,
            PAD_TOKEN_ID
        ] = float("-inf")

        logits[
            :,
            UNK_TOKEN_ID
        ] = float("-inf")

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probs = F.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Sample next token
        # ----------------------------------------------------

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        # ----------------------------------------------------
        # Add token
        # ----------------------------------------------------

        idx = torch.cat(
            [
                idx,
                next_token
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Stop at END
        # ----------------------------------------------------

        if (
            next_token.item()
            == END_TOKEN_ID
        ):

            break


    # ========================================================
    # DECODE ONLY GENERATED TOKENS
    # ========================================================

    generated_ids = idx[
        0,
        prompt_length:
    ]

    answer = decode_tokens(
        generated_ids
    )


    # ========================================================
    # CLEAN ANSWER
    # ========================================================

    answer = clean_answer(
        answer
    )


    # ========================================================
    # EMPTY ANSWER
    # ========================================================

    if not answer:

        return (
            "Sorry, I don't know the answer."
        )

    return answer


# ============================================================
# CHAT FUNCTION
# ============================================================

def chat(question):

    question = str(
        question
    ).strip()

    if not question:

        return (
            "Please ask a sports question."
        )

    answer = generate_answer(
        question
    )

    return answer

