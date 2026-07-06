import csv
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from Extraction import extract as zhang_extract

TASK_NAMES = ["parity", "contains_11", "mod3"]
MODEL_KINDS = ["direct", "cot_true", "cot_pad"]
SEEDS = [0, 1, 2, 3, 4]

TRAIN_MIN_LEN = 2
TRAIN_MAX_LEN = 32
IID_MIN_LEN = 2
IID_MAX_LEN = 32
OOD_MIN_LEN = 33
OOD_MAX_LEN = 64

TRAIN_SAMPLES = 20_000
DCSA_TRAIN_SAMPLES = 10_000
FAITHFULNESS_SAMPLES = 1000
EXTRACTION_EVAL_SAMPLES = 3_000

D_MODEL = 128
N_HEADS = 2
DROPOUT = 0.1
BATCH_SIZE = 64
N_EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_BASE_LEN = 129   # [CLS] + up to 128 bits
MAX_COT_LEN = 257    # [CLS] + 2 * up to 128 bits

DCSA_EPOCHS = 300
DCSA_LR = 1e-3
DCSA_BATCH_SIZE = 64
DCSA_L1_WEIGHT = 1.0
DCSA_CE_WEIGHT = 1.0
FAITHFULNESS_AGREEMENT_THRESHOLD = 0.95
FAITHFULNESS_L1_THRESHOLD = 0.20

ZHANG_TIME_LIMIT = 50
ZHANG_INITIAL_SPLIT_DEPTH = 10
ZHANG_STARTING_EXAMPLES = ["", "0", "1", "00", "01", "10", "11"]

OUT_DIR = "analysis"
CHECKPOINT_DIR = "checkpoints_dfa"
DFA_DIR = "dfas"
RESULTS_CSV = "analysis/dfa_extraction_results.csv"
LOAD_EXISTING_CHECKPOINTS = True
SAVE_CHECKPOINTS = True

RESULT_COLUMNS = [
    "task", "seed", "model_kind", "backend",
    "dcsa_agreement", "dcsa_mean_l1", "faithfulness_pass",
    "extraction_success", "n_extracted_states", "n_minimal_states", "state_count_error",
    "fidelity_to_transformer_iid", "fidelity_to_transformer_ood",
    "ground_truth_accuracy_iid", "ground_truth_accuracy_ood",
    "notes",
]

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_result(row, csv_path=RESULTS_CSV):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})

def parity_label(s):
    return 1 if sum(s) % 2 == 0 else 0

def parity_transition(tok, state):
    return state ^ int(tok)

def parity_state_to_label(state):
    return 1 if int(state) == 0 else 0


def contains_11_label(s):
    return int(any(s[i] == 1 and s[i + 1] == 1 for i in range(len(s) - 1)))

def contains_11_transition(tok, state):
    tok, state = int(tok), int(state)
    if state == 2:
        return 2
    if state == 0:
        return 1 if tok == 1 else 0
    return 2 if tok == 1 else 0

def contains_11_state_to_label(state):
    return 1 if int(state) == 2 else 0


def mod3_label(s):
    return 1 if sum(s) % 3 == 0 else 0

def mod3_transition(tok, state):
    return (int(state) + int(tok)) % 3

def mod3_state_to_label(state):
    return 1 if int(state) == 0 else 0


TASKS = {
    "parity": {
        "name": "parity",
        "label_fn": parity_label,
        "transition_fn": parity_transition,
        "state_to_label_fn": parity_state_to_label,
        "n_states": 2,
        "minimal_dfa_states": 2,
    },
    "contains_11": {
        "name": "contains_11",
        "label_fn": contains_11_label,
        "transition_fn": contains_11_transition,
        "state_to_label_fn": contains_11_state_to_label,
        "n_states": 3,
        "minimal_dfa_states": 3,
    },
    "mod3": {
        "name": "mod3",
        "label_fn": mod3_label,
        "transition_fn": mod3_transition,
        "state_to_label_fn": mod3_state_to_label,
        "n_states": 3,
        "minimal_dfa_states": 3,
    },
}

VOCAB_BASE = {0: 0, 1: 1, "[PAD]": 2, "[CLS]": 3}

def cot_vocab_for_task(task):
    vocab = dict(VOCAB_BASE)
    for st in range(task["n_states"]):
        vocab[f"s{st}"] = len(vocab)
    return vocab


def tokenise(tokens, vocab, max_length):
    seq = ["[CLS]"] + list(tokens)
    ids = [vocab[t] for t in seq]
    mask = [1] * len(ids)
    pad = max_length - len(ids)
    ids += [vocab["[PAD]"]] * pad
    mask += [0] * pad
    return {"input_ids": ids, "attention_mask": mask}


def build_interleaved_sequence(string, transition_fn, cot_mode="cot_true"):
    # cot_true inserts the true state token; cot_pad inserts a constant dummy
    # token s0 while keeping the same sequence length and prediction targets.
    state = 0
    interleaved = []
    labels = [-1]  # [CLS]
    pred_positions = []
    pos = 1

    for tok in string:
        interleaved.append(int(tok))
        state = transition_fn(int(tok), state)
        labels.append(state)
        pred_positions.append(pos)
        pos += 1

        inserted_state = state if cot_mode == "cot_true" else 0
        interleaved.append(f"s{inserted_state}")
        labels.append(-1)
        pos += 1

    return interleaved, labels, pred_positions

def generate_dataset(min_len, max_len, n, label_fn, seed=None, p_one=0.5):
    rng = random.Random(seed)
    strings, labels = [], []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        s = [1 if rng.random() < p_one else 0 for _ in range(length)]
        strings.append(s)
        labels.append(label_fn(s))
    return strings, labels


def sample_eval_strings(task, split, seed):
    if split == "iid":
        min_len, max_len, offset = IID_MIN_LEN, IID_MAX_LEN, 10_000
    else:
        min_len, max_len, offset = OOD_MIN_LEN, OOD_MAX_LEN, 20_000
    return generate_dataset(min_len, max_len, EXTRACTION_EVAL_SAMPLES, task["label_fn"], seed=seed + offset)


class DirectDataset(Dataset):
    def __init__(self, strings, labels, max_length):
        self.samples = []
        for s, y in zip(strings, labels):
            tok = tokenise(s, VOCAB_BASE, max_length)
            self.samples.append({
                "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
                "label": torch.tensor(int(y), dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CoTDataset(Dataset):
    def __init__(self, strings, task, max_length, cot_mode):
        self.samples = []
        vocab = cot_vocab_for_task(task)
        for s in strings:
            interleaved, labels, _ = build_interleaved_sequence(s, task["transition_fn"], cot_mode=cot_mode)
            tok = tokenise(interleaved, vocab, max_length)
            labels = labels + [-1] * (max_length - len(labels))
            self.samples.append({
                "input_ids": torch.tensor(tok["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(tok["attention_mask"], dtype=torch.long),
                "state_labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=128, n_heads=2, max_len=512, dropout=0.1, pad_idx=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask, causal=False):
        x = self.pos_enc(self.embedding(input_ids))
        key_padding_mask = attention_mask == 0
        attn_mask = None
        if causal:
            L = input_ids.size(1)
            attn_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=input_ids.device), diagonal=1)
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

    def classify_direct(self, input_ids, attention_mask):
        hidden = self.forward(input_ids, attention_mask, causal=False)
        return self.classifier(hidden[:, 0, :])

    def classify_cot(self, input_ids, attention_mask):
        hidden = self.forward(input_ids, attention_mask, causal=True)
        return self.classifier(hidden)

    def get_cls_representation(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask, causal=False)[:, 0, :]

    def get_hidden_states(self, input_ids, attention_mask, causal):
        return self.forward(input_ids, attention_mask, causal=causal)


def train_direct(model, train_loader, val_loader, device, save_path):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = CrossEntropyLoss()
    best_val = -1.0

    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model.classify_direct(ids, mask), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        val_acc = evaluate_direct_accuracy(model, val_loader, device)
        print(f"      direct epoch {epoch + 1}/{N_EPOCHS} | val acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            if SAVE_CHECKPOINTS:
                torch.save({"model_state": model.state_dict()}, save_path)

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device)["model_state"])
    return model


def train_cot(model, train_loader, val_loader, device, save_path):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = CrossEntropyLoss(ignore_index=-1)
    best_val = -1.0

    for epoch in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["state_labels"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.classify_cot(ids, mask)
            loss = crit(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        val_acc = evaluate_cot_state_accuracy(model, val_loader, device)
        print(f"      cot epoch {epoch + 1}/{N_EPOCHS} | val state acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            if SAVE_CHECKPOINTS:
                torch.save({"model_state": model.state_dict()}, save_path)

    if save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device)["model_state"])
    return model


@torch.no_grad()
def evaluate_direct_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        preds = model.classify_direct(ids, mask).argmax(dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_cot_state_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["state_labels"].to(device)
        preds = model.classify_cot(ids, mask).argmax(dim=-1)
        m = labels != -1
        correct += int((preds[m] == labels[m]).sum().item())
        total += int(m.sum().item())
    return correct / max(total, 1)


def get_or_train_transformer(task, model_kind, seed, device, train_strings, train_labels, val_strings, val_labels):
    ckpt_path = Path(CHECKPOINT_DIR) / f"transformer_{task['name']}_{model_kind}_seed{seed}.pt"

    if model_kind == "direct":
        model = TransformerEncoder(len(VOCAB_BASE), 2, D_MODEL, N_HEADS, max(MAX_BASE_LEN, MAX_COT_LEN), DROPOUT)
    else:
        model = TransformerEncoder(len(cot_vocab_for_task(task)), task["n_states"], D_MODEL, N_HEADS, max(MAX_BASE_LEN, MAX_COT_LEN), DROPOUT)

    if LOAD_EXISTING_CHECKPOINTS and ckpt_path.exists():
        print(f"      loading Transformer checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        return model.to(device)

    print(f"      training Transformer: task={task['name']}, model={model_kind}, seed={seed}")
    if model_kind == "direct":
        train_loader = DataLoader(DirectDataset(train_strings, train_labels, MAX_BASE_LEN), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(DirectDataset(val_strings, val_labels, MAX_BASE_LEN), batch_size=BATCH_SIZE, shuffle=False)
        return train_direct(model, train_loader, val_loader, device, ckpt_path)

    cot_mode = "cot_true" if model_kind == "cot_true" else "cot_pad"
    train_loader = DataLoader(CoTDataset(train_strings, task, MAX_COT_LEN, cot_mode), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CoTDataset(val_strings, task, MAX_COT_LEN, cot_mode), batch_size=BATCH_SIZE, shuffle=False)
    return train_cot(model, train_loader, val_loader, device, ckpt_path)


def transformer_predict_label(model, string, task, model_kind, device):
    model.eval()
    with torch.no_grad():
        if model_kind == "direct":
            tok = tokenise(string, VOCAB_BASE, MAX_BASE_LEN)
            ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.tensor(tok["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
            return int(model.classify_direct(ids, mask).argmax(dim=-1).item())

        cot_mode = "cot_true" if model_kind == "cot_true" else "cot_pad"
        interleaved, _, pred_positions = build_interleaved_sequence(string, task["transition_fn"], cot_mode=cot_mode)
        tok = tokenise(interleaved, cot_vocab_for_task(task), MAX_COT_LEN)
        ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(tok["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
        logits = model.classify_cot(ids, mask)
        pred_state = 0 if not pred_positions else int(logits[0, pred_positions[-1]].argmax(dim=-1).item())
        return int(task["state_to_label_fn"](pred_state))


def transformer_final_representation_and_pred(model, string, task, model_kind, device):
    # direct: class = accept/reject label, representation = CLS hidden state.
    # cot: class = predicted DFA state, representation = last real-token hidden state.
    model.eval()
    with torch.no_grad():
        if model_kind == "direct":
            tok = tokenise(string, VOCAB_BASE, MAX_BASE_LEN)
            ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.tensor(tok["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
            hidden = model.get_cls_representation(ids, mask).squeeze(0)
            pred = int(model.classifier(hidden).argmax(dim=-1).item())
            return pred, hidden.detach()

        cot_mode = "cot_true" if model_kind == "cot_true" else "cot_pad"
        interleaved, _, pred_positions = build_interleaved_sequence(string, task["transition_fn"], cot_mode=cot_mode)
        tok = tokenise(interleaved, cot_vocab_for_task(task), MAX_COT_LEN)
        ids = torch.tensor(tok["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(tok["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)
        hidden_all = model.get_hidden_states(ids, mask, causal=True)
        last_pos = pred_positions[-1] if pred_positions else 0
        hidden = hidden_all[0, last_pos, :]
        pred = int(model.classifier(hidden).argmax(dim=-1).item())
        return pred, hidden.detach()


class DCSAClassifier(nn.Module):
    # Recurrent surrogate for the Transformer. The classifier head is copied
    # from the Transformer and frozen; the RNN and pre-classifier learn to
    # reproduce the Transformer's predictions and final hidden representation.
    def __init__(self, vocab_size, d_model, classifier_w, classifier_b):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.pre_classifier = nn.Linear(d_model, d_model)
        nn.init.eye_(self.pre_classifier.weight)
        nn.init.zeros_(self.pre_classifier.bias)
        self.classifier = nn.Linear(d_model, classifier_w.size(0))
        self.classifier.weight = nn.Parameter(classifier_w.detach().clone(), requires_grad=False)
        self.classifier.bias = nn.Parameter(classifier_b.detach().clone(), requires_grad=False)

    def forward(self, input_ids, lengths=None):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if lengths is None:
                lengths = torch.tensor([input_ids.size(1)], device=input_ids.device)

        if input_ids.size(1) == 0:
            h_final = torch.zeros(input_ids.size(0), self.d_model, device=input_ids.device)
        else:
            embedded = self.embedding(input_ids)
            if lengths is not None:
                lengths = torch.clamp(lengths, min=1)
                packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, h_n = self.rnn(packed)
            else:
                _, h_n = self.rnn(embedded)
            h_final = h_n.squeeze(0)

        pre_out = self.pre_classifier(h_final)
        return self.classifier(pre_out), pre_out

    def classify_state_tensor(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(self.pre_classifier(state))
        return int(logits.argmax(dim=-1).item())

    def get_initial_state(self):
        return [0.0] * self.d_model

    def get_next_state(self, state, char, device):
        token = int(char)
        with torch.no_grad():
            token_tensor = torch.tensor([[token]], dtype=torch.long, device=device)
            embedded = self.embedding(token_tensor)
            h_prev = torch.tensor(state, dtype=torch.float32, device=device).view(1, 1, self.d_model)
            _, h_next = self.rnn(embedded, h_prev)
            h_vec = h_next.squeeze(0).squeeze(0)
            pred = self.classify_state_tensor(h_vec)
        return h_vec.detach().cpu().tolist(), pred


def precompute_transformer_targets(transformer, strings, task, model_kind, device):
    token_ids, tf_preds, tf_reprs = [], [], []
    transformer.eval()
    with torch.no_grad():
        for s in strings:
            pred, rep = transformer_final_representation_and_pred(transformer, s, task, model_kind, device)
            token_ids.append(torch.tensor(s, dtype=torch.long))
            tf_preds.append(pred)
            tf_reprs.append(rep.detach().cpu())
    return token_ids, torch.tensor(tf_preds, dtype=torch.long), torch.stack(tf_reprs)


def train_dcsa(dcsa, transformer, strings, task, model_kind, device):
    dcsa.to(device)
    token_ids, tf_preds, tf_reprs = precompute_transformer_targets(transformer, strings, task, model_kind, device)
    lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long)
    padded_ids = pad_sequence(token_ids, batch_first=True, padding_value=0).to(device)
    lengths, tf_preds, tf_reprs = lengths.to(device), tf_preds.to(device), tf_reprs.to(device)

    params = list(dcsa.embedding.parameters()) + list(dcsa.rnn.parameters()) + list(dcsa.pre_classifier.parameters())
    opt = optim.AdamW(params, lr=DCSA_LR)
    ce = CrossEntropyLoss()
    l1 = nn.L1Loss()
    n = padded_ids.size(0)

    for epoch in range(DCSA_EPOCHS):
        dcsa.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, DCSA_BATCH_SIZE):
            idx = perm[start:start + DCSA_BATCH_SIZE]
            logits, h_final = dcsa(padded_ids[idx], lengths[idx])
            loss = DCSA_CE_WEIGHT * ce(logits, tf_preds[idx]) + DCSA_L1_WEIGHT * l1(h_final, tf_reprs[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)

        if (epoch + 1) % max(1, DCSA_EPOCHS // 10) == 0 or epoch == DCSA_EPOCHS - 1:
            print(f"      DCSA epoch {epoch + 1}/{DCSA_EPOCHS} | loss={total_loss / n:.5f}")

    return dcsa


@torch.no_grad()
def evaluate_dcsa_faithfulness(dcsa, transformer, strings, task, model_kind, device):
    dcsa.eval()
    transformer.eval()
    agreements = 0
    total_l1 = 0.0
    for s in strings:
        tf_pred, tf_repr = transformer_final_representation_and_pred(transformer, s, task, model_kind, device)
        token_ids = torch.tensor(s, dtype=torch.long, device=device)
        logits, h_final = dcsa(token_ids)
        dcsa_pred = int(logits.argmax(dim=-1).item())
        agreements += int(dcsa_pred == tf_pred)
        total_l1 += float(torch.mean(torch.abs(h_final.squeeze(0) - tf_repr)).item())
    n = max(len(strings), 1)
    return agreements / n, total_l1 / n


class DCSANetworkAdapter:
    def __init__(self, dcsa, task, model_kind, device):
        self.dcsa = dcsa
        self.task = task
        self.model_kind = model_kind
        self.device = device
        self.alphabet = "01"
        self.dcsa.eval()

    def _pred_to_label(self, pred):
        if self.model_kind == "direct":
            return int(pred)
        return int(self.task["state_to_label_fn"](pred))

    def classify_word(self, word):
        if len(word) == 0:
            state = torch.zeros(self.dcsa.d_model, device=self.device)
            pred = self.dcsa.classify_state_tensor(state)
        else:
            ids = torch.tensor([int(c) for c in word], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits, _ = self.dcsa(ids)
            pred = int(logits.argmax(dim=-1).item())
        return self._pred_to_label(pred)

    def get_first_RState(self):
        state = self.dcsa.get_initial_state()
        pred = self.dcsa.classify_state_tensor(torch.tensor(state, dtype=torch.float32, device=self.device))
        return state, self._pred_to_label(pred)

    def get_next_RState(self, state, char):
        next_state, pred = self.dcsa.get_next_state(state, char, self.device)
        return next_state, self._pred_to_label(pred)


def dfa_predict(dfa, string):
    word = "".join(str(int(x)) for x in string)
    for method_name in ["classify_word", "classify", "accepts", "accept", "predict", "__call__"]:
        if hasattr(dfa, method_name):
            method = getattr(dfa, method_name)
            for arg in (word, list(string)):
                try:
                    out = method(arg)
                    if isinstance(out, torch.Tensor):
                        out = out.item()
                    return int(out)
                except Exception:
                    pass
    raise AttributeError("Could not classify a string with this DFA.")


def get_num_states(dfa):
    for attr in ["states", "Q", "_states"]:
        if hasattr(dfa, attr):
            try:
                return len(getattr(dfa, attr))
            except Exception:
                pass
    for method_name in ["num_states", "get_num_states"]:
        if hasattr(dfa, method_name):
            try:
                return int(getattr(dfa, method_name)())
            except Exception:
                pass
    return None


def evaluate_dfa_ground_truth(dfa, strings, task):
    correct = total = 0
    for s in strings:
        try:
            correct += int(dfa_predict(dfa, s) == task["label_fn"](s))
            total += 1
        except Exception:
            continue
    return correct / max(total, 1)


def evaluate_dfa_fidelity(dfa, transformer, strings, task, model_kind, device):
    correct = total = 0
    for s in strings:
        try:
            correct += int(dfa_predict(dfa, s) == transformer_predict_label(transformer, s, task, model_kind, device))
            total += 1
        except Exception:
            continue
    return correct / max(total, 1)


def save_dfa(dfa, path):
    import pickle
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dfa, f)


def run_whitebox_extraction(dcsa, task, model_kind, device):
    adapter = DCSANetworkAdapter(dcsa=dcsa, task=task, model_kind=model_kind, device=device)
    dfa = zhang_extract(
        adapter,
        time_limit=ZHANG_TIME_LIMIT,
        initial_split_depth=ZHANG_INITIAL_SPLIT_DEPTH,
        starting_examples=list(ZHANG_STARTING_EXAMPLES),
    )
    return dfa


def make_base_row(task, seed, model_kind, backend, agreement, mean_l1, faithfulness_pass, notes=""):
    return {
        "task": task["name"],
        "seed": seed,
        "model_kind": model_kind,
        "backend": backend,
        "dcsa_agreement": f"{agreement:.6f}",
        "dcsa_mean_l1": f"{mean_l1:.6f}",
        "faithfulness_pass": int(bool(faithfulness_pass)),
        "extraction_success": 0,
        "n_extracted_states": "",
        "n_minimal_states": task["minimal_dfa_states"],
        "state_count_error": "",
        "fidelity_to_transformer_iid": "",
        "fidelity_to_transformer_ood": "",
        "ground_truth_accuracy_iid": "",
        "ground_truth_accuracy_ood": "",
        "notes": notes,
    }


def fill_extraction_metrics(row, dfa, transformer, task, model_kind, device, iid_strings, ood_strings):
    n_states = get_num_states(dfa)
    row["extraction_success"] = 1
    row["n_extracted_states"] = "" if n_states is None else int(n_states)
    row["state_count_error"] = "" if n_states is None else int(n_states) - task["minimal_dfa_states"]
    row["fidelity_to_transformer_iid"] = f"{evaluate_dfa_fidelity(dfa, transformer, iid_strings, task, model_kind, device):.6f}"
    row["fidelity_to_transformer_ood"] = f"{evaluate_dfa_fidelity(dfa, transformer, ood_strings, task, model_kind, device):.6f}"
    row["ground_truth_accuracy_iid"] = f"{evaluate_dfa_ground_truth(dfa, iid_strings, task):.6f}"
    row["ground_truth_accuracy_ood"] = f"{evaluate_dfa_ground_truth(dfa, ood_strings, task):.6f}"
    return row


def run_condition(task_name, seed, model_kind, device):
    task = TASKS[task_name]
    print("\n" + "=" * 88)
    print(f"TASK={task['name']} | SEED={seed} | MODEL={model_kind}")
    print("=" * 88)
    set_seed(seed)

    strings, labels = generate_dataset(TRAIN_MIN_LEN, TRAIN_MAX_LEN, TRAIN_SAMPLES, task["label_fn"], seed=seed + 1_000)
    train_strings, val_strings, train_labels, val_labels = train_test_split(
        strings, labels, test_size=0.2, random_state=seed, stratify=labels if len(set(labels)) > 1 else None,
    )

    transformer = get_or_train_transformer(task, model_kind, seed, device, train_strings, train_labels, val_strings, val_labels)

    dcsa_strings, _ = generate_dataset(TRAIN_MIN_LEN, TRAIN_MAX_LEN, DCSA_TRAIN_SAMPLES, task["label_fn"], seed=seed + 3_000)
    faith_strings, _ = generate_dataset(IID_MIN_LEN, IID_MAX_LEN, FAITHFULNESS_SAMPLES, task["label_fn"], seed=seed + 4_000)
    iid_strings, _ = sample_eval_strings(task, "iid", seed)
    ood_strings, _ = sample_eval_strings(task, "ood", seed)

    print("      training DCSA surrogate")
    dcsa = DCSAClassifier(
        vocab_size=2,  # DCSA reads raw bits
        d_model=D_MODEL,
        classifier_w=transformer.classifier.weight.data,
        classifier_b=transformer.classifier.bias.data,
    )
    dcsa = train_dcsa(dcsa, transformer, dcsa_strings, task, model_kind, device)

    agreement, mean_l1 = evaluate_dcsa_faithfulness(dcsa, transformer, faith_strings, task, model_kind, device)
    faithfulness_pass = agreement >= FAITHFULNESS_AGREEMENT_THRESHOLD and mean_l1 <= FAITHFULNESS_L1_THRESHOLD
    print(f"      DCSA faithfulness: agreement={agreement:.4f}, mean_l1={mean_l1:.4f}, pass={faithfulness_pass}")

    if not faithfulness_pass:
        row = make_base_row(task, seed, model_kind, backend="none", agreement=agreement, mean_l1=mean_l1,
                             faithfulness_pass=False, notes="DCSA failed faithfulness gate; extraction skipped.")
        append_result(row)
        return

    print("      running whitebox extraction")
    row = make_base_row(task, seed, model_kind, backend="zhang_lstar", agreement=agreement, mean_l1=mean_l1, faithfulness_pass=True)
    try:
        dfa = run_whitebox_extraction(dcsa, task, model_kind, device)
        row = fill_extraction_metrics(row, dfa, transformer, task, model_kind, device, iid_strings, ood_strings)
        save_dfa(dfa, Path(DFA_DIR) / f"dfa_{task['name']}_{model_kind}_seed{seed}.pkl")
    except Exception as exc:
        row["notes"] = f"extraction failed: {type(exc).__name__}: {exc}"

    append_result(row)

# MAIN
def run_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Results CSV: {RESULTS_CSV}")

    for task_name in TASK_NAMES:
        for seed in SEEDS:
            for model_kind in MODEL_KINDS:
                run_condition(task_name, seed, model_kind, device)

# run_all()
