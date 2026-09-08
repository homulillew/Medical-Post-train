"""Deterministic SFT governance and supervision; no benchmark labels consumed."""
import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def normalize_question(text):
    text = unicodedata.normalize('NFKC', text).strip().lower()
    text = re.sub(r'^(?:第\s*\d+\s*题\s*[:、.]?|\d+\s*[、)]\s*|\d+\.(?!\d)\s*)', '', text)
    # Option labels are presentation, not question identity. Exclusion is deliberately
    # stem-conservative: the same question with reordered/changed options is excluded.
    option_markers = list(re.finditer(r'(?:\s+|(?<=[?？]))[a-e]\s*[.、:)](?!\d)', text))
    if len(option_markers) >= 2:
        text = text[:option_markers[0].start()]
    text = re.sub(r'\s+', '', text)
    keep = []
    for i, c in enumerate(text):
        if unicodedata.category(c).startswith('P'):
            # Preserve decimal values, signs, ratios and unit notation.
            if c in '-+/%' or (c == '.' and i and i+1 < len(text) and text[i-1].isdigit() and text[i+1].isdigit()):
                keep.append(c)
        else:
            keep.append(c)
    return ''.join(keep)

def grams(text, n):
    return {text[i:i+n] for i in range(max(1, len(text)-n+1))}

def quality_reason(messages):
    values = [m['content'] for m in messages]
    if any(not isinstance(v, str) or not v.strip() for v in values):
        return 'empty_question_or_answer'
    text = '\n'.join(values)
    if '\ufffd' in text or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        return 'encoding_corruption'
    if any('<|im_' in v or '<|endoftext|>' in v for v in values):
        return 'reserved_chat_delimiter'
    if len(text) > 40000:
        return 'severe_character_length'
    if re.search(r'(.)\1{29,}', text) or any(n >= 10 for s, n in Counter(re.split(r'[。！？\n]', text)).items() if len(s) >= 12):
        return 'extreme_repetition'
    if re.search(r'写.{0,8}(?:python|java|c\+\+|代码)|股票.{0,8}(?:走势|买卖)|天气预报|写一首.{0,4}诗', messages[0]['content'], re.I) and not re.search(r'医|药|病|患者|症|治疗|健康|心理|营养|手术', text):
        return 'obvious_nonmedical'
    return None

def encode_conversation(tokenizer, messages):
    """Use the native full template and mask each actual assistant span.

    Historical reasoning removal is native Qwen behavior, explicitly counted.
    No source in the pinned Huatuo snapshot is multi-turn; future multi-turn
    records retain all role boundaries/answers, with native history formatting.
    """
    for m in messages:
        if m['role'] not in ('system', 'user', 'assistant') or '<|im_' in m['content']:
            raise ValueError('Unsupported role or source control token')
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if full != encoded['input_ids']:
        raise ValueError('Native template/tokenization mismatch')
    spans = [(m.start(1), m.end(1)) for m in re.finditer(r'<\|im_start\|>assistant\n([\s\S]*?<\|im_end\|>\n)', rendered)]
    if len(spans) != sum(m['role'] == 'assistant' for m in messages):
        raise ValueError('Assistant span count mismatch')
    labels = [token if any(start <= a and b <= end and b > a for start, end in spans) else -100 for token, (a, b) in zip(full, encoded['offset_mapping'])]
    for start, end in spans:
        supervised = [token for token, (a, b) in zip(full, encoded['offset_mapping']) if start <= a and b <= end and b > a]
        if tokenizer.eos_token_id not in supervised:
            raise ValueError('Assistant EOS not supervised')
    reasoning = re.findall(r'<think>\n([\s\S]*?)\n</think>', rendered)
    answers = re.findall(r'<answer>([\s\S]*?)</answer>', rendered)
    if len(answers) != len(spans):
        raise ValueError('Final answer boundary lost')
    stats = dict(total_tokens=len(full), supervised_tokens=sum(x != -100 for x in labels), reasoning_tokens=sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in reasoning), answer_tokens=sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in answers), assistant_turns=len(spans), native_history_think_blocks_removed=len(spans)-len(reasoning))
    return dict(input_ids=full, labels=labels, **stats)

def collate(examples, pad_token_id):
    import torch
    length = max(len(x['input_ids']) for x in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        n = len(row['input_ids'])
        ids[i, :n] = torch.tensor(row['input_ids'])
        labels[i, :n] = torch.tensor(row['labels'])
        attention[i, :n] = 1
    return dict(input_ids=ids, labels=labels, attention_mask=attention)

class DuplicateGraph:
    """Complete prefix join for char-3 Jaccard >= .65 candidate retrieval.

    A common global gram order gives a shared prefix for every qualifying pair.
    Candidates are confirmed using char-5 Jaccard >= .85 or normalized edit-like
    SequenceMatcher >= .90 (the latter within the .65 retrieval boundary).
    Exact hashes bypass this retrieval boundary. This is lexical, not semantic.
    """
    candidate_threshold = .65
    def __init__(self, texts):
        self.texts = texts
        self.parents = list(range(len(texts)))
        self.sets = [grams(t, 3) for t in texts]
        frequency = Counter(g for s in self.sets for g in s)
        self.order = {g: i for i, g in enumerate(sorted(frequency, key=lambda g: (frequency[g], g)))}

    def root(self, i):
        while self.parents[i] != i:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def join(self, callback, progress=lambda n: None):
        index = defaultdict(list)
        exact = {}
        for i, (text, gs) in enumerate(zip(self.texts, self.sets)):
            if text in exact:
                j = exact[text]
                self.parents[self.root(i)] = self.root(j)
                callback(i, j, 'exact_question', 1.0)
                continue
            exact[text] = i
            prefix = sorted(gs, key=self.order.__getitem__)[:len(gs)-math.ceil(self.candidate_threshold*len(gs))+1]
            candidates = set(j for g in prefix for j in index[g])
            for j in sorted(candidates):
                if self.root(i) == self.root(j):
                    continue
                other = self.sets[j]
                if min(len(gs), len(other)) < self.candidate_threshold*max(len(gs), len(other)):
                    continue
                j3 = len(gs & other)/len(gs | other)
                if j3 < self.candidate_threshold:
                    continue
                a, b = grams(text, 5), grams(self.texts[j], 5)
                j5 = len(a & b)/len(a | b)
                if j5 >= .85:
                    reason, similarity = 'near_char5_jaccard', j5
                else:
                    similarity = SequenceMatcher(None, text, self.texts[j], autojunk=False).ratio()
                    if similarity < .90:
                        continue
                    reason = 'near_sequence_similarity'
                self.parents[self.root(i)] = self.root(j)
                callback(i, j, reason, similarity)
            for g in prefix:
                index[g].append(i)
            if i % 10000 == 0:
                progress(i)
