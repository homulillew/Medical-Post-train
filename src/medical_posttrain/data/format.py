"""One explicit target format for SFT and RL. No invented Huatuo reasoning."""
import re

def target(answer, reasoning=""):
    for value in (answer, reasoning):
        if any(tag in value for tag in ("<think>", "</think>", "<answer>", "</answer>", "<|im_")):
            raise ValueError("Reserved output delimiter in source text")
    return f"<think>\n{reasoning.strip()}\n</think>\n\n<answer>{answer.strip()}</answer>"

def parse_answer(text):
    if text.count('<think>') != text.count('</think>') or text.count('<think>') > 1:
        return None
    match = re.fullmatch(r"\s*(?:<think>[\s\S]*?</think>\s*)?<answer>\s*([A-Ea-e\s,，、;；]+)\s*</answer>\s*", text)
    if not match or text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    letters = re.sub(r"[\s,，、;；]", "", match[1]).upper()
    if not letters or len(set(letters)) != len(letters):
        return None
    return "".join(sorted(letters))

def messages(question, answer=None, reasoning=""):
    rows = [{"role":"user", "content":question}]
    if answer is not None:
        rows.append({"role":"assistant", "content":target(answer, reasoning)})
    return rows

def sft_tokens(tokenizer, question, answer, reasoning=""):
    prompt = tokenizer.apply_chat_template(messages(question), tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    full = tokenizer.apply_chat_template(messages(question,answer,reasoning), tokenize=True, return_dict=False, add_generation_prompt=False, enable_thinking=True)
    if full[:len(prompt)] != prompt:
        raise ValueError("Template prompt/target boundary mismatch")
    return full, [-100]*len(prompt)+full[len(prompt):]
