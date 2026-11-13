# siglip qwen vlm for task 2
# frozen encoder then connector then frozen lm with lora

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# spec settings

QWEN = "Qwen/Qwen3-0.6B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
SEQ_QUESTION_CAP = 80
CAPTION_CAP = 64
ANSWER_CAP = 16


# connector

class Connector(nn.Module):
    def __init__(self, in_dim=768, hid=1024, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, out_dim))

    def forward(self, x):
        return self.net(x)


# model

class VLM(nn.Module):
    def __init__(self, device, n_visual=196):
        super().__init__()
        self.device = device
        self.n_visual = n_visual
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        # lm frozen
        base = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16)
        for p in base.parameters():
            p.requires_grad = False
        self.embed = base.get_input_embeddings()
        # lora starts at zero
        self.lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                   target_modules=LORA_TARGETS, bias="none")
        self.lm = get_peft_model(base, self.lora_cfg).to(device)
        self.core = self.lm.get_base_model()
        self.core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.core.config.use_cache = False
        self.connector = Connector().to(device)
        self.scale = 1.0
        self.scale_rule = None
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.set_stage(1)

    # stages

    def set_stage(self, stage):
        # stage 1 connector only
        lora_on = stage >= 2
        for n, p in self.core.named_parameters():
            if "lora_" in n:
                p.requires_grad = lora_on
        for p in self.connector.parameters():
            p.requires_grad = True

    def trainable(self, stage):
        conn = list(self.connector.parameters())
        lora = [p for n, p in self.core.named_parameters() if "lora_" in n] if stage >= 2 else []
        return conn, lora

    # calibration

    @torch.no_grad()
    def calibrate(self, feats, text_ids):
        # training captions only
        vis = self.connector(feats.to(self.device).float())
        txt = self.embed(text_ids.to(self.device)).float()
        self.scale = (txt.std() / vis.std()).item()
        self.scale_rule = ("scale = std(text token embeddings) / std(connector output), "
                           "initial connector, training caption tokens only")
        return self.scale

    # sequence assembly

    def encode_text(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def build_example(self, feat, question, answer, cap_answer, with_image=True):
        # returns prefix suffix and target ids
        q_ids = self.encode_text(question)[:SEQ_QUESTION_CAP]
        a_ids = self.encode_text(answer)[:cap_answer] + [self.im_end]
        prefix = self.encode_text("<|im_start|>user\n")
        # empty think block
        suffix = q_ids + self.encode_text("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        return prefix, suffix, a_ids

    def assemble(self, feats, parts, with_image=True, left_pad=False, targets=True):
        B = len(parts)
        vis = self.connector(feats.to(self.device).float()) * self.scale if with_image else None
        seqs, labs = [], []
        for b, (prefix, suffix, target) in enumerate(parts):
            emb = [self.embed(torch.tensor(prefix, device=self.device))]
            lab = [-100] * len(prefix)
            if with_image:
                emb.append(vis[b])
                lab += [-100] * self.n_visual
            tail = suffix + (target if targets else [])
            emb.append(self.embed(torch.tensor(tail, device=self.device)).float())
            lab += [-100] * len(suffix) + (target if targets else [])
            seqs.append(torch.cat(emb, dim=0))
            labs.append(torch.tensor(lab, device=self.device))
        # pad to longest
        L = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        embeds = torch.zeros(B, L, D, device=self.device)
        mask = torch.zeros(B, L, dtype=torch.long, device=self.device)
        labels = torch.full((B, L), -100, dtype=torch.long, device=self.device)
        for b, (s, l) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            # left pad for generation
            if left_pad:
                embeds[b, L - n:] = s
                mask[b, L - n:] = 1
                labels[b, L - n:] = l
            # right pad for training
            else:
                embeds[b, :n] = s
                mask[b, :n] = 1
                labels[b, :n] = l
        return embeds, mask, labels

    # loss

    def example_nll(self, embeds, mask, labels):
        # nll per example
        hidden = self.core.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        # shift by one
        tgt = labels[:, 1:]
        valid = tgt != -100
        # lm head on supervised positions
        logits = self.core.lm_head(hidden[:, :-1][valid])
        tok_loss = F.cross_entropy(logits.float(), tgt[valid], reduction="none")
        ex = torch.arange(embeds.shape[0], device=self.device).unsqueeze(1).expand_as(valid)[valid]
        sums = torch.zeros(embeds.shape[0], device=self.device).index_add(0, ex, tok_loss)
        return sums / valid.sum(1).clamp(min=1)

    def loss(self, feats, parts, with_image=True):
        # example weighted loss
        embeds, mask, labels = self.assemble(feats, parts, with_image=with_image)
        return self.example_nll(embeds, mask, labels).mean()
