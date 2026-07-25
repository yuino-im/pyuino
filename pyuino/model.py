import torch
from torch import nn
from typing import Optional
from transformers import GenerationMixin, GPTNeoXPreTrainedModel, GPTNeoXModel, GPTNeoXConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache


class YuinoModel(GPTNeoXPreTrainedModel, GenerationMixin):
    ft_vec_size = 128
    word_emb_size = 64
    pos_ids_size = 1588

    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        self.model = GPTNeoXModel(config)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.p_loss_func = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        self.sigmoid = nn.Sigmoid()

        self.word_emb = nn.Linear(self.ft_vec_size, self.word_emb_size)
        self.lm_win = nn.Linear(self.word_emb_size, int(config.hidden_size / 2))
        self.pos_emb = nn.Embedding(self.pos_ids_size, int(config.hidden_size / 2), dtype=config.dtype)
        self.lm_head = nn.Linear(config.hidden_size, (self.word_emb_size + self.pos_ids_size), bias=False)
        self.post_init()

    def forward(
            self,
            inputs_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            labels: Optional[torch.Tensor] = None,
            inputs_poss: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> CausalLMOutputWithPast:

        if labels is not None:
            input_w_embs = self.sigmoid(self.word_emb(labels))
            inputs_embeds = torch.where((input_w_embs > 0.5), 1., 0.).to(input_w_embs.dtype)

        input_w_embs = self.lm_win(inputs_embeds)
        inputs_embeds_in = torch.cat((input_w_embs, self.pos_emb(inputs_poss)), dim=2)

        # training model
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds_in,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # get loss word emb
            word_logits = logits[:, :, :self.word_emb_size]
            shift_emb_labels = inputs_embeds[:, 1:].contiguous()
            emp_emb_labels = torch.zeros((inputs_embeds.shape[0], 1, inputs_embeds.shape[2]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            shift_emb_labels = torch.cat((shift_emb_labels, emp_emb_labels), dim=1)
            loss_w = self.loss_func(word_logits, shift_emb_labels)
            loss_w = loss_w.view(loss_w.size(0), -1).sum(dim=1).mean()

            # get loss pos emb
            pos_logits = logits[:, :, self.word_emb_size:]
            shift_pos_labels = inputs_poss[:, 1:].contiguous()
            emp_pos_labels = torch.zeros((inputs_poss.shape[0], 1), dtype=torch.long, device=inputs_poss.device)
            shift_pos_labels = torch.cat((shift_pos_labels, emp_pos_labels), dim=1)
            loss_p = self.p_loss_func(pos_logits.view(-1, self.pos_ids_size), shift_pos_labels.view(-1))

            loss = loss_w + loss_p
            #print("loss_w: %f * loss_p: %f = %f" % (loss_w.item(), loss_p.item(), loss.item()))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def get_word_id(self, labels: torch.Tensor) -> int:
        y = self.sigmoid(self.word_emb(labels.squeeze()))
        y = torch.where((y > 0.5), 1, 0)
        return sum(x * (1 << i) for i, x in enumerate(reversed(y.squeeze().tolist())))


class YuinoConvModel(torch.nn.Module):
    def __init__(self, model: YuinoModel):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        # transformers>=4.5x builds the causal mask with torch.vmap inside
        # create_causal_mask(), which torch.jit.trace cannot trace (it fails with
        # "RuntimeError: unordered_map::at"). Passing an already-4D additive mask
        # makes create_causal_mask early-exit and return it as-is, avoiding vmap.
        b, s, _ = inputs_embeds.shape
        mask = torch.full((s, s), float("-inf"), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0).expand(b, 1, s, s)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=False)
        return outputs.logits
