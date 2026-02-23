import torch
from torch import nn
from typing import Optional
from transformers import Qwen3Model, Qwen3PreTrainedModel, Qwen3Config
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache


class YuinoOnnx(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values_flat):
        past_key_values = None
        if len(past_key_values_flat) > 0:
            past_key_values = []
            for i in range(self.num_layers):
                k = past_key_values_flat[i * 2]
                v = past_key_values_flat[i * 2 + 1]
                past_key_values.append((k, v))
            past_key_values = tuple(past_key_values)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )

        present_key_values_flat = []
        for k, v in outputs.past_key_values:
            present_key_values_flat.append(k)
            present_key_values_flat.append(v)

        return outputs.logits, *present_key_values_flat


class YuinoModel(Qwen3PreTrainedModel):
    word_emb_size = 32
    pos_ids_size = 57
    label_emb_size = 768

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.sigmoid = nn.Sigmoid()

        self.word_enc = nn.Linear(self.label_emb_size, self.word_emb_size)
        self.pos_emb = nn.Embedding(self.pos_ids_size, self.word_emb_size)
        self.lm_in = nn.Linear((self.word_emb_size * 2), config.hidden_size, bias=False)
        self.lm_out = nn.Linear(config.hidden_size, (self.word_emb_size * 2), bias=False)
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
            x_in = self.sigmoid(self.word_enc(labels))
            x_in = torch.where((x_in > 0.5), 1., 0.)
            input_p_embs = self.sigmoid(self.pos_emb(inputs_poss))
            input_p_embs = torch.where((input_p_embs > 0.5), 1., 0.)
            inputs_embeds = torch.cat((x_in, input_p_embs), dim=2)

        inputs_embeds_in = self.lm_in(inputs_embeds)

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
        logits = self.lm_out(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # get loss for Kana-Kanji conversion
            shift_emb_labels = inputs_embeds[:, 1:].contiguous()
            emp_emb_labels = torch.zeros((inputs_embeds.shape[0], 1, inputs_embeds.shape[2]), dtype=torch.float, device=inputs_embeds.device)
            shift_emb_labels = torch.cat((shift_emb_labels, emp_emb_labels), dim=1)
            loss = self.loss_func(logits, shift_emb_labels)
            loss = loss.view(loss.size(0), -1).sum(dim=1).mean()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

    def get_uint_id(self, labels: torch.Tensor) -> int:
        y = self.sigmoid(self.word_enc(labels))
        y = torch.where((y > 0.5), 1, 0)
        powers = 2 ** torch.arange(y.size(2) - 1, -1, -1)
        return (y * powers).sum().item()

    def get_pos_id(self, inputs_poss: torch.LongTensor) -> int:
        y = self.sigmoid(self.pos_emb(inputs_poss))
        y = torch.where((y > 0.5), 1, 0)
        powers = 2 ** torch.arange(y.size(0) - 1, -1, -1)
        return (y * powers).sum().item()
