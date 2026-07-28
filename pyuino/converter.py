import time
import torch
from logging import getLogger
import torch.nn.functional as F
from .model import YuinoModel
from .dictionary import YuinoDictionary


class YuinoConverter:
    def __init__(self, model_path: str, device="cpu"):
        self._logger = getLogger('YuinoServer')
        self._dict = YuinoDictionary(model_path=model_path)
        self._model = YuinoModel.from_pretrained(model_path).to(device).eval()
        self._loss_func = torch.nn.BCEWithLogitsLoss()
        self._device = device

        self._kana = ""
        self._preedit = ""
        self._past_key_values = None
        self._candidates = [(0., [self._dict.bos_id], None)]

    @torch.no_grad()
    def convert(self, text, removed_check=True):
        start_time = time.time()
        word_tree = self._dict.build_word_tree(text)
        removed = self._set_kana(text) if removed_check else False

        if not removed:
            for i, yomi_s in enumerate(word_tree):
                if i < self.len_fixed:
                    # 既に予測済みのため次のフレーズへ進む
                    continue

                min_cost = 0.
                min_loss = 0.
                min_pos_prob = 0.
                min_words = []
                min_past_key_values = None
                for yomi in yomi_s:
                    # Predict the next word vector from the previous words
                    pre_words = self.get_candidate(i - len(yomi))
                    word_pred, pos_pred, past_key_values = self.predict(pre_words[1][-1], pre_words[2])
                    _, pos_top_k_indices = torch.topk(pos_pred, k=5, dim=1)

                    for wid in self._dict.gets(yomi):
                        loss = self.cost(word_pred, wid)
                        pos_prob = pos_pred.squeeze()[self._dict.pos(wid)].item()
                        cost = loss + pre_words[0]
                        if self._dict.pos(wid) not in pos_top_k_indices:
                            # ほぼありえない品詞なためコスト無効化
                            cost += 0xff
                        if min_cost == 0. or cost < min_cost:
                            min_cost = cost
                            min_loss = loss
                            min_pos_prob = pos_prob
                            min_words = pre_words[1] + [wid]
                            min_past_key_values = past_key_values

                # fixed this index
                self._candidates.append((min_cost, min_words, min_past_key_values))
                self._logger.debug("%f (%f/%f) %s" % (min_cost, min_loss, min_pos_prob, str([self._dict.surface(wid) for wid in min_words])))

        fixed_words = self._fixed_text()
        self._logger.info("%s : %f sec" % (fixed_words, time.time() - start_time))
        return fixed_words

    def predict(self, wid: int, past_key_values):
        wt, pos = self._dict.embed([wid])
        y = self._model(
            inputs_embeds=wt.to(self._device),
            inputs_poss=pos.to(self._device),
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = y.logits[:, -1, :]
        word_pred = logits[:, :64]
        pos_pred = F.softmax(logits[:, 64:])
        return word_pred, pos_pred, y.past_key_values

    def cost(self, pred, wid):
        embed = self._dict.word_embed(wid)
        loss = self._loss_func(embed, pred).item()
        return loss

    @property
    def len_fixed(self):
        return len(self._candidates)

    def _set_kana(self, kana: str):
        removed = False
        if len(self._kana) > 0:
            if len(kana) < len(self._kana):
                # 1文字消されている
                self._candidates.pop()
                removed = True
        else:
            # 初回時なのでリセット
            self._kana = ""
            self._preedit = ""
            self._past_key_values = None
            self._candidates = [(0., [self._dict.bos_id], None)]
            removed = True

        self._kana = kana
        return removed

    def _fixed_text(self):
        fixed_words = ""
        for i, word in enumerate(self._candidates[-1][1]):
            if i != 0:
                fixed_words += self._dict.surface(word)
        return fixed_words

    def get_candidate(self, idx):
        return self._candidates[idx]
