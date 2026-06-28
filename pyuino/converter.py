import time
import torch
from logging import getLogger
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
                min_words = []
                min_past_key_values = None
                for yomi in yomi_s:
                    # Predict the next word vector from the previous words
                    pre_words = self.get_candidate(i - len(yomi))
                    pred, past_key_values = self.predict(pre_words[1][-1], pre_words[2])

                    for wid in self._dict.gets(yomi):
                        embed = self._dict.embed([wid]).squeeze(0)
                        cost = self.loss(pred, embed) + pre_words[0]
                        if min_cost == 0. or cost < min_cost:
                            min_cost = cost
                            min_words = pre_words[1] + [wid]
                            min_past_key_values = past_key_values

                # fixed this index
                self._candidates.append((min_cost, min_words, min_past_key_values))
                self._logger.debug("%f %s" % (min_cost, str([self._dict.surface(wid) for wid in min_words])))

        fixed_words = self._fixed_text()
        self._logger.info("%s : %f sec" % (fixed_words, time.time() - start_time))
        return fixed_words

    def predict(self, wid: int, past_key_values):
        wt = self._dict.embed([wid]).to(self._device)
        y = self._model(inputs_embeds=wt, past_key_values=past_key_values, use_cache=True)
        return y.logits[:, -1, :], y.past_key_values

    def loss(self, y, y_hat):
        return self._loss_func(y, y_hat).item()

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
