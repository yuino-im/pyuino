import time
import torch
from .model import YuinoModel
from .dictionary import YuinoDictionary


class YuinoConverter:
    def __init__(self, model_path: str, device="cpu"):
        self._dict = YuinoDictionary(model_path=model_path)
        self._model = YuinoModel.from_pretrained(model_path).to(device).eval()
        self._loss_func = torch.nn.BCEWithLogitsLoss()
        self._device = device

    def convert(self, text):
        past_key_values = None
        start_time = time.time()
        word_tree = self._dict.build_word_tree(text)

        with torch.no_grad():
            candidates = []
            for i, yomi_s in enumerate(word_tree):
                if i == 0:
                    # Since [CLS] is included, the 0th one is ignored.
                    candidates.append((0., [self._dict.bos_id], past_key_values),)
                    continue

                min_cost = None
                min_words = None
                min_past_key_values = None
                for yomi in yomi_s:
                    # Predict the next word vector from the previous words
                    pre_words = candidates[i - len(yomi)]
                    pred, past_key_values = self.predict(pre_words[1][-1], pre_words[2])

                    for wid in self._dict.gets(yomi):
                        embed = self._dict.embed([wid]).squeeze(0)
                        cost = self.loss(pred, embed) + pre_words[0]
                        if min_cost is None or cost < min_cost:
                            min_cost = cost
                            min_words = pre_words[1] + [wid]
                            min_past_key_values = past_key_values

                # fixed this index
                candidates.append((min_cost, min_words, min_past_key_values))
                print(min_cost, [self._dict.surface(wid) for wid in min_words])

        fixed_words = ""
        for i, word in enumerate(candidates[-1][1]):
            if i != 0:
                fixed_words += self._dict.surface(word)

        print("%s : %f sec" % (fixed_words, time.time() - start_time))
        return fixed_words

    def predict(self, wid: int, past_key_values):
        wt = self._dict.embed([wid]).to(self._device)
        y = self._model(inputs_embeds=wt, past_key_values=past_key_values, use_cache=True)
        return y.logits[:, -1, :], y.past_key_values

    def loss(self, y, y_hat):
        return self._loss_func(y, y_hat).item()



