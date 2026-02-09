import time
from .dictionary import YuinoDictionary


class YuinoConverter:
    def __init__(self, model_path: str, device="cpu"):
        self._dict = YuinoDictionary(model_path=model_path, device=device)

    def convert(self, text):
        start_time = time.time()
        word_tree = self._dict.build_word_tree(text)

        candidates = []
        for i, yomi_s in enumerate(word_tree):
            if i == 0:
                # Since [CLS] is included, the 0th one is ignored.
                candidates.append((0., [self._dict.bos_id]))
                continue

            min_cost = None
            min_words = None
            for yomi in yomi_s:
                # Predict the next word vector from the previous words
                pre_words = candidates[i - len(yomi)]
                pred = self._dict.predict(pre_words[1])

                for wid in self._dict.gets(yomi):
                    embed = self._dict.embed([wid]).squeeze(0)
                    cost = self._dict.loss(pred, embed) + pre_words[0]
                    if min_cost is None or cost < min_cost:
                        min_cost = cost
                        min_words = pre_words[1] + [wid]

            # fixed this index
            candidates.append((min_cost, min_words))
            print(min_cost, [self._dict.surface(wid) for wid in min_words])

        fixed_words = ""
        for i, word in enumerate(candidates[-1][1]):
            if i != 0:
                fixed_words += self._dict.surface(word)

        print("%s : %f sec" % (fixed_words, time.time() - start_time))
        return fixed_words


