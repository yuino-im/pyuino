import time
from .dictionary import YuinoDictionary


class YuinoConverter:
    def __init__(self, model_path: str, device="cpu"):
        self._dict = YuinoDictionary(model_path=model_path, device=device)

    def convert(self, text, top_k=1):
        start_time = time.time()
        word_tree = self._dict.build_word_tree(text)
        print(word_tree)

        candidates = []
        for ym in word_tree[0]:
            next_idx = len(ym)
            for wid in self._dict.gets(ym):
                words = [self._dict.bos_id, wid]
                embed = self._dict.embed([wid]).squeeze(0)
                loss = self._dict.loss(self._dict.first_embed, embed)
                candidates.append((loss.item(), words, next_idx, [self._dict.surface(wid) for wid in words]))

        fixed = []
        while True:
            cand_tmp = []
            for pre in candidates:
                if pre[2] >= len(text):
                    continue

                pred = self._dict.predict(pre[1])
                for ym in word_tree[pre[2]]:
                    next_idx = pre[2] + len(ym)
                    for wid in self._dict.gets(ym):
                        words = pre[1] + [wid]
                        embed = self._dict.embed([wid]).squeeze(0)
                        loss = self._dict.loss(pred, embed)
                        if next_idx < len(text):
                            cand_tmp.append((loss.item(), words, next_idx, [self._dict.surface(wid) for wid in words]))
                        else:
                            fixed.append((loss.item(), words, next_idx, [self._dict.surface(wid) for wid in words]))

            if len(cand_tmp) > 0:
                candidates = sorted(cand_tmp)[:top_k]
                print("-------------------------------------")
                print(candidates)
            else:
                break

        fixed_words = ""
        for word in sorted(fixed)[0][3][1:]:
            fixed_words += word

        print("%s : %f msec" % (fixed_words, time.time() - start_time))


