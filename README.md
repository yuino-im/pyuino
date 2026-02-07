# pyuino
pyuino は LLM の仕組みを利用したかな漢字変換です  
厳密には Qwen3 モデルがベースになっていますが、入出力が変わっているため llama.cpp 等では **動かない** ことを確認済みです  

まだお試し版です。まともに動かないです。  

### モデル・辞書ファイルの準備
下記の場所からモデルファイル及び辞書ファイルをダウンロードしてください
https://www.dropbox.com/scl/fo/03sverk4gsj3l8qmx9ltw/ACueNBsN8EwwSYP18v2a1lQ?rlkey=99ki15e75q3cx9ddmotzgcpah&st=2n8rooyz&dl=0

* config.json
* model.safetensors
* yuino_dict.pb

ダウンロードしたファイルは `YuinoLM` 下に配置してください


### サーバーの起動
現在 IBus-Anya を利用することを想定します  
IBus-Anya 起動前に、下記コマンドでサーバー側を起動してください

```shell
docker run -d -p 30055:30055 -v $HOME/.local/share/yuino:/opt/pyuino/YuinoLM ghcr.io/yuino-im/pyuino -m /opt/pyuino/YuinoLM
```
