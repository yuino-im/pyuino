# pyuino
pyuino は LLM の仕組みを利用したかな漢字変換です  
厳密には Qwen3 モデルがベースになっていますが、入出力が変わっているため llama.cpp 等では **動かない** ことを確認済みです  

まだお試し版です。まともに動かないです。  

### インストール
PyPi からインストールできます  
```shell
pip install pyuino
```

### モデル・辞書ファイルの準備
下記の場所からモデルファイル及び辞書ファイルをダウンロードしてください  
※モデル・辞書ファイルは最新版のリリースタグのコードと紐づいており、mainブランチのコードとは互換性がない場合もあります    
https://www.dropbox.com/scl/fo/03sverk4gsj3l8qmx9ltw/ACueNBsN8EwwSYP18v2a1lQ?rlkey=99ki15e75q3cx9ddmotzgcpah&st=2n8rooyz&dl=0

* config.json
* model.safetensors
* yuino_dict.pb
* pos_id.csv

ダウンロードしたファイルは `YuinoLM` 下に配置してください  

### ToyBox の起動
ToyBox は pyuino のデモアプリケーションです  
かなを入力すると、変換後のかな漢字を返します  

```shell
$ pyuino-toybox
INFO:     Started server process [21290]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```
→ 起動後に http://localhost:8888 へアクセスするとライブ変換で遊べます。  

### サーバーの起動
現在 IBus-Anya を利用することを想定します  
IBus-Anya 起動前に、下記コマンドでサーバー側を起動してください

```shell
docker run -d -p 30055:30055 -v $HOME/.local/share/yuino:/opt/pyuino/YuinoLM ghcr.io/yuino-im/pyuino -m /opt/pyuino/YuinoLM
```

### 使用モデル・データセット
Yuino では下記のモデル、及びデータセットを使用して学習しました  
（ありがとうございます！！）  

#### 辞書
SudachiDict (WorksApplications)  
https://github.com/WorksApplications/SudachiDict

#### データセット
CC100(Japanese)  
https://huggingface.co/datasets/range3/cc100-ja


