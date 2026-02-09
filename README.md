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
https://www.dropbox.com/scl/fo/03sverk4gsj3l8qmx9ltw/ACueNBsN8EwwSYP18v2a1lQ?rlkey=99ki15e75q3cx9ddmotzgcpah&st=2n8rooyz&dl=0

* config.json
* model.safetensors
* yuino_dict.pb

ダウンロードしたファイルは `YuinoLM` 下に配置してください  

### ToyBox の起動
ToyBox は pyuino のデモアプリケーションです  
かなを入力すると、変換後のかな漢字を返します  

```shell
$ pyuino-toybox
--Yuino TOY-BOX--
かな > はこねおんせんへようこそ
0.5473639369010925 ['[CLS]', '羽']
0.5501382946968079 ['[CLS]', '筐']
0.6909381747245789 ['[CLS]', 'はこね']
1.3568682670593262 ['[CLS]', 'はこね', 'お']
1.3223243355751038 ['[CLS]', '筐', 'ネオン']
1.4771115183830261 ['[CLS]', 'はこね', 'オンセ']
0.6929624676704407 ['[CLS]', '箱根温泉']
1.1179965436458588 ['[CLS]', '箱根温泉', 'へ']
1.5004573464393616 ['[CLS]', '箱根温泉', 'へよ']
1.525723159313202 ['[CLS]', '箱根温泉', 'へよう']
1.8744302093982697 ['[CLS]', '箱根温泉', 'へ', '楊子']
1.8722382485866547 ['[CLS]', '箱根温泉', 'へ', 'ようこそ']
箱根温泉へようこそ : 0.990405 sec
漢字: 箱根温泉へようこそ
```

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

#### LLMトークナイザ
LINE DistilBERT Japanese (LINE Corporation)  
https://huggingface.co/line-corporation/line-distilbert-base-japanese

#### データセット
CC100(Japanese)  
https://huggingface.co/datasets/range3/cc100-ja


