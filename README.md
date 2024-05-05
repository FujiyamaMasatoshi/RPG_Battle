# RPG Battle
マルチプレイヤRPGバトルの環境です。PyTorchを使用した深層強化学習用の環境として作成しました。このリポジトリにはRPGバトルのデモプレイを行うことができます。

実際のデモプレイ動画を以下のURLにて公開しております。

https://youtu.be/L2V-pbWtrGk


## キャラクタ画面を表示させながらのプレイ

### 初めて起動する際の準備
1. Anacondaをインストールしてください。
2. conda仮想環境を作成するので、ターミナルを起動して、以下のコマンドを打ち込んでください。

```
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

`env_name`は`conda create -n env_name python=3.10`した時の環境名ですので、適切に決めてください。

### ゲームの起動方法
ターミナルで 
```
conda activate env_name
```
と打ち込んで`env_name`を起動させてください。

その後、ターミナルで
```
python test_run.py
```
と打ち込んで実行すると、
```
ゲームシチュエーションを選択してください (1, 2, 3, 4)
```
とターミナル上にメッセージが表示されるので、()の中から数字を選んでターミナルに打ち込んでください。`1-4`の数字がゲームシチュエーションを表している`situation01.csv - situation04.csv`に対応しています。




### ゲームプレイの仕方


#### 基本的な操作
* スペースキー: ゲームの進行、アクション選択やターゲット選択の決定
* 数字入力: アクションやキャラクタの選択のために使用します。


#### ゲームの進行方法

ゲームを起動すると、ゲームウィンドウが立ち上がります。(キーボード入力しても反応しない場合は立ち上がったウィンドウをマウス等でウィンドウをクリックしてください。)

ウィンドウの下部はメッセージウィンドウとなっており、入力した数字やゲームの進行状況を確認することができます。

ゲームを進行する際は、**スペースキー**を押すとゲームは進行します。

ゲームが進行すると、アクションや敵キャラクタ、味方キャラクタを選択するようにメッセージウィンドウが更新されます。

キーボードから数字を打ち込んでエンターキーorリターンキーを押すことでアクション選択やターゲット選択を行うことができます。

入力された数字はメッセージウィンドウに表示されますので、数字を入力してアクションやターゲットを選択したり、決定するまでは入力した数字をバックスペースキーから削除したりすることができます。
