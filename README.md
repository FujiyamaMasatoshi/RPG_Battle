# RPG Battle
マルチプレイヤRPGバトルの環境です。PyTorchを使用した深層強化学習用の環境として作成しました。このリポジトリにはRPGバトルのデモプレイを行うことができます。

<!--実際のデモプレイ動画を以下のURLにて公開しております。-->
<!--https://youtu.be/L2V-pbWtrGk --> 

## デモ動画

https://github.com/FujiyamaMasatoshi/RPG_Battle/assets/168803336/6a8be18e-6d7e-4450-97ae-fc2d39027336



## RPGバトルのプレイ方法

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

## 模倣学習によって学習したAIとの共闘プレイ方法
ターミナルを立ち上げてconda仮想環境の`env_name`を起動します。

```
conda activate env_name
```

`rpg_battle`のディレクトリまでcdコマンドで移動してください。

```
cd /your/home/directory/.../rpg_battle
```
`python rpg_battle/ImitationLearning/GAIL_test_run.py`は自身のディレクトリに変更してください。

`rpg_battle/ImitationLearning/GAIL_test_run.py`を起動させるため以下のプロンプトを打ち込んでください。

pwdコマンドでターミナルから
```
/Users/.../rpg_battle
```
となっているのを確認できたら以下のコマンドをターミナルに打ち込んでAIとの共闘プレイを行います。

```
python ./ImitationLearning/GAIL_test_run.py
```

上記のコマンドをターミナルに打ち込むと以下のようなメッセージが出力されます。

```
操作するキャラクタidを選択してください [10, 20, 30, 40, a]
(aを選択した場合は全てのキャラクタがAIによるでもプレイを見ることができます)
```

この時、[]の中身の選択肢からターミナルに入力を行い、ゲームを起動します。
### 各選択肢の内容
* 10: 戦士をプレイ、他キャラクタはAI
* 20: 僧侶をプレイ、他キャラクタはAI
* 30: 魔法使いをプレイ、他キャラクタはAI
* 40: 旅芸人をプレイ、他キャラクタはAI
* a: 全てのキャラクタがAIによって操作され、デモプレイを見ることができる

その後、ゲームシチュエーションを選択するようターミナルに指示が出力されるので、`1-4`で選択してください。
```
ゲームシチュエーションを選択してください, [1, 2, 3, 4]
(situatio=4は学習させていない状況です。)
```
`1-4`が各ゲームシチュエーション`situation01.csv - situation04.csv`に対応しています。

洗濯後、ゲームが開始します。
