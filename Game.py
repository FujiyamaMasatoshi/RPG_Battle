# Game.pyとの違い
# initメソッドでゲームsituationを選択できる
# init in reset()でも同様

import random
# np.random.seed(123)
import torch
from collections import deque
import csv
from Character import Character
from GameState import GameState


# ゲームの流れ
class Game():
    """ ゲーム本体"""

    def __init__(self, situation):

        # キャラクタデータ読み込む =====

        load_file_path = f"./situation0{situation}.csv"
        
        

        # csv読み込み
        rawdata = []
        with open(load_file_path, encoding='utf8', newline='') as f:
            csvreader = csv.reader(f)
            for row in csvreader:
                rawdata.append(row)

        # chracter情報入力
        self.characters = []
        
        # 現在のキャラクタ
        self.now_character = None
        
        
        # 登場している初期キャラクターにflagを立てる
        self.characters_flag = {}
        self.enemy_flag = {}
        # print(self.characters_flag)
        
        for i in range(len(rawdata)):
            if i != 0:
                here = int(rawdata[i][1])
                flag_key = int(rawdata[i][7])
                self.characters_flag[flag_key] = here
                side = int(rawdata[i][4])
                if side == 1:
                    self.enemy_flag[flag_key] = here
                if here == 1:
                    obj_path = str(rawdata[i][0])           # obj_path
                    playable = int(rawdata[i][2])           # playable
                    # playstyle (playable == 0)
                    playstyle = str(rawdata[i][3])
                    if playable == 0:
                        self.PLAYSTYLE = playstyle
                    side = int(rawdata[i][4])               # side
                    role = str(rawdata[i][5])               # role
                    name = str(rawdata[i][6])               # name
                    id = int(rawdata[i][7])                 # id (side == 0 -> id = i0 (i >= 1), side == 1 -> id = i1, None -> id = -1, 味方全体 -> 0, 敵全体 -> 1)
                    hp = int(rawdata[i][8])                 # hp
                    max_hp = int(rawdata[i][9])             # max_hp
                    mp = int(rawdata[i][10])                 # mp
                    max_mp = int(rawdata[i][11])            # max_mp
                    attack = int(rawdata[i][12])            # attack
                    defense = int(rawdata[i][13])           # defense
                    agility = int(rawdata[i][14])           # agility
                    magicpower_atk = int(rawdata[i][15])    # magicATK
                    magicpower_recovery = int(rawdata[i][16])  # magicREC

                    # 耐性入力
                    registance = [int(rawdata[i][17]), int(rawdata[i][18]), int(rawdata[i][19]), int(rawdata[i][20]), int(rawdata[i][21]), int(rawdata[i][22]), int(rawdata[i][23])]

                    # アクションセット入力
                    action_set = []
                    for j in range(len(rawdata[i]) - 24):
                        action_set.append(int(rawdata[i][j + 24]))

                    self.characters.append(Character(obj_path, playable, playstyle, side, role, name, id, hp, max_hp, mp, max_mp, attack, defense, agility, magicpower_atk, magicpower_recovery, registance, action_set))

        # ゲーム開始時の味方数(self.N_myparty, self.N_enemys)を取得
        self.N_myparty = 0
        self.N_enemys = 0
        for i in range(len(self.characters)):
            if self.characters[i].SIDE == 0:
                self.N_myparty += 1
            elif self.characters[i].SIDE == 1:
                self.N_enemys += 1
                
        # 生存しているキャラクターリスト
        self.alived_characters = []
        for i in range(len(self.characters)):
            self.alived_characters.append(self.characters[i])

        # 死んでいるキャラクターリスト を作成
        self.dead_characters = []

        # 状態遷移用の変数を用意
        self.game_state = GameState.TURN_START

        # ターン数
        self.turn = 0
        
        # ターン最大数
        self.MAX_TURN = 35
        # 味方と敵のキャラクタをリスト化
        mymembers, enemys = self.split_characters(self.characters)
        
        # ゲームのプレイスタイルを取得
        for c in mymembers:
            if c.Playable == 0:
                self.PLAYSTYLE = c.PlayStyle
                break
            else:
                self.PLAYSTYLE = "RANDOM"
        
        # 敵味方のステータス状態
        self.mymembers_state = self.get_mymembers_state(mymembers)
        self.enemys_state = self.get_enemys_state()
        
        # 状態
        self.state = {"mymembers": self.mymembers_state,
                      "enemys": self.enemys_state,
                      "turn": self.turn,
                    }

        
        # 報酬
        self.reward = 0
        
    
    def update_state(self):
        # 敵、味方を判別
        mymembers, enemys = self.split_characters(self.characters)
        # 敵味方状態の更新
        self.state["mymembers"] = self.get_mymembers_state(mymembers)
        self.state["enemys"] = self.get_enemys_state()
        
        # turn
        self.state["turn"] = self.turn
        
        
    

    # 味方の状態を返すメソッド
    def get_mymembers_state(self, mymembers):
        mymembers_state = []
        for c in mymembers:
            id = c.id
            if c.HP <= 0:
                alive = 0
            else:
                alive = 1
            
            # HP状態
            hp_state = c.HP
            
            # MP状態
            mp_state = c.MP
            
            # まとめる
            data = {
                "id": id,
                "alive": alive,
                "hp_state": hp_state,
                "mp_state": mp_state,
                "atk_state_state": c.STATE_ATK["state"],
                "atk_state_turn": c.STATE_ATK["turn"],
                "def_state_state": c.STATE_DEF["state"],
                "def_state_turn": c.STATE_DEF["turn"],
                "agi_state_state": c.STATE_AGI["state"],
                "agi_state_turn": c.STATE_AGI["turn"],
                "mgcATK_state_state": c.STATE_MagicATK["state"],
                "mgcATK_state_turn": c.STATE_MagicATK["turn"],
                "mgcREC_state_state": c.STATE_MagicREC["state"],
                "mgcREC_state_turn": c.STATE_MagicREC["turn"],
                "registance_state_state": c.REGISTANCE["state"],
                "registance_state_turn": c.REGISTANCE["turn"]
            }
            
            # 追加
            mymembers_state.append(data)
        
        return mymembers_state
    
    def get_enemys_state(self):
        enemys_state = []
        for key, value in self.enemy_flag.items():
            id = key
            if value == 1:
                chara = self.id_to_charaObj(id, self.characters)
                # alive
                if chara.HP <= 0:
                    alive = 0
                else:
                    alive = 1
                
                hp_state = chara.HP
                # parameter state
                atk_state_state =  chara.STATE_ATK["state"]
                atk_state_turn =  chara.STATE_ATK["turn"]
                def_state_state = chara.STATE_DEF["state"]
                def_state_turn = chara.STATE_DEF["turn"]
                agi_state_state = chara.STATE_AGI["state"]
                agi_state_turn = chara.STATE_AGI["turn"]
                mgcATK_state_state = chara.STATE_MagicATK["state"]
                mgcATK_state_turn = chara.STATE_MagicATK["turn"]
                mgcREC_state_state = chara.STATE_MagicREC["state"]
                mgcREC_state_turn = chara.STATE_MagicREC["turn"]
                registance_state_state = chara.REGISTANCE["state"]
                registance_state_turn = chara.REGISTANCE["turn"]
                
            else:
                alive = -1
                hp_state = -1
                atk_state_state = -1
                atk_state_turn =  -1
                def_state_state = -1
                def_state_turn = -1
                agi_state_state = -1
                agi_state_turn = -1
                mgcATK_state_state = -1
                mgcATK_state_turn = -1
                mgcREC_state_state = -1
                mgcREC_state_turn = -1
                registance_state_state = -1
                registance_state_turn = -1
                
            # まとめる
            data = {
                "id": id,
                "alive": alive,
                "hp_state": hp_state,
                "atk_state_state": atk_state_state,
                "atk_state_turn": atk_state_turn,
                "def_state_state": def_state_state,
                "def_state_turn": def_state_turn,
                "agi_state_state": agi_state_state,
                "agi_state_turn": agi_state_turn,
                "mgcATK_state_state": mgcATK_state_state,
                "mgcATK_state_turn": mgcATK_state_turn,
                "mgcREC_state_state": mgcREC_state_state,
                "mgcREC_state_turn": mgcREC_state_turn,
                "registance_state_state": registance_state_state,
                "registance_state_turn": registance_state_turn
            }
            enemys_state.append(data)
        return enemys_state
    
    # 状態のtensor化 - stateをtorch.tensorに変換する
    def convert_state(self, state):
        # 各要素をtorch.tensorに変換する
        
        mymembers = state["mymembers"]
        enemys = state["enemys"]
        turn = state["turn"]
        
        
        # 味方情報
        mymembers_list = []
        for member in mymembers:
            member_tensor = torch.tensor(list(member.values()), dtype=torch.long)
            mymembers_list.append(member_tensor)
        mymembers_tensor = torch.stack(mymembers_list, dim=0)
        mymembers_tensor_1d = mymembers_tensor.reshape(-1)
        mymembers_tensor_1d = torch.tensor(mymembers_tensor_1d, dtype=torch.float)
        
        # 敵情報
        enemys_list = []
        for enemy in enemys:
            enemy_tensor = torch.tensor(list(enemy.values()), dtype=torch.long)
            enemys_list.append(enemy_tensor)
        enemys_tensor = torch.stack(enemys_list, dim=0)
        enemys_tensor_1d = enemys_tensor.reshape(-1)
        enemys_tensor_1d = torch.tensor(enemys_tensor_1d, dtype=torch.float)
        
        # turn state
        turn_tensor = torch.tensor(turn/self.MAX_TURN, dtype=torch.float).unsqueeze(0)
        
        
        state_tensor = torch.cat((mymembers_tensor_1d, enemys_tensor_1d, turn_tensor))
        
        return state_tensor
        
        

    def State(self):
        # stateのupdate
        self.update_state()
        
        temp_state = self.convert_state(self.state)
        
        return temp_state

    def update_characters_state(self, characters):
        # 各キャラクタに対して、状態をチェックしていく
        for c in characters:
            # 各状態のturnを-1する

            # 攻撃力状態
            if c.STATE_ATK["turn"] > 0:
                c.STATE_ATK["turn"] -= 1
            else:
                c.STATE_ATK["state"] = 0

            # 守備力状態
            if c.STATE_DEF["turn"] > 0:
                c.STATE_DEF["turn"] -= 1
            else:
                c.STATE_DEF["state"] = 0

            # 素早さ状態
            if c.STATE_AGI["turn"] > 0:
                c.STATE_AGI["turn"] -= 1
            else:
                c.STATE_AGI["state"] = 0

            # 攻撃魔力状態
            if c.STATE_MagicATK["turn"] > 0:
                c.STATE_MagicATK["turn"] -= 1
            else:
                c.STATE_MagicATK["state"] = 0

            # 回復魔力状態
            if c.STATE_MagicREC["turn"] > 0:
                c.STATE_MagicREC["turn"] -= 1
            else:
                c.STATE_MagicREC["state"] = 0

            # 耐性ダウンのかかりやすさ
            if c.REGISTANCE["turn"] > 0:
                c.REGISTANCE["turn"] -= 1
            else:
                c.REGISTANCE["state"] = 0

    # 死んだキャラをself.charactersから削除するキャラクタ更新

    def update_characters(self):
        # 死んだキャラの添字を保持するtemp_idx
        dead_chara = []

        # 死んでいるキャラクタを発見
        for c in self.alived_characters:
            if c.HP <= 0:
                # queueからHP= 0の要素を削除
                dead_chara.append(c)

        # 死んでいるキャラクターをcharacter_queから削除
        for c in dead_chara:
            if c in self.character_que:
                self.character_que.remove(c)

        # self.alived_characters からdead_charaを削除、self.dead_characterに追加
        for c in dead_chara:
            self.dead_characters.append(c)
            self.alived_characters.remove(c)

    
    def N_mymember(self):
        n = 0
        for c in self.alived_characters:
            if c.SIDE == 0:
                n += 1
        return n
    def N_enemy(self):
        n = 0
        for c in self.alived_characters:
            if c.SIDE == 1:
                n += 1
        return n
                

    def step(self, action_flow):
        reward = 0
        if self.game_state == GameState.TURN_START:
            # ターン開始時に現在の味方の数を取得
            self.n_mymembers = self.N_mymember()
            
            
            self.turn_start()
            done = False
            # return done
            return self.State(), self.reward, done
        elif self.game_state == GameState.ACTION_ORDER:
            self.action_order()
            done = False
            # return done
            return self.State(), self.reward, done
        elif self.game_state == GameState.POP_CHARACTER:
            self.pop_character()
            done = False
            return self.State(), self.reward, done
            # return self.state, reward, done
        elif self.game_state == GameState.TURN_NOW:
            # 与える報酬の初期化
            reward = 0

            result_action = self.turn_now(action_flow)
            
            valid_action = result_action[5]
            reward += (2*valid_action - 1)  # (0,1) to (-1, 1)
            
            
            action = result_action[3]
            target_id = result_action[4]
            done = False
            return self.State(), reward, done, action, target_id
        elif self.game_state == GameState.TURN_END:
            self.turn_end()
            done = False
            return self.State(), reward, done
        elif self.game_state == GameState.GAME_END:
            win = self.game_end()
            if win:
                reward = (0.95**self.turn) * 50
            else:
                reward = -10
            done = True
            return self.State(), reward, done, win
    
    
    # result_actionを返す
    def step_get_log(self, action_flow):
        if self.game_state == GameState.TURN_START:
            self.turn_start()
            done = False
            # return done
            return self.State(), self.reward, done
        elif self.game_state == GameState.ACTION_ORDER:
            self.action_order()
            done = False
            # return done
            return self.State(), self.reward, done
        elif self.game_state == GameState.POP_CHARACTER:
            self.pop_character()
            done = False
            return self.State(), self.reward, done
            # return self.state, reward, done
        elif self.game_state == GameState.TURN_NOW:
            result_action = self.turn_now(action_flow)
            # action = result_action[3]
            # target_id = result_action[4]
            done = False
            return self.State(), self.reward, done, result_action
        elif self.game_state == GameState.TURN_END:
            self.turn_end()
            self.reward = self.eval_playstyle(self.state)
            
            done = False
            return self.State(), self.reward, done
        elif self.game_state == GameState.GAME_END:
            win = self.game_end()
            self.reward = self.eval_playstyle(self.state)
            done = True
            return self.State(), self.reward, done, win
    
    
    def step_reward(self, result_action):
        reward = 0
        damages = result_action[1]
        recoveries = result_action[2]
        action = result_action[3]
        target_id = result_action[4]
        valid_action = result_action[5]
        # 有効なアクションかどうか
        if valid_action == 0:
            reward = -1
        else:
            if target_id == -1:
                reward = 0
            elif target_id == 1: #的全体攻撃の場合
                sum_hp = 0
                for c in self.alived_characters:
                    if c.SIDE == 1:
                        sum_hp += c.MAX_HP
                if sum_hp == 0:
                    reward += 0
                else:
                    reward += damages/sum_hp
            elif target_id == 0:
                sum_hp = 0
                for c in self.alived_characters:
                    if c.SIDE == 0:
                        sum_hp += c.MAX_HP
                if sum_hp == 0:
                    reward += 0
                else:
                    reward += recoveries/sum_hp
            else:
                reward += (damages + recoveries)/self.id_to_charaObj(target_id, self.characters).MAX_HP

        return reward
    
    def eval_playstyle(self, state):
        reward = 0
        if self.PLAYSTYLE == "ATTACKING": # なるはやで倒す
            if self.game_state == GameState.TURN_END:
                # 死んだ敵の数nを求める
                n = 0
                for data in state["enemys"]:
                    if data["alive"] == 0:
                        n += 1
                reward = n
            if self.game_state == GameState.GAME_END:
                reward = self.MAX_TURN - self.turn
        
        elif self.PLAYSTYLE == "SaveHP": 
            if self.game_state == GameState.TURN_END:
                # HPが50％以上のキャラクタ数nを求める
                n = 0
                for data in state["mymembers"]:
                    if data["hp"]/data["max_hp"] >= 0.5:
                        n += 1
                reward = n*10 / (len(state["mymembers"]) - n + 1)
        
        
        return reward

    def reset(self, situation):
        self.__init__(situation=situation)
        return self.State()
    

    # 生存キャラクの表示
    def print_characters(self, characters):
        for c in characters:
            print("{}".format(c.get_status()))

    # 味方と敵のキャラクタをそれぞれ、分けて、mymebers, enemysとして返す
    def split_characters(self, characters):
        mymembers = []
        enemys = []
        for c in characters:
            if c.SIDE == 0:
                mymembers.append(c)
            elif c.SIDE == 1:
                enemys.append(c)

        return mymembers, enemys

    

    def turn_start(self):
        self.reward = 0
        self.turn += 1
        if self.turn > self.MAX_TURN:
            self.reward = -1
            s = "ターン制限によりゲームオーバー"
            self.game_state = GameState.GAME_END
            
        else:
            # ターン宣言
            s = "*** ターン" + str(self.turn) + " ***\n"

            # 報酬のリセット
            self.reward = 0

            # 状態遷移
            self.game_state = GameState.ACTION_ORDER
        return s
    def action_order(self):
        # 行動選択順の決定
        self.character_que = deque(
            sorted(self.alived_characters, key=lambda c: c.AGI * random.randrange(75, 150, 1)/100, reverse=True))

        # 状態遷移
        self.game_state = GameState.POP_CHARACTER
    
    def pop_character(self):
        # stepごとの報酬初期化
        self.reward = 0
        
        # キャラクターキューを設定
        self.now_character = self.character_que.popleft()
        
        # ########################
        # ##  action前の状態の更新   
        # #########################
        
        # 敵、味方を判別
        mymembers, enemys = self.split_characters(self.characters)
        # 敵味方状態の更新
        self.state["mymembers"] = self.get_mymembers_state(mymembers)
        self.state["enemys"] = self.get_enemys_state()
        

        # ###############
        # 状態遷移
        # ###############
        self.game_state = GameState.TURN_NOW

    def turn_now(self, action_flow):
        # stepごとに報酬を渡すので、self.reward = 0で初期化
        self.reward = 0
        
        self.recoveries_enemy = 0
        self.damages_enemy = 0
        # しんでいる味方と敵の数を取得
        n_dead_mymembers = 0
        n_dead_enemys = 0
        for c in self.dead_characters:
            if c.SIDE == 0:
                n_dead_mymembers += 1
            elif c.SIDE == 1:
                n_dead_enemys += 1
                
        # キャラクターキューから逐次行動
        if len(self.character_que) >= 0:
            # self.now_character = self.character_que.popleft()

            # 防御状態なら解除する
            if self.now_character.PROTECT == True:
                self.now_character.PROTECT = False

            # 行動しているキャラクタを保存し、行動選択を表示
            # self.__save_log("現在の行動キャラクター: {}".format(now_character.name))
            s = "** now_character_id -> {}\n".format(self.now_character.id)
            # self.log += s
            # print(s)

            # actionから帰ってくる要素
            # s, damages_enemy, recoveires_enemy, action, subaction, target

            # 味方ならactionを自分で選択する、敵ならランダムに選択させる
            # 味方キャラクの場合
            if self.now_character.SIDE == 0:
                if self.now_character.Playable == 1: 
                # print("確認: action = {}, target={}".format(action_flow[0], action_flow[1]))
                    result_action = self.now_character.action(self, action_flow)
                    
                    # 報酬について
                    # ターゲットのHPの何割を削ったかを報酬とする
                    self.reward += self.step_reward(result_action)
                
                else:
                    action_flow = self.now_character.action_select(self.now_character.PlayStyle, self)
                    result_action = self.now_character.action(self, action_flow)
                
                # # demo play
                # action_flow = self.now_character.action_select(self.now_character.PlayStyle, self)
                # result_action = self.now_character.action(self, action_flow)
            # 敵キャラの場合
            elif self.now_character.SIDE == 1:
                action_flow = self.now_character.action_select(self.now_character.PlayStyle, self)
                # print("確認: action = {}, target={}".format(action_flow[0], action_flow[1]))
                result_action = self.now_character.action(self, action_flow)

        
        self.update_characters()
        
        # 味方が死んでいたら、self.reward -= 1、敵が死んでいたら、self.reward += 1
        for c in self.dead_characters:
            if c.SIDE == 0:
                n_dead_mymembers -= 1 # 死んだ味方数が増えているとこの値は"-"となる
            elif c.SIDE == 1:
                n_dead_enemys -= 1 # 死んだ敵の数が経ていたら、この値は"-"となる
        self.reward += n_dead_mymembers - n_dead_enemys

        
        # 味方が全滅したか判別

        self.n_dead_mymember = 0
        self.n_dead_enemys = 0

        for c in self.dead_characters:
            if c.SIDE == 0:
                self.n_dead_mymember += 1
            elif c.SIDE == 1:
                self.n_dead_enemys += 1

        if self.n_dead_mymember == self.N_myparty:
            self.game_state = GameState.GAME_END
            return result_action
        elif self.n_dead_enemys == self.N_enemys:
            self.game_state = GameState.GAME_END
            return result_action

        # 全員行動終了したらターンエンド
        if len(self.character_que) == 0:
            self.game_state = GameState.TURN_END
            return result_action
        else:
            self.game_state = GameState.POP_CHARACTER
            # return self.state, self.reward
            return result_action

    def turn_end(self):
        # 報酬の初期化
        self.reward = 0
        
        # キャラクターキューの初期化
        self.character_que = deque()

        # ステータス状態の更新
        self.update_characters_state(self.alived_characters)
        
        # self.print_characters(self.characters)

        # 状態遷移
        self.game_state = GameState.TURN_START

    def game_end(self):
        # 報酬の初期化
        self.reward = 0
        
        if self.n_dead_mymember == self.N_myparty:
            s = "*** 全滅しました... ***\n"
            # print(s)
            # self.log += s
            self.reward = -1
            win = False

        elif self.n_dead_enemys == self.N_enemys:
            s = "*** まもの をやっつけた!! ***\n"
            # print(s)
            # self.log += s
            self.reward = 1
            win = True
        
        elif self.turn > self.MAX_TURN:
            s = "*** time up ***"
            # print(s)
            # self.log += s
            self.reward = -1
            win = False

        # terminal 表示
        s = "-----ゲームエンド-----\n\n"

        return win
        
    def id_to_name(self, id, characters):
        if id == -1:
            return "None"
        elif id == 0:
            return "All mymembers"
        elif id == 1:
            return "All enemys"
        for c in characters:
            if c.id == id:
                return c.name
    
    def id_to_charaObj(self, id, characters):
        if id == -1:
            return "None"
        elif id == 0:
            return "All mymembers"
        elif id == 1:
            return "All enemys"
        for c in characters:
            if c.id == id:
                return c
    
    def characters_to_ids(self, characters):
        ids = []
        for c in characters:
            ids.append(c.id)
        return ids

    def print_state(self):
        print("*** 現在の状態 ***")
        for key in self.state.keys():
            if key == "mymembers" or key == "enemys":
                print("** {}".format(key))
                charas = []
                for c in self.state[key]:
                    id = c["id"]
                    chara = self.id_to_charaObj(id, self.characters)
                    charas.append(chara)
                self.print_characters(charas)
                

            elif key == "target":
                target_name = self.id_to_name(self.state[key], self.characters)
                print("** {}: {}".format(key, target_name))
            elif key == "now_character":
                print("** {}: {}".format(key, self.now_character.name))
            else:
                print("** {}: {}".format(key, self.state[key]))
        print("\n")
    
    def print_state_given_agent(self):
        print("*** now state ***")
        for key in self.state.keys():
            
            if key == "mymembers" or key == "enemys":
                component = self.state[key]
                print("* {}".format(key))
                for c in component:
                    print(c)
            else:
                print("* {}: {}".format(key, self.state[key]))

    # ゲーム進行
    def game_statement(self):
        state = self.state
        # print(state)
        s = "** ENEMY **\n"
        for c in state["enemys"]:
            id = c["id"]
            name = self.id_to_name(id, self.characters)
            alive = c["alive"]
            if alive == 1:
                hp = c["hp_state"]
                s += "* [{} (id:{}) ] -> [HP]{}\n".format(name, id, hp)
            elif alive == 0:
                s += "* [{} (id:{}) ] -> dead\n".format(name, id)
        
        s += "\n"
        
        s += "** MYPARTY **\n"
        for c in state["mymembers"]:
            id = c["id"]
            chara = self.id_to_charaObj(id, self.characters)
            alive = c["alive"]
            if alive == 1:
                s += "* [{} (id:{}, role:{})] -> HP: {}/{}, MP: {}/{}, ATK: {}({}), DEF: {}({}), SPD: {}({}), MPW: {}({}), MRC: {}({}), RGS: {}({})\n".format(chara.name, id, chara.ROLE, chara.HP, chara.MAX_HP, chara.MP, chara.MAX_MP, chara.STATE_ATK["state"], chara.STATE_ATK["turn"], chara.STATE_DEF["state"], chara.STATE_DEF["turn"], chara.STATE_AGI["state"], chara.STATE_AGI["turn"], chara.STATE_MagicATK["state"], chara.STATE_MagicATK["turn"], chara.STATE_MagicREC["state"], chara.STATE_MagicREC["turn"], chara.REGISTANCE["state"], chara.REGISTANCE["turn"])
            elif alive == 0:
                s += "* [{} (id:{})] -> X\n".format(chara.name, id)
            
    
        return s
    
    
    # ##############
    # pygame用
    # ##############
    def s_action_set(self, character):
        s = ""
        cnt = 0
        for key, value in character.ACTION_SET.items():
            cnt += 1
            mp = character.MP_Actions[key]
            s += "{}: {} (MP:{})   ".format(key, value, mp)
            if cnt >= 2:
                s += "\n"
                cnt = 0
        return s

    def s_target_set(self, characters):
        s = ""
        cnt = 0
        for key, value in characters.items():
            cnt += 1
            s += "{}: {}   ".format(key, value)
            if cnt == 3:
                s += "\n"
                cnt = 0
        return s

    def check_status(self, character):
        s = ""
        if character.STATE_ATK["state"] != 0:
            s += "ATK:{}({})\n".format(character.STATE_ATK["state"], character.STATE_ATK["turn"])
        if character.STATE_DEF["state"] != 0:
            s += "DEF:{}({})\n".format(character.STATE_DEF["state"], character.STATE_DEF["turn"])
        if character.STATE_AGI["state"] != 0:
            s += "SPD:{}({})\n".format(character.STATE_AGI["state"], character.STATE_AGI["turn"])
        if character.STATE_MagicATK["state"] != 0:
            s += "MATK:{}({})\n".format(character.STATE_MagicATK["state"], character.STATE_MagicATK["turn"])
        if character.STATE_MagicREC["state"] != 0:
            s += "MREC:{}({})\n".format(character.STATE_MagicREC["state"], character.STATE_MagicREC["turn"])
        if character.REGISTANCE["state"] != 0:
            s += "RGS:{}({})\n".format(character.REGISTANCE["state"], character.REGISTANCE["turn"])
        return s

if __name__ == "__main__":
    print("press situation number (1 - 4)")
    while True:
        situation = int(input())
        if situation in [1, 2, 3, 4]:
            break
    game = Game(situation)
    game.convert_state(game.state)