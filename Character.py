import pygame
from pygame.locals import *
import random


class Character:
    """キャラクタークラス"""
    ACTIONS = {0: "ぼうぎょ", 1: "単体攻撃", 2: "全体攻撃", 3: "単体回復",
               4: "攻撃魔法(弱)", 5: "攻撃魔法(中)", 6: "攻撃魔法(全体)", 7: "攻撃魔法(単体)",
               8: "ATKアップ", 9: "DEFアップ(全)", 10: "SPDアップ(全体)", 11: "MagicATKアップ", 12: "MagicRECアップ",
               13: "ATKダウン", 14: "DEFダウン", 15: "SPDダウン", 16: "MagicATKダウン", 17: "MagicRECダウン",
               18: "耐性アップ", 19: "耐性ダウン", 20: "強攻撃", 21: "強全体攻撃", 22: "全体回復"}

    # 各アクションの属性
    Attribute_Actions = {0: "protect", 1: "attacking", 2: "attacking", 3: "healing",
                            4: "attacking", 5: "attacking", 6: "attacking", 7: "attacking",
                            8: "bufsupport", 9: "bufsupport", 10: "bufsupport", 11: "bufsupport", 12: "bufsupport",
                            13: "debufsupport", 14: "debufsupport", 15: "debufsupport", 16: "debufsupport", 17: "debufsupport",
                            18: "bufsupport", 19: "debufsupport", 20: "attacking", 21: "attacking", 22: "healing"}

    MP_Actions = {0: 0, 1: 0, 2: 8, 3: 8,
                4: 6, 5: 7, 6: 10, 7: 8,
                8: 5, 9: 5, 10: 5, 11: 5, 12: 5,
                13: 5, 14: 5, 15: 5, 16: 5, 17: 5,
                18: 7, 19: 8, 20: 10, 21: 15, 22: 25}
    
    def __init__(self, obj_path, playable, playstyle, side, role, name, id, hp, max_hp, mp, max_mp,
                 attack, defense, agility, magicpower_atk, magicpower_recovery, registance, action_set):
        self.Obj = pygame.transform.scale(
            pygame.image.load(obj_path), (200, 200))  # object画像読みこみ
        self.Playable = playable    # playstyleで動かす場合0, そうでない場合(エージェントで動かす場合)1
        self.PlayStyle = playstyle  # プレイスタイル
        self.SIDE = side            # キャラクターの属性(味方0 or 敵1)
        self.name = name            # キャラクターname
        self.id = id                # キャラクター識別id (int型)
        self.ROLE = role            # キャラクターrole
        self.MAX_HP = max_hp        # 最大HP
        self.HP = hp                # 現在のHP
        self.MAX_MP = max_mp        # 最大MP
        self.MP = mp                # 現在のMP
        self.ATK = attack           # 攻撃力
        self.DEF = defense          # 防御力
        self.AGI = agility          # 素早さ
        self.MAGIC_ATK = magicpower_atk  # 攻撃魔力 MAX値: 999
        self.MAGIC_REC = magicpower_recovery    # 回復魔力
        self.PROTECT = False        # 防御しているかどうか(基本的にはしていないのでFalse)
        # ステータス状態(ATK, DEF, AGI, INTE 上下に2段階)
        self.STATE_ATK = {"state": 0, "turn": 0}          # 攻撃力状態
        self.STATE_DEF = {"state": 0, "turn": 0}          # 防御力状態
        self.STATE_AGI = {"state": 0, "turn": 0}          # 素早さ状態
        self.STATE_MagicATK = {"state": 0, "turn": 0}     # 攻撃魔力状態
        self.STATE_MagicREC = {"state": 0, "turn": 0}     # 回復魔力状態
        
        # 耐性
        self.REGISTANCE = {'state': registance[0], 'turn': registance[1], 'FIRE': registance[2], 'ICE': registance[3], 'DARK': registance[4], 'SHINE': registance[5],  'DOWN_ZOKUSEI': registance[6]}

        # キャラクタのアクションセット
        self.ACTION_SET = {}
        for i in range(len(action_set)):
            # key
            key = action_set[i]
            self.ACTION_SET[key] = self.ACTIONS[key]

    def status_rate(self, status_state):
        if status_state == 0:
            return 1.0
        elif status_state == 1:
            return 1.5
        elif status_state == 2:
            return 2.0
        elif status_state == -1:
            return 0.75
        elif status_state == -2:
            return 0.5
        
    
    def registance_rate(self, regi_state):
        if regi_state == 2:
            return 0.5
        elif regi_state == 1:
            return 0.75
        elif regi_state == 0:
            return 1.0
        elif regi_state == -1:
            return 1.5
        elif regi_state == -2:
            return 2.0

    # ステータス取得
    def get_status(self):
        return "[{}] HP:{}/{} MP:{}/{} ATK:{} ({} ({})) DEF:{} ({} ({})) AGI:{} ({} ({})) MAGIC_ATK:{} ({} ({})) MAGIC_REC:{} ({} ({})) ".format(
            self.name, self.HP, self.MAX_HP, self.MP, self.MAX_MP, self.ATK, self.STATE_ATK["state"], self.STATE_ATK["turn"], self.DEF, self.STATE_DEF["state"], self.STATE_DEF["turn"], self.AGI, self.STATE_AGI["state"], self.STATE_AGI["turn"], self.MAGIC_ATK, self.STATE_MagicATK["state"], self.STATE_MagicATK["turn"], self.MAGIC_REC, self.STATE_MagicREC["state"], self.STATE_MagicREC["turn"],)

    # partyキャラクタをdict()にまとめて返す
    def myParty(self, mySide, game):
        myparty = []
        key = []
        for i in range(len(game.alived_characters)):
            chara = game.alived_characters[i]
            if chara.SIDE == mySide:
                myparty.append(chara)

        for i in range(len(myparty)):
            key.append(i)

        return dict(zip(key, myparty))

    def select_mymember(self, myparty, auto):
        # Character型のdict() (== mymem)を受け取っているのでCharacter型のnameのみをname_myparty[]に保管する
        name_myparty = []
        for i in range(len(myparty)):
            name_myparty.append(myparty[i].name)

        name_myparty_key = [i for i in range(len(myparty))]
        name_myparty_dict = dict(zip(name_myparty_key, name_myparty))

        # randでランダムに選択するか、手動で選択するかを決定する
        if auto == True:
            key = random.randrange(0, len(myparty), 1)
            mymember = myparty[key]

        # 手動でtarget選択
        elif auto == False:
            print("キャラクタを選択してください。", name_myparty_dict)
            mymember = myparty[int(input())]
            print("{} が選択されました".format(mymember.name))

        return mymember

    
    
    # enemyキャラクタをdict()にまとめて返す
    def target_enemy(self, mySide, game):
        targets = []
        key = []
        for i in range(len(game.alived_characters)):
            chara = game.alived_characters[i]
            if chara.SIDE != mySide:
                targets.append(chara)

        for i in range(len(targets)):
            key.append(i)

        return dict(zip(key, targets))

    # 敵キャラのリストを取得し、その中から攻撃する相手を選択する

    def select_target(self, targets, auto):

        # Character型のdict() (== targets)を受け取っているのでCharacter型のnameのみをname_targets[]に保管する
        name_target = []
        for i in range(len(targets)):
            name_target.append(targets[i].name)

        name_target_key = [i for i in range(len(targets))]
        name_target_dict = dict(zip(name_target_key, name_target))

        # randでランダムに選択するか、手動で選択するかを決定する
        if auto == True:
            key = random.randrange(0, len(targets), 1)
            target = targets[key]
            # print("* 対象キャラクタとして {} が選択されました".format(target.name))

        # 手動でtarget選択
        elif auto == False:
            print("* 対象キャラクタを選択してください。", name_target_dict)
            target = targets[int(input())]
            print("* 対象キャラクタとして {} が選択されました".format(target.name))

        return target
    
    def select_target_id(self, mySide, action, game, agent=False):
        targets_id = []
        if self.Attribute_Actions[action] == "protect":
            return -1   # target選択なしとして返す
        
        elif self.Attribute_Actions[action] == "attacking" or self.Attribute_Actions[action] == "debufsupport":
            if action == 2 or action == 6 or action == 19 or action == 21:
                return 1
            for c in game.alived_characters:
                if c.SIDE != mySide:
                    targets_id.append(c.id)
        
        elif self.Attribute_Actions[action] == "healing" or self.Attribute_Actions[action] == "bufsupport":
            if action == 9 or action == 10 or action == 18 or action == 22:
                return 0
            for c in game.alived_characters:
                if c.SIDE == mySide:
                    targets_id.append(c.id)

        if agent == False:
            return random.choice(targets_id)
        
        
        
        
    def opposite_side(self, myside):
        if myside == 0:
            return 1
        elif myside == 1:
            return 0

    # プレイアブルキャラクタの行動選択 --- 手入力で毎回学習させられないため、
    # これを用いて、プレイアブルキャラクターの操作を行う

    def action_select(self, playstyle, game):
        # ACTION, target を返す
        # ACTION_SETの中をアクション属性によって分類する
        atk_actions = []
        healing_actions = []
        buf_actions = []
        debuf_actions = []
        protect_actions = []
        for key in self.ACTION_SET:
            if self.Attribute_Actions[key] == "attacking":
                atk_actions.append(key)
            elif self.Attribute_Actions[key] == "healing":
                healing_actions.append(key)
            elif self.Attribute_Actions[key] == "bufsupport":
                buf_actions.append(key)
            elif self.Attribute_Actions[key] == "debufsupport":
                debuf_actions.append(key)
            elif self.Attribute_Actions[key] == "protect":
                protect_actions.append(key)
        all_actions = atk_actions + healing_actions + buf_actions + debuf_actions + protect_actions
        # self.ACTION_SETの中から完全ランダムに選択
        if playstyle == "RANDOM":
            p = random.random()
            # 7割の確率でたたかうを選択
            if p < 0.85:
                action = random.randrange(1, len(self.ACTION_SET), 1)
                
            # 3割の確率でぼうぎょを選択
            else:
                action = 0
                # target = None
            
            # ターゲット選択
            target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id

        # playstyle == ATTACKING => 8割の確率でattackingを選択する残りの2割はattacking以外の中から選択
        elif playstyle == "ATTACKING":
            # attacking以外のアクションセット
            temp = healing_actions + buf_actions + debuf_actions + protect_actions

            p = random.random()
            if p < 0.9 and len(atk_actions) > 0:
                action = atk_actions[random.randrange(0, len(atk_actions), 1)]
                # action = random.choice(atk_actions)
            elif p >= 0.8 and len(temp) > 0:
                action = temp[random.randrange(0, len(temp), 1)]
                # action = random.choice(temp)
            else:
                # 例外処理
                action = self.ACTION_SET[random.randrange(
                    0, len(self.ACTION_SET), 1)]

            # ターゲット選択
            target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id

        # いろいろやろうぜ
        elif playstyle == "MULTI":
            while True:
                p = random.random()
                if p < 0.1:
                    if len(protect_actions) != 0:
                        action = protect_actions[random.randrange(0, len(protect_actions), 1)]
                        # action = random.choice(protect_actions)
                        break
                elif p < 0.225:
                    if len(atk_actions) != 0:
                        # action = random.choice(atk_actions)
                        action = atk_actions[random.randrange(
                            0, len(atk_actions), 1)]
                        break
                elif 0.225 <= p < 0.45:
                    if len(healing_actions) != 0:
                        action = healing_actions[random.randrange(
                            0, len(healing_actions), 1)]
                        # action = random.choice(healing_actions)
                        break
                elif 0.45 <= p < 0.725:
                    if len(buf_actions) != 0:
                        action = buf_actions[random.randrange(
                            0, len(buf_actions), 1)]
                        # action = random.choice(buf_actions)
                        break
                elif 0.725 <= p < 1.0:
                    if len(debuf_actions) != 0:
                        action = debuf_actions[random.randrange(
                            0, len(debuf_actions), 1)]
                        # action = random.choice(debuf_actions)
                        break

            target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id

        # 命大事に
        elif playstyle == "SAVE_HP":
            # action = healing_actions[random.randrange(0, len(healing_actions), 1)]
            action = random.choice(healing_actions)
            temp = atk_actions + buf_actions + debuf_actions + protect_actions

            # 回復対象を把握
            recovery_target_id = []
            for c in game.alived_characters:
                if c.SIDE == self.SIDE:
                    if c.HP < int(c.MAX_HP*0.5):
                        recovery_target_id.append(c.id)

            

            if len(recovery_target_id) > 0:
                target_id = random.choice(recovery_target_id)
            else:
                action = random.choice(temp)
                target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id
        
        elif playstyle == "SUPPORT":
            mymember_targets, enemy_targets = game.split_characters(game.alived_characters)
            p = random.random()
            if (0 <= p < 0.85) and (len(buf_actions) > 0):
                action = random.choice(buf_actions)
                # 同sideのキャラクターからtarget_idを選択
                # attackerを選択 いなければ attacker > SAVE_HP > other でtarget選択
                
                if self.SIDE == 0:
                    target_ids = game.characters_to_ids(mymember_targets)
                    target_id = random.choice(target_ids)
                    for c in mymember_targets:
                        if c.PlayStyle == "ATTACKING" or c.PlayStyle == "MAOU":
                            target_id = c.id
                        
                elif self.SIDE == 1:
                    target_ids = game.characters_to_ids(enemy_targets)
                    target_id = random.choice(target_ids)
                    for c in enemy_targets:
                        if c.PlayStyle == "ATTACKING" or c.PlayStyle == "MAOU":
                            target_id = c.id
                
                target_chara = self.get_charaObj_by_id(target_id, game.alived_characters)
                if target_chara.HP/target_chara.MAX_HP < 0.5 and len(healing_actions) > 0:
                    q = random.random()
                    if 0 <= q < 0.3:
                        action = random.choice(healing_actions)
                        target_id = target_chara.id
                        # print("healing action")
                        
            elif (0.85 <= p < 0.95) and (len(debuf_actions) > 0):
                action = random.choice(debuf_actions)
                if self.SIDE == 0:
                    target_ids = game.characters_to_ids(enemy_targets)
                    target_id = random.choice(target_ids)
                elif self.SIDE == 1:
                    target_ids = game.characters_to_ids(mymember_targets)
                    target_id = random.choice(target_ids)
            # ランダム選択
            else:
                action = random.choice(all_actions)
                target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id

        elif playstyle == "MAOU":
            mymember_targets, enemy_targets = game.split_characters(game.alived_characters)
            p = random.random()
            if self.HP/self.MAX_HP < 0.5 and p < 0.3:
                action = random.choice(healing_actions)
                target_id = self.select_target_id(self.SIDE, action, game)
            elif 0<= p < 0.15:
                action = random.choice(debuf_actions)
                target_id = self.select_target_id(self.SIDE, action, game)
            elif 0.15 <= p < 0.3:
                action = random.choice(buf_actions)
                target_id = self.id
            elif 0.3 <= p < 0.95:
                action = random.choice(atk_actions)
                target_id = self.select_target_id(self.SIDE, action, game)
            else:
                action = random.choice(all_actions)
                target_id = self.select_target_id(self.SIDE, action, game)
            
            return action, target_id

    # 敵の行動記述s, 敵に与えたダメージ量damages_enemy, 敵の回復した量recoveries_enemy
    
    def get_name_by_id(self, id, characters):
        if id == -1:
            return "None"
        elif id == 0:
            return "All mymembers"
        elif id == 1:
            return "All enemys"
        else:
            for c in characters:
                if c.id == id:
                    return c.name
            
    # 明確なターゲットがある場合のみ呼び出す
    def get_charaObj_by_id(self, id, characters):
        for c in characters:
            if c.id == id:
                return c


    def action(self, game, action_flow):
        s = ""
        damages_enemy = 0
        recoveries_enemy = 0
        
        action, target_id = action_flow # これらを返す
        
        # ターゲットのidとオブジェクトを得る
        target_name = self.get_name_by_id(target_id, game.characters)
        target = self.get_charaObj_by_id(target_id, game.characters)        
        

        selected_Action = action
        
        s += "* Action -> {}\n".format(self.ACTIONS[selected_Action])
        s += "* Target -> {}\n".format(target_name)
        
        # ぼうぎょ
        if selected_Action == 0:
            valid_action = 1
            self.PROTECT = True
            s += "* Result -> {} は {} している.\n".format(
                self.name, self.ACTIONS[selected_Action])
            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 単体攻撃
        elif selected_Action == 1:
            

            # 有効なアクションかどうか
            valid_action = 1
            
            # ダメージ計算
            damage = int(((self.ATK*self.status_rate(self.STATE_ATK["state"]))/2 - (
                target.status_rate(target.STATE_DEF["state"]) * target.DEF)/4)*random.randrange(94, 106, 1)/100)

            # ダメージは0以上
            if damage <= 0:
                damage = 0

            if target.PROTECT == True:
                damage = int(damage/2)

            # ログ用
            log_damage = damage

            # 与えたダメージ
            damages_enemy = damage

            # target.HPは0未満にならない
            if target.HP < damage:
                damage = target.HP

            # ダメージを与える
            target.HP -= damage

            # game alived_charactersの更新
            game.update_characters()

            s += "* Result -> {} は {} に {} のダメージを与えた.\n".format(
                self.name, target.name, log_damage)

            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # グループ攻撃
        elif selected_Action == 2:
            # 必要MP
            needMP = self.MP_Actions[selected_Action]
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} をしようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            
            else:
                valid_action = 1
                # MP計算
                self.MP -= needMP

                # 攻撃する敵キャラクタを取得
                targets = self.target_enemy(self.SIDE, game)

                # 添字によって与えるダメージを補完していく
                damages = []

                # 敵1キャラずつダメージ処理を行う
                for i in range(len(targets)):
                    # ターゲットとなるキャラクタ
                    target = targets[i]

                    # ダメージ計算
                    damage = int(((self.ATK * self.status_rate(self.STATE_ATK["state"]))/2 - (target.status_rate(target.STATE_DEF["state"]) * target.DEF)/4)*random.randrange(65, 76, 1)/100)

                    # ダメージは0以上
                    if damage <= 0:
                        damage = 0

                    if target.PROTECT == True:
                        damage = int(damage/2)

                    # ログ用
                    log_damage = damage

                    # target.HPは0未満にならない
                    if target.HP < damage:
                        damage = target.HP

                    # ダメージを与える
                    target.HP -= damage

                    # 与えるダメージをdamagesに補完
                    damages.append(log_damage)

            for i in range(len(targets)):
                s += "* Result -> {} は {} に {} のダメージを与えた.\n".format(
                    self.name, targets[i].name, damages[i])

            # 的に与えた総ダメージを計算
            damages_enemy = sum(damages)

            # game alived_charactersの更新
            game.update_characters()

            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 回復魔法
        elif selected_Action == 3:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                # ターゲット(味方キャラクタ)
                mymember = target
                # 残りHPの計算
                recovery_hp = int(random.randrange(
                    28, 31, 1) * (1 + (self.MAGIC_REC * self.status_rate(self.STATE_MagicREC["state"]) / 300)) * random.randrange(87, 102, 1)/100)
                after_hp = mymember.HP + recovery_hp
                log_hp = recovery_hp  # ログ用
                recoveries_enemy = recovery_hp

                # HP回復 --- after_hpがMAX_HPを超えていないか判定
                if mymember.MAX_HP < after_hp:
                    mymember.HP = mymember.MAX_HP
                else:
                    mymember.HP += recovery_hp

                # game alived_charactersの更新
                game.update_characters()

                # 戦闘ログを表示
                if self.SIDE == 0:
                    s += "* Result -> {} は {} に {} を唱えた.\nHPは {} 回復した！(HP:{}/{})\n".format(
                    self.name, mymember.name, self.ACTIONS[selected_Action], log_hp, mymember.HP, mymember.MAX_HP)
                elif self.SIDE == 1:
                    s += "* Result -> {} は {} に {} を唱えた.\nHPは {} 回復した！\n".format(
                    self.name, mymember.name, self.ACTIONS[selected_Action], log_hp)
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 攻撃呪文(弱)
        elif selected_Action == 4:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  
                # 唱えられる
                valid_action = 1

                # MP計算
                self.MP -= needMP


                # ダメージ計算
                damage = int(random.randrange(15, 25, 1) * target.registance_rate(target.REGISTANCE['FIRE']) * (
                    1 + (self.MAGIC_ATK * self.status_rate(self.STATE_MagicATK["state"])/500)*random.randrange(92, 102, 1)/100))

                # ダメージは0以上
                if damage <= 0:
                    damage = 0

                if target.PROTECT == True:
                    damage = int(damage/2)

                # ログ用
                log_damage = damage

                damages_enemy = damage

                # target.HPは0未満にならない
                if target.HP < damage:
                    damage = target.HP

                # ダメージを与える
                target.HP -= damage

                # game alived_charactersの更新
                game.update_characters()

                s += "* Result -> {} は {} に {} のダメージを与えた\n".format(
                    self.name, target.name, log_damage)

                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 攻撃呪文(中)
        elif selected_Action == 5:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # MP計算
                self.MP -= needMP

               
                damage = int(random.randrange(25, 35, 1) * target.registance_rate(target.REGISTANCE['ICE']) * (1 + (self.MAGIC_ATK * self.status_rate(self.STATE_MagicATK["state"])/500))*random.randrange(92, 103, 1)/100)

                

                # ダメージは0以上
                if damage <= 0:
                    damage = 0

                if target.PROTECT == True:
                    damage = int(damage/2)

                # ログ用
                log_damage = damage

                damages_enemy = damage

                # target.HPは0未満にならない
                if target.HP < damage:
                    damage = target.HP

                # ダメージを与える
                target.HP -= damage

                # game alived_charactersの更新
                game.update_characters()

                s += "* Result -> {} は {} に {} のダメージを与えた\n".format(
                    self.name, target.name, log_damage)
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 攻撃魔法 (全体)
        elif selected_Action == 6:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:
                valid_action = 1
                # MP計算
                self.MP -= needMP

                # 攻撃する敵キャラクタを取得
                targets = self.target_enemy(self.SIDE, game)

                # 添字によって与えるダメージを補完していく
                damages = []

                # 敵1キャラずつダメージ処理を行う
                for i in range(len(targets)):
                    # ターゲットとなるキャラクタ
                    target = targets[i]

                    # ダメージ計算
                    damage = int(random.randrange(22, 33, 1) * target.registance_rate(target.REGISTANCE['SHINE']) * (
                        1 + (self.MAGIC_ATK * self.status_rate(self.STATE_MagicATK["state"])/500)*random.randrange(65, 76, 1)/100))

                    # ダメージは0以上
                    if damage <= 0:
                        damage = 0

                    if target.PROTECT == True:
                        damage = int(damage/2)

                    # ログ用
                    log_damage = damage

                    # target.HPは0未満にならない
                    if target.HP < damage:
                        damage = target.HP

                    # ダメージを与える
                    target.HP -= damage

                    # 与えるダメージをdamagesに補完
                    damages.append(damage)

            s += "* Result ->\n"
            for i in range(len(targets)):
                s += "* {} は {} に {} のダメージを与えた.\n".format(
                    self.name, targets[i].name, damages[i])

            # 与えた総ダメージ
            damages_enemy = sum(damages)

            # game alived_charactersの更新
            game.update_characters()

            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 攻撃魔法(強)
        elif selected_Action == 7:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # MP計算
                self.MP -= needMP

                

                # ダメージ計算
                damage = int(random.randrange(27, 38, 1) * target.registance_rate(target.REGISTANCE['DARK']) * (
                    1 + (self.MAGIC_ATK * self.status_rate(self.STATE_MagicATK["state"])/500)*random.randrange(92, 102, 1)/100))

                # ダメージは0以上
                if damage <= 0:
                    damage = 0

                if target.PROTECT == True:
                    damage = int(damage/2)

                # ログ用
                log_damage = damage

                damages_enemy = damage
                # target.HPは0未満にならない
                if target.HP < damage:
                    damage = target.HP

                # ダメージを与える
                target.HP -= damage

                # game alived_charactersの更新
                game.update_characters()

                s += "* Result -> {} は {} に {} のダメージを与えた.\n".format(
                    self.name, target.name, log_damage)
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # # ATKアップ
        elif selected_Action == 8:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                mymember = target
                
                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                # バフが乗っていたら、さらにプラスする
                if mymember.STATE_ATK["state"] >= 0:
                    mymember.STATE_ATK["turn"] += n_turn
                # デバフが乗っていたら、上書きする
                else:
                    mymember.STATE_ATK["turn"] = n_turn

                # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                if mymember.STATE_ATK["turn"] > 5:
                    mymember.STATE_ATK["turn"] = 5

                # 攻撃力状態を参照
                if mymember.STATE_ATK["state"] == -2:
                    mymember.STATE_ATK["state"] = -1

                elif mymember.STATE_ATK["state"] == -1:
                    mymember.STATE_ATK["state"] = 0

                elif mymember.STATE_ATK["state"] == 0:
                    mymember.STATE_ATK["state"] = 1

                elif mymember.STATE_ATK["state"] == 1:
                    mymember.STATE_ATK["state"] = 2

                # 最終的に"state" == 0 ならば、"turn"を0に戻す
                if mymember.STATE_ATK["state"] == 0:
                    mymember.STATE_ATK["turn"] = 0

                s += "* Result -> {} は {} に {} を唱えた.\n{} の攻撃力が {} 倍に上がった.\n({}ターン)\n".format(
                    self.name, mymember.name, self.ACTIONS[selected_Action], mymember.name, mymember.status_rate(mymember.STATE_ATK["state"]), mymember.STATE_ATK["turn"])
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
        # # DEFアップ
        elif selected_Action == 9:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                s += "{} は {} を唱えた.\n".format(self.name, self.ACTIONS[selected_Action])
                
                mymembers = []
                for c in game.alived_characters:
                    if c.SIDE == self.SIDE:
                        mymembers.append(c)

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                # 味方キャラクタ一体ずつ処理
                for mymember in mymembers:
                # バフが乗っていたら、さらにプラスする
                    if mymember.STATE_DEF["state"] >= 0:
                        mymember.STATE_DEF["turn"] += n_turn
                    # デバフが乗っていたら、上書きする
                    else:
                        mymember.STATE_DEF["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if mymember.STATE_DEF["turn"] > 5:
                        mymember.STATE_DEF["turn"] = 5

                    # 攻撃力状態を参照
                    if mymember.STATE_DEF["state"] == -2:
                        mymember.STATE_DEF["state"] = -1

                    elif mymember.STATE_DEF["state"] == -1:
                        mymember.STATE_DEF["state"] = 0  # STATE_ATK == 1

                    elif mymember.STATE_DEF["state"] == 0:
                        mymember.STATE_DEF["state"] = 1

                    elif mymember.STATE_DEF["state"] == 1:
                        mymember.STATE_DEF["state"] = 2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if mymember.STATE_DEF["state"] == 0:
                        mymember.STATE_DEF["turn"] = 0

                    s += "* Result -> {} の守備力が {} 倍に上がった.\n({}ターン)\n".format(mymember.name, mymember.status_rate(mymember.STATE_DEF["state"]), mymember.STATE_DEF["turn"])
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # AGIアップ
        elif selected_Action == 10:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                s += "{} は {} を唱えた.\n".format(self.name, self.ACTIONS[selected_Action])
                
                # mymember = target
                mymembers = []
                for c in game.alived_characters:
                    if c.SIDE == self.SIDE:
                        mymembers.append(c)

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                for mymember in mymembers:
                    # バフが乗っていたら、さらにプラスする
                    if mymember.STATE_AGI["state"] >= 0:
                        mymember.STATE_AGI["turn"] += n_turn
                    # デバフが乗っていたら、上書きする
                    else:
                        mymember.STATE_AGI["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if mymember.STATE_AGI["turn"] > 5:
                        mymember.STATE_AGI["turn"] = 5

                    # 攻撃力状態を参照
                    if mymember.STATE_AGI["state"] == -2:
                        mymember.STATE_AGI["state"] = -1

                    elif mymember.STATE_AGI["state"] == -1:
                        mymember.STATE_AGI["state"] = 0  # STATE_ATK == 1

                    elif mymember.STATE_AGI["state"] == 0:
                        mymember.STATE_AGI["state"] = 1

                    elif mymember.STATE_AGI["state"] == 1:
                        mymember.STATE_AGI["state"] = 2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if mymember.STATE_AGI["state"] == 0:
                        mymember.STATE_AGI["turn"] = 0

                    s += "* Result -> {} の素早さが {} 倍に上がった.\n({}ターン)\n".format(
                        mymember.name, mymember.status_rate(mymember.STATE_AGI["state"]), mymember.STATE_AGI["turn"])
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # MagicATKアップ
        elif selected_Action == 11:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                
                mymember = target

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                # バフが乗っていたら、さらにプラスする
                if mymember.STATE_MagicATK["state"] >= 0:
                    mymember.STATE_MagicATK["turn"] += n_turn
                # デバフが乗っていたら、上書きする
                else:
                    mymember.STATE_MagicATK["turn"] = n_turn

                # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                if mymember.STATE_MagicATK["turn"] > 5:
                    mymember.STATE_MagicATK["turn"] = 5

                # 攻撃力状態を参照
                if mymember.STATE_MagicATK["state"] == -2:
                    mymember.STATE_MagicATK["state"] = -1

                elif mymember.STATE_MagicATK["state"] == -1:
                    # STATE_ATK == 1
                    mymember.STATE_MagicATK["state"] = 0

                elif mymember.STATE_MagicATK["state"] == 0:
                    mymember.STATE_MagicATK["state"] = 1

                elif mymember.STATE_MagicATK["state"] == 1:
                    mymember.STATE_MagicATK["state"] = 2

                # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                if mymember.STATE_MagicATK["state"] == 0:
                    mymember.STATE_MagicATK["turn"] = 0
                s += "* Result -> {} は {} に {} を唱えた.\n{} の攻撃魔力が {} 倍に上がった.\n({}ターン)\n".format(
                    self.name, mymember.name, self.ACTIONS[selected_Action], mymember.name, mymember.status_rate(mymember.STATE_MagicATK["state"]), mymember.STATE_MagicATK["turn"])
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # MagicRECアップ
        elif selected_Action == 12:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                # ターゲット(味方キャラクタ)
                mymember = target

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                # バフが乗っていたら、さらにプラスする
                if mymember.STATE_MagicREC["state"] >= 0:
                    mymember.STATE_MagicREC["turn"] += n_turn
                # デバフが乗っていたら、上書きする
                else:
                    mymember.STATE_MagicREC["turn"] = n_turn

                # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                if mymember.STATE_MagicREC["turn"] > 5:
                    mymember.STATE_MagicREC["turn"] = 5

                # 攻撃力状態を参照
                if mymember.STATE_MagicREC["state"] == -2:
                    mymember.STATE_MagicREC["state"] = -1

                elif mymember.STATE_MagicREC["state"] == -1:
                    mymember.STATE_MagicREC["state"] = 0

                elif mymember.STATE_MagicREC["state"] == 0:
                    mymember.STATE_MagicREC["state"] = 1

                elif mymember.STATE_MagicREC["state"] == 1:
                    mymember.STATE_MagicREC["state"] = 2

                # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                if mymember.STATE_MagicREC["state"] == 0:
                    mymember.STATE_MagicREC["turn"] = 0

                s += "* Result -> {} は {} に {} を唱えた.\n{} の回復魔力が {} 倍に上がった.\n({}ターン)\n".format(
                    self.name, mymember.name, self.ACTIONS[selected_Action], mymember.name, mymember.status_rate(mymember.STATE_MagicREC["state"]), mymember.STATE_MagicREC["turn"])
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # ATKダウン
        elif selected_Action == 13:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 1
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                self.MP -= needMP

                target = target

                s += "* Result -> {} は {} に {} を唱えた.\n".format(
                    self.name, target.name, self.ACTIONS[selected_Action])

                flag = 1

                if flag == 0:
                    s += " しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:
                    # 持続ターン数を決める 3-5turn
                    n_turn = random.randrange(3, 6, 1)

                    # デバフが乗っていたら、さらにプラスする
                    if target.STATE_ATK["state"] <= 0:
                        target.STATE_ATK["turn"] += n_turn
                    # バフが乗っていたら、上書きする
                    else:
                        target.STATE_ATK["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if target.STATE_ATK["turn"] > 5:
                        target.STATE_ATK["turn"] = 5

                    # 攻撃力状態を参照
                    if target.STATE_ATK["state"] == 2:
                        target.STATE_ATK["state"] = 1

                    elif target.STATE_ATK["state"] == 1:
                        target.STATE_ATK["state"] = 0  # STATE_ATK == 1

                    elif target.STATE_ATK["state"] == 0:
                        target.STATE_ATK["state"] = -1

                    elif target.STATE_ATK["state"] == -1:
                        target.STATE_ATK["state"] = -2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if target.STATE_ATK["state"] == 0:
                        target.STATE_ATK["turn"] = 0

                    s += "* {} の攻撃力が {} 倍に下がった.\n({}ターン)\n".format(
                        target.name, target.status_rate(target.STATE_ATK["state"]), target.STATE_ATK["turn"])
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
        # DEFダウン
        elif selected_Action == 14:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                self.MP -= needMP


                s += "* Result -> {} は {} に {} を唱えた.\n".format(
                    self.name, target.name, self.ACTIONS[selected_Action])


                flag = 1
                
                if flag == 0:
                    s += "しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:

                    # 持続ターン数を決める 3-5turn
                    n_turn = random.randrange(3, 6, 1)

                    # デバフが乗っていたら、さらにプラスする
                    if target.STATE_DEF["state"] <= 0:
                        target.STATE_DEF["turn"] += n_turn
                    # バフが乗っていたら、上書きする
                    else:
                        target.STATE_DEF["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if target.STATE_DEF["turn"] > 5:
                        target.STATE_DEF["turn"] = 5

                    # 攻撃力状態を参照
                    if target.STATE_DEF["state"] == 2:
                        target.STATE_DEF["state"] = 1

                    elif target.STATE_DEF["state"] == 1:
                        target.STATE_DEF["state"] = 0  # STATE_ATK == 1

                    elif target.STATE_DEF["state"] == 0:
                        target.STATE_DEF["state"] = -1

                    elif target.STATE_DEF["state"] == -1:
                        target.STATE_DEF["state"] = -2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if target.STATE_DEF["state"] == 0:
                        target.STATE_DEF["turn"] = 0

                    s += "* {} の守備力が {} 倍に下がった.\n({}ターン)\n".format(
                        target.name, target.status_rate(target.STATE_DEF["state"]), target.STATE_DEF["turn"])

                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # AGIダウン
        elif selected_Action == 15:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                self.MP -= needMP


                s += "* Result -> \n"
                s += "* {} は {} に {} を唱えた.\n".format(
                    self.name, target.name, self.ACTIONS[selected_Action])

                
                flag = 1

                if flag == 0:
                    s += "*しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:

                    # 持続ターン数を決める 3-5turn
                    n_turn = random.randrange(3, 6, 1)

                    # デバフが乗っていたら、さらにプラスする
                    if target.STATE_AGI["state"] <= 0:
                        target.STATE_AGI["turn"] += n_turn
                    # バフが乗っていたら、上書きする
                    else:
                        target.STATE_AGI["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if target.STATE_AGI["turn"] > 5:
                        target.STATE_AGI["turn"] = 5

                    # 攻撃力状態を参照
                    if target.STATE_AGI["state"] == 2:
                        target.STATE_AGI["state"] = 1

                    elif target.STATE_AGI["state"] == 1:
                        target.STATE_AGI["state"] = 0  # STATE_ATK == 1

                    elif target.STATE_AGI["state"] == 0:
                        target.STATE_AGI["state"] = -1

                    elif target.STATE_AGI["state"] == -1:
                        target.STATE_AGI["state"] = -2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if target.STATE_AGI["state"] == 0:
                        target.STATE_AGI["turn"] = 0

                    s += "* {} の素早さが {} 倍に下がった.\n({}ターン)\n".format(
                        target.name, target.status_rate(target.STATE_AGI["state"]), target.STATE_AGI["turn"])

                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # MagicATKダウン
        elif selected_Action == 16:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                self.MP -= needMP


                s += "* Result -> {} は {} に {} を唱えた.\n".format(
                    self.name, target.name, self.ACTIONS[selected_Action])

                
                flag = 1

                if flag == 0:
                    s += "しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:

                    # 持続ターン数を決める 3-5turn
                    n_turn = random.randrange(3, 6, 1)

                    # デバフが乗っていたら、さらにプラスする
                    if target.STATE_MagicATK["state"] <= 0:
                        target.STATE_MagicATK["turn"] += n_turn
                    # バフが乗っていたら、上書きする
                    else:
                        target.STATE_MagicATK["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if target.STATE_MagicATK["turn"] > 5:
                        target.STATE_MagicATK["turn"] = 5

                    # 攻撃力状態を参照
                    if target.STATE_MagicATK["state"] == 2:
                        target.STATE_MagicATK["state"] = 1

                    elif target.STATE_MagicATK["state"] == 1:
                        # STATE_ATK == 1
                        target.STATE_MagicATK["state"] = 0

                    elif target.STATE_MagicATK["state"] == 0:
                        target.STATE_MagicATK["state"] = -1

                    elif target.STATE_MagicATK["state"] == -1:
                        target.STATE_MagicATK["state"] = -2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if target.STATE_MagicATK["state"] == 0:
                        target.STATE_MagicATK["turn"] = 0

                    s += "* {} の攻撃魔力が {} 倍に下がった.\n({}ターン)\n".format(
                        target.name, target.status_rate(target.STATE_MagicATK["state"]), target.STATE_MagicATK["turn"])

                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # MagicRECダウン
        elif selected_Action == 17:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

            else:  # 唱えられる
                valid_action = 1
                self.MP -= needMP


                s += "* Result -> {} は {} に {} を唱えた.\n".format(
                    self.name, target.name, self.ACTIONS[selected_Action])

                flag = 1

                if flag == 0:
                    s += "しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:

                    # 持続ターン数を決める 3-5turn
                    n_turn = random.randrange(3, 6, 1)

                    # デバフが乗っていたら、さらにプラスする
                    if target.STATE_MagicREC["state"] <= 0:
                        target.STATE_MagicREC["turn"] += n_turn
                    # バフが乗っていたら、上書きする
                    else:
                        target.STATE_MagicREC["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if target.STATE_MagicREC["turn"] > 5:
                        target.STATE_MagicREC["turn"] = 5

                    # 攻撃力状態を参照
                    if target.STATE_MagicREC["state"] == 2:
                        target.STATE_MagicREC["state"] = 1

                    elif target.STATE_MagicREC["state"] == 1:
                        # STATE_ATK == 1
                        target.STATE_MagicREC["state"] = 0

                    elif target.STATE_MagicREC["state"] == 0:
                        target.STATE_MagicREC["state"] = -1

                    elif target.STATE_MagicREC["state"] == -1:
                        target.STATE_MagicREC["state"] = -2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if target.STATE_MagicREC["state"] == 0:
                        target.STATE_MagicREC["turn"] = 0

                    s += "* {} の回復魔力が {} 倍に下がった.\n({}ターン)\n".format(
                        target.name, target.status_rate(target.STATE_MagicREC["state"]), target.STATE_MagicREC["turn"])
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 耐性アップ
        elif selected_Action == 18:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                s += "* Result -> {} は {} を唱えた.\n".format(
                    self.name, self.ACTIONS[selected_Action])

                # 味方キャラクター
                mymembers = []
                for c in game.alived_characters:
                    if c.SIDE == self.SIDE:
                        mymembers.append(c)

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                # 味方キャラクタ1体ずつ処理する
                for mymember in mymembers:
                    # バフが乗っていたら、さらにプラスする
                    if mymember.REGISTANCE["state"] <= 0:
                        mymember.REGISTANCE["turn"] += n_turn
                    # デバフが乗っていたら、上書きする
                    else:
                        mymember.REGISTANCE["turn"] = n_turn

                    # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                    if mymember.REGISTANCE["turn"] > 5:
                        mymember.REGISTANCE["turn"] = 5

                    # 状態を参照
                    if mymember.REGISTANCE["state"] == -2:
                        mymember.REGISTANCE["state"] = -1

                    elif mymember.REGISTANCE["state"] == -1:
                        mymember.STATE_DEF["state"] = 0  # STATE_ATK == 1

                    elif mymember.REGISTANCE["state"] == 0:
                        mymember.REGISTANCE["state"] = 1

                    elif mymember.REGISTANCE["state"] == 1:
                        mymember.REGISTANCE["state"] = 2

                    # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                    if mymember.REGISTANCE["state"] == 0:
                        mymember.REGISTANCE["turn"] = 0

                    s += "* {} の耐性ダウン 成功率 が {}% に下がった!\n({}ターン)\n".format(mymember.name, int(
                        mymember.registance_rate(mymember.REGISTANCE["state"])*40), mymember.REGISTANCE["turn"])

                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 耐性ダウン (全体)
        elif selected_Action == 19:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかどうかの判定
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:  # 唱えられる
                valid_action = 1
                # 残りMPの計算
                self.MP -= needMP

                s += "* Result -> {} は {} を唱えた.\n".format(
                    self.name, self.ACTIONS[selected_Action])

                # 敵キャラクター
                enemys = []
                for c in game.alived_characters:
                    if c.SIDE == self.opposite_side(self.SIDE):
                        enemys.append(c)

                # 持続ターン数を決める 3-5turn
                n_turn = random.randrange(3, 6, 1)

                
                flag = 1

                if flag == 0:
                    s += "しかし, 効かない.\n"
                    return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
                else:
                    # 敵キャラクタ1体ずつ処理する
                    for enemy in enemys:
                        # デバフが乗っていたら、さらにプラスする
                        if enemy.REGISTANCE["state"] >= 0:
                            enemy.REGISTANCE["turn"] += n_turn
                        # デバフが乗っていたら、上書きする
                        else:
                            enemy.REGISTANCE["turn"] = n_turn

                        # 持続の最大ターンは5にするので、その判定も行う --- 重ねがけ対策
                        if enemy.REGISTANCE["turn"] > 5:
                            enemy.REGISTANCE["turn"] = 5

                        # 状態を参照
                        if enemy.REGISTANCE["state"] == -2:
                            enemy.REGISTANCE["state"] = -1

                        elif enemy.REGISTANCE["state"] == -1:
                            # STATE_ATK == 1
                            enemy.STATE_DEF["state"] = 0

                        elif enemy.REGISTANCE["state"] == 0:
                            enemy.REGISTANCE["state"] = 1

                        elif enemy.REGISTANCE["state"] == 1:
                            enemy.REGISTANCE["state"] = 2

                        # 最終的に"state" == 1.0 ならば、"turn"を0に戻す
                        if enemy.REGISTANCE["state"] == 0:
                            enemy.REGISTANCE["turn"] = 0

                        s += "* {} の耐性ダウン 成功率 が {}% に上がった!\n({}ターン)\n".format(enemy.name, int(
                            enemy.REGISTANCE["state"]*40), enemy.REGISTANCE["turn"])

                        return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 強攻撃
        if selected_Action == 20:
            needMP = self.MP_Actions[selected_Action]
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} をしようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:
                valid_action = 1
                
                # mp
                self.MP -= needMP

                # ダメージ計算
                damage = int(((self.status_rate(self.STATE_ATK["state"])*self.ATK)/2 - (
                    target.status_rate(target.STATE_DEF["state"]) * target.DEF)/4)*random.randrange(135, 170, 1)/100)

                # ダメージは0以上
                if damage <= 0:
                    damage = 0

                if target.PROTECT == True:
                    damage = int(damage/2)

                # ログ用
                log_damage = damage

                damages_enemy = damage

                # target.HPは0未満にならない
                if target.HP < damage:
                    damage = target.HP

                # ダメージを与える
                target.HP -= damage

                # game alived_charactersの更新
                game.update_characters()

                s += "* Result -> {} は {} に {} のダメージを与えた.\n".format(
                    self.name, target.name, log_damage)
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 全体強攻撃
        elif selected_Action == 21:
            # 必要MP
            needMP = self.MP_Actions[selected_Action]
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} をしようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:
                valid_action = 1
                # MP計算
                self.MP -= needMP

                s += "* Result -> {} は {} を唱えた.\n".format(
                    self.name, self.ACTIONS[selected_Action])

                # 攻撃する敵キャラクタを取得
                targets = self.target_enemy(self.SIDE, game)

                # 添字によって与えるダメージを補完していく
                damages = []

                # 敵1キャラずつダメージ処理を行う
                for i in range(len(targets)):
                    # ターゲットとなるキャラクタ
                    target = targets[i]

                    # ダメージ計算
                    damage = int(((self.status_rate(self.STATE_ATK["state"]) * self.ATK)/2 - (
                        target.status_rate(target.STATE_DEF["state"]) * target.DEF)/4)*random.randrange(72, 91, 1)/100)

                    # ダメージは0以上
                    if damage <= 0:
                        damage = 0

                    if target.PROTECT == True:
                        damage = int(damage/2)

                    # ログ用
                    log_damage = damage

                    # target.HPは0未満にならない
                    if target.HP < damage:
                        damage = target.HP

                    # ダメージを与える
                    target.HP -= damage

                    # 与えるダメージをdamagesに補完
                    damages.append(damage)

            # 総ダメージ計算
            damages_enemy = sum(damages)

            for i in range(len(targets)):
                s += "* {} は {} に {} のダメージを与えた.\n".format(
                    self.name, targets[i].name, damages[i])

            # game alived_charactersの更新
            game.update_characters()

            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action

        # 全体回復
        elif selected_Action == 22:
            needMP = self.MP_Actions[selected_Action]
            # 呪文が唱えられるかMPの計算
            if self.MP < needMP:
                valid_action = 0
                s += "* Result -> {} は {} を唱えようとした.\nしかしMPが足りない!\n".format(
                    self.name, self.ACTIONS[selected_Action])
                target_id = -1
                return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
            else:
                valid_action = 1
                # MP計算
                self.MP -= needMP

                s += "* Result -> {} は {} を唱えた.\n".format(
                    self.name, self.ACTIONS[selected_Action])

                # 回復キャラクタを取得
                mymembers = []
                for c in game.alived_characters:
                    if c.SIDE == self.SIDE:
                        mymembers.append(c)

                # キャラクタ別回復量
                recoveries = []

                # 回復させるキャラクター1体ずつ処理
                for mymember in mymembers:
                    # 残りHPの計算
                    recovery_hp = int(random.randrange(
                        35, 45, 1) * (1 + (self.MAGIC_REC * self.status_rate(self.STATE_MagicREC["state"]) / 300)) * random.randrange(45, 66, 1)/100)
                    after_hp = mymember.HP + recovery_hp
                    log_hp = recovery_hp  # ログ用
                    recoveries.append(recovery_hp)

                    # HP回復 --- after_hpがMAX_HPを超えていないか判定
                    if mymember.MAX_HP < after_hp:
                        mymember.HP = mymember.MAX_HP
                    else:
                        mymember.HP += recovery_hp

                    s += "* {} の HPは {} 回復した！\n".format(
                        mymember.name, log_hp)

            # 総回復量計算
            recoveries_enemy = sum(recoveries)

            # game alived_charactersの更新
            game.update_characters()

            return s, damages_enemy, recoveries_enemy, action, target_id, valid_action
