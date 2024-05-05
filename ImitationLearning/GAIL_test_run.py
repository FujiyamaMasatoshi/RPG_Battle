# パスの情報を管理するConfig.pyを自身のディレクトリに変更してimportしてください

from Config import Config
configuration=Config()
import sys
sys.path.append(configuration.parent_dir)  # 親ディレクトリを追加
import pygame
from pygame.locals import *
from CommandWindow import *

# 環境
from env.Game import Game
from env.GameState import GameState


# GAIL Agent
from GAIL_REINFORCE import *
import torch

# util
from function import *



def gail_test_run(playable_id=None, situation=2):
    # モジュールの初期化
    pygame.init()  # pygameの初期化
    game = Game(situation)
    
    dir = configuration.gail_learned_path
    
    # エージェントインスタンス作成
    # LSTM
    agents = {}
    agents_action_memory = {}
    lrs = (1e-4, 1e-4, 1e-4)
    for c in game.characters:
        if c.SIDE == 0:
            key = c.id
            agents[key] = RLAgent(game, key, lrs=lrs, argmax_episode=0, argmax_interval=10, exp_data_path=configuration.expert_data_path)
            agents[key].pi_action.load_state_dict(torch.load( dir+'agent_id={}_action.pth'.format(key), map_location=torch.device('cpu')))
            agents[key].pi_target1.load_state_dict(torch.load(dir+'agent_id={}_target1.pth'.format(key), map_location=torch.device('cpu')))
            agents[key].pi_target2.load_state_dict(torch.load(dir+'agent_id={}_target2.pth'.format(key), map_location=torch.device('cpu')))
            # 各エージェントは前回の行動から次の行動の直前の状態を保持する
            agents_action_memory[key] = []
    
    
    
    # スクリーンの設定
    screen = pygame.display.set_mode(SCREEN_SIZE)  # 横20, 縦15の幅をとる

    # タイトル表示
    pygame.display.set_caption("RPG")

    # 時間管理を行うオブジェクト
    clock = pygame.time.Clock()

    # command windowに表示した文字
    statement = ""
    
    # inputs
    action_txt = ""
    target_txt = ""
    
    # turn count
    cnt_turn = 0
    
    # error message
    error_txt = ""
    
    action = None
    target_id = None
    
    blue_window = 1 # now characterに青色で表示
    
    # 各ゲーム状態
    state_turn_start = 0
    state_turn_now = 0
    state_pop_character = 0
    
    # 敵の数
    # n_enemy
    n_enemy = 0
    for c in game.alived_characters:
        if c.SIDE == 1:
            n_enemy += 1
    
    # 敵のimgの大きさ
    enemy_img_width = int(SCREEN_SIZE_WIDTH/n_enemy*0.8)
    if enemy_img_width > 200:
        enemy_img_width = 200
    
    while True:
        # fps管理
        clock.tick(30)  
        
        # 背景の表示
        # しろ背景
        # screen.fill((255, 255, 255))
        # 黒背景
        screen.fill((0, 0, 0))


        # chara表示
        
        # alived characterの表示
        j = 0
        k = 0
        s_windows = []
        enemy_windows = []
        
        for c in game.alived_characters:
            start_x = 50
            if c.SIDE == 1:
                
                obj = pygame.transform.scale(c.Obj, (enemy_img_width, enemy_img_width))
                screen.blit(obj, (int(CMD_WINDOW_WIDTH/(n_enemy))*j, 50))
                # width = int((SCREEN_SIZE_WIDTH-start_x*2)/3)
                width = enemy_img_width
                rect = pygame.Rect((int(CMD_WINDOW_WIDTH/(n_enemy))*j, enemy_img_width+50, width, 100))
                enemy_windows.append(StatusWindow(rect))
                
                status = "{}\n".format(c.name)
                status += game.check_status(c)
                hp_rate = c.HP/c.MAX_HP
                if hp_rate > 0.5:
                    color = (255, 255, 255)
                elif 0.3 < hp_rate <=0.5:
                    color = (255, 225, 0)
                elif 0.1 < hp_rate <= 0.3:
                    color = (225, 170, 0)
                elif hp_rate <= 0.1:
                    color = (225, 0, 0)
                
                    
                if game.now_character is not None:
                    if game.now_character.id == c.id and blue_window == 1:
                        enemy_windows[j].draw(screen, status, color, now_turn=True)
                    else:
                        enemy_windows[j].draw(screen, status, color)
                else:
                    enemy_windows[j].draw(screen, status, color)
                
                j += 1
            else:
                # character表示
                obj = pygame.transform.scale(c.Obj, (100, 100))
                screen.blit(obj, (int((CMD_WINDOW_WIDTH-100)/4)*k+50, 400))
                width = int((SCREEN_SIZE_WIDTH-start_x*2)/4)
                rect = pygame.Rect((start_x+width*k, 500, width-10, 1.7*(width-10)))
                s_windows.append(StatusWindow(rect))
                # status = "{}\nHP:{}\nMP:{}\nATK:{}({})\nDEF:{}({})\nSPD:{}({})\nMPW:{}({})\nMRC:{}({})\nRGS:{}({})".format(c.name, c.HP, c.MP, c.STATE_ATK["state"], c.STATE_ATK["turn"], c.STATE_DEF["state"], c.STATE_DEF["turn"], c.STATE_AGI["state"], c.STATE_AGI["turn"], c.STATE_MagicATK["state"], c.STATE_MagicATK["turn"], c.STATE_MagicREC["state"], c.STATE_MagicREC["turn"], c.REGISTANCE["state"], c.REGISTANCE["turn"])
                status = "{}\nHP: {}/{}\nMP: {}/{}\n".format(c.name, c.HP, c.MAX_HP, c.MP, c.MAX_MP)
                status += game.check_status(c)
                hp_rate = c.HP/c.MAX_HP
                if hp_rate > 0.5:
                    color = (255, 255, 255)
                elif 0.3 < hp_rate <=0.5:
                    color = (255, 225, 0)
                elif 0.1 < hp_rate <= 0.3:
                    color = (225, 170, 0)
                elif hp_rate <= 0.1:
                    color = (225, 0, 0)
                
                
                if game.now_character is not None:
                    if game.now_character.id == c.id and blue_window == 1:
                        s_windows[k].draw(screen, status, color, now_turn=True)
                    else:
                        s_windows[k].draw(screen, status, color)
                else:
                    s_windows[k].draw(screen, status, color)  
                
                
                k += 1
                
        
        # for i in range(k):
            
        #     # 4枠の幅を計算
        #     width = int((SCREEN_SIZE_WIDTH-start_x*2)/4)
        #     rect = pygame.Rect((start_x+width*i, 600, width-10, width-10))
        #     s_windows.append(StatusWindow(rect))
        # for i in range(k):
            
        #     s_windows[i].draw(screen, "HP: 100")
        
        # command window
        c_window = CommandWindow(pygame.Rect(COMMAND_WINDOW_RECT))
        # c_window.draw(screen, "")
        # c_window.draw(screen, "")
        # # #################
        # # ゲーム状態管理 
        # # #################
        
        c_window.draw(screen, statement)
        
        # print(game.game_state, state_turn_now)
        
        if game.game_state == GameState.TURN_START:
            if cnt_turn == 0:
                game.turn += 1
                cnt_turn = 1
            if state_turn_start == 0:
                blue_window = 0
                statement = "TURN {}".format(game.turn)
            elif state_turn_start == 1:
                game.game_state = GameState.ACTION_ORDER
        elif game.game_state == GameState.ACTION_ORDER:
            # state_turn_start = 0
            game.action_order()
        elif game.game_state == GameState.POP_CHARACTER:
            if state_pop_character == 0 and blue_window == 0:
                game.pop_character()
                state_pop_character = -1
                print(game.now_character.name)
            
            
        elif game.game_state == GameState.TURN_NOW:
            # playable characterの行動
            if game.now_character.SIDE == 0 and (playable_id is not None) and game.now_character.id == playable_id:
                if state_turn_now == 0:
                    blue_window = 1
                    statement = "現在の行動キャラクタ: {}\n".format(game.now_character.name)
                    statement += "\nアクションを選択してください\n\n"
                    statement += game.s_action_set(game.now_character)
                    statement += "\n"
                    statement += action_txt
                    statement += error_txt
                    c_window.draw(screen, statement)
                elif state_turn_now == 1: # action入力完了
                    statement = "{} を選択しました．\n".format(game.now_character.ACTION_SET[action])
                    # print(game.now_character.name)
                    if game.now_character.Attribute_Actions[action] == "protect":
                        target_id = -1
                        state_turn_now = 2
                        
                    elif game.now_character.Attribute_Actions[action] == "attacking" or game.now_character.Attribute_Actions[action] == "debufsupport":
                        if action == 2 or action == 6 or action ==19 or action == 21:
                            target_id = 1
                            state_turn_now = 2
                        else:
                            alived_enemy = {}
                            for c in game.alived_characters:
                                if c.SIDE == 1:
                                    key = c.id
                                    alived_enemy[key] = c.name
                            targets = alived_enemy
                            statement += "ターゲットを選択してください\n\n"
                            statement += game.s_target_set(targets)
                            statement += "\n\n"
                            statement += target_txt
                            statement += error_txt
                    elif game.now_character.Attribute_Actions[action] == "healing" or game.now_character.Attribute_Actions[action] == "bufsupport":
                        if action == 9 or action == 10 or action == 18 or action == 22:
                            target_id = 0
                            state_turn_now = 2
                        else:
                            alived_mymembers = {}
                            for c in game.alived_characters:
                                if c.SIDE == 0:
                                    key = c.id
                                    alived_mymembers[key] = c.name
                            targets = alived_mymembers
                            statement += "ターゲットを選択してください\n\n"
                            statement += game.s_target_set(targets)
                            statement += "\n\n"
                            statement += target_txt
                            statement += error_txt
                        
                    # c_window.draw(screen, statement)
                
                elif state_turn_now == 2: # target_id入力完了
                    # ####################
                    # データの追加
                    # ####################
                    state = game.State()
                    for key, _ in agents_action_memory.items():
                        data = (state, action, target_id)
                        agents_action_memory[key].append(data)
                    # ####################
                    
                    
                    action_flow = (action, target_id)
                    
                    result_action = game.turn_now(action_flow)
                    
                    statement = result_action[0]
                    print(statement)
                    state_turn_now = -1
                    c_window.draw(screen, statement)
                    state_pop_character = 0
            # 味方エージェントの行動
            elif game.now_character.SIDE == 0 and game.now_character.id != playable_id:
                if state_turn_now == 0:
                    blue_window = 1
                    state = game.State()
                    now_agent_id = game.now_character.id
                    now_agent = agents[now_agent_id]
                    
                    # 現在の状態state, action=999, target=999をdataに追加する
                    temp_datas = copy.deepcopy(agents_action_memory[now_agent_id])
                    data = (state, 999, 999)
                    temp_datas.append(data)
                    seqs = [temp_datas]
                    
                    
                    # エージェントのアクション選択
                    # action = now_agent.select_action(state)
                    # 入力シーケンス
                    inputs = pre_process(seqs)
                    action, _ = now_agent.select_action(inputs, episode=1)
                    # action決定後 -> target memoryに追加する
                    
                    temp_datas.clear()
                    
                    # ターゲット選択
                    if game.now_character.Attribute_Actions[action] == "protect":
                        target_id = -1
                    elif game.now_character.Attribute_Actions[action] == "attacking" or game.now_character.Attribute_Actions[action] == "debufsupport":
                        if action == 2 or action == 6 or action ==19 or action == 21:
                            target_id = 1
                        else:
                            temp_datas = copy.deepcopy(agents_action_memory[now_agent_id])
                            data = (state, action, 999)
                            temp_datas.append(data)
                            seqs = [temp_datas]
                            inputs = pre_process(seqs)
                            target_id, _ = now_agent.select_target2(inputs, episode=1)
                            temp_datas.clear()
                    elif game.now_character.Attribute_Actions[action] == "healing" or game.now_character.Attribute_Actions[action] == "bufsupport":
                        if action == 9 or action == 10 or action == 18 or action == 22:
                            target_id = 0
                        else:
                            temp_datas = copy.deepcopy(agents_action_memory[now_agent_id])
                            data = (state, action, 999)
                            temp_datas.append(data)
                            seqs = [temp_datas]
                            inputs = pre_process(seqs)
                            target_id, _ = now_agent.select_target1(inputs, episode=1)
                            temp_datas.clear()
                    
                    # 行動したエージェントのinputs_sequenceをclearする
                    agents_action_memory[now_agent_id].clear()
                    
                    
                    
                    # ####################
                    # データの追加
                    # ####################
                    for key, _ in agents_action_memory.items():
                        data = (state, action, target_id)
                        agents_action_memory[key].append(data)
                    # ####################
                    
                    # 行動の実行
                    action_flow = (action, target_id)
                    result_action = game.turn_now(action_flow)
                    
                    state_turn_now = -1
                    # _action = result_action[3]
                    # _target_id = result_action[4]
                    _s = result_action[0]
                    print(_s)
                    statement = "味方エージェントの行動\n"
                    statement += "行動キャラクタ: {}\n".format(game.now_character.name)
                    # statement += "* action -> {}\n\n".format(game.now_character.ACTION_SET[_action])
                    # statement += "* target -> {}\n\n".format(game.id_to_name(_target_id, game.characters))
                    
                    statement += _s
               
                    c_window.draw(screen, statement)
                    state_pop_character = 0
            # 敵の行動
            elif game.now_character.SIDE == 1:
                if state_turn_now == 0:
                    blue_window = 1
                    
                    result_action = game.turn_now(action_flow=None)
                    
                    
                    state_turn_now = -1
                    # _action = result_action[3]
                    # _target_id = result_action[4]
                    _s = result_action[0]
                    statement = "敵の行動\n"
                    statement += "行動キャラクタ: {}\n".format(game.now_character.name)
                    # statement += "* action -> {}\n\n".format(game.now_character.ACTION_SET[_action])
                    # statement += "* target -> {}\n\n".format(game.id_to_name(_target_id, game.characters))
                    print(_s)
                    statement += _s
                    

                    c_window.draw(screen, statement)
                    state_pop_character = 0
                
        elif game.game_state == GameState.TURN_END:
            game.turn_end()
            state_turn_start = -1
            cnt_turn = 0
        elif game.game_state == GameState.GAME_END:
            win = game.game_end()
            if win:
                statement = "Game Clear !!"
            else:
                statement = "Game Over ..."
            c_window.draw(screen, statement)
                        
            
        
        

        # イベント処理
        for event in pygame.event.get():
            # バツボタンで終了
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if game.game_state == GameState.TURN_START:
                if state_turn_start == -1:
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        state_turn_start = 0
                elif state_turn_start == 0:
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        state_turn_start = 1
            elif game.game_state == GameState.POP_CHARACTER:
                if event.type == KEYDOWN and event.key == K_SPACE:
                    blue_window = 0
                    state_turn_now = 0
                    
        
            elif game.game_state == GameState.TURN_NOW:
                if game.now_character.SIDE == 0:
                    if state_turn_now == -1:
                        if event.type == KEYDOWN and event.key == K_SPACE:
                            state_turn_now = 0
                    if state_turn_now == 0:
                        c_window.draw(screen, statement)
                        if event.type == KEYDOWN:
                            # 数字入力を実行する
                            if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                                if not action_txt == "":
                                    action = int(action_txt)
                                    
                                    # actionの有効性を確認
                                    if action in game.now_character.ACTION_SET:
                                        
                                        action_txt = ""
                                        error_txt = ""
                                        state_turn_now = 1
                                        break
                                    else:
                                        action_txt = ""
                                        error_txt = "無効なアクションです.有効なアクションを選択してください"
                                else:
                                    error_txt = "無効な入力です．もう一度入力してください"
                            elif event.key == pygame.K_BACKSPACE:
                                action_txt = action_txt[:-1]
                            else:
                                if event.unicode.isnumeric():
                                    error_txt = ""
                                    action_txt += event.unicode
                    elif state_turn_now == 1:
                        c_window.draw(screen, statement)
                        if event.type == KEYDOWN:
                            if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                                if not target_txt == "":
                                    target_id = int(target_txt)
                                
                                    if target_id in targets.keys():
                                        target_txt = ""
                                        error_txt = ""
                                        state_turn_now = 2
                                    else:
                                        target_txt = ""
                                        error_txt = "無効なターゲットです.有効なターゲットを選択してください"
                                else:
                                    error_txt = "無効な入力です．もう一度入力してください"
                            elif event.key == pygame.K_BACKSPACE:
                                target_txt = target_txt[:-1]
                            else:
                                if event.unicode.isnumeric():
                                    error_txt = ""
                                    target_txt += event.unicode
                # 敵キャラクタの時
                else:
                    if state_turn_now == -1:
                        c_window.draw(screen, statement)
                        if event.type == KEYDOWN and event.key == K_SPACE:
                            state_turn_now = 0

                    
            elif game.game_state == GameState.GAME_END:
                if event.type == KEYDOWN and event.key == K_SPACE:
                    pygame.quit()
                    sys.exit()   

        pygame.display.update()




if __name__ == '__main__':
    selection = [10, 20, 30, 40, "x"]
    print("操作するキャラクタidを選択してください [10, 20, 30, 40, a]\n(aを選択した場合は全てのキャラクタがAIによるでもプレイを見ることができます)")
    agent_id = input()
    print("ゲームシチュエーションを選択してください, [1, 2, 3, 4]\n(situatio=4は学習させていない状況です。)")
    situation=int(input())
    if agent_id == "a":
        gail_test_run(situation=situation)
    else:
        agent_id = int(agent_id)
        gail_test_run(agent_id, situation=situation)