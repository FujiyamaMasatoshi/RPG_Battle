# パスの情報を管理するConfig.pyを自身のディレクトリに変更してimportしてください
from Config import Config
configuration = Config()
# GAILによるシミュレーション学習

# ライブラリimport
import copy
import numpy as np
# np.random.seed(123) # シード固定

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 環境
import sys
# /rpg_battle/env/をimportするので親ディレクトリを指定してください ↓
# sys.path.append('/Users/fujiyamax/home/myProjects/rpg_battle/')  # 親ディレクトリを追加
sys.path.append(configuration.parent_dir)  # 親ディレクトリを追加
from env.Game import Game
from env.GameState import GameState
from env.Character import Character

# GAILの学習
from GAIL_exp import GAIL_ExpData # expertデータを保持
from function import pre_process # シーケンスデータの前処理に使用

# エージェント
from GAIL_REINFORCE import * # algo=REINFORCE



# # 学習ログ保持
# import wandb
# wandb.init(
#     project="GAIL"
# )

# #################################
# lossなどの学習ログを表記するメソッド
# #################################

# def plot_loss(agent_id, outputs_disc_generator, outputs_disc_expert):
#     wandb.log({
#         "agent_id={}: generator_action".format(agent_id): outputs_disc_generator[0],
#         "agent_id={}: generator_target1".format(agent_id): outputs_disc_generator[1],
#         "agent_id={}: generator_target2".format(agent_id): outputs_disc_generator[2],
#         "agent_id={}: expert_action".format(agent_id): outputs_disc_expert[0],
#         "agent_id={}: expert_target1".format(agent_id): outputs_disc_expert[1],
#         "agent_id={}: expert_target2".format(agent_id): outputs_disc_expert[2]
#     })

# def plot_clear_turn(clear_turn, episode):
#     # 現在の終了turn
#     turn = clear_turn[episode-1]
#     if turn == 0:
#         turn = 35 #
    
#     # 過去100episodeのうち、winした時の平均クリアターンを取る
#     if len(clear_turn) < 100:
#         n_win = len([x for x in clear_turn if x > 0])
#         if n_win == 0:
#             ave_clear_turn = None
#         else:
#             ave_clear_turn = sum(clear_turn)/n_win
#     else:
#         n_win = len([x for x in clear_turn[episode-100:episode] if x > 0])
#         if n_win == 0:
#             ave_clear_turn = None
#         else:
#             ave_clear_turn = sum(clear_turn[episode-100:episode])/n_win
    
#     # 過去100episodeでの最小クリアターンを求める
#     if len(clear_turn) < 100:
#         sublist = []
#         for t in clear_turn[:episode]:
#             if t > 0:
#                 sublist.append(t)
#         if len(sublist) > 0:
#             min_clear_turn = np.min(sublist)
#         else:
#             min_clear_turn = None
#     else:
#         sublist = []
#         for t in clear_turn[episode-100:episode]:
#             if t > 0:
#                 sublist.append(t)
#         if len(sublist) > 0:
#             min_clear_turn = np.min(sublist)
#         else:
#             min_clear_turn = None
    
#     # 最大値を取得 # game overによる35turnは除く
#     if len(clear_turn) < 100:
#         sublist = []
#         for t in clear_turn[:episode]:
#             if 0 < t <= 35:
#                 sublist.append(t)
#         if len(sublist) > 0:
#             max_clear_turn = np.max(sublist)
#         else:
#             max_clear_turn = None
#     else:
#         sublist = []
#         for t in clear_turn[episode-100:episode]:
#             if 0 < t <= 35:
#                 sublist.append(t)
#         if len(sublist) > 0:
#             max_clear_turn = np.max(sublist)
#         else:
#             max_clear_turn = None
#     # wandbにplot
#     wandb.log({
#         "clear turn": turn,
#         "mean of clear turn (before 100 episode)": ave_clear_turn,
#         "min of clear turn (before 100 episodes)": min_clear_turn,
#         "max of clear turn (before 100 episodes)": max_clear_turn
#     })

# def plot_reward(agent_id, rewards):
#     wandb.log({f"agent_id={agent_id}: action reward": rewards[0],
#                f"agent_id={agent_id}: target1 reward": rewards[1],
#                f"agent_id={agent_id}: target2 reward": rewards[2]})

# def plot_loss_agent_update(agent_id, loss_agent_update):
#     loss_a = loss_agent_update[0]
#     loss_t1 = loss_agent_update[1]
#     loss_t2 = loss_agent_update[2]
#     wandb.log({
#         f"agent_id={agent_id}: loss pi action": loss_a,
#         f"agent_id={agent_id}: loss pi target1": loss_t1,
#         f"agent_id={agent_id}: loss pi target2": loss_t2,
#     })

# def plot_loss_disc_update(loss_disc_update):
#     loss_a = loss_disc_update[0]
#     loss_t1 = loss_disc_update[1]
#     loss_t2 = loss_disc_update[2]
    
#     wandb.log({
#         "loss disc action": loss_a
#     })
#     if not loss_t1 == 0.0:
#         wandb.log({
#             "loss disc target1": loss_t1
#         })
#     if not loss_t2 == 0.0:
#         wandb.log({
#             "loss disc target2": loss_t2
#         })
        

# def plot_pre_training_loss(agent_id, loss):
#     wandb.log({
#         f"agent_id={agent_id}: loss action": loss[0],
#         f"agent_id={agent_id}: loss target1": loss[1],
#         f"agent_id={agent_id}: loss target2": loss[2]
#     })

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self, x):
        # x = torch.stack(x, dim=0)
        x = x.to(torch.float)
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        
        return out # 偽物である確率を出力

  
        



# 全てのエージェントがGAILによって動く
class GAIL:
    def __init__(self):
        # 各RLエージェントの学習率
        self.config = {
            "lr_pre_training": {"10": {"action": 1e-3, "target1": 1e-3, "target2": 1e-3}, 
                                "20": {"action": 1e-3, "target1": 1e-3, "target2": 1e-3}, 
                                "30": {"action": 1e-3, "target1": 1e-3, "target2": 1e-3},
                                "40": {"action": 1e-3, "target1": 1e-3, "target2": 1e-3}},
            "lr_generator": {"10": {"action": 1e-5, "target1": 1e-5, "target2": 1e-5}, 
                             "20": {"action": 1e-5, "target1": 1e-5, "target2": 1e-5}, 
                             "30": {"action": 1e-5, "target1": 1e-5, "target2": 1e-5},
                             "40": {"action": 1e-5, "target1": 1e-5, "target2": 1e-5}},
            "argmax_episode": 0,
            "argmax_interval":0,
            "lr_discriminator": {"action": 5e-6, "target1": 5e-6, "target2": 5e-6},
            "batch_size": 32,
            "update_interval": 45,
            "episodes": 300,
            "load_init_params_agents": False, # 保存されている初期値をloadするかどうか
            "load_init_params_discriminator": False, # 保存されている初期値をloadするかどうか
            "game_situation": 1,
            "use_env_reward": True,
            "negative_point": 2.0,
            "reward_log": True, #報酬にlogをとる(True)か 1-sigmoidをとる(False)か
            
        }
        
        print("config:\n", self.config)
        

        self.env = Game(1) # situation initialize
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 生成器 -- RL agents
        # self.learning_id = 20 # 指定したidのみDemoPlayエージェント
        self.agents = {}
        self.agents_id = {}
        self.agents_action_sequence = {} # lstm入力用にmemoryを用意
        
        self.load_init_params_agents = self.config["load_init_params_agents"]
        self.load_init_params_discriminator = self.config["load_init_params_discriminator"]
        # init_params_dir = "/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/GAIL_Init_Params_model/"
        init_params_dir = configuration.pre_train_path
        
        
        
        
        # updateの頻度 (何episode毎にupdateを呼ぶか)
        self.update_interval = self.config["update_interval"]
        
        # 環境からの報酬値
        self.NP = self.config["negative_point"]
        
        
        # self.expert_dir="/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/ExpertData/situation=all_ALL_PLAYERS_all.txt"
        self.expert_dir=configuration.expert_data_path
        
        for c in self.env.characters:
            if c.SIDE == 0:
                key = c.id
                lr_a = self.config["lr_generator"][f"{key}"]["action"]
                lr_t1 = self.config["lr_generator"][f"{key}"]["target1"]
                lr_t2 = self.config["lr_generator"][f"{key}"]["target2"]
                lrs = (lr_a, lr_t1, lr_t2)
                argmax_interval = self.config["argmax_interval"]
                argmax_episode = self.config["argmax_episode"]
                print("expert data path:", self.expert_dir)
                self.agents[key] = RLAgent(self.env, key, lrs=lrs, argmax_episode=argmax_episode, argmax_interval=argmax_interval, exp_data_path=self.expert_dir)
                self.agents_id[key] = key
                self.agents_action_sequence[key] = []
                
                
                # 初期値を読み込むかどうか
                if self.load_init_params_agents is True:
                    self.agents[key].pi_action.load_state_dict(torch.load(init_params_dir+'agent_id={}_action.pth'.format(key)))
                    self.agents[key].pi_target1.load_state_dict(torch.load(init_params_dir+'agent_id={}_target1.pth'.format(key)))
                    self.agents[key].pi_target2.load_state_dict(torch.load(init_params_dir+'agent_id={}_target2.pth'.format(key)))
        
        self.buffer_size = 40000
        self.batch_size = self.config["batch_size"]
        self.expert_data = GAIL_ExpData(self.buffer_size, self.batch_size, self.expert_dir, list(self.agents.keys()))

    
        
        # 生成データを保持
        self.memory = []

        
        # 識別器
        
        # loss function
        self.loss_fn = nn.BCELoss()
        self.hidden_dim = 128
        self.input_disc = self.expert_data.buffer.shape[-1]
        self.disc_action = Discriminator(self.input_disc, self.hidden_dim, 1, 1)
        self.disc_action.to(self.device)
        self.optim_disc_action = optim.Adam(self.disc_action.parameters(), lr=self.config["lr_discriminator"]["action"])
        self.disc_target1 = Discriminator(self.input_disc, self.hidden_dim, 1, 1)
        self.disc_target1.to(self.device)
        self.optim_disc_target1 = optim.Adam(self.disc_target1.parameters(), lr=self.config["lr_discriminator"]["target1"])
        self.disc_target2 = Discriminator(self.input_disc, self.hidden_dim, 1, 1)
        self.disc_target2.to(self.device)
        self.optim_disc_target2 = optim.Adam(self.disc_target2.parameters(), lr=self.config["lr_discriminator"]["target2"])
        
        # 初期値の読み込み
        if self.load_init_params_discriminator is True:
            self.disc_action.load_state_dict(torch.load(init_params_dir+"disc_action.pth"))
            self.disc_target1.load_state_dict(torch.load(init_params_dir+"disc_target1.pth"))
            self.disc_target2.load_state_dict(torch.load(init_params_dir+"disc_target2.pth"))
        
        # ###############################################
        # エージェントnn, discriminator nnの初期値を保存する
        # ###############################################
        # save agents nn parameters 
        if self.load_init_params_agents is False:
            for key, agent in self.agents.items():
                torch.save(self.agents[key].pi_action.state_dict(), init_params_dir+'agent_id={}_action.pth'.format(key))
                torch.save(self.agents[key].pi_target1.state_dict(), init_params_dir+'agent_id={}_target1.pth'.format(key))
                torch.save(self.agents[key].pi_target2.state_dict(), init_params_dir+'agent_id={}_target2.pth'.format(key))
        # save discriminator params
        if self.load_init_params_discriminator is False:
            torch.save(self.disc_action.state_dict(), init_params_dir+'disc_action.pth')
            torch.save(self.disc_target1.state_dict(), init_params_dir+'disc_target1.pth')
            torch.save(self.disc_target2.state_dict(), init_params_dir+'disc_target2.pth')
    
    
    
    # データ(agent_id, state, action, target_id, actioin_prob, taregt_prob, reward)からシーケンスデータを作成して返す
    def make_sequence(self, datas, agent_id):
        # エージェント毎にデータを作成
        # 前の行動後の(状態,行動)から時行動直前の(状態,行動)までを保持
        seqs = [] # (seq, action_prob, reward)を持つlist
        seq = []
        found = False
        for data in datas:
            # # if isinstance(data, str):
            # #     continue
            # else:
            if isinstance(data, str):
                if data == "GAME_END":
                    seq = []
                else:
                    continue
                # continue
            else:
                
                data_id = data[0]
                if found:
                    if data_id == agent_id:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        action_prob = data[4]
                        reward = data[6]
                        
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        seqs.append((seq, action_prob, reward))
                        seq = []
                        
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                else:
                    if data_id == agent_id:
                        found = True
                        
                        if not seq == []:
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            action_prob = data[4]
                            reward = data[6]
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            
                            seqs.append((seq, action_prob, reward))
                            seq = []
                            # クリアしたら次のseq用にseqに追加
                            seq.append((s, action, target_id))
                        else:
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            action_prob = data[4]
                            reward = data[6]
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            
                            seqs.append((seq, action_prob, reward))
                            
                            # seqsに追加したらseqをクリアしてseqに新たにs, a, tを追加
                            seq = []
                            seq.append((s, action, target_id))
                            
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))

        # breakpoint()
        return seqs
    
    # target1 network用のシーケンスデータ抽出
    def make_sequence_for_target1(self, datas, agent_id):
        # print("self.data:\n", self.datas[1])
        # エージェント毎にデータを作成
        # 前の行動後の(状態,行動)から時行動直前の(状態,行動)までを保持
        seqs = []
        seq = []
        found = False
        cnt = 0
        for data in datas:
            cnt += 1
            if isinstance(data, str):
                if data == "GAME_END":
                    seq = []
                else:
                    continue
                # continue
            else:
            
                
                # print("type of data[0] {}, ({}/{})".format(type(data[1]), cnt, len(datas)))
                # if len(datas) == 1:
                    # print("data",data)
                data_id = data[0]
                if found:
                    if data_id == agent_id:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        target_prob = data[5]
                        reward = data[6]
                        
                        # seqの最後のデータ
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        # target1を選択しているなら、seqsに追加する
                        # そうでなければ、追加しない
                        if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                            seqs.append((seq, target_prob, reward))
                        seq = []
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                else:
                    if data_id == agent_id:
                        found = True
                        
                        if not seq == []:
                            # dataを追加
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            target_prob = data[5]
                            reward = data[6]
                            
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            
                            # 最後のデータを確認
                            last_seq = seq[-1]
                            state = last_seq[0]
                            action = last_seq[1]
                            target_id = last_seq[2]
                            
                            if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                                seqs.append((seq, target_prob, reward))
                            seq = []
                            # クリアしたら次のseq用にseqに追加
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                        
                        else:
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            target_prob = data[5]
                            reward = data[6]
                            
                            # target_idを確認してtarget1ならばseqsに追加
                            if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                                seq.append((s, action, target_id))
                                # seq.append((data_id, s, action, target_id))
                                seqs.append((seq, target_prob, reward))
                            seq = []
                            # seqsに追加したらseqをクリアして
                            # 次のseq用に(s, a, t)をseqに追加
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                    
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
        return seqs
    
    
    # target2 network用のシーケンスデータ抽出
    def make_sequence_for_target2(self, datas, agent_id):
        # print("self.data:\n", self.datas[1])
        # エージェント毎にデータを作成
        # 前の行動後の(状態,行動)から時行動直前の(状態,行動)までを保持
        seqs = []
        seq = []
        found = False
        cnt = 0
        for data in datas:
            cnt += 1
            if isinstance(data, str):
                if data == "GAME_END":
                    seq = []
                else:
                    continue
                # continue
            else:
                
                # print("type of data[0] {}, ({}/{})".format(type(data[1]), cnt, len(datas)))
                # if len(datas) == 1:
                    # print("data",data)
                data_id = data[0]
                if found:
                    if data_id == agent_id:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        target_prob = data[5]
                        reward = data[6]
                                                
                        # seqの最後のデータ
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        # target1を選択しているなら、seqsに追加する
                        # そうでなければ、追加しない
                        if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                            seqs.append((seq, target_prob, reward))
                        seq = []
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                else:
                    if data_id == agent_id:
                        found = True
                        if not seq == []:
                            # dataを追加
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            target_prob = data[5]
                            reward = data[6]
                            
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            
                            if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                                seqs.append((seq, target_prob, reward))
                            seq = []
                            # クリアしたら次のseq用にseqに追加
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                        else:
                            s = data[1]
                            action = data[2]
                            target_id = data[3]
                            target_prob = data[5]
                            reward = data[6]
                            
                            # target_idを確認してtarget1ならばseqsに追加
                            if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                                seq.append((s, action, target_id))
                                # seq.append((data_id, s, action, target_id))
                                seqs.append((seq, target_prob, reward))
                            seq = []
                            # seqsに追加したらseqをクリアして
                            # 次のseq用に(s, a, t)をseqに追加
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                    
                    else:
                        s = data[1]
                        action = data[2]
                        target_id = data[3]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
        return seqs
    
    
    # turn start - turn endまでをシーケンスデータとする
    def make_sequence_ver_2(self, datas, agent_id):
        result_sequence = []
        # (state, action, target)の列をresult_sequenceに保持していく
        sequence = [] # 1sequenceを作成していく
        for data in datas:
            if isinstance(data, str):
                if data == "GAME_END":
                    if sequence == []:
                        continue
                    else:
                        result_sequence.append(sequence)
                        sequence = []
                elif data == "TURN_START":
                    if sequence == []:
                        continue
                    else:
                        result_sequence.append(sequence)
                        sequence = []
            else:
                # dataを分解
                id = data[0]
                state = data[1]
                action = data[2]
                target_id = data[3]
                next_state = data[4]
                reward = data[5]
                done = data[6]
                # 1sequenceに(state, action, target_id)を追加
                sequence.append((state, action, target_id))
        
        return result_sequence
    
    # updateメソッド
    def update(self, agent_id):
        # print("update!!")
        # discriminator update
        
        # #####################
        # action
        # #####################
        
        
        # generator sequence
        action_sequence_probs = self.make_sequence(self.memory, agent_id)
        action_sequence = [item[0] for item in action_sequence_probs]
        # action_probs = [item[1] for item in action_sequence_probs]
        valid_action_rewards = [self.NP*item[2]-self.NP for item in action_sequence_probs] # valid actionによるreward (0, -2)
        inputs_generator = pre_process(action_sequence)
        
       
        
        # breakpoint()
        
        # エージェントupdate
        with torch.no_grad():
            inputs_generator = inputs_generator.to(self.device)
            
            # agentに用いる報酬
            if self.config["reward_log"] is True:
                action_rewards = -torch.log(F.sigmoid(self.disc_action(inputs_generator))).cpu() # reward = log(disc)
                
            else:
                action_rewards = F.sigmoid(-self.disc_action(inputs_generator)).cpu()
            
            if self.config["use_env_reward"] is True:
                action_rewards = action_rewards + torch.tensor(valid_action_rewards,dtype=torch.float).unsqueeze(1)
            
            action_rewards = action_rewards.detach().numpy()
        
        # ###################
        # pi action update
        # ####################
        loss_a = self.agents[agent_id].pi_action_update(action_rewards)
        
        # expert sequence
        
        expert_action_sequence = self.expert_data.get_batch_action_seq(agent_id)
        inputs_exp = pre_process(expert_action_sequence)
        inputs_exp = inputs_exp.to(self.device)
        
        # 表示のためのoutputs
        with torch.no_grad():
            outputs_disc_exp_action = F.sigmoid(self.disc_action(inputs_exp)).mean()
            outputs_disc_generator_action = F.sigmoid(self.disc_action(inputs_generator)).mean()
        
        
        # ######################
        # disc action update
        # #######################
        
        # loss
        loss_exp_action = -F.logsigmoid(-self.disc_action(inputs_exp).cpu()).mean() # expertデータを入力とした時のloss
        loss_pi_action = -F.logsigmoid(self.disc_action(inputs_generator).cpu()).mean() # 生成データを入力とした時のloss
        
        loss_disc_action = loss_pi_action + loss_exp_action
        
        # breakpoint()
        
        self.optim_disc_action.zero_grad()
        loss_disc_action.backward()
        self.optim_disc_action.step()
        
        # breakpoint()
        
        loss_disc_action.cpu().detach().numpy()
        action_rewards = action_rewards.mean()
        
        
        # #####################
        # target1
        # #####################

        # generator 
        target1_sequence_probs = self.make_sequence_for_target1(self.memory, agent_id)
        target1_sequence = [item[0] for item in target1_sequence_probs]
        inputs_generator = pre_process(target1_sequence)
        
        # expert sequence
        # expert_target1_sequence = self.expert_data.target1_seqs[agent_id]
        expert_target1_sequence = self.expert_data.get_batch_target1_seq(agent_id)
        # print("len of exp_target1_seqs:", len(expert_target1_sequence))
        inputs_exp = pre_process(expert_target1_sequence)
        
        
        if inputs_generator != [] and inputs_exp != []:
            
            inputs_generator = inputs_generator.to(self.device)
            
            # 報酬
            with torch.no_grad():
                
                
                if self.config["reward_log"] is True:
                    target1_rewards = -torch.log(F.sigmoid(self.disc_target1(inputs_generator)))
                else:
                    target1_rewards = F.sigmoid(-self.disc_target1(inputs_generator))
                
                target1_rewards.cpu().detach().numpy()
            
            
            # ##################
            # pi target1 update
            # ###################
            loss_t1 = self.agents[agent_id].pi_target1_update(target1_rewards)

            # expert loss
            # to gpu
            inputs_exp = inputs_exp.to(self.device)
            
            # 表示のためのoutputs
            with torch.no_grad():
                outputs_disc_exp_target1 = F.sigmoid(self.disc_target1(inputs_exp).cpu()).mean()
                outputs_disc_generator_target1 = F.sigmoid(self.disc_target1(inputs_generator).cpu()).mean()
            
            
            # loss disc
            loss_pi_target1 = -F.logsigmoid(self.disc_target1(inputs_generator)).mean()
            loss_exp_target1 = -F.logsigmoid(-self.disc_target1(inputs_exp)).mean()
            loss_disc_target1 = loss_pi_target1 + loss_exp_target1
            
            # breakpoint()
            
            self.optim_disc_target1.zero_grad()
            loss_disc_target1.backward()
            self.optim_disc_target1.step()
            
            # breakpoint()
            
            loss_disc_target1.cpu().detach().numpy()
            target1_rewards = target1_rewards.mean()
        else:
            loss_disc_target1 = 0.0
            loss_generator_target1 = 0.0
            outputs_disc_generator_target1 = 0.0
            outputs_disc_exp_target1 = 0.0
            target1_rewards = 0.0
            loss_t1 = 0.0
        
        # #####################
        # target2
        # #####################
        
        # discriminator to gpu
        # self.disc_target2.to(self.device)
        
        # generator
        target2_sequence_probs = self.make_sequence_for_target2(self.memory, agent_id)
        target2_sequence = [item[0] for item in target2_sequence_probs]
        inputs_generator = pre_process(target2_sequence)
        
        # expert sequence
        expert_target2_sequence = self.expert_data.get_batch_target2_seq(agent_id)
        inputs_exp = pre_process(expert_target2_sequence)
        
        
        if inputs_generator != [] and inputs_exp != []:
            
            inputs_generator = inputs_generator.to(self.device)
            
            # 報酬
            with torch.no_grad():
                if self.config["reward_log"] is True:
                    target2_rewards = -torch.log(F.sigmoid(self.disc_target2(inputs_generator)))
                else:
                    target2_rewards = F.sigmoid(-self.disc_target2(inputs_generator))
                
                target2_rewards.cpu().detach().numpy()
            
            
            # ##################
            # pi target2 update
            # ##################
            loss_t2 = self.agents[agent_id].pi_target2_update(target2_rewards)
            
            inputs_exp = inputs_exp.to(self.device)
            
            # 表示のためのoutputs
            with torch.no_grad():
                outputs_disc_exp_target2 = F.sigmoid(self.disc_target2(inputs_exp).cpu()).mean()
                outputs_disc_generator_target2 = F.sigmoid(self.disc_target2(inputs_generator).cpu()).mean()            
            # loss disc
            loss_pi_target2 = -F.logsigmoid(self.disc_target2(inputs_generator)).mean()
            loss_exp_target2 = -F.logsigmoid(-self.disc_target2(inputs_exp)).mean()
            # generator_label = torch.zeros((len(inputs_generator), 1), dtype=torch.float).to(self.device)
            # exp_label = torch.ones((len(inputs_exp), 1), dtype=torch.float).to(self.device)
            # loss_pi_target2 = self.loss_fn(self.disc_target2(inputs_generator), generator_label).mean()
            # loss_exp_target2 = self.loss_fn(self.disc_target2(inputs_exp), exp_label).mean()
            loss_disc_target2 = loss_pi_target2 + loss_exp_target2
            
            # breakpoint()
            
            self.optim_disc_target2.zero_grad()
            loss_disc_target2.backward()
            self.optim_disc_target2.step()
            
            # breakpoint()
            
            # loss disc to cpu
            loss_disc_target2.cpu().detach().numpy()
            target2_rewards = target2_rewards.mean()
        else:
            loss_disc_target2 = 0.0
            loss_generator_target2 = 0.0
            outputs_disc_generator_target2 = 0.0
            outputs_disc_exp_target2 = 0.0
            target2_rewards = 0.0
            loss_t2 = 0.0
        
        
        # lossをまとめる
        # loss_generator = (action_rewards, target1_rewards, target2_rewards)
        # loss_discriminator = (loss_disc_action, loss_disc_target1, loss_disc_target2)
        
        outputs_disc_generator = (outputs_disc_generator_action, outputs_disc_generator_target1, outputs_disc_generator_target2)
        outputs_disc_expert = (outputs_disc_exp_action, outputs_disc_exp_target1, outputs_disc_exp_target2)
        rewards = (action_rewards, target1_rewards, target2_rewards)
        loss_agent_update = (loss_a, loss_t1, loss_t2)
        loss_disc_update = (loss_disc_action, loss_disc_target1, loss_disc_target2)
        
        return outputs_disc_generator, outputs_disc_expert, rewards, loss_agent_update, loss_disc_update
    
    
    def changed_env(self, env):
        self.env = env
    
    def learning(self):
        # episodes
        episodes = self.config["episodes"]
        
        # training result
        n_win = 0
        clear_turn = []
        n_wins = {1:0, 2:0, 3:0}

        # start episode
        for episode in range(1, 1+episodes):
            # 環境のreset
            self.situation = ((episode-1) % 3) + 1
            env = Game(self.situation)
            self.changed_env(env) # self.env = env
            state = env.reset(situation=self.situation)
            done = False
            for key, _ in self.agents.items():
                self.agents[key].update_changed_env(env)
            
            # agents_action_memoryを空にする
            for key, _ in self.agents_action_sequence.items():
                self.agents_action_sequence[key] = []
            
            # start simulation
            while not done:
                if env.game_state is GameState.TURN_START or env.game_state is GameState.ACTION_ORDER:
                    action_flow = None
                    next_state, reward, done = env.step(action_flow)
                    
                    self.memory.append("TURN_START")
                elif env.game_state is GameState.POP_CHARACTER:
                    action_flow = None
                    next_state, reward, done = env.step(action_flow)
                elif env.game_state == GameState.TURN_NOW:
                    state = self.env.State()
                    # enemyの行動
                    if env.now_character.SIDE == 1:
                        action_flow = None
                        next_state, reward, done, action, target_id = env.step(action_flow)
                        
                        # 敵の行動も追加 する設定の時は追加
                        # データの追加 --- 全てのエージェントにs, a, tを追加する
                        for key, _ in self.agents.items():
                            data = (state, action, target_id)
                            self.agents_action_sequence[key].append(data)
                        
                    # エージェントの行動
                    else:
                        # 現在のエージェント
                        now_agent_id = env.now_character.id
                        # print("now agent id",now_agent_id)
                        now_agent = self.agents[now_agent_id]
                        
                        
                        
                        # 現在の状態state, action=999, target=999をtemp_dataに追加する
                        
                        temp_datas = copy.deepcopy(self.agents_action_sequence[now_agent_id])
                        
                        data = (state, 999, 999)
                        temp_datas.append(data)
                        
                        
                        # action選択
                        # seqs = make_sequence(temp_datas, now_agent_id)
                        seqs = [temp_datas]
                        inputs = pre_process(seqs)
                        # print("inputs_seq size", inputs.size())
                        action, action_prob = now_agent.select_action(inputs, episode)
                        
                        
                        # ##############################################
                        # updateで使用するタプルデータ tmp: (入力データ, action)
                        # tmp -> memory_actionに追加
                        # ##############################################
                        tmp = (copy.copy(temp_datas), action)
                        now_agent.memory_action.append(tmp)
                        
                        temp_datas.clear() # 使用したデータのクリア
                        
                        
                        # ターゲット選択
                        if Character.Attribute_Actions[action] == "protect":
                            target_id = -1
                            target_prob = 1.0
                        elif Character.Attribute_Actions[action] == "attacking" or Character.Attribute_Actions[action]== "debufsupport":
                            if action == 2 or action == 6 or action == 19 or action == 21:
                                target_id = 1
                                target_prob = 1.0
                            else:
                                temp_datas = copy.deepcopy(self.agents_action_sequence[now_agent_id])
                                # data = (env.now_character.SIDE, env.now_character.id, env.State(), action, 999, None, None, None)
                                data = (state, action, 999) # target idの埋め込み
                                temp_datas.append(data)
                                # data to seq and nn input
                                seqs = [temp_datas]
                                inputs = pre_process(seqs)
                                target_id, target_prob = now_agent.select_target2(inputs, episode)
                                
                                
                                # #####################################################
                                # updateで使用するタプルデータ tmp: (入力データ, target_id)
                                # tmp -> memory_target2に追加
                                # #####################################################
                                tmp = (copy.copy(temp_datas), target_id)
                                now_agent.memory_target2.append(tmp)
                                temp_datas.clear()
                                
                        elif Character.Attribute_Actions[action] == "healing" or Character.Attribute_Actions[action] == "bufsupport":
                            if action == 9 or action == 10 or action == 18 or action == 22:
                                target_id = 0
                                target_prob = 1.0
                            else:
                                temp_datas = copy.deepcopy(self.agents_action_sequence[now_agent_id])
                                data = (state, action, 999)
                                temp_datas.append(data)
                                # data to seq and nn input
                                
                                seqs = [temp_datas]
                                inputs = pre_process(seqs)
                                target_id, target_prob = now_agent.select_target1(inputs, episode)
                                
                                # ##############################################
                                # updateで使用するタプルデータ tmp: (入力データ, target_id)
                                # tmp -> memory_target1に追加
                                # ##############################################
                                tmp = (copy.copy(temp_datas), target_id)
                                now_agent.memory_target1.append(tmp)
                                temp_datas.clear()
                        
                        
                    
                    
                        # 行動したエージェントのinputs_sequneceをclearする
                        self.agents_action_sequence[now_agent_id].clear()
                        
                        
                        # データの追加 --- 全てのエージェントの行動シーケンスメモリにs, a, tを追加する
                        for key, _ in self.agents.items():
                            # data = (env.now_character.SIDE, env.now_character.id, env.State(), action, target_id, None, None, None)
                            data = (state, action, target_id)
                            self.agents_action_sequence[key].append(data)
                        
                        # action, target_idが確定
                        action_flow = (action, target_id)
                        next_state, reward, done, action, target_id = env.step(action_flow)

                        # GAIL update用 memoryにdataを追加
                        self.memory.append((now_agent_id, state, action, target_id, action_prob, target_prob, reward))
                
                        
                        
                elif env.game_state is GameState.TURN_END:
                    action_flow = None
                    next_state, reward, done = env.step(action_flow)
                elif env.game_state is GameState.GAME_END:
                    action_flow = None
                    next_state, reward, done, win = env.step(action_flow)

                    if win:
                        n_win += 1
                        n_wins[self.situation] += 1
                        clear_turn.append(env.turn)
                    else:
                        clear_turn.append(0) #負けたら0
                    
                
                    self.memory.append("GAME_END")
            
            
            # 1episdoe終了後
            # plot_clear_turn(clear_turn, episode) # to use -> import wandb
            
            # self.update_interval毎にupdateを呼ぶ
            if episode % self.update_interval == 0:
                for key, _ in self.agents.items():
                    out_disc_generator, out_disc_expert, rewards, loss_agent_update, loss_disc_update = self.update(key)
                    
                    # plot_loss(key, out_disc_generator, out_disc_expert) # to use -> import wandb
                    # plot_reward(key, rewards) # to use -> import wandb
                    # plot_loss_agent_update(key, loss_agent_update) # to use -> import wandb
                    # plot_loss_disc_update(loss_disc_update) # to use -> import wandb
                
                # gailの持つ生成データを保持していたself.memoryをclearする
                self.memory.clear()
            
            if episode % 100 == 0:
                # 過去100epiosdeの平均クリアターンを取得
                n = 0 # 過去100episodeの平均を取るためのn
                for t in clear_turn[episode-100:episode]:
                    if t > 0:
                        n += 1
                if n == 0:
                    ave_clear_turn = None
                else:
                    ave_clear_turn = sum(clear_turn[episode-100:episode])/n
                print("Episode:", episode, "situation:",self.situation, "n_win:", n_win, "average of clear_turn:", ave_clear_turn)

            
        
        # 勝率
        print("rate of win: {}/{}".format(n_win, episodes))
        print("n_wins", n_wins)
        # 平均クリアターン
        if len(clear_turn) == 0:
            print("num of win: 0")
        else:
            print("average of clear turn: {}".format(sum(clear_turn)/n_win))

        
        # パラメータ保存
        # dir="/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/GAIL_learned_model/"
        dir=configuration.gail_learned_path
        print("save model parameters in ", dir)
        for key, value in self.agents.items():
            torch.save(self.agents[key].pi_action.state_dict(), dir+'agent_id={}_action.pth'.format(key))
            torch.save(self.agents[key].pi_target1.state_dict(), dir+'agent_id={}_target1.pth'.format(key))
            torch.save(self.agents[key].pi_target2.state_dict(), dir+'agent_id={}_target2.pth'.format(key))
        

        print("save successfully")
        
        # configの情報を保存
        # テキストファイルに書き込むファイルパス
        file_path = dir+'config.txt'
        # テキストファイルに書き込み
        with open(file_path, 'w') as file:
            for key, value in self.config.items():
                file.write(f"{key}: {value}\n")
        
    
    
if __name__ == "__main__":
    gail = GAIL()
    gail.learning()