# ライブラリimport
# パスの情報を管理するConfig.pyを自身のディレクトリに変更してimportしてください
from Config import Config
configuration = Config() # configを設定

import numpy as np
# np.random.seed(123)
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(configuration.parent_dir)  # 親ディレクトリを追加

from env.Character import Character
from function import pre_process
from SequenceData import SequenceData



# output -> action, target_id
class PolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PolicyNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
       
        
    
    def forward(self, x):
        # x = torch.stack(x, dim=0)
        x = x.to(torch.float)
        out, _ = self.lstm(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :])
        return out


# 強化学習エージェント
class RLAgent:
    def __init__(self, env, agent_id, lrs, argmax_episode, argmax_interval, exp_data_path):
        self.env = env
        self.agent_id = agent_id
        # lr
        self.lr_action = lrs[0]
        self.lr_target1 = lrs[1]
        self.lr_target2 = lrs[2]
        
        self.gamma = 0.95
        self.batch_size = 128
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is", self.device)
        self.ACTION_SET = self.env.id_to_charaObj(agent_id, env.characters).ACTION_SET
        # action mask
        self.ACTION_MASK = {}
        i = 0
        for key in self.ACTION_SET:
            self.ACTION_MASK[i] = key
            i += 1
        print(self.ACTION_MASK)
        
        # target mask
        self.target1_id_idx = [] # 味方id
        self.target1_id_mask = []
        self.target2_id_idx = [] # 敵id
        self.target2_id_mask = [] # 現在の環境にいる敵のみ1が立つ
        self.init_target_mask()
        
        
        # network input
        self.state_dim = len(self.env.State())
        self.action_dim = len(self.ACTION_SET)
        self.input_pi_target1_dim = self.state_dim + self.action_dim
        self.input_pi_target2_dim = self.state_dim + self.action_dim
        
        
        # network output
        self.target1_dim = len(self.target1_id_idx)
        self.target2_dim = len(self.target2_id_idx)
        
        # memory update用のmemory
        self.memory_action = []
        self.memory_target1 = []
        self.memory_target2 = []
        
        
        
        self.input_dim = len(self.env.State()) + len(Character.ACTIONS) + 3 + len(self.target1_id_idx) + len(self.target2_id_idx)
        print("LSTM: input_dim", self.input_dim)
        self.hidden_dim = 128
        self.lstm_layer = 1
        
        # 方策
        # argmaxを取るインターバル
        self.argmax_interval = argmax_interval
        self.argmax_episode = argmax_episode
        # action policy
        self.pi_action = PolicyNet(self.input_dim, self.hidden_dim, self.lstm_layer, self.action_dim)
        self.pi_action.to(self.device)
        self.optim_a = torch.optim.Adam(self.pi_action.parameters(), lr=self.lr_action)
        self.cnt_action = 0
        
        # target1 policy
        self.pi_target1 = PolicyNet(self.input_dim, self.hidden_dim, self.lstm_layer, self.target1_dim)
        self.pi_target1.to(self.device)
        self.optim_1 = torch.optim.Adam(self.pi_target1.parameters(), lr=self.lr_target1)
        self.cnt_target1 = 0
        
        # target2 policy
        self.pi_target2 = PolicyNet(self.input_dim, self.hidden_dim, self.lstm_layer, self.target2_dim)
        self.pi_target2.to(self.device)
        self.optim_2 = torch.optim.Adam(self.pi_target2.parameters(), lr=self.lr_target2)
        self.cnt_target2 = 0
        
        self.exp_batch_size = 64 # pre train用のbatch size
        self.buffer_exp = SequenceData(40000, self.exp_batch_size, exp_data_path, self.agent_id, self.env)
        
        
        
    def init_target_mask(self):
        self.target1_id_idx = [] # 味方id
        self.target1_id_mask = []
        self.target2_id_idx = [] # 敵id
        self.target2_id_mask = [] # 現在の環境にいる敵のみ1が立つ
        for c in self.env.characters:
            if c.SIDE == 0:
                self.target1_id_idx.append(c.id)
                self.target1_id_mask.append(1)
        # print("target1 id idx: ", self.target1_id_idx)
        # print("target1 id mask: ", self.target1_id_mask)
        
        
        # print("env.enemy_flag: ", env.enemy_flag)
        # print("env.enemy_flag.type(): ", type(env.enemy_flag))
        for key, value in self.env.enemy_flag.items():
            self.target2_id_idx.append(key)
            self.target2_id_mask.append(value)
        # print("target2 id idx: ", self.target2_id_idx)
        # print("target2 id mask: ", self.target2_id_mask)
    
    
    # action id -> idx of ActionSet
    def action_to_idx(self, action):
        i = 0
        for key in self.ACTION_SET:
            if action == key:
                return i
            i += 1
    
    def action_to_idx_batch(self, actions):
        result = []
        for action in actions:
            idx = self.action_to_idx(action)
            result.append(idx)
        return result
    
    def action_to_one_hot_encoding(self, action):
        one_hot = [0]*self.action_dim
        if action == 999:
            return one_hot
        else:
            i = 0
            for a, _ in self.ACTION_SET.items():
                if action == a:
                    one_hot[i] = 1
                    return one_hot
                i += 1
            return None
    
    def action_to_one_hot_encoding_batch(self, actions):
        result = []
        for action in actions:
            one_hot = self.action_to_one_hot_encoding(action)
            result.append(one_hot)
        return result
    
    
    def target_id_to_one_hot_encoding(self, target_id, target_id_idx):
        one_hot = [0]*len(target_id_idx)
        if target_id == 999:
            return one_hot
        else:
            i = 0
            for id in target_id_idx:
                if target_id == id:
                    one_hot[i] = 1
                    return one_hot
                i += 1
            return None
    
    def target_id_to_one_hot_encoding_batch(self, target_ids, target_id_idx):
        result = []
        for target_id in target_ids:
            one_hot = self.target_id_to_one_hot_encoding(target_id, target_id_idx)
            result.append(one_hot)
        return result
    
    # target_id_idx -> id
    def idx_to_id(self, idx, target_id_idx):
        return target_id_idx[idx]
    
    def id_to_idx(self, id, target_id_idx):
        for i in range(len(target_id_idx)):
            if id == target_id_idx[i]:
                return i
        return None
    
    def id_to_idx_batch(self, target_ids, target_id_idx):
        result = []
        for id in target_ids:
            idx = self.id_to_idx(id, target_id_idx)
            result.append(idx)
        return result
    
    # mask update --- (deadChara -> 0)
    # def update_mask(self, target_id_idx, target_id_mask):
    #     for id, m in zip(target_id_idx, target_id_mask):
    #         for c in self.env.dead_characters:
    #             if c.id == id and m == 1:
    #                 m = 0
    
    def update_target1_id_mask(self):
        for i in range(len(self.target1_id_mask)):
            id = self.target1_id_idx[i]
            m = self.target1_id_mask[i]
            # print("update mask1 id", id)
            c = self.env.id_to_charaObj(id, self.env.characters)
            if c is not None:
                if c.HP <= 0 and m == 1:
                    self.target1_id_mask[i] = 0
    
    def update_target2_id_mask(self):
        for i in range(len(self.target2_id_mask)):
            id = self.target2_id_idx[i]
            m = self.target2_id_mask[i]
            # print("before update mask2 (id, m)", id, m)
            c = self.env.id_to_charaObj(id, self.env.characters)
            if c is not None:
                if c.HP <= 0 and m == 1:
                    self.target2_id_mask[i] = 0
            # m = self.target2_id_mask[i]
            # print("after update mask2 (id, m)", id, m)
    
    # use mask form env to select target
    def masked_probs(self, probs, mask):
        masked1 = []
        for m, p in zip(mask, probs):
            masked1.append(m*p)
        masked2 = []
        for m in masked1:
            if m == 0:
                masked2.append(0.0)
            else:
                masked2.append(m + m/sum(masked1) * (1-sum(masked1)))
        return masked2

    
    # 0 -> -infにしてsoftmaxをかける
    def masked_probs_ver2(self, probs, mask):
        masked_1 = []
        for m, p in zip(mask, probs):
            if m == 0:
                masked_1.append(float('-inf'))
            else:
                masked_1.append(p)
        masked_1 = torch.tensor(masked_1)
        masked_probs = F.softmax(masked_1)
        
        return masked_probs
    
    # select action -> PolicyGradient method
    def select_action(self, inputs, episode):
        self.cnt_action += 1
        # to gpu
        inputs = inputs.to(self.device)
        self.pi_action.to(self.device)
        
        output = self.pi_action(inputs)
        # print("size of output", output.size())
        output = output.view(-1)
        action_probs = F.softmax(output)
        
        action_probs = action_probs.cpu()
        temp_probs = action_probs.clone().detach().numpy()
        temp_probs /= sum(temp_probs)
            
        # if episode % 100 == 0:
        #     print("action_probs", action_probs)
        # print("agent id", self.agent_id)
        # print("len of action", len(self.ACTION_SET))
        # print("action_probs", action_probs)
        
        # action_probs = action_probs.flatten()
        
        # print("action_probs.flatten()", action_probs)
        # argmaxを取らずに常に確率的に選択する
        if self.argmax_episode == 0 or self.argmax_interval == 0:
            selected_action_idx = np.random.choice(len(temp_probs), p=temp_probs)
            action_prob = action_probs[selected_action_idx]
        else:
            if episode % self.argmax_episode == 0 or self.cnt_action % self.argmax_interval == 0:
                selected_action_idx = np.argmax(temp_probs)
                action_prob = action_probs[selected_action_idx]
                self.cnt_action = 0
            else:
                selected_action_idx = np.random.choice(len(temp_probs), p=temp_probs)
                action_prob = action_probs[selected_action_idx]
        
        selected_action = self.ACTION_MASK[selected_action_idx]
        
        return selected_action, action_prob

    
    
    def select_target1(self, inputs, episode):
        self.cnt_target1 += 1
        # maskの更新
        self.update_target1_id_mask()
        
        # to gpu
        self.pi_target1.to(self.device)
        inputs = inputs.to(self.device)
        output = self.pi_target1(inputs)
        output = output.view(-1)
        target1_probs = F.softmax(output)
        
        # to cpu and detach numpy
        target1_probs = target1_probs.cpu()
        
        # masked_probs = self.masked_probs(target1_probs, self.target1_id_mask)
        masked_probs = self.masked_probs_ver2(target1_probs, self.target1_id_mask)
        temp_probs = masked_probs.clone().detach().numpy()
        temp_probs = temp_probs/sum(temp_probs)
        
        # argmaxを取らずに常に確率的に選択する
        if self.argmax_episode == 0 or self.argmax_interval == 0:
            # print("target1 net output:\n", output)
            # print("unmasked target1 probs:\n", target1_probs)
            # print("target1 mask:\n", self.target1_id_mask)
            # print("target1 masked probs:\n",temp_probs)
            selected_target1_idx = np.random.choice(len(temp_probs), p=temp_probs)
            target1_prob = target1_probs[selected_target1_idx]
        else:
            if episode % self.argmax_episode == 0 or self.cnt_target1 % self.argmax_interval == 0:
                selected_target1_idx = np.argmax(temp_probs)
                target1_prob = target1_probs[selected_target1_idx]
                self.cnt_target1 = 0
            else:
                selected_target1_idx = np.random.choice(len(temp_probs), p=temp_probs)
                target1_prob = target1_probs[selected_target1_idx]
        
        selected_target1_id = self.idx_to_id(selected_target1_idx, self.target1_id_idx)
        return selected_target1_id, target1_prob
    
    def select_target2(self, inputs, episode):
        self.cnt_target2 += 1
        # maskの更新
        self.update_target2_id_mask()
        
        
        # to gpu
        self.pi_target2.to(self.device)
        inputs = inputs.to(self.device)
        output = self.pi_target2(inputs)
        output = output.view(-1)
        target2_probs = F.softmax(output)
        
        
        
        # to cpu and detach numpy
        # target2_probs = target2_probs.cpu().detach().numpy().flatten()
        target2_probs = target2_probs.cpu()
        
        # masked_probs = self.masked_probs(target2_probs, self.target2_id_mask)
        masked_probs = self.masked_probs_ver2(target2_probs, self.target2_id_mask)
        # print("maked_probs", masked_probs)
        
        temp_probs = masked_probs.clone().detach().numpy()
        temp_probs = temp_probs/sum(temp_probs)
        # print("episode", episode)
        # print("target2 id mask", self.target2_id_mask)
        # print("temp_probs in target2", temp_probs)
        
        
        
        # argmaxを取らずに常に確率的に選択する
        if self.argmax_episode == 0 or self.argmax_interval == 0:
            # print("target2 net output:\n", output)
            # print("unmasked target2 probs:\n", target2_probs)
            # print("target2 mask:\n", self.target2_id_mask)
            # print("target2 masked probs:\n",temp_probs)
            selected_target2_idx = np.random.choice(len(temp_probs), p=temp_probs)
            target2_prob = target2_probs[selected_target2_idx]
        else:
            if episode % self.argmax_episode == 0 or self.cnt_target2 % self.argmax_interval == 0:
                selected_target2_idx = np.argmax(temp_probs)
                target2_prob = target2_probs[selected_target2_idx]
                self.cnt_target2 = 0
            else:
                selected_target2_idx = np.random.choice(len(temp_probs), p=temp_probs)
                target2_prob = target2_probs[selected_target2_idx]
        
        selected_target2_id = self.idx_to_id(selected_target2_idx, self.target2_id_idx)
        
        return selected_target2_id, target2_prob

    # input_data, action in memory
    def pi_action_update(self, rewards):
        # self.memory_actionからinputs_seqsとactionsを取り出す
        inputs_datas = [item[0] for item in self.memory_action]
        actions = [item[1] for item in self.memory_action]
        # print("inputs_data:\n",inputs_datas)
        # print("actions:\n", actions)
        
        # lstm inputs
        inputs = pre_process(inputs_datas)
        inputs = inputs.to(self.device)
        
        
        actions_idxs = self.action_to_idx_batch(actions) # inputsに対して選択したactionのインデックス
        probs = F.softmax(self.pi_action(inputs)).cpu() # nn output probs
        # print("probs:\n", probs)
        
        selected_probs = torch.stack([row[i] for row, i in zip(probs, actions_idxs)])
        # print("selected_probs:\n", selected_probs, type(selected_probs), selected_probs.size())
        
        
        # 方策勾配法によるlossの計算
        G = 0.0
        loss = torch.tensor(0.0, requires_grad=True)
        for reward, prob in reversed(list(zip(rewards, selected_probs))):
            G = reward + self.gamma*G
            loss = loss - torch.log(prob)*torch.tensor(G)
    
        # nn paramters 確認
        
        # breakpoint()
        
        self.optim_a.zero_grad()
        loss.backward()
        self.optim_a.step()
        
        # nn paramters 確認
        # breakpoint()
        
        # updateに使用したmemoryをクリアする
        self.memory_action.clear()
        
        return loss.mean()
    
    
    
    
    def pi_target1_update(self, rewards):
        inputs_datas = [item[0] for item in self.memory_target1]
        target_ids = [item[1] for item in self.memory_target1]
        inputs = pre_process(inputs_datas)
        inputs = inputs.to(self.device)
        
        probs = F.softmax(self.pi_target1(inputs)).cpu()
        
        target_idxs = self.id_to_idx_batch(target_ids, self.target1_id_idx)
        
        
        
        G = 0
        loss = torch.tensor(0.0, requires_grad=True)
        for reward, prob, target_idx in zip(rewards, probs, target_idxs):
            p = prob[target_idx]
            G = reward + self.gamma*G
            loss = loss - torch.log(p)*torch.tensor(G)
        
        # breakpoint()
        
        self.optim_1.zero_grad()
        loss.backward()
        self.optim_1.step()
        
        # breakpoint()
        
        self.memory_target1.clear()
        
        return loss.mean()


    
    def pi_target2_update(self, rewards):
        inputs_datas = [item[0] for item in self.memory_target2]
        target_ids = [item[1] for item in self.memory_target2]
        # print("target2 inputs_datas:\n", inputs_datas)
        # print("target_ids:\n", target_ids)
        
        # lstm inputs
        inputs = pre_process(inputs_datas)
        inputs = inputs.to(self.device)
        
        target_idxs = self.id_to_idx_batch(target_ids, self.target2_id_idx)
        # print("target2_idxs:\n", target_idxs)
        
        
        probs = F.softmax(self.pi_target2(inputs)).cpu()
        # print("probs:\n", probs, probs.requires_grad)

        
        G = 0
        loss = torch.tensor(0.0, requires_grad=True)
        for reward, prob, target_idx in zip(rewards, probs, target_idxs):
            p = prob[target_idx]
            G = reward + self.gamma*G
            loss = loss - torch.log(p)*torch.tensor(G)
        # print("update target2")
        
        # breakpoint()
        
        self.optim_2.zero_grad()
        loss.backward()
        self.optim_2.step()
        
        # breakpoint()
        
        self.memory_target2.clear()
        
        return loss.mean()
        
    
    # GAILで学習させる前に事前学習を行う (Behavior Cloning)
    def pre_training(self):
        loss_fn = nn.CrossEntropyLoss()
        # ###########
        # action
        # ###########
        mini_batch_a = self.buffer_exp.get_batch_action()
        if len(mini_batch_a) > 0:
            sequences_action = [item[0] for item in mini_batch_a]
            action_answers = [item[1] for item in mini_batch_a]
            
            inputs = pre_process(sequences_action).to(self.device)
            action_batch = self.action_to_one_hot_encoding_batch(action_answers)
            action_batch_tensor = torch.tensor(action_batch, dtype=torch.float).to(self.device)
            
            loss_action = loss_fn(F.softmax(self.pi_action(inputs)), action_batch_tensor)
            
            self.optim_a.zero_grad()
            loss_action.backward()
            self.optim_a.step()
            
            loss_action = loss_action.cpu()
        else:
            loss_action = 0.0
        
        # ############
        # target1 
        # ############
        mini_batch_1 = self.buffer_exp.get_batch_target1()
        if len(mini_batch_1) > 0:
            # 正解データ
            sequences_target1 = [item[0] for item in mini_batch_1]
            target1_answers = [item[1] for item in mini_batch_1]
            
            inputs = pre_process(sequences_target1).to(self.device)
            target1_batch = self.target_id_to_one_hot_encoding_batch(target1_answers, self.target1_id_idx)
            target1_batch_tensor = torch.tensor(target1_batch, dtype=torch.float).to(self.device)
            
            loss_target1 = loss_fn(F.softmax(self.pi_target1(inputs)), target1_batch_tensor)
            
            
            self.optim_1.zero_grad()
            loss_target1.backward()
            self.optim_1.step()
            
            loss_target1 = loss_target1.cpu()
        else:
            loss_target1 = 0.0
        
        # #############
        # target2 
        # ##############
        mini_batch_2 = self.buffer_exp.get_batch_target2()
        if len(mini_batch_2) > 0:
            # 正解データ
            sequences_target2 = [item[0] for item in mini_batch_2]
            target2_answers = [item[1] for item in mini_batch_2]
            
            inputs = pre_process(sequences_target2).to(self.device)
            target2_batch = self.target_id_to_one_hot_encoding_batch(target2_answers, self.target2_id_idx)
            target2_batch_tensor = torch.tensor(target2_batch, dtype=torch.float).to(self.device)
            
            loss_target2 = loss_fn(F.softmax(self.pi_target2(inputs)), target2_batch_tensor)
            
            self.optim_2.zero_grad()
            loss_target2.backward()
            self.optim_2.step()
            
            loss_target2 = loss_target2.cpu()
        else:
            loss_target2 = 0.0
        
        # lossまとめる
        loss = (loss_action, loss_target1, loss_target2)
        return loss
    
    # 事前学習の学習結果の検証
    def eval_pre_train(self):
        loss_fn = nn.CrossEntropyLoss()
        # ############
        # action
        # ############
        test_data = self.buffer_exp.batch_datas_action_test
        if len(test_data) > 0:
            with torch.no_grad():
                seq = [item[0] for item in test_data]
                answer = [item[1] for item in test_data]
                
                inputs = pre_process(seq).to(self.device)
                answer_batch_data = self.action_to_one_hot_encoding_batch(answer)
                answer_batch_data_tensor = torch.tensor(answer_batch_data, dtype=torch.float).to(self.device)
                
                test_loss_action = loss_fn(F.softmax(self.pi_action(inputs)), answer_batch_data_tensor).mean()
        else:
            test_loss_action = 0.0
        
        # ###########
        # target1
        # ###########
        test_data = self.buffer_exp.batch_datas_target1_test
        if len(test_data) > 0:
            with torch.no_grad():
                seq = [item[0] for item in test_data]
                answer = [item[1] for item in test_data]
                
                inputs = pre_process(seq).to(self.device)
                answer_batch_data = self.target_id_to_one_hot_encoding_batch(answer, self.target1_id_idx)
                answer_batch_data_tensor = torch.tensor(answer_batch_data, dtype=torch.float).to(self.device)
                
                test_loss_target1 = loss_fn(F.softmax(self.pi_target1(inputs)), answer_batch_data_tensor).mean()
        else:
            test_loss_target1 = 0.0

        # ##########
        # target2 
        # ##########
        test_data = self.buffer_exp.batch_datas_target2_test
        if len(test_data) > 0:
            with torch.no_grad():
                seq = [item[0] for item in test_data]
                answer = [item[1] for item in test_data]
                
                inputs = pre_process(seq).to(self.device)
                answer_batch_data = self.target_id_to_one_hot_encoding_batch(answer, self.target2_id_idx)
                answer_batch_data_tensor = torch.tensor(answer_batch_data, dtype=torch.float).to(self.device)
                
                test_loss_target2 = loss_fn(F.softmax(self.pi_target2(inputs)), answer_batch_data_tensor).mean()
        else:
            test_loss_target2 = 0.0
        
        test_loss = (test_loss_action, test_loss_target1, test_loss_target2)
        
        return test_loss
    # envのsituationを変更させる時
    # enemy_flagを更新する。
    # episode更新のたびに呼び出す。
    def update_changed_env(self, env):
        self.env = env
        self.init_target_mask()
    
    
    
    
      