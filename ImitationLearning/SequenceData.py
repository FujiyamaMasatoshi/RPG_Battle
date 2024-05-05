from collections import deque
import random
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from env.Character import Character


from function import *

class SequenceData:
    def __init__(self, buffer_size, batch_size, file_path, agent_id, env):
        # datas -> fileよりデータを読み込む
        f = open(file_path, "rb")
        self.datas = pickle.load(f)
        
        # batch size
        self.batch_size = batch_size
        # buffer size
        self.buffer_size = buffer_size
        # agent id
        self.agent_id = agent_id
        
        
        # ##############
        # action tensor
        # ##############
        self.ACTION_TENSOR = torch.tensor([0]*len(Character.ACTIONS))
        
        # ##############
        # target tensor
        # ##############
        keys_to_zero = env.characters_flag.keys()
        a = {-1: 0, 0: 0, 1: 0}
        # env.characters_flagをコピーして新たな辞書を作成し、キーに対応する値を0に設定
        result_dict = dict(env.characters_flag)
        for key in keys_to_zero:
            result_dict[key] = 0
        # aとresult_dictを連結させる
        self.TARGET_TENSOR_dict = {**a, **result_dict}
        # print("TARGET_TENSOR_dict:", TARGET_TENSOR_dict)
        
        # シーケンスデータ
        seqs = self.make_sequence(self.datas)
        self.buffer = self.pre_process(seqs)
        # print("len of buffer", len(self.buffer))
        self.seqs_for_action = self.make_sequence(self.datas)
        self.seqs_for_target1 = self.make_sequence_for_target1(self.datas)
        self.seqs_for_target2 = self.make_sequence_for_target2(self.datas)
        
        # バッチ用データ
        self.batch_datas_action = self.make_batch_action()
        self.batch_datas_target1 = self.make_batch_target1()
        self.batch_datas_target2 = self.make_batch_target2()
        
        self.test_size = 0.2
        
        # テストデータ
        # action
        if len(self.seqs_for_action) > 0:
            self.batch_datas_action_train, self.batch_datas_action_test = train_test_split(self.batch_datas_action, test_size=self.test_size, random_state=123)
        else:
            self.batch_datas_action_train, self.batch_datas_action_test = [], []
        # target1
        if len(self.seqs_for_target1) > 0:
            self.batch_datas_target1_train, self.batch_datas_target1_test = train_test_split(self.batch_datas_target1, test_size=self.test_size, random_state=123)
        else:
            self.batch_datas_target1_train, self.batch_datas_target1_test = [], []
        # target2
        if len(self.seqs_for_target2) > 0:
            self.batch_datas_target2_train, self.batch_datas_target2_test = train_test_split(self.batch_datas_target2, test_size=self.test_size, random_state=123)
        else:
            self.batch_datas_target2_train, self.batch_datas_target2_test = [], []
        
        
    # シーケンスデータを作成して返す 
    # data = (side, agent_id, state, acion, target, ...) -> (state, action, target)
    # のみを取得し，sequencesを作成
    def make_sequence(self, datas):
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
                data_id = data[1]
                if found:
                    if data_id == self.agent_id:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        
                        
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                else:
                    if data_id == self.agent_id:
                        found = True
                        if not seq == []:
                            seqs.append(seq)
                            seq = []
                        else:
                            s = data[2]
                            action = data[3]
                            target_id = data[4]
                            seq.append((s, action, target_id)) # (s, a, t)全て
        return seqs
    
    
    # 味方選択用のseqsを作成
    def make_sequence_for_target1(self, datas):
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
                data_id = data[1]
                if found:
                    if data_id == self.agent_id:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        
                        # seqの最後のデータ
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        # target1を選択しているなら、seqsに追加する
                        # そうでなければ、追加しない
                        if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                            seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                else:
                    if data_id == self.agent_id:
                        found = True
                        if not seq == []:
                            last_seq = seq[-1]
                            state = last_seq[0]
                            action = last_seq[1]
                            target_id = last_seq[2]
                            if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                                seqs.append(seq)
                            seq = []
                        else:
                            s = data[2]
                            action = data[3]
                            target_id = data[4]
                            seq.append((s, action, target_id)) # (s, a, t)全て
                            # seq.append((data_id, s, action, target_id))
        return seqs

    # 敵選択用のseqsを作成
    def make_sequence_for_target2(self, datas):
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
                data_id = data[1]
                if found:
                    if data_id == self.agent_id:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        
                        # seqの最後のデータ
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        # target2を選択しているなら、seqsに追加する
                        # そうでなければ、追加しない
                        if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                            seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id)) # (s, a, t)全て
                        # seq.append((data_id, s, action, target_id))
                else:
                    if data_id == self.agent_id:
                        found = True
                        if not seq == []:
                            last_seq = seq[-1]
                            state = last_seq[0]
                            action = last_seq[1]
                            target_id = last_seq[2]
                            if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                                seqs.append(seq)
                            seq = []
                        else:
                            s = data[2]
                            action = data[3]
                            target_id = data[4]
                            seq.append((s, action, target_id)) # (s, a, t)全て
                            # seq.append((data_id, s, action, target_id))
        return seqs

        
        
    def pre_process(self, seqs):
        # sequencesにエージェントの行動決定までの(s, a)シーケンスが保持される
        # sequences = self.make_sequence(datas)
        sequences = seqs
        # print("len of sequences", len(sequences))
        # padding
        if len(sequences) > 0:
            pre_process_sequences = deque(maxlen=self.buffer_size)
            for seq in sequences:
                pre_process_seq = []
                for s in seq:
                    # #########
                    # state
                    # #########
                    s_state = torch.tensor(s[0])
                    
                    # ############################
                    # s_action -> action tensor
                    # ############################
                    s_action = s[1] # (s, a, t)全て
                    # s_action = s[0] # (a, t)のみ
                    action_tensor_temp = copy.deepcopy(self.ACTION_TENSOR)
                    if s_action != 999:
                        action_tensor_temp[int(s_action)] = 1
                    s_action_tensor = torch.tensor(action_tensor_temp)
                    # print("action tensor",s_action_tensor)
                    
                    # ##########################
                    # s_target -> target tensor
                    # ###########################
                    s_target = s[2] # (s, a, t)全て
                    # s_target = s[1] # (a, t)のみ
                    target_tensor_dict_temp = copy.deepcopy(self.TARGET_TENSOR_dict)
                    if s_target != 999:
                        target_tensor_dict_temp[int(s_target)] = 1
                    s_target_tensor = torch.tensor(list(target_tensor_dict_temp.values()))
                    
                    # ####################
                    # cat merge
                    # #####################
                    s_cat_tensor = torch.tensor(torch.cat((s_state, s_action_tensor, s_target_tensor))) #(s, a, t)全て
                    # s_cat_tensor = torch.tensor(torch.cat((s_action_tensor, s_target_tensor))) # (a, t)のみ
                    pre_process_seq.append(s_cat_tensor)
                pre_process_sequences.append(pre_process_seq)
            # print("len of preprocessSeqs",len(pre_process_sequences))
            padded_sequences = pad_sequence([torch.stack(seq) for seq in pre_process_sequences], batch_first=True, padding_value=0)
            # print(padded_sequences)
            # print(padded_sequences.size())
            return padded_sequences
        else:
            return []
        
        


   
    
    def get_batch_datas(self):
        if len(self.datas) < self.batch_size:
            mini_batch = random.sample(self.datas, len(self.datas))
        else:
            mini_batch = random.sample(self.datas, self.batch_size)
        if len(self.datas) == 0:
            return None
        else:
            return mini_batch

    def get_batch_seq_and_label(self):
        sequences = self.make_sequence(self.datas)
        label = [[1, 0]]*len(sequences) # [answer, generator]
        
        # merge answer_action_list and sequence
        seqs_and_label = []
        for s, l in zip(sequences, label):
            seqs_and_label.append((s, l))
        
        # ミニバッチを取り出す
        if len(seqs_and_label) <= self.batch_size:
            mini_batch = random.sample(seqs_and_label, len(seqs_and_label))
        else:
            mini_batch = random.sample(seqs_and_label, self.batch_size)
        return mini_batch
    
    def make_batch_action(self):
        # self.agent_idエージェントのsequecneデータを作る
        sequences = self.seqs_for_action
        answer_action_list = []
        # sequenceデータから正解ラベルとなるアクションnumを取り出す
        
        # 加工用のsequences -> padded_sequences
        padded_sequences = []
        
        for seq in sequences:
            pad_seq = []
            for i in range(len(seq)):
                if i == len(seq)-1:
                    s = seq[i]
                    answer_action = s[1] # (s, a, t)全て
                    answer_action_list.append(answer_action)
                    
                    # pad_seqの処理
                    state = s[0]
                    action = 999
                    target_id = 999
                    pad_seq.append((state, action, target_id))
                else:
                    pad_seq.append(copy.copy(seq[i]))
            padded_sequences.append(pad_seq)
        
        # merge answer_action_list and sequence
        answer_action_and_seqs = []
        for s, a in zip(padded_sequences, answer_action_list):
            answer_action_and_seqs.append((s, a))
        
        return answer_action_and_seqs

    def get_batch_action(self):
        if len(self.batch_datas_action_train) <= self.batch_size:
            mini_batch = random.sample(self.batch_datas_action_train, len(self.batch_datas_action_train))
        else:
            mini_batch = random.sample(self.batch_datas_action_train, self.batch_size)
        return mini_batch
    
    def make_batch_target1(self):
        sequences = self.seqs_for_target1
        answer_target_list = []
        # sequenceデータから正解ラベルとなるアクションnumを取り出す
        
        # 加工用のsequences -> padded_sequences
        padded_sequences = []
        
        for seq in sequences:
            pad_seq = []
            for i in range(len(seq)):
                if i == len(seq)-1:
                    s = seq[i]
                    answer_target = s[2] # (s, a, t)全て
                    answer_target_list.append(answer_target)
                    
                    # pad_seqの処理
                    state = s[0]
                    action = s[1]
                    target_id = 999
                    pad_seq.append((state, action, target_id))
                else:
                    pad_seq.append(copy.copy(seq[i]))
            padded_sequences.append(pad_seq)
        
        # merge answer_action_list and sequence
        answer_target_and_seqs = []
        for s, a in zip(padded_sequences, answer_target_list):
            answer_target_and_seqs.append((s, a))
        
        return answer_target_and_seqs
    
    def get_batch_target1(self):
        if len(self.batch_datas_target1_train) <= self.batch_size:
            mini_batch = random.sample(self.batch_datas_target1_train, len(self.batch_datas_target1_train))
        else:
            mini_batch = random.sample(self.batch_datas_target1_train, self.batch_size)
        
        return mini_batch
    
    def make_batch_target2(self):
        sequences = self.seqs_for_target2
        answer_target_list = []
        # sequenceデータから正解ラベルとなるアクションnumを取り出す
        
        # 加工用のsequences -> padded_sequences
        padded_sequences = []
        
        for seq in sequences:
            pad_seq = []
            for i in range(len(seq)):
                if i == len(seq)-1:
                    s = seq[i]
                    answer_target = s[2] # (s, a, t)全て
                    answer_target_list.append(answer_target)
                    
                    # pad_seqの処理
                    state = s[0]
                    action = s[1]
                    target_id = 999
                    pad_seq.append((state, action, target_id))
                else:
                    pad_seq.append(copy.copy(seq[i]))
            padded_sequences.append(pad_seq)
        
        # merge answer_action_list and sequence
        answer_target_and_seqs = []
        for s, a in zip(padded_sequences, answer_target_list):
            answer_target_and_seqs.append((s, a))
        
        return answer_target_and_seqs
    
    def get_batch_target2(self):
        if len(self.batch_datas_target2_train) <= self.batch_size:
            mini_batch = random.sample(self.batch_datas_target2_train, len(self.batch_datas_target2_train))
        else:
            mini_batch = random.sample(self.batch_datas_target2_train, self.batch_size)
        
        return mini_batch
    
