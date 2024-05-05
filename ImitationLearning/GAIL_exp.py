import pickle
import random
random.seed(123)
from function import *

class GAIL_ExpData:
    def __init__(self, buffer_size, batch_size, file_path, agents_id_list):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.agents_ids = agents_id_list # [10, 20, 30, 40]
        # data読み込み
        f = open(file_path, "rb")
        self.datas = pickle.load(f)
        
        seqs = make_sequence(self.datas, self.agents_ids[0])
        self.buffer = pre_process(seqs)
        
        # expert data
        self.action_seqs = {}
        self.target1_seqs = {}
        self.target2_seqs = {}
        # data = (state, action, target)が格納されている
        for id in self.agents_ids:
            self.action_seqs[id] = make_sequence(self.datas, id)
            self.target1_seqs[id] = make_sequence_for_target1(self.datas, id)
            self.target2_seqs[id] = make_sequence_for_target2(self.datas, id)
        

    def make_label(self, seqs):
        result_seqs = []
        for seq in seqs:
            label = "expert"
            data = (seq, label)
            result_seqs.append(data)
        return result_seqs
    
    # バッチサイズ分seqsを取り出す
    def get_batch_action_seq(self, agent_id):
        if len(self.action_seqs[agent_id]) <= self.batch_size:
            mini_batch = random.sample(self.action_seqs[agent_id], len(self.action_seqs[agent_id]))
        else:
            mini_batch = random.sample(self.action_seqs[agent_id], self.batch_size)
        return mini_batch
    
    def get_batch_target1_seq(self, agent_id):
        if len(self.target1_seqs[agent_id]) <= self.batch_size:
            mini_batch = random.sample(self.target1_seqs[agent_id], len(self.target1_seqs[agent_id]))
        else:
            mini_batch = random.sample(self.target1_seqs[agent_id], self.batch_size)
        return mini_batch
    
    def get_batch_target2_seq(self, agent_id):
        if len(self.target2_seqs[agent_id]) <= self.batch_size:
            mini_batch = random.sample(self.target2_seqs[agent_id], len(self.target2_seqs[agent_id]))
        else:
            mini_batch = random.sample(self.target2_seqs[agent_id], self.batch_size)
        return mini_batch