# パスの情報を管理するConfig.pyを自身のディレクトリに変更してimportしてください
from Config import Config
configuration = Config()
# ライブラリimport
import copy
import torch
from torch.nn.utils.rnn import pad_sequence


import sys
sys.path.append(configuration.parent_dir)  # 親ディレクトリを追加
from env.Game import Game
from env.Character import Character

env = Game(1) # 1, 2, 3

def ave_loss(loss_list):
    loss_action = 0
    loss_mymember = 0
    loss_enemy = 0
    cnt_action = 0
    cnt_mymember = 0
    cnt_enemy = 0
    for row in loss_list:
        if not row[0] is None:
            loss_action += row[0]
            cnt_action += 1
        if not row[1] is None:
            loss_mymember += row[1]
            cnt_mymember += 1
        if not row[2] is None:
            loss_enemy += row[2]
            cnt_enemy += 1
    if not cnt_action == 0:
        ave_loss_action = loss_action/cnt_action
    else:
        ave_loss_action = 1.0
    if not cnt_mymember == 0:
        ave_loss_mymember = loss_mymember/cnt_mymember
    else:
        ave_loss_mymember = 1.0
    if not cnt_enemy == 0:
        ave_loss_enemy = loss_enemy/cnt_enemy
    else:
        ave_loss_enemy = 1.0
    return (ave_loss_action, ave_loss_mymember, ave_loss_enemy)



def print_dict(d):
    for key, value in d.items():
        print(f"{key}: {value}")

def split_list(input_list, split_sizes):
    result = []
    index = 0
    for size in split_sizes:
        result.append(input_list[index : index + size])
        index += size
    return result

def convert_state_to_origin(state_tensor):
    state_list = state_tensor.tolist()
    # print(state_list)
    # split_sizes = [16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    # split_sizes = [16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1, 1, 23, 1, 1]
    split_sizes = [16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 1] # idはそのまま
    # split_sizes = [15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 1] # idを削除
    
    split_result = split_list(state_list, split_sizes)
    # print(split_result)
    mymembers_state = []
    enemys_state = []
    
    for i in range(len(split_result)):
        if 0 <= i < 4: # mymember_id == 10, 20, 30, 40
            # keys = ["id", "alive", "hp", "max_hp", "mp", "max_mp", "atk_state_state", "atk_state_turn", "def_state_state", "def_state_turn", "agi_state_state", "agi_state_turn", "mgcATK_state_state", "mgcATK_state_turn", "mgcREC_state_state", "mgcREC_state_turn", "registance_state_state", "registance_state_turn"]
            keys = ["id", "alive", "hp_state", "mp_state", "atk_state_state", "atk_state_turn", "def_state_state", "def_state_turn", "agi_state_state", "agi_state_turn", "mgcATK_state_state", "mgcATK_state_turn", "mgcREC_state_state", "mgcREC_state_turn", "registance_state_state", "registance_state_turn"]
            # keys = ["alive", "hp_state", "mp_state", "atk_state_state", "atk_state_turn", "def_state_state", "def_state_turn", "agi_state_state", "agi_state_turn", "mgcATK_state_state", "mgcATK_state_turn", "mgcREC_state_state", "mgcREC_state_turn", "registance_state_state", "registance_state_turn"]
            data = {key: value for key, value in zip(keys, split_result[i])}
            mymembers_state.append(data)
        elif 4 <= i < 13: # enemy_id == 1101, 1102, 1102, 1201, 1202, 1301, 1302, 1401, 1501
            keys = ["id", "alive", "hp_state", "atk_state_state", "atk_state_turn", "def_state_state", "def_state_turn", "agi_state_state", "agi_state_turn", "mgcATK_state_state", "mgcATK_state_turn", "mgcREC_state_state", "mgcREC_state_turn", "registance_state_state", "registance_state_turn"]
            # keys = ["alive", "hp_state", "atk_state_state", "atk_state_turn", "def_state_state", "def_state_turn", "agi_state_state", "agi_state_turn", "mgcATK_state_state", "mgcATK_state_turn", "mgcREC_state_state", "mgcREC_state_turn", "registance_state_state", "registance_state_turn"]
            data = {key: value for key, value in zip(keys, split_result[i])}
            enemys_state.append(data)    
        # elif i == 13: # now character id
        #     now_character_id = split_result[i][0]
        # elif i == 14: # now character action set
            now_character_action_set = split_result[i]
        # elif i == 15: # damages_enemy
        #     damages_enemy = split_result[i][0]
        # elif i == 16: # recoveries_enemy  
        #     recoveries_enemy = split_result[i][0]
        elif i == 13:
            turn = int(split_result[i][0]*env.MAX_TURN)
    
    # now_character_id = None
    # now_character_action_set = None
    # damages_enemy = None
    # recoveries_enemy = None
    
    original_state = {
        "mymembers": mymembers_state,
        "enemys": enemys_state,
        "turn": turn,
        # "now_character_id": now_character_id,
        # "now_character_action_set": now_character_action_set,
        # "damages_enemy": damages_enemy,
        # "recoveries_enemy": recoveries_enemy
    }
    
    return original_state


# データ(side, agent_id, state, action, target_id, next_state, reward, done)からシーケンスデータを作成して返す
def make_sequence(datas, agent_id):
    # エージェント毎にデータを作成
    # 前の行動後の(状態,行動)から時行動直前の(状態,行動)までを保持
    seqs = []
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
            
            data_id = data[1]
            if found:
                if data_id == agent_id:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    
                    seq.append((s, action, target_id))
                    # seq.append((data_id, s, action, target_id))
                    
                    seqs.append(seq)
                    seq = []
                    seq.append((s, action, target_id))
                    # seq.append((data_id, s, action, target_id))
                    
                else:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    seq.append((s, action, target_id)) # (s, a, t)全て
                    # seq.append((data_id, s, action, target_id))
            else:
                if data_id == agent_id:
                    found = True
                    
                    if not seq == []:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                        seqs.append(seq)
                        seq = []
                        # クリアしたら次のseq用にseqに追加
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                        seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                        
                else:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    seq.append((s, action, target_id)) # (s, a, t)全て
                    # seq.append((data_id, s, action, target_id))
    
    return seqs

# 味方選択用のseqsを作成
def make_sequence_for_target1(datas, agent_id):
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
                if data_id == agent_id:
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
                if data_id == agent_id:
                    found = True
                    
                    if not seq == []:
                        # dataを追加
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                        if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                            seqs.append(seq)
                        seq = []
                        # クリアしたら次のseq用にseqに追加
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        # target_idを確認してtarget1ならばseqsに追加
                        if target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40:
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            
                            seqs.append(seq)
                        seq = []
                        # seqsに追加したらseqをクリアして
                        # 次のseq用に(s, a, t)をseqに追加
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                else:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    seq.append((s, action, target_id)) # (s, a, t)全て
                    # seq.append((data_id, s, action, target_id))
    return seqs

# 敵選択用のseqsを作成
def make_sequence_for_target2(datas, agent_id):
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
                if data_id == agent_id:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    
                    # seqの最後のデータ
                    seq.append((s, action, target_id)) # (s, a, t)全て
                    # seq.append((data_id, s, action, target_id))
                    
                    # target1を選択しているなら、seqsに追加する
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
                if data_id == agent_id:
                    found = True
                    if not seq == []:
                        # dataを追加
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                        
                        if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                            seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                            
                    else:
                        s = data[2]
                        action = data[3]
                        target_id = data[4]
                        # target_idを確認してtarget1ならばseqsに追加
                        if not (target_id == -1 or target_id == 0 or target_id == 1 or target_id == 10 or target_id == 20 or target_id == 30 or target_id == 40):
                            seq.append((s, action, target_id))
                            # seq.append((data_id, s, action, target_id))
                            seqs.append(seq)
                        seq = []
                        seq.append((s, action, target_id))
                        # seq.append((data_id, s, action, target_id))
                            
                else:
                    s = data[2]
                    action = data[3]
                    target_id = data[4]
                    seq.append((s, action, target_id)) # (s, a, t)全て
                    # seq.append((data_id, s, action, target_id))
    return seqs

ACTION_TENSOR = torch.tensor([0]*len(Character.ACTIONS))
# target tensor
keys_to_zero = env.characters_flag.keys()
a = {-1: 0, 0: 0, 1: 0}
result_dict = dict(env.characters_flag)
for key in keys_to_zero:
    result_dict[key] = 0
# aとresult_dictを連結させる
TARGET_TENSOR_dict = {**a, **result_dict}
print("TARGET_TENSOR_dict:", TARGET_TENSOR_dict)
TARGET_TENSOR = list(TARGET_TENSOR_dict.values())


def pre_process(seqs):
    
    sequences = seqs
    # print("len of sequences", len(sequences))
    # padding
    if len(sequences) > 0:
        pre_process_sequences = []
        for seq in sequences:
            pre_process_seq = []
            for s in seq:
                # print("len of s", len(s))
                
                # #########
                # state
                # #########
                s_state = torch.tensor(s[0], dtype=torch.float)
                # print("size of s_state", s_state.size())
                
                # ############################
                # s_action -> action tensor
                # ############################
                s_action = s[1] # (s, a, t)全て
                # print("s_action", s_action)
                action_tensor_temp = copy.deepcopy(ACTION_TENSOR)
                if s_action != 999:
                    action_tensor_temp[int(s_action)] = 1
                s_action_tensor = torch.tensor(action_tensor_temp, dtype=torch.float)
                # print("size of s_action_tensor",s_action_tensor.size())
                
                # ##########################
                # s_target -> target tensor
                # ###########################
                s_target = s[2] # (s, a, t)全て
                # s_target = s[1] # (a, t)のみ
                target_tensor_dict_temp = copy.deepcopy(TARGET_TENSOR_dict)
                if s_target != 999:
                    target_tensor_dict_temp[int(s_target)] = 1
                s_target_tensor = torch.tensor(list(target_tensor_dict_temp.values()), dtype=torch.float)
                # print("size of s_target_tensor",s_target_tensor.size())
                
                # ####################
                # cat merge
                # #####################
                s_cat_tensor = torch.tensor(torch.cat((s_state, s_action_tensor, s_target_tensor))) #(s, a, t)全て
                pre_process_seq.append(s_cat_tensor)
            pre_process_sequences.append(pre_process_seq)
        # print("len of preprocessSeqs",len(pre_process_sequences))
        padded_sequences = pad_sequence([torch.stack(seq) for seq in pre_process_sequences], batch_first=True, padding_value=0)
        # print(padded_sequences)
        # print(padded_sequences.size())
        return padded_sequences
    else:
        return []







# env.characters_flagをコピーして新たな辞書を作成し、キーに対応する値を0に設定




if __name__ == "__main__":
    file_path = "./ExpertData/Agent_exp_action.txt"
    f = open(file_path, "rb")
    env = Game(1)
    state = env.State()
    action = 999
    target_id = 999
    seq = [[(state, action, target_id)]]
    print("pre process(seq)\n", pre_process(seq))
    