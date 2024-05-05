from Config import Config
configuration=Config()
from GAIL_REINFORCE import *

import sys
sys.path.append(configuration.parent_dir)  # 親ディレクトリを追加
from env.Game import Game

# wandbで学習ログをとる
# import wandb
# wandb.init(
#     project="GAIL PreTraining"
# )
# def plot_pre_training_loss(agent_id, loss):
#     wandb.log({
#         f"agent_id={agent_id}: loss action": loss[0],
#         f"agent_id={agent_id}: loss target1": loss[1],
#         f"agent_id={agent_id}: loss target2": loss[2]
#     })

config = {
    "lr_pre_training": {"10": {"action": 1e-4, "target1": 1e-4, "target2": 1e-4}, 
                        "20": {"action": 1e-4, "target1": 1e-4, "target2": 1e-4}, 
                        "30": {"action": 1e-4, "target1": 1e-4, "target2": 1e-4},
                        "40": {"action": 1e-4, "target1": 1e-4, "target2": 1e-4}},
    "expert_data_path": configuration.expert_data_path,
    "save_paremters_dir": configuration.pre_train_path,
    "epochs": 1000
}

# LSTMによるBC
def gail_pre_train():
    print(config)
    env = Game(1)
    # 学習するエージェントの用意
    agents = {}
    for key in [10, 20, 30, 40]:
        lr_a = config["lr_pre_training"][f"{key}"]["action"]
        lr_t1 = config["lr_pre_training"][f"{key}"]["target1"]
        lr_t2 = config["lr_pre_training"][f"{key}"]["target2"]
        lrs = (lr_a, lr_t1, lr_t2)
        expert_data_path = config["expert_data_path"]
        agents[key] = RLAgent(env, key, lrs, 0, 0, expert_data_path)
    epochs = config["epochs"]
    
    # 学習率長せscheduler
    
    
    for epoch in range(1, 1+epochs):
        if epoch % int(epochs/10) == 0:
            print("Epoch:", epoch)
        
        for key, _ in agents.items():
            loss = agents[key].pre_training()
            # plot_pre_training_loss(key, loss) # to use -> import wandb
            if epoch % int(epochs/10) == 0:
                print(f"agent_id={key}: action: {loss[0]}, target1: {loss[1]}, target2: {loss[2]}")

    
    # 評価
    for key, _ in agents.items():
        loss = agents[key].eval_pre_train()
        print(f"agent_id={key}\nLoss: ({loss})")
        
    
    
    # 学習済みモデルを保存
    dir=config["save_paremters_dir"]
    print("save in ", dir)
    for key, value in agents.items():
        torch.save(agents[key].pi_action.state_dict(), dir+'agent_id={}_action.pth'.format(key))
        torch.save(agents[key].pi_target1.state_dict(), dir+'agent_id={}_target1.pth'.format(key))
        torch.save(agents[key].pi_target2.state_dict(), dir+'agent_id={}_target2.pth'.format(key))                    
    print("save successfully")
    
    
    
    # configの情報を保存
    # テキストファイルに書き込むファイルパス
    file_path = dir+'config.txt'
    # テキストファイルに書き込み
    with open(file_path, 'w') as file:
        for key, value in config.items():
            file.write(f"{key}: {value}\n")

    
        

if __name__ == "__main__":
    gail_pre_train()