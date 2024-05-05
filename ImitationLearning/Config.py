
class Config:
    def __init__(self):
        # `rpg_battle`があるパスを指定してください。パスの最後は"/"を追加してください
        self.parent_dir = "/Users/fujiyamax/home/myProjects/rpg_battle/"

        # エキスパートデータファイルのパスを指定してください
        self.expert_data_path = "/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/ExpertData/situation=all_ALL_PLAYERS_all.txt"

        # GAIL 事前学習パラメータが保持してあるPATH
        # `GAIL_Init_Params_model` があるパスを指定してください。パスの最後は"/"を追加してください
        self.pre_train_path = "/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/GAIL_Init_Params_model/"

        # GAIL シミュレーション学習パラメータが保持してあるPATH
        # `GAIL_learned_model`があるパスを指定してください。パスの最後は"/"を追加してください。
        self.gail_learned_path = "/Users/fujiyamax/home/myProjects/rpg_battle/ImitationLearning/GAIL_learned_model/"