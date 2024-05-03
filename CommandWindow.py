# pygame
import pygame
from pygame.locals import *
import sys


# screen サイズ
SCREEN_SIZE_WIDTH = 700
SCREEN_SIZE_HEIGHT = 1000
SCREEN_SIZE = (SCREEN_SIZE_WIDTH, SCREEN_SIZE_HEIGHT)

# コマンドウィンドウ サイズ
CMD_WINDOW_INIT_X = 10  # コマンドウィンドウの左上x座標
CMD_WINDOW_INIT_Y = 700  # コマンドウィンドウの左上y座標
CMD_WINDOW_WIDTH = 680  # コマンドウィンドウの横幅
CMD_WINDOW_HEIGHT = 280  # コマンドウィンドの縦幅
COMMAND_WINDOW_RECT = (CMD_WINDOW_INIT_X, CMD_WINDOW_INIT_Y,
                       CMD_WINDOW_WIDTH, CMD_WINDOW_HEIGHT)

pygame.font.init()  # フォントの初期化
# font = pygame.font.Font("./ipaexg00401/ipaexg.ttf", 10)



class CommandWindow():
    """ウィンドウの基本クラス"""
    EDGE_WIDTH = 4  # 白枠の幅

    # message 対応
    def __init__(self, rect):
        # 外側の白い矩形
        self.rect = rect
        # 内側の黒い矩形
        self.inner_rect = self.rect.inflate(-self.EDGE_WIDTH * 2,
                                            -self.EDGE_WIDTH*2)

        self.is_visible = False  # ウィンドウを表示中か？

    # テキストをコマンドウィンドウないに表示させる
    def draw(self, screen, text):
        # fontサイズの指定
        font_size = 20
        font_path = "./ipaexg00401/ipaexg.ttf"
        font = pygame.font.Font(font_path, font_size)

        """ウィンドウを描画"""
        # if self.is_visible == False:
        #     return
        # else:
        #     pygame.draw.rect(screen, (255, 255, 255), self.rect, 0)
        #     pygame.draw.rect(screen, (0, 0, 0), self.inner_rect, 0)

        # コマンドウィンドウ表示
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 0)
        pygame.draw.rect(screen, (0, 0, 0), self.inner_rect, 0)
        # self.inner_rectのx, y座標、width, heightを取得
        rect_x = self.inner_rect.x
        rect_y = self.inner_rect.y
        # rect_width = self.inner_rect.width
        # rect_height = self.inner_rect.height

        # テキストを改行文字で分割
        lines = text.split("\n")

        y = rect_y  # テキストの開始位置（縦方向）
        for line in lines:
            text_surface = font.render(line, True, (255, 255, 255))  # テキストカラーを指定

            # テキスト表示
            screen.blit(text_surface, (rect_x, y))
            # font_size = 20
            y += font_size  # 次の行の開始位置を更新

        # pygame.display.flip()

    def show(self):
        """ウィンドウを表示"""
        if self.is_visible:
            self.is_visible = False
        else:
            self.is_visible = True
        

    # def hide(self):
    #     """ウィンドウを隠す"""
    #     self.is_visible = False

class StatusWindow:
    EDGE_WIDTH = 3
    def __init__(self, rect):
        # 外側の白い矩形
        self.rect = rect
        # 内側の黒い矩形
        self.inner_rect = self.rect.inflate(-self.EDGE_WIDTH*2, -self.EDGE_WIDTH*2)
        self.is_visible = False
    
    def draw(self, screen, text, color=(255, 255, 255), now_turn=False):
        # フォントサイズの指定
        # font_size = 20
        font_size = 17
        font_path = "./ipaexg00401/ipaexg.ttf"
        font = pygame.font.Font(font_path, font_size)

        # status windowの表示
        if now_turn is True:
            pygame.draw.rect(screen, (0, 0, 255), self.rect, 0)
        else:
            pygame.draw.rect(screen, (255, 255, 255), self.rect, 0)
        pygame.draw.rect(screen, (0, 0, 0), self.inner_rect, 0)
        rect_x = self.inner_rect.x
        rect_y = self.inner_rect.y
        
        # 改行文字で区切る
        lines = text.split("\n")
        
        # テキストの開始位置
        y = rect_y
        for line in lines:
            text_surface = font.render(line, True, color)
            # while text_surface.get_width() > self.rect.width or text_surface.get_height() > self.rect.height:
            #     font_size -= 1
            #     font = pygame.font.Font(font_path, font_size)
            #     text_surface = font.render(line, True, (255, 255, 255))
            # # もう1size小さくする
            # font_size -= 1
            # font = pygame.font.Font(font_path, font_size)
            # text_surface = font.render(line, True, (255, 255, 255))
            
            # rect内に表示させる
            screen.blit(text_surface, (rect_x, y))
            
            y += font_size
        # pygame.display.flip()


