import pettingzoo
from pettingzoo.classic import connect_four_v3

def main():
    try:
        # 创建一个PettingZoo环境实例
        env = connect_four_v3.env()
        # 重置环境以开始新的游戏
        env.reset()
        print("PettingZoo库安装成功！")
    except ImportError as e:
        print(f"导入失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()

