# 导入必要的库
from langchain_openai import ChatOpenAI  # 连接本地 Ollama 的 OpenAI 兼容接口
from langchain.memory import ConversationBufferMemory  # 管理对话记忆
from langchain.chains import ConversationChain  # 把记忆和模型串成对话链
import json  # 处理 JSON 文件（保存/读取对话历史）
import os  # 检查文件是否存在
import sys  # 程序退出时使用
import logging  # 日志系统
import time  # 时间处理，用于日志文件名格式化
from logging.handlers import TimedRotatingFileHandler  # 按时间滚动的日志处理器基类
from typing import Optional, TextIO


# ==================== 1. 配置日志 ====================
# 自定义按天滚动，备份文件名为 chat.log.2026-03-24 格式
class DailyRotatingFileHandler(TimedRotatingFileHandler):
    # 重新声明 stream 为 Optional 类型，覆盖基类的严格类型
    stream: Optional[TextIO] = None

    def doRollover(self):
        """滚动时，将当前日志文件重命名为带日期的备份文件，然后新建空文件"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # 计算备份文件的时间戳（滚动前一刻）
        current_time = int(self.rolloverAt - self.interval)
        time_tuple = time.localtime(current_time)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, time_tuple)

        # 如果备份文件已存在，先删除（避免冲突）
        if os.path.exists(dfn):
            os.remove(dfn)

        # 重命名当前日志文件为备份文件
        os.rename(self.baseFilename, dfn)

        # 清理超过 backupCount 的旧文件
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

        # 重新打开新日志文件
        self.stream = self._open()
        self.rolloverAt = self.rolloverAt + self.interval


# 创建 handler 实例
file_handler = DailyRotatingFileHandler(
    filename="chat.log",  # 基础文件名，实际写入的文件是 chat.log
    when="midnight",  # 滚动时间点：每天午夜
    interval=1,  # 间隔 1 天
    backupCount=30,  # 保留最近 30 个备份文件
    encoding="utf-8",  # 编码
)
suffix = "%Y-%m-%d"  # 备份文件后缀格式，如 chat.log.2026-03-24

# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 配置 root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler])
# ====================================================


# ==================== 2. 对话会话类 ====================
class ChatSession:
    """封装对话会话，支持持久化记忆和日志"""

    def __init__(self, config_path="config.json", memory_file=None):
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # 记忆文件路径（如果外部传入则使用，否则从配置读取）
        self.memory_file = memory_file or self.config["memory_file"]

        # 加载历史对话
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                logging.info(f"加载历史对话，共 {len(self.history)} 条")
            except json.JSONDecodeError:
                logging.error("历史文件损坏，将重新初始化")
                self.history = []
        else:
            self.history = []
            logging.info("未找到历史文件，初始化空历史")

        # 初始化模型
        self.llm = ChatOpenAI(
            model=self.config["model"],
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            temperature=self.config["temperature"],
        )
        logging.info("LLM 初始化完成")

        # 创建记忆对象并加载历史
        self.memory = ConversationBufferMemory()
        for turn in self.history:
            self.memory.chat_memory.add_user_message(turn["user"])
            self.memory.chat_memory.add_ai_message(turn["ai"])
        logging.info(f"已将 {len(self.history)} 条历史加载到记忆")

        # 角色设定
        self.memory.chat_memory.add_ai_message(
            "我是小七，昨天是我的生日。我是用户的AI助手。"
        )
        logging.info("角色设定已添加")

        # 创建对话链
        self.conversation = ConversationChain(
            llm=self.llm, memory=self.memory, verbose=False
        )

    def chat(self, user_input: str) -> str:
        """接收用户输入，返回 AI 回复，并自动更新历史"""
        logging.info(f"用户输入: {user_input}")
        response = self.conversation.predict(input=user_input)
        logging.info(f"AI回复: {response}")

        # 更新内存中的历史（供后续保存用）
        self.history.append({"user": user_input, "ai": response})
        return response

    def save_history(self):
        """保存当前对话历史到文件"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        logging.info(f"历史已保存，共 {len(self.history)} 条")


# ====================================================

# ==================== 8. 主循环 ====================
if __name__ == "__main__":
    session = ChatSession()
    print("持久对话助手启动，输入 'save' 保存并退出，输入 'exit' 不保存退出")
    while True:
        user = input("你: ")
        if user.lower() == 'exit':
            logging.info("用户选择退出，程序结束")
            break
        if user.lower() == 'save':
            session.save_history()
            print("历史已保存")
            continue

        try:
            reply = session.chat(user)
            print(f"AI: {reply}")
        except Exception as e:
            logging.error(f"对话出错: {e}", exc_info=True)
            print(f"出错: {e}，请检查日志文件 chat.log")
# ====================================================