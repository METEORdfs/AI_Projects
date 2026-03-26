# 导入必要的库
from langchain_openai import ChatOpenAI          # 连接本地 Ollama 的 OpenAI 兼容接口
from langchain.memory import ConversationBufferMemory  # 管理对话记忆
from langchain.chains import ConversationChain          # 把记忆和模型串成对话链
import json          # 处理 JSON 文件（保存/读取对话历史）
import os            # 检查文件是否存在
import sys           # 程序退出时使用
import logging       # 日志系统
import time          # 时间处理，用于日志文件名格式化
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
    filename="chat.log",          # 基础文件名，实际写入的文件是 chat.log
    when="midnight",              # 滚动时间点：每天午夜
    interval=1,                   # 间隔 1 天
    backupCount=30,               # 保留最近 30 个备份文件
    encoding="utf-8"             # 编码
)
suffix="%Y-%m-%d"             # 备份文件后缀格式，如 chat.log.2026-03-24

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 配置 root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler]
)
# ====================================================

# ==================== 2. 加载配置文件 ====================
try:
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)          # 读取配置
    logging.info("配置文件加载成功")     # 记录日志
except FileNotFoundError:
    logging.error("未找到 config.json 文件，请确保它在项目根目录下。")
    sys.exit(1)
except json.JSONDecodeError:
    logging.error("config.json 格式不正确，请检查 JSON 语法。")
    sys.exit(1)

MEMORY_FILE = config["memory_file"]   # 历史对话保存的文件名
# ====================================================

# ==================== 3. 加载历史对话 ====================
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)           # 读取已有历史
        logging.info(f"加载历史对话，共 {len(history)} 条")
    except json.JSONDecodeError:
        logging.error("历史文件损坏，将重新初始化")
        history = []                         # 文件损坏，从头开始
else:
    history = []                              # 首次运行，无历史
    logging.info("未找到历史文件，初始化空历史")
# ====================================================

# ==================== 4. 初始化模型 ====================
llm = ChatOpenAI(
    model=config["model"],           # 模型名，从配置读取
    base_url=config["base_url"],     # Ollama API 地址
    api_key=config["api_key"],       # 随便填，本地不需要真实 key
    temperature=config["temperature"] # 温度，控制随机性
)
logging.info("LLM 初始化完成")
# ====================================================

# ==================== 5. 将历史对话加载到记忆 ====================
memory = ConversationBufferMemory()
for turn in history:
    memory.chat_memory.add_user_message(turn["user"])  # 加入用户消息
    memory.chat_memory.add_ai_message(turn["ai"])      # 加入 AI 消息
logging.info(f"已将 {len(history)} 条历史加载到记忆")
# ====================================================

# ==================== 6. 设定 AI 的角色身份 ====================
# 这里在记忆里预先加一条 AI 说的话，告诉它自己的身份
memory.chat_memory.add_ai_message("我是小七，昨天是我的生日。我是用户的AI助手。")
logging.info("角色设定已添加")
# ====================================================

# ==================== 7. 创建对话链 ====================
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False   # 不打印中间步骤，保持输出简洁
)
# ====================================================

# ==================== 8. 主循环 ====================
print("持久对话助手启动，输入 'save' 保存并退出，输入 'exit' 不保存退出")
while True:
    user = input("你: ")
    if user.lower() == 'exit':
        logging.info("用户选择退出，程序结束")
        break
    if user.lower() == 'save':
        # 保存当前所有对话到 JSON 文件
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logging.info(f"历史已保存，共 {len(history)} 条")
        print("历史已保存")
        continue

    try:
        logging.info(f"用户输入: {user}")           # 记录用户输入
        response = conversation.predict(input=user)  # 调用模型
        logging.info(f"AI回复: {response}")          # 记录 AI 回复
        print(f"AI: {response}")
        history.append({"user": user, "ai": response})  # 追加到历史列表
    except Exception as e:
        logging.error(f"对话出错: {e}", exc_info=True)   # 记录详细错误
        print(f"出错: {e}，请检查日志文件 chat.log")
# ====================================================