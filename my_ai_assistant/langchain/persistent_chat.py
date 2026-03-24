from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json
import os

MEMORY_FILE = "chat_history.json"

# 加载历史对话
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
else:
    history = []

# 连接本地 Ollama
llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="dummy",
    temperature=0.7
)

# 创建记忆对象
memory = ConversationBufferMemory()

# 第一步：注入角色设定（这是关键！）
# 用 AI 消息告诉模型它的身份，确保模型记住
memory.chat_memory.add_ai_message("我是小七，昨天是我的生日。我是用户的AI助手。")
# 也可以加一条用户消息强化
memory.chat_memory.add_user_message("你是小七，昨天是你的生日。我是你的创造者。")
memory.chat_memory.add_ai_message("好的，我记住了。我是小七，昨天是我的生日，你是我的创造者。")

# 将之前保存的历史对话加载到 memory
for turn in history:
    memory.chat_memory.add_user_message(turn["user"])
    memory.chat_memory.add_ai_message(turn["ai"])

# 创建对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False   # 如果你想看内部步骤，可改为 True
)

print("持久对话助手启动，输入 'save' 保存并退出，输入 'exit' 不保存退出")
while True:
    user = input("你: ")
    if user.lower() == 'exit':
        break
    if user.lower() == 'save':
        # 保存历史
        # 注意：这里保存的是用户和AI的对话历史，不包括角色设定消息
        # 但角色设定已经通过 memory 中的消息在后续对话中持续生效
        # 如果希望下次启动时角色设定也保留，可以把 memory 中的所有消息都保存，但复杂一些，暂不实现
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print("历史已保存")
        continue
    response = conversation.predict(input=user)
    print(f"AI: {response}")
    # 记录本次对话到历史列表（仅用于持久化）
    history.append({"user": user, "ai": response})