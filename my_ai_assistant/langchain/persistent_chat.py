# persistent_chat.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import json
import os

MEMORY_FILE = "chat_history.json"

# 加载历史
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r") as f:
        history = json.load(f)
else:
    history = []

llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="dummy",
    temperature=0.7
)

memory = ConversationBufferMemory()
# 将历史加载到 memory
for turn in history:
    memory.chat_memory.add_user_message(turn["user"])
    memory.chat_memory.add_ai_message(turn["ai"])

conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

print("持久对话助手启动，输入 'save' 保存并退出，输入 'exit' 不保存退出")
while True:
    user = input("你: ")
    if user.lower() == 'exit':
        break
    if user.lower() == 'save':
        # 保存历史
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print("历史已保存")
        continue
    response = conversation.predict(input=user)
    print(f"AI: {response}")
    history.append({"user": user, "ai": response})