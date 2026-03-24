from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 连接本地 Ollama
llm = ChatOpenAI(
    model="qwen2.5:7b",       # 可换成 qwen2.5:14b
    base_url="http://localhost:11434/v1",
    api_key="dummy",
    temperature=0.7
)

# 创建内存（存储对话历史）
memory = ConversationBufferMemory()

# 创建对话链，自动管理记忆
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True   # 打印内部步骤，方便调试
)

print("AI 助手已启动，输入 'exit' 退出")
while True:
    user_input = input("你: ")
    if user_input.lower() == 'exit':
        break
    response = conversation.predict(input=user_input)
    print(f"AI: {response}")