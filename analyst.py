import getpass, os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [SystemMessage(content="You are a professional technical financial analyst."),
            HumanMessage(content =""),
            ]

print(model.invoke(messages).content)