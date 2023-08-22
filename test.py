from langchain.llms import ChatGLM
from langchain import PromptTemplate, LLMChain

template = """{question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

endpoint_url = "http://0.0.0.0:9996/v1"

llm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    history=[["我将从美国到中国来旅游，出行前希望了解中国的城市", "欢迎问我任何问题。"]],
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "北京和上海两座城市有什么不同？"

llm_chain.run(question)

