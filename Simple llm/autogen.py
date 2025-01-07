from autogen import AssistantAgent, UserProxyAgent

config_list = [{
    "model": "codellama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "NA",
}]

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})

user_proxy = UserProxyAgent("user", code_execution_config={"work_dir": "/home/devops/test", "use_docker": False})
user_proxy.initiate_chat(assistant, message="Create a report on deforestation in India")