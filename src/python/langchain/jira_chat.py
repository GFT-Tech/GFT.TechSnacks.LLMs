from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.utilities.jira import JiraAPIWrapper
from dotenv import load_dotenv, find_dotenv
from langchain.llms import AzureOpenAI
#check .env
_ = load_dotenv(find_dotenv())

llm = AzureOpenAI(
    deployment_name="gpt-tech-snacks",
    model_name="gpt-35-turbo",
    temperature=0
)

jira = JiraAPIWrapper()
toolkit = JiraToolkit.from_jira_api_wrapper(jira)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

#Worked!
agent.run("How many issues are the in Project SSP")
agent.run("What can you tell me about SSP-1")