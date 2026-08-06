from langchain.agents import  create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from ..tools.tools import web_search, scrape_url
import os


load_dotenv()


#setting up the model
load_dotenv()

#setting up llm
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


#creating agents::(1st agent) :search agent
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search],
    )


#second agents:: (2nd agent)  :reader agent
def build_reader_agent():
    return create_agent(
        model = llm,
        tools= [scrape_url],
    )



#writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system","You are an Expert Research writer. Write Clear, structured and Insightful"),
    ("human", """Write a detailed research report on the topic below.
    
    Topic: {topic}
    
    Research Gathered:
    {research}
    
    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - conclusion
    - sources (list all urls found in the research)
    
    Be detailed, factual and professional """),
    ])




writer_chain = writer_prompt | llm | StrOutputParser()


#critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system","you are a sharp and constructive research critic. Be honest and specific"),
    ("human", """Review the research report below and evaluate it strictly.
    Report: {report}
    
    Respond in this exact format:
    Score: X/10
    
    Strengths:
    - ....
    -....
    
    Areas to Improve: 
    -.....
    -.....
    One line verdict:
    ......
    """),
])

critic_chain = critic_prompt | llm | StrOutputParser()








