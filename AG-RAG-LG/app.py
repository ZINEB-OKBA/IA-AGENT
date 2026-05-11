from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv.ipython import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()


loader = PyPDFLoader("cvzineb.pdf")

tokennizer = tiktoken.encoding_for_model("gpt-4o-mini")



splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name = tokennizer.name,
    chunk_size = 300,
    chunk_overlap = 20
)


chunks= loader.load_and_split(splitter)

embeddings_model = OpenAIEmbeddings()


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model, 
    collection_name="cvzineb")


retriever = vectorstore.as_retriever(kwargs={'K': 3})

@tool
def retriever_tool(query: str) ->str:
   """
   permet de chercher des informations sur candidate:
   -nom
   -prenom
   -formation
   -expérience professionnelle
   """

   relevent_chunks=retriever.invoke(query)
   context_list=[d.page_content for d in relevent_chunks]
   context = "\n".join(context_list)
   return context
 
@tool
def get_company_info(comapny_name:str):
    """consulter les informations d'une entreprise donner"""
    return{
    "company_name": comapny_name,
    "domain": "IT",
    "turnover": 123_447_860
    }

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
agent=create_agent(
    tools=[retriever_tool,get_company_info],
    model=llm,
    system_prompt="repondre à la question de l'utilisateur en utilisant les informations du cv de la candidate. Si les informations ne sont pas disponibles dans le cv, répondre honnêtement que vous ne savez pas. en utilisant les tools founies pour chercher les informations dans le cv. Ne pas inventer des informations qui ne sont pas dans le cv."
)




 
