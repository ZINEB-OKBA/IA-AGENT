

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# Utilisation de create_react_agent (standard actuel)
from langgraph.prebuilt import create_react_agent 

# 1. Configuration LLM (Correction gpt-4o)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. VectorStore et Retriever
texts = ["Je m'appelle Mohamed Youssfi, Je suis Professeur en Informatique et Intelligence artificielle",
"Je travaille à l'ENSET Mohammedia, Université Hassan II de Casablanca",
"J'ai obtenu mon doctorat d'état en 2015, Mon doctorat de troisième cycle en 1996 et mon"
"Master en 1992, tous de l'Université Hassan II de Casablanca",
"Mes domaines de recherche comprennent l'intelligence artificielle, l'apprentissage automatique, le traitement du langage naturel et les systèmes de recommandation",
"J'ai publié de nombreux articles dans des revues et des conférences internationales, et j'ai participé à plusieurs projets de recherche dans le domaine de l'informatique et de l'intelligence artificielle",
"Je suis également membre de plusieurs associations professionnelles dans le domaine de l'informatique et de l'intelligence artificielle, et j'ai encadré de nombreux étudiants de master et de doctorat dans leurs recherches",
"Je suis passionné par l'enseignement et la recherche, et je m'efforce de contribuer au développement de l'informatique et de l'intelligence artificielle au Maroc et dans le monde entier"]
 
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_texts(texts, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Définition des outils
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="cv_retriever",
    description="useful for when you want to answer questions about Mohamed Youssfi's CV"
)

@tool
def get_employee_info(name: str):
    """Retrieves information about an employee given their name."""
    return {"name": name, "position": "Software Engineer", "email": f"{name.lower()}@example.com"}

@tool
def send_mail(email: str, subject: str, content: str):
    """Sends an email."""
    return f"Email sent to {email}"

# 4. Initialisation de l'agent (Correction du nom de la variable tool)
tools = [get_employee_info, retriever_tool, send_mail]
agent_executor = create_react_agent(llm, tools)

# 5. Invocation
resp = agent_executor.invoke(
    input={"messages": [HumanMessage(content="what is the position of Mohamed Youssfi?")]}
)

print(resp["messages"][-1].content)