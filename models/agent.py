from typing import TypedDict
from models.prompts import AgentPrompts

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

class AgentState(TypedDict):
	pregunta : str
	fuente : str
	contexto : str
	respuesta : str

class FederalLaborAgent:
	def __init__(self, chat_model: BaseChatModel, retriever_rag : BaseRetriever):
		self.agent_prompts = AgentPrompts()
		self.chat_model = chat_model
		self.retriever_rag = retriever_rag


	def agent_node(self, state : AgentState):
		pregunta = state["pregunta"]

		prompt = AgentPrompts.PROMPT_CLASSIFIER.replace("$PREGUNTA$", pregunta)
		respuesta = self.chat_model.invoke(prompt)
		fuente = respuesta.content#.strip()

		if isinstance(fuente, str):
			fuente = fuente.strip().replace("\n", " ")
		else:
			raise TypeError(f"Se detectó salida compleja del tipo '{type(fuente)}'")

		if "rag" in fuente.lower():
			fuente = "RAG"
		else:
			fuente = "WEB"

		print(f"El agente decidió '{fuente}'")

		return {
			"fuente" : fuente
		}

	def nodo_rag(self, state : AgentState):
		pregunta = state["pregunta"]

		docs = self.retriever_rag.invoke(pregunta)
		contexto = "\n\n---\n\n".join(doc.page_content for doc in docs)

		return {
			"contexto" : contexto
		}