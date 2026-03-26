from typing import TypedDict
from models.prompts import AgentPrompts

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from models.search_web import BaseSearchModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage

class AgentState(TypedDict):
	pregunta: str
	fuente: str
	contexto: str
	respuesta: AIMessage

class FederalLaborAgent:
	def __init__(
		self,
		chat_model: BaseChatModel,
		retriever_rag: BaseRetriever,
		search_class: BaseSearchModel,
	):
		self._agent_prompts = AgentPrompts()
		self._chat_model = chat_model
		self._retriever_rag = retriever_rag
		self._search_class = search_class

		self._graph = StateGraph(AgentState)
		self._app: CompiledStateGraph
		self._agent_state: dict

	def _agent_node(self, state: AgentState):
		pregunta = state["pregunta"]

		prompt = self._agent_prompts.PROMPT_CLASSIFIER.replace("$PREGUNTA$", pregunta)
		respuesta = self._chat_model.invoke(prompt)
		fuente = respuesta.content  # .strip()

		if isinstance(fuente, str):
			fuente = fuente.strip().replace("\n", " ")
		else:
			raise TypeError(f"Se detectó salida compleja del tipo '{type(fuente)}'")

		if "rag" in fuente.lower():
			fuente = "RAG"
		else:
			fuente = "WEB"

		print(f"El agente decidió '{fuente}'")

		return {"fuente": fuente}

	def _rag_node(self, state: AgentState):
		pregunta = state["pregunta"]

		docs = self._retriever_rag.invoke(pregunta)
		contexto = "\n\n---\n\n".join(doc.page_content for doc in docs)

		return {"contexto": contexto}

	def _search_node(self, state: AgentState):
		contexto = self._search_class.search(state["pregunta"])

		return {"contexto": contexto}

	def _generate_node(self, state: AgentState):
		contexto = state["contexto"]
		pregunta = state["pregunta"]
		prompt = (
			self._agent_prompts.PROMPT_GENERATE
				.replace("$CONTEXTO$", contexto)
				.replace("$PREGUNTA$", pregunta)
		)

		respuesta = self._chat_model.invoke(prompt)

		return {"respuesta" : respuesta}

	def _choice_source(self, state: AgentState):
		if state["fuente"] == "RAG":
			return "RAG"
		elif state["fuente"] == "WEB":
			return "WEB"
		else:
			return "ERROR"

	def build_graph(self):
		self._graph.add_node("Agent", self._agent_node)
		self._graph.add_node("RAG", self._rag_node)
		self._graph.add_node("WEB", self._search_node)
		self._graph.add_node("Generator", self._generate_node)

		self._graph.add_edge(START, "Agent")
		self._graph.add_conditional_edges(
			"Agent", self._choice_source,
			{
				"RAG" : "RAG",
				"WEB" : "WEB"
			}
		)
		self._graph.add_edge("RAG", "Generator")
		self._graph.add_edge("WEB", "Generator")
		self._graph.add_edge("Generator", END)

		self._app = self._graph.compile()

	def execute(self, query: str):

		resultado = self._app.invoke(
			{
				"pregunta" : query,
				"fuente" : "",
				"contexto" : "",
				"respuesta" : "",
			}
		)

		self._agent_state = resultado

		return resultado