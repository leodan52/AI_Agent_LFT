from datetime import date, datetime
from typing import TypedDict

import pdfkit
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from markdown import markdown

from models.prompts import AgentPrompts
from models.search_web import BaseSearchModel


class AgentState(TypedDict):
	pregunta: str
	fuente: str
	contexto: str
	respuesta: str
	bibliografia: str
	metadata_respuesta: dict

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
		self._agent_state = dict()
		self._app: CompiledStateGraph

	@property
	def answer(self):
		try:
			return self._agent_state["respuesta"]
		except KeyError:
			return "No hay respuesta aún"

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
		elif "web" in fuente.lower():
			fuente = "WEB"
		else:
			fuente = "ERROR"

		print(f"El agente decidió '{fuente}'")

		return {"fuente": fuente}

	def _rag_node(self, state: AgentState):
		pregunta = state["pregunta"]

		docs = self._retriever_rag.invoke(pregunta)
		contexto = "\n\n---\n\n".join(doc.page_content for doc in docs)
		bibliografia = self._get_metadata(docs)
		#contexto += "\n\n---\n\n"*2 + f"Bibliografía:\n{self._get_metadata(docs)}"

		return {"contexto": contexto, "bibliografia" : bibliografia}

	def _search_node(self, state: AgentState):
		contexto = self._search_class.search(state["pregunta"])

		return {"contexto": contexto, "bibliografia" : ""}

	def _error_classifier_node(self, state: AgentState):
		respuesta = "La pregunta realizada es ambigua. Realiza otra con más información"

		return {"respuesta" : respuesta, "bibliografia" : ""}

	def _generate_node(self, state: AgentState):
		contexto = state["contexto"]
		pregunta = state["pregunta"]
		bibliografia = state["bibliografia"]

		if bibliografia:
			contexto += f"\nBibliografía:\n{bibliografia}\n"

		prompt = (
			self._agent_prompts.PROMPT_GENERATE
				.replace("$CONTEXTO$", contexto)
				.replace("$PREGUNTA$", pregunta)
		)

		messege = self._chat_model.invoke(prompt)

		return {"respuesta" : messege.content, "metadata_respuesta" : messege.usage_metadata}

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
		self._graph.add_node("ERROR", self._error_classifier_node)
		self._graph.add_node("Generator", self._generate_node)

		self._graph.add_edge(START, "Agent")
		self._graph.add_conditional_edges(
			"Agent", self._choice_source,
			{
				"RAG" : "RAG",
				"WEB" : "WEB",
				"ERROR" : "ERROR"
			}
		)
		self._graph.add_edge("RAG", "Generator")
		self._graph.add_edge("WEB", "Generator")
		self._graph.add_edge("ERROR", END)
		self._graph.add_edge("Generator", END)

		self._app = self._graph.compile()

	def execute(self, query: str):

		resultado = self._app.invoke(
			{
				"pregunta" : query,
				"fuente" : "",
				"contexto" : "",
				"respuesta" : "",
				"bibliografia" : "",
				"metadata_respuesta" : dict(),
			}
		)

		self._agent_state = resultado

		return resultado

	@staticmethod
	def _get_metadata(docs: list[Document]) -> str:
		biblio = dict()
		for d in docs:
			doc_pdf = d.metadata.get("source", "Sin información")
			pag = d.metadata.get("page_label", "Sin información")

			if str(pag).isnumeric():
				pag = int(pag)

			if doc_pdf in biblio:
				biblio[doc_pdf].add(pag)
			else:
				biblio[doc_pdf] = {pag}

		lineas = "\n".join(
			[f"    * Del archivo '{k}' en las páginas {str(v)}" for k, v in biblio.items()]
		)

		return lineas

	def response_to_pdf(self):
		respuesta = self._agent_state["respuesta"]
		respuesta = respuesta.strip()
		html_text = markdown(respuesta, extensions=["extra", "sane_lists"])
		now = datetime.now()

		if not respuesta:
			return

		full_html = f"""
		<html>
		<head>
			<meta charset="UTF-8">
			<style>
				body {{
					background-color: white !important;
					font-family: 'Helvetica', sans-serif;
					line-height: 1.6;
					color: #333;
					margin: 50px;
				}}
				h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
				code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
				blockquote {{ border-left: 5px solid #ccc; margin-left: 0; padding-left: 15px; color: #666; }}
			</style>
		</head>
		<body>
			{html_text}
		</body>
		</html>
		"""

		options = {
			'encoding': "UTF-8",
			'margin-top': '0.75in',
			'margin-right': '0.75in',
			'margin-bottom': '0.75in',
			'margin-left': '0.75in',
		}

		pdfkit.from_string(full_html, f"Report_{now}.pdf", options=options)

	def response_to_markdown(self):
		respuesta = self._agent_state["respuesta"]
		respuesta = respuesta.strip()
		now = datetime.now()
		with open(f"Reporte_{now}.md", "w") as f:
			f.write(respuesta)
