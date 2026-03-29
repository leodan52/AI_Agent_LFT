from datetime import datetime
from typing import TypedDict
import re

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
	"""
	Orchestrator for the Mexican Federal Labor Law (LFT) AI Agent.

	This class manage the lifecycle of a RAG-based workflow. It integrates
	language models with vector databases to provide legal insights,
	specifically focused on Mexican Labor Regulations documentation.

	Properties:
			answer (str): The final response generad by the LLM

	Methods:
			build_graph() -> None: Construct AI Agent workflow graph
			execute(query: str) -> dict: Run the Agent with an input query
									and return the complete workflow data.
	"""

	def __init__(
		self,
		chat_model: BaseChatModel,
		retriever_rag: BaseRetriever,
		search_class: BaseSearchModel,
	) -> None:
		"""Class Constructor

		Args:
				chat_model (BaseChatModel): The LLM provider used for
						generating responses.
				retriever_rag (BaseRetriever): The retrieval interface
						connected to the vector store
				search_class (BaseSearchModel): The Search Model class that
						encapsulates the Web Search API logic.
		"""
		self._agent_prompts = AgentPrompts()
		self._chat_model = chat_model
		self._retriever_rag = retriever_rag
		self._search_class = search_class

		self._graph = StateGraph(AgentState)
		self._agent_state = dict()
		self._app: CompiledStateGraph

	@property
	def answer(self) -> str:
		"""AI Agent's final response

		Returns:
				str: The response formatted like-Markdown syntax
		"""
		try:
			return self._agent_state["respuesta"]
		except KeyError:
			return "No hay respuesta aún"

	def build_graph(self) -> None:
		"""
		Build the AI Agent workflow using a like-graph structure.

		This method initializes the StateGraph, adds nodes for agent logic,
		and defines the edges that control the execution flow.
		"""
		self._graph.add_node("Agent", self._agent_node)
		self._graph.add_node("RAG", self._rag_node)
		self._graph.add_node("WEB", self._search_node)
		self._graph.add_node("ERROR", self._error_classifier_node)
		self._graph.add_node("Generator", self._generate_node)

		self._graph.add_edge(START, "Agent")
		self._graph.add_conditional_edges(
			"Agent", self._choice_source, {"RAG": "RAG", "WEB": "WEB", "ERROR": "ERROR"}
		)
		self._graph.add_edge("RAG", "Generator")
		self._graph.add_edge("WEB", "Generator")
		self._graph.add_edge("ERROR", END)
		self._graph.add_edge("Generator", END)

		self._app = self._graph.compile()

	def execute(self, query: str) -> dict:
		"""
		Execute the AI Agent workflow with a given input query.

		Args:
				query (str): The user's natural language question or instruction

		Returns:
				dict: The final state of the workflow contained the answer,
						context and metadata
		"""

		resultado = self._app.invoke(
			{
				"pregunta": self._clean_text(query),
				"fuente": "",
				"contexto": "",
				"respuesta": "",
				"bibliografia": "",
				"metadata_respuesta": dict(),
			}
		)

		self._agent_state = resultado

		return resultado

	def response_to_pdf(self) -> None:
		"""
		Export the last AI Agent response to a PDF file.

		The file will be saved in the main project path using the
		following naming convention: 'Reporte_{datetime_data}.pdf'

		"""
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
			"encoding": "UTF-8",
			"margin-top": "0.75in",
			"margin-right": "0.75in",
			"margin-bottom": "0.75in",
			"margin-left": "0.75in",
		}

		pdfkit.from_string(full_html, f"Report_{now}.pdf", options=options)

	def response_to_markdown(self) -> None:
		"""
		Export the last AI Agent response to a Markdown file.

		The file will be saved in the main project path using the
		following naming convention: 'Reporte_{datetime_data}.md'
		"""
		respuesta = self._agent_state["respuesta"]
		respuesta = respuesta.strip()
		now = datetime.now()
		with open(f"Reporte_{now}.md", "w") as f:
			f.write(respuesta)

	def _agent_node(self, state: AgentState) -> dict:
		"""
		Agent node. This node use the LLM provider to decide whether the workflow
		should use RAG or Web Search to obtain the context.

		Args:
				state (AgentState): The current workflow state

		Raises:
				TypeError: If the LLM returns data that is not a string

		Returns:
				dict: The updated workflow state
		"""
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

	def _rag_node(self, state: AgentState) -> dict:
		"""
		RAG node. This node searches the vector store within our file database
		to obtain relevant context for the query.

		Args:
				state (AgentState): The current workflow state

		Returns:
				dict: The updated workflow state. Add retrieved context and source.
		"""
		pregunta = state["pregunta"]

		docs = self._retriever_rag.invoke(pregunta)
		contexto = "\n\n---\n\n".join(
			self._clean_text(doc.page_content) for doc in docs
		)
		bibliografia = self._get_metadata(docs)
		# contexto += "\n\n---\n\n"*2 + f"Bibliografía:\n{self._get_metadata(docs)}"

		return {"contexto": contexto, "bibliografia": bibliografia}

	def _search_node(self, state: AgentState) -> dict:
		"""
		Search node. This node searches the web for relevant context to answer
		the user query when local documentation is insufficient.

		Args:
				state (AgentState): The current workflow state

		Returns:
				dict: The updated workflow state. Add web-based context.
		"""
		contexto = self._search_class.search(state["pregunta"])

		return {"contexto": contexto, "bibliografia": ""}

	def _error_classifier_node(self, state: AgentState) -> dict:
		"""
		Error handling node. This node manages cases where the user input is ambiguos,
		nonsensical, or out of the agent's scope.

		Args:
				state (AgentState): The current workflow state

		Returns:
				dict: The updated workflow state. Add a fallback responses.
		"""
		respuesta = "La pregunta realizada es ambigua. Realiza otra con más información"

		return {"respuesta": respuesta, "bibliografia": ""}

	def _generate_node(self, state: AgentState) -> dict:
		"""
		Generate node. This node sends the user query and the retrieved context to
		the LLM provider to generate a final responses in Markdown syntax.

		Args:
				state (AgentState): The current workflow state

		Returns:
				dict: The updated workflow state. Add the final response from the LLM.
		"""
		contexto = state["contexto"]
		pregunta = state["pregunta"]
		bibliografia = state["bibliografia"]

		if bibliografia:
			contexto += f"\nBibliografía:\n{bibliografia}\n"

		prompt = self._agent_prompts.PROMPT_GENERATE.replace(
			"$CONTEXTO$", contexto
		).replace("$PREGUNTA$", pregunta)

		messege = self._chat_model.invoke(prompt)

		return {
			"respuesta": messege.content,
			"metadata_respuesta": messege.usage_metadata,
		}

	def _choice_source(self, state: AgentState) -> str:
		"""
		Routing function for conditional edges.

		This method reads the 'fuente' field from the current state to
		determine the next node in the workflow: RAG, WEB, or ERROR.

		Args:
				state (AgentState): The current workflow state

		Returns:
				str: The name of the next node to execute.
		"""
		if state["fuente"] == "RAG":
			return "RAG"
		elif state["fuente"] == "WEB":
			return "WEB"
		else:
			return "ERROR"

	@staticmethod
	def _get_metadata(docs: list[Document]) -> str:
		"""
		Extract metadata from a list of retrieved documents.

		This method identifies the file names and page numbers from
		the documents obtained through similarity search.

		Args:
				docs (list[Document]): A List of Document objects from the vector
						store search.

		Returns:
				str: A formatted string listing files a pages, e.g._
							>	* Del archivo <file_name.pdf> en las páginas {<list>}
		"""
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
			[f"	* Del archivo '{k}' en las páginas {str(v)}" for k, v in biblio.items()]
		)

		return lineas

	@staticmethod
	def _clean_text(text: str) -> str:
		"""
		Sanitize and normalize input text.

		This method removes non-ASCII characters, collapses multiple
		whitespaces or line breaks, and prunes repetitive character
		sequences to optimize the context for the LLM.

		Args:
			text (str): The raw text string to be processed.

		Returns:
			str: The cleaned and normalized text.
		"""
		text = re.sub(r"[^\x00-\x7F]+", " ", text)
		text = re.sub(r"\n+", "\n", text)
		text = re.sub(r" +", " ", text)
		text = re.sub(r"(.+?)\1{2,}", r"\1", text)

		return text.strip()
