import json
import os
from time import sleep

from dotenv import load_dotenv
from google.genai.types import ProminentPeople
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)
from prompt_toolkit import prompt

from models.agent import FederalLaborAgent
from models.search_web import BaseSearchModel
from models.vector_storage import VectorStorage


class SerpAPIModel(BaseSearchModel):
	def __init__(self, model):
		self._model = model

	def search(self, query: str):
		return self._model.run(query)

class AgentCommandLine:
	def __init__(self) -> None:
		self._agent: FederalLaborAgent
		self._settings: dict

		self._get_settings()

	def run(self) -> None:
		embeddings = GoogleGenerativeAIEmbeddings(
			model=self._settings["embedding_model"]
		)
		vector_store = VectorStorage(embeddings)
		vector_store.load_existing_vector_store(self._settings["vector_stage_path"])
		retriever = vector_store.as_retriever({
			"k" : self._settings["k_retriever"]
		})

		web_searcher = SerpAPIModel(SerpAPIWrapper())
		chat_model = ChatGoogleGenerativeAI(
			model=self._settings["llm_model"],
			temperature=0.2
		)

		self._agent = FederalLaborAgent(
			chat_model,
			retriever,
			web_searcher
		)

		self._agent.build_graph()

	def exec(self):
		print("\n**Bienvenido a la consola del Agente de IA**")
		print("Responderé las preguntas que tengas respecto a la Ley Federal de Trabajo (LFT)")

		print("\nAcciones que puedes tomar:")
		print("  exit     Para salir de la consola.")
		print("  export   Para iniciar el motor para exportar la última respuesta.")
		print("           Solo soporta salida para PDF y Markdown.\n")

		print("\nComenzamos!\n")

		while True:
			user_input = prompt(">>> ").strip()

			if user_input.lower() == "exit":
				break
			elif user_input.lower() == "export":
				self._export_menu()
				continue

			print("Procesando...")
			self._agent.execute(user_input)
			print("Listo!\n\nRespuesta:\n")
			for char in self._agent.answer:
				print(char, end="", flush=True)
				sleep(0.01)
			print()
			print()

		print("\nAdios!\n")

	def _get_settings(self):
		default_settings = {
			"embedding_model" : "models/gemini-embedding-001",
			"llm_model" : "gemini-2.5-flash",
			"vector_stage_path" : "vector_store/",
			"k_retriever" : 4,
		}

		file_name = "settings/agent.json"
		if not os.path.isdir(os.path.dirname(file_name)):
			os.mkdir(os.path.dirname(file_name))

		if os.path.isfile(file_name):
			with open(file_name, "r") as f:
				self._settings = json.load(f)
		else:
			self._settings = default_settings
			with open(file_name, "w", encoding="utf-8") as f:
				json.dump(self._settings, f, indent=4)

	def _export_menu(self):

		user_input = prompt("Deseas exportar a PDF (ingresa pdf) o Markdown (ingresa md)? Ingresa otra cosa para salir>> ")
		user_input = user_input.strip().lower()

		if user_input == "pdf":
			self._agent.response_to_pdf()
		elif user_input == "md":
			self._agent.response_to_markdown()
		else:
			print("No se exportó nada")
			return

		print("Se ha creado el reporte en la carpeta raíz.")


if __name__ == "__main__":
	load_dotenv()

	app = AgentCommandLine()
	app.run()
	app.exec()