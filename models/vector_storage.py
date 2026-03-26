import os
from math import floor
from time import sleep

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStorage:

	def __init__(self, embedding_model):
		self._path_files: str
		self._vectors_store: FAISS

		self._embeddings: Embeddings = embedding_model

	@property
	def vector_store(self):
		return self._vectors_store

	def as_retriever(self, search_kwargs):
		return self.vector_store.as_retriever(
			search_kwargs=search_kwargs
		)

	def generate_embeddings(
		self, path_pdf_files: str, rpm: int = -1, chunk_size=400, chunk_overlap=40
	):
		self._path_files = path_pdf_files

		documents = self._load_pdf_files()
		chunks = self._create_chunks(documents, chunk_size, chunk_overlap)

		del documents

		print(f"\nEl número de chunks obtenidos es de: {len(chunks)}")
		print(f"Para:\n\t{chunk_size = }\n\t{chunk_overlap = }\n")

		if (rpm <= 0) or (len(chunks) < rpm):
			print("Sin limite")
			self._vectors_store = FAISS.from_documents(
				documents=chunks, embedding=self._embeddings
			)
		else:
			print(f"Creando embeddings: Tiempo estimado: {len(chunks)/rpm} min")
			clusters = self._rpm_limits(len(chunks), rpm)

			self._vectors_store = FAISS.from_documents(
				documents=chunks[0 : clusters[0]], embedding=self._embeddings
			)

			aux = clusters[0] + 1
			for i in clusters[1:]:
				sleep(60)
				self._vectors_store.merge_from(
					FAISS.from_documents(
						documents=chunks[aux:i], embedding=self._embeddings
					)
				)

				aux = i + 1

		del chunks

	def load_existing_vector_store(self, path_data: str = "data/"):
		self._vectors_store = FAISS.load_local(
			path_data, self._embeddings, allow_dangerous_deserialization=True
		)

	def store_vectors(self, path_data: str = "data/"):
		self._vectors_store.save_local(path_data)

	def _load_pdf_files(self):
		documents: list[Document] = []

		for root, _, files in os.walk(self._path_files):
			for f in files:
				if not f.endswith(".pdf"):
					continue

				ruta = os.path.join(root, f)
				loader = PyPDFLoader(ruta)
				paginas = loader.load()
				documents.extend(paginas)

		return documents

	def _create_chunks(
		self,
		documents,
		chunk_size,
		chunk_overlap,
		separators=["\n\n", "\n", ". ", " ", ""],
	):

		divisor = RecursiveCharacterTextSplitter(
			chunk_size=chunk_size,  # Tamaño del fragmento
			chunk_overlap=chunk_overlap,  # Superposicion de los fragmentos
			separators=separators,
		)

		chunks: list[Document] = divisor.split_documents(documents)

		return chunks

	@staticmethod
	def _rpm_limits(len_chunks, rpm):
		len_step = floor(rpm * 0.8)

		output = []
		aux = len_step
		while True:
			if aux <= len_chunks:
				output.append(aux - 1)
				aux += len_step
			else:
				output.append(len_chunks)
				break

		return output
