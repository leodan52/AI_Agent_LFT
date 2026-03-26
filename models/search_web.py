from abc import ABC, abstractmethod


class BaseSearchModel(ABC):
	"""
	Abstract base class for custom web search providers. Subclasses
	must encapsulate a search model and implement the search method.
	"""

	@abstractmethod
	def search(self, query: str):
		"""This method must be implemented by child classes to return
		   web search results.

		Args:
				query (str): The search term or question to be process
		"""
		pass
