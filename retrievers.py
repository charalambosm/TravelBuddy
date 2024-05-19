from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
import requests
from os import getenv
from langchain_core.retrievers import BaseRetriever
from typing import List
import itertools
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

class GooglePlaceSearchRetriever(BaseRetriever):
    """
    A retriever class for fetching information about places using the Google Places API.
    
    Attributes:
        k (int): Maximum number of results to return.
    """
    k: int = 20
    """Maximum number of results to return"""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Retrieve relevant documents based on a query.

        Args:
            query (str): The search query.
            run_manager (CallbackManagerForRetrieverRun): Manager for handling callback operations during retrieval.

        Returns:
            List[Document]: A list of Document objects containing information about relevant places.
        """
        # Adapt query to google place search input
        queries = self._adapt_query_for_google_place_search(query)
        
        # Search google places
        results = [self._search_google_places(query) for query in queries]
        
        # Keep a maximum of k results (k // number of places types for each type)
        no_search_results = self.k // len(queries)
        results = [result[:no_search_results] for result in results]

        # Merge lists of results
        results = list(itertools.chain.from_iterable(results))

        # Create Document objects and return
        return self._create_docs_from_search_results(results)

    def _adapt_query_for_google_place_search(self, query: str) -> List[str]:
        """
        Adapt a user query for Google Place Search.

        Args:
            query (str): The user query.

        Returns:
            List[str]: A list of adapted queries suitable for Google Place Search.
        """
        prompt = ChatPromptTemplate.from_template(
            """
            Given an input form extract the types of
            places and location as specified in the instructions.

            Input:
            ####
            {input}
            ####

            Instructions:
            ####
            {instructions}
            ####
            """
        )
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        output_parser = JsonOutputParser(pydantic_object=self.GooglePlaceSearchQueryModel)
        chain = prompt | model | output_parser
        response = chain.invoke({
            "input": query,
            "instructions": output_parser.get_format_instructions()
        })
        return [place_type + " in " + response["location"] for place_type in response["types"]]

    class GooglePlaceSearchQueryModel(BaseModel):
        """
        A Pydantic model for parsing the query results from the language model.
        
        Attributes:
            types (List[str]): List of place types mentioned in the input.
            location (str): The location mentioned in the input.
        """
        types: List[str] = Field(description="What are the types of places mentioned in the input? For example: ['Museum', 'Art gallery'].")
        location: str = Field(description="Which city and country is mentioned in the input? For example: 'Paris,France'.")

    def _search_google_places(self, searchQuery: str) -> dict:
        """
        Fetch information about a set of places for a given search query.

        Args:
            searchQuery (str): The search query.

        Returns:
            dict: A dictionary containing information about places returned by the Google Places API.
        """
        BASE_URL = "https://places.googleapis.com/v1/places:searchText"

        params = {
            "textQuery": searchQuery,
            "openNow": "false",
            "key": getenv("GOOGLEMAPS_API_KEY"),
        }

        headers = {
            "accept": "application/json",
            "X-Goog-FieldMask": "places.displayName.text,places.id,places.formattedAddress"
        }

        response = requests.post(BASE_URL, headers=headers, params=params)

        return response.json()['places']

    def _create_docs_from_search_results(self, results: List[dict]) -> List[Document]:
        """
        Create Document objects from search results.

        Args:
            results (List[dict]): A list of dictionaries containing search results.

        Returns:
            List[Document]: A list of Document objects containing the processed search results.
        """
        return [Document(
            page_content=result["displayName"]["text"],
            metadata={
                "google_id": result["id"],
                "address": result["formattedAddress"]
            } )
            for result in results]