from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from place import PlaceModel
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from retrievers import GooglePlaceSearchRetriever
from typing import List
from langchain_core.runnables.base import Runnable

class Chain(ABC):
    """
    Abstract base class for defining a chain of operations to process and retrieve information.
    
    Attributes:
        system_message (SystemMessagePromptTemplate): System message prompt template for the chat.
    """
    system_message = SystemMessagePromptTemplate.from_template(
        """ 
        You are a helpful travel agent. Your job is to recommend places to visit in travel destinations.
        """
    )

    @property
    @abstractmethod
    def human_message(self) -> HumanMessagePromptTemplate:
        """
        Abstract property for the human message prompt template.

        Returns:
            HumanMessagePromptTemplate: Human message prompt template.
        """
        pass

    @property
    @abstractmethod
    def llm(self) -> Runnable:
        """
        Abstract property for the language model.

        Returns:
            Runnable: Language model instance.
        """
        pass

    @property
    @abstractmethod
    def output_parser(self) -> Runnable:
        """
        Abstract property for the output parser.

        Returns:
            Runnable: Output parser instance.
        """
        pass

    @property
    def prompt(self) -> ChatPromptTemplate:
        """
        Property to get the combined prompt template for the chat.

        Returns:
            ChatPromptTemplate: Combined chat prompt template.
        """
        return ChatPromptTemplate.from_messages([
            self.system_message, 
            self.human_message
        ])

    @property
    @abstractmethod
    def chain(self) -> Runnable:
        """
        Abstract property for the chain of operations.

        Returns:
            Runnable: A LangChain chain.
        """
        pass

    @abstractmethod
    def invoke(self):
        """
        Abstract method to invoke the chain.

        Returns:
            Any: Result of invoking the chain.
        """
        pass


class RetrievalChain(Chain):
    """
    Concrete implementation of the Chain class for retrieving travel recommendations.

    Args:
        location (str): The travel location.
        previous_places (List[str]): List of places previously visited.
        interests (List[str]): List of interests for travel recommendations.
    """
    def __init__(self, location: str, previous_places: List[str], interests: List[str]):
        self._location = location
        self._previous_places = previous_places
        self._interests = interests

    @property
    def human_message(self):
        """
        Property to get the human message prompt template.

        Returns:
            HumanMessagePromptTemplate: Human message prompt template.
        """
        prompt = PromptTemplate(
            template = """
        Answer the following question based only on the following context:

        QUESTION:
        ####
        {input}
        Do not include {previous_places}.
        ####

        CONTEXT:
        ####
        {context}
        ####

        INSTRUCTIONS:
        ####
        {instructions}
        ####
        """,
        input_variables=[],
        partial_variables={
            "previous_places": self._previous_places,
            "instructions": self.output_parser.get_format_instructions()
        }
        )
        return HumanMessagePromptTemplate(prompt=prompt)
    
    @property
    def llm(self):
        """
        Property to get the language model.

        Returns:
            ChatOpenAI: Language model instance.
        """
        return ChatOpenAI(
            model = "gpt-3.5-turbo",
            temperature = 0.0
        )
    
    @property
    def output_parser(self):
        """
        Property to get the output parser.

        Returns:
            JsonOutputParser: Output parser instance.
        """
        return JsonOutputParser(
            pydantic_object=PlaceModel
        )
    
    def _create_interests_string(self):
        """
        Create a string from the list of interests.

        Returns:
            str: Formatted string of interests.
        """
        interests = [x.lower() for x in self._interests]
        if len(interests) == 1:
            return interests[0]
        else:
            return ', '.join(interests[:-1]) + ' or ' + interests[-1]

    @property
    def question(self):
        """
        Property to get the formatted question for the retriever.

        Returns:
            str: Formatted question string.
        """
        return f"""Recommend a {self._create_interests_string()} to visit in {self._location}."""

    @property
    def retriever(self):
        """
        Property to get the retriever instance.

        Returns:
            GooglePlaceSearchRetriever: Instance of the retriever.
        """
        return GooglePlaceSearchRetriever()

    @property
    def chain(self):
        """
        Property to get the chain of operations.

        Returns:
            Chain: Chain of operations for retrieval and combination.
        """
        combine_docs_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt,
            output_parser=self.output_parser
        )
        return create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_docs_chain
        )
    
    def invoke(self):
        """
        Invoke the chain to get travel recommendations.

        Returns:
            Any: Result of the chain invocation, typically the answer to the travel question.
        """
        return self.chain.invoke({"input":self.question})['answer']