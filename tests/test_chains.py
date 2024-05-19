import pytest
from unittest.mock import patch, Mock
from chains import RetrievalChain, PlaceModel

def test_initialization():
    location = "Paris"
    previous_places = ["Louvre", "Eiffel Tower"]
    interests = ["museum", "art gallery"]
    chain = RetrievalChain(location, previous_places, interests)
    
    assert chain._location == location
    assert chain._previous_places == previous_places
    assert chain._interests == interests

def test_llm():
    location = "Paris"
    previous_places = ["Louvre", "Eiffel Tower"]
    interests = ["museum", "art gallery"]
    chain = RetrievalChain(location, previous_places, interests)

    with patch('chains.ChatOpenAI') as mock_chat_openai:
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm

        llm = chain.llm
        assert llm == mock_llm
        mock_chat_openai.assert_called_once_with(model="gpt-3.5-turbo", temperature=0.0)

def test_output_parser():
    location = "Paris"
    previous_places = ["Louvre", "Eiffel Tower"]
    interests = ["museum", "art gallery"]
    chain = RetrievalChain(location, previous_places, interests)

    with patch('chains.JsonOutputParser') as mock_json_output_parser:
        mock_parser = Mock()
        mock_json_output_parser.return_value = mock_parser

        output_parser = chain.output_parser
        assert output_parser == mock_parser
        mock_json_output_parser.assert_called_once_with(pydantic_object=PlaceModel)

def test_create_interests_string():
    chain = RetrievalChain("Paris", [], ["museum"])
    assert chain._create_interests_string() == "museum"

    chain = RetrievalChain("Paris", [], ["museum", "art gallery"])
    assert chain._create_interests_string() == "museum or art gallery"

    chain = RetrievalChain("Paris", [], ["museum", "art gallery", "park"])
    assert chain._create_interests_string() == "museum, art gallery or park"

def test_question():
    chain = RetrievalChain("Paris", [], ["museum"])
    assert chain.question == "Recommend a museum to visit in Paris."

    chain = RetrievalChain("Paris", [], ["museum", "art gallery"])
    assert chain.question == "Recommend a museum or art gallery to visit in Paris."

    chain = RetrievalChain("Paris", [], ["museum", "art gallery", "park"])
    assert chain.question == "Recommend a museum, art gallery or park to visit in Paris."

def test_retriever():
    location = "Paris"
    previous_places = ["Louvre", "Eiffel Tower"]
    interests = ["museum", "art gallery"]
    chain = RetrievalChain(location, previous_places, interests)

    with patch('chains.GooglePlaceSearchRetriever') as mock_retriever:
        mock_retriever_instance = Mock()
        mock_retriever.return_value = mock_retriever_instance

        retriever = chain.retriever
        assert retriever == mock_retriever_instance
        mock_retriever.assert_called_once()

def test_chain():
    location = "Paris"
    previous_places = ["Louvre", "Eiffel Tower"]
    interests = ["museum", "art gallery"]
    chain = RetrievalChain(location, previous_places, interests)

    with patch('chains.create_stuff_documents_chain') as mock_create_stuff_chain, \
         patch('chains.create_retrieval_chain') as mock_create_retrieval_chain:
        mock_docs_chain = Mock()
        mock_retrieval_chain = Mock()
        mock_create_stuff_chain.return_value = mock_docs_chain
        mock_create_retrieval_chain.return_value = mock_retrieval_chain

        chain_obj = chain.chain
        assert chain_obj == mock_retrieval_chain
        mock_create_stuff_chain.assert_called_once_with(llm=chain.llm, prompt=chain.prompt, output_parser=chain.output_parser)
        mock_create_retrieval_chain.assert_called_once_with(retriever=chain.retriever, combine_docs_chain=mock_docs_chain)

@pytest.mark.parametrize("input",[
    ({
        "location": "Paris, France",
        "previous_places": [],
        "interests": ["museum", "art gallery"]
    }),
    ({
        "location": "Paris, France",
        "previous_places": ["Louvre", "Eiffel Tower"],
        "interests": ["museum", "art gallery"]        
    }),
    ({
        "location": "Athens, Greece",
        "previous_places": ["Voodoo"],
        "interests": ["cocktail bar", "night club"]        
    }),
    ({
        "location": "Rome, Italy",
        "previous_places": ["Colosseum"],
        "interests": ["landmark"]        
    }),
    ({
        "location": "New York, United States",
        "previous_places": [],
        "interests": ["restaurant"]        
    }),            
    ])
def test_prompt_invoke(input):
    chain = RetrievalChain(
        location = input["location"],
        previous_places = input["previous_places"],
        interests = input["interests"]
    )

    result = chain.chain.invoke({"input": chain.question})['answer']

    # Assert that the result is a dict
    assert isinstance(result, dict)
    # Assert that it has keys 'place' and 'category'
    assert 'place' in result
    assert 'category' in result
    assert 'address' in result
    # Assert that 'category' is either 'museum' or 'art gallery'
    assert result['category'].lower() in input["interests"]
    # Assert that 'place' is not in 'previous_places'
    assert result['place'] not in input["previous_places"] 