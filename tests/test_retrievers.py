from retrievers import GooglePlaceSearchRetriever
import pytest
from langchain_core.documents import Document

## Tests for _adapt_query_for_google_place_search method
@pytest.mark.parametrize(
        "input, expected_output",
        [
            (
                "Recommend a museum or art gallery to visit in Paris, France.",
                {
                    "types": ["museum", "art gallery"], 
                    "location":"paris, france",
                    "length": 2,
                }
            ),
            (
                "Recommend a theatre to visit in Athens, Greece",
                {
                    "types": ["theatre"],
                    "location": "athens, greece",
                    "length": 1
                }
            ),
            (
                "Recommend a restaurant to visit in London, United Kingdom",
                {
                    "types": ["restaurant"],
                    "location": "london, united kingdom",
                    "length": 1
                }
            ),
            (
                "Recommend a museum, landmark or shopping area to visit in Rome, Italy.",
                {
                    "types": ["museum", "landmark", "shopping area"],
                    "location": "rome, italy",
                    "length": 3
                }
            )
        ]
)
def test_adapt_query_for_google_place_search(input, expected_output):
    retriever = GooglePlaceSearchRetriever()
    actual_output = retriever._adapt_query_for_google_place_search(input)

    assert len(actual_output) == expected_output["length"]

    for output in actual_output:
        found = [place_type in output.lower() for place_type in expected_output["types"]]
        assert sum(found) == 1
        assert expected_output["location"] in output.lower()


## Tests for _search_google_places method
@pytest.mark.parametrize(
        "input",
        [
            ("restaurant in Paris, France"),
            ("theatre in Paris, France"),
            ("museum in Barcelona, Spain"),
            ("art gallery in New York, United States of America"),
            ("landmark in London, United Kingdom"),
            ("bar in Rome, Italy"),
            ("night club in Athens, Greece")
        ]
)
def test_search_google_places(input):
    retriever = GooglePlaceSearchRetriever()
    output = retriever._search_google_places(input)

    # assert the output is a list of dict objects
    assert isinstance(output, list)
    assert all(isinstance(place, dict) for place in output)

    # assert the output contains the correct keys (id, displayName, displayName.text, formattedAddress)
    for place in output:
        assert 'id' in place
        assert 'displayName' in place
        assert 'text' in place['displayName']
        assert 'formattedAddress' in place


## Tests for _create_docs_from_search_results method
@pytest.mark.parametrize(
    "input,expected_output",
    [
        (
            [{'id': 'ChIJ1Ur6S0VgLxMR5KB4WcXKA-s', 'displayName': {'text': 'La Botticella of Poggi Giovanni'},'formattedAddress':'mockAddress'}],
            [Document(page_content='La Botticella of Poggi Giovanni', metadata={'google_id': 'ChIJ1Ur6S0VgLxMR5KB4WcXKA-s','address': 'mockAddress'})]
        ), 
        (
            [
                {'id': 'ChIJq6NczaVhLxMR9L2mPSrTd5g', 'displayName': {'text': 'Apotheke Cocktail Bar'}, 'formattedAddress': 'mockAddress1'},
                {'id': 'ChIJ2Ti8x_hgLxMRk3c0LjnarPI', 'displayName': {'text': 'Stravinskij Bar'}, 'formattedAddress': 'mockAddress2'}
            ],
            [
                Document(page_content='Apotheke Cocktail Bar', metadata={'google_id': 'ChIJq6NczaVhLxMR9L2mPSrTd5g', 'address': 'mockAddress1'}),
                Document(page_content='Stravinskij Bar', metadata={'google_id': 'ChIJ2Ti8x_hgLxMRk3c0LjnarPI', 'address': 'mockAddress2'})
            ]
        ),
    ]
)
def test_create_docs_from_search_results(input, expected_output):
    retriever = GooglePlaceSearchRetriever()
    output = retriever._create_docs_from_search_results(input)

    # assert the class of the output is a list of Document
    assert isinstance(output, list)
    assert all(isinstance(doc, Document) for doc in output)

    # assert the number of Document objects is equal to the length of the input
    assert len(output) == len(input)

    # assert page_content and metadata are correct
    assert output == expected_output


## Tests for the GooglePlaceSearchRetriever invoke method
@pytest.mark.parametrize(
    "input",[
    ("Recommend a museum or art gallery to visit in Paris, France."),
    ("Recommend a theatre to visit in Athens, Greece"),
    ("Recommend a restaurant to visit in London, United Kingdom"),
    ("Recommend a museum, landmark or shopping area to visit in Rome, Italy."),
])
def test_GooglePlaceSearchRetriever_invoke(input):
    retriever = GooglePlaceSearchRetriever(k=20)
    output = retriever.invoke(input)

    assert len(output) <= 20