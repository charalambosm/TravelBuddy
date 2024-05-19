# Meet your new travel buddy!

## Overview
This repository contains a travel recommendation system that uses the Google Places API (exploiting Retrieval Augmented Generation) and ChatGPT to suggest places to visit based on user queries. The system is designed to help you to recommend travel destinations and points of interest tailored to users' preferences and previous travel history.

## Features
- **Google Places Search Integration**: Utilizes the Google Places API to search for places based on user queries.
- **Language Model Processing**: Leverages the GPT-3.5-turbo model to process natural language queries and generate formatted responses.
- **Flexible Query Adaptation**: Transforms user queries into search-friendly formats for the Google Places API.
- **Customizable Recommendations**: Takes into account user interests and previously visited places to provide personalized recommendations.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/travel-recommendation-system.git
    cd travel-recommendation-system
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    Ensure you have a valid Google Maps API key. Create a `.env` file in the root directory and add your API key:
    ```env
    GOOGLEMAPS_API_KEY=your_google_maps_api_key
    ```

## Usage

Define a RetrievalChain and invoke it to get a recommendation. An example is included in the `run.py` script.
```python
from chains import RetrievalChain

location = "Paris, France"
previous_places = ["Louvre Museum"]
interests = ["art", "history"]

chain = RetrievalChain(location, previous_places, interests)
recommendation = chain.invoke()
print(recommendation)
```

## Classes and Methods

### RetrievalChain

This class defines a LangChain chain to process user queries and retrieve relevant travel recommendations.

- `__init__(location: str, previous_places: List[str], interests: List[str])`: Initializes the RetrievalChain with location, previous places, and interests.
- `human_message`: Property to get the human message prompt template.
- `llm`: Property to get the language model instance.
- `output_parser`: Property to get the output parser instance.
- `_create_interests_string()`: Creates a string from the list of interests.
- `question`: Property to get the formatted question for the retriever.
- `retriever`: Property to get the retriever instance.
- `chain`: Property to get the chain of operations for retrieval and combination.
- `invoke()`: Invokes the chain to get travel recommendations.

### GooglePlaceSearchRetriever

This class is responsible for interacting with the Google Places API to fetch information about places based on a search query.

- `k`: Maximum number of results to return.
- `_get_relevant_documents(query: str, run_manager: CallbackManagerForRetrieverRun)`: Retrieves relevant documents based on a query.
- `_adapt_query_for_google_place_search(query: str)`: Adapts a user query for Google Place Search.
- `_search_google_places(searchQuery: str)`: Fetches information about a set of places for a given search query.
- `_create_docs_from_search_results(results: List[dict])`: Creates Document objects from search results.

## Contributing
Contributions are welcome! If you have any suggestions, issues, or would like to contribute, please contact me at [cmaxoutis79@gmail.com](mailto:cmaxoutis79@gmail.com).

## Acknowledgements
- [LangChain](https://www.langchain.com/): For providing a robust framework for building language model applications.
- [OpenAI](https://www.openai.com/): For the GPT-3.5-turbo model.
- [Google Places API](https://developers.google.com/maps/documentation/places/web-service/overview): For providing place search functionality.
