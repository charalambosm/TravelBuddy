from chains import RetrievalChain

# Specify location and interests
location = "Athens, Greece"
interests = ["Cocktail bar, museum"]

# Create an instrance of the RAG chain
# Specify no previous places
chain = RetrievalChain(
    location=location,
    previous_places=[],
    interests=interests
)

# Invoke chain and print result
result = chain.invoke()
print(
    f"""
    Place:    {result["place"]}
    Address:  {result["address"]}
    Category: {result["category"]}
    """
    )
