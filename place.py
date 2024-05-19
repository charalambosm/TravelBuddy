from pydantic import BaseModel, Field

_SIGHT_INTERESTS = [
    'Art gallery', 'Museum', 'Landmark', 'Shopping area', 'Outdoor activity'
]

_FOOD_INTERESTS = [
    'Restaurant', 'Cafe', 'Brasserie'
]

_NIGHT_INTERESTS = [
    'Cocktail bar', 'Club'
]

class PlaceModel(BaseModel):
    place: str = Field(description="What is the name of the place?")
    category: str = Field(
        description=
        f"""
        What is the type of the place? Pick one of the following:
        '{"', '".join(_SIGHT_INTERESTS)}', '{"', '".join(_FOOD_INTERESTS)}', '{"', '".join(_NIGHT_INTERESTS)}'.
        """)
    address: str = Field(description="What is the address of the place?")