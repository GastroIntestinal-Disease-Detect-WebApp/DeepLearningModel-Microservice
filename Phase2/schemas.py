from pydantic import BaseModel

class ImageLinkInput(BaseModel):
    image_link: str