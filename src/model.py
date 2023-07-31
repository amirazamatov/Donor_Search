from detected_table import extract_image
from extract_text import extract_data

def process_image(file: str):
    image = extract_image(file)
    df = extract_data(image)
    return df
