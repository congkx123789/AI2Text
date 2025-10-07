from PIL import Image
import numpy as np

def load_and_flatten(img_path, size=(28,28)):
    img = Image.open(img_path).convert("L").resize(size)
    return np.array(img).flatten() / 255.0
