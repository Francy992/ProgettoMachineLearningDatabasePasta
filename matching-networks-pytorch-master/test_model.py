from matching_networks import MatchingNetwork
import torch
from PIL import Image

model = MatchingNetwork(0.0, 20, 1, 1e-3, True, 20, 1, 28, False)

model.load_state_dict(torch.load("model.pth"))
img = Image.open("./w.png")
model(img, img, img, img)

print(model)