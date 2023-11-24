from model.resnet18 import ResNet18
from torchinfo import summary

# details = summary(ResNet18(1), input_size=(64,1,224,224))

# print(details)

from torchvision import transforms
from PIL import Image
import os




trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop((224,224), scale=(0.1,1)),
    transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5),
    transforms.ToTensor(),transforms.ToPILImage()])

path = os.path.join(os.getcwd(), 'data/classify-leaves/images/0.jpg')
img = Image.open(path).convert('RGB')

img_trans = trans(img)
img_trans.show()
