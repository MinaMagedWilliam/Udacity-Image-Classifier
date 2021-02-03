import argparse
import json
import torch
from torchvision import transforms, models
import PIL as Image
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store',
                        default='checkpoint.pth',
                        help='type the checkpoint')
    parser.add_argument('--gpu', default='cuda',
                        action='store', dest='gpu',
                        help='type cpu or gpu')
    parser.add_argument('--path', dest='path',
                        help='type the image path',
                        required='true')
    parser.add_argument('--top_k', dest='top_k', default='3',
                        help='chose top k', type=int)
    parser.add_argument('--category_names', dest='cat',
                        default='cat_to_name.json',
                        help='type category file')
    return parser.parse_args()

args = get_args()

if args.gpu == "cuda":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
else:
    device = "cpu"


def process_image(image):
    img_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    img = img_transforms(Image.open(image))
    return img


def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')
    model=models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    return model

model = load_checkpoint(args.checkpoint)  



def predict(image_path, model, topk=args.top_k):

    model.to(device)
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.from_numpy(np.array([img])).float()
    img = img.to(device)
    with torch.no_grad():
        logps = model.forward(img.cuda())
    probability = torch.exp(logps).data
    return probability.topk(topk)

with open(args.cat, 'r') as f:
    cat_to_name = json.load(f)


def prediction():
    ps = predict(args.path, model)
    a = np.array(ps[0][0])
    b = [cat_to_name[str(index+1)] for index in np.array(ps[1][0])]
    for label in range(len(labels)):
        x = round(a[label]*100, 2)
        print("{b[label]} with a probability of {x}")

prediction()
