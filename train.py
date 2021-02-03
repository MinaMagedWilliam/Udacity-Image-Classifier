import argparse
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, action='store',
                        dest="ddir", help='type the data directory')
    parser.add_argument('--epochs', type=int, action='store',
                        dest="epochs", default=3,
                        help='type the number of epochs')
    parser.add_argument('--arch', type=str, action='store',
                        dest="arch", default='vgg19',
                        help='type the CNN model architecture to use')
    parser.add_argument('--learning_rate', type=float, action='store',
                        dest="lr", default=0.001,
                        help='type the learning rate')
    parser.add_argument('--save_dir', type=str, action='store',
                        dest="sdir", default="checkpoint.pth",
                        help='type the saved data directory')
    parser.add_argument('--hidden_unit', type=int, action='store',
                        dest="hu", default=2048,
                        help='type the hidden unit')
    parser.add_argument('--gpu', type=str, action='store',
                        dest="gpu", default='cuda',
                        help='type cpu or gpu')
    return parser.parse_args()
args = get_args()

if args.gpu == "cuda":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
else:
    device = "cpu"


def train(dr):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(dr+"/train", transform=train_transforms)
    test_data = datasets.ImageFolder(dr+"/test", transform=test_transforms)
    valid_data = datasets.ImageFolder(dr+"/valid", transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=64,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=64,)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=64,)

    model = getattr(models, args.arch)(pretrained=True)
    images, labels = next(iter(train_loader))

    for para in model.parameters():
        para.requires_grad = False

    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(
                                    nn.Linear(feature_num, 1024),
                                    nn.Dropout(p=0.5),
                                    nn.ReLU(),
                                    nn.Linear(1024, 102),
                                    nn.LogSoftmax(dim=1))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(nn.Linear(1024, 500),
                                   nn.Dropout(p=0.6),
                                   nn.ReLU(),
                                   nn.Linear(500, 102),
                                   nn.LogSoftmax(dim=1))
    elif args.arch == "vgg19":
   
        classifier = nn.Sequential(
                                    nn.Linear(25088,2048),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(2048,1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1024,102),
                                    nn.LogSoftmax(dim=1),)
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    model.to(device)

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 10

    for epo in range(epochs):
        for inputs, labels in train_loader:

            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epo+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()

    test_accuracy = 0
    test_steps = 0
    with torch.no_grad():
        model.to(device)
        model.eval()
        for inputs, labels in test_loader:
            test_steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("test steps: " + str(test_steps),"accuracy: "+ str(test_accuracy/len(test_loader)*100)+"%")

    torch.save({'structure': args.arch,
                'hidden_layer1': args.hu,
                'droupout': 0.2,
                'epochs': args.epochs,
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict()}, args.sdir)
    
    print("CHECKPOINT HAS BEEN SAVED SUCCESSFULLY")

train(args.ddir)
