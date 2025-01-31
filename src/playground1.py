import numpy as np
import torch 
from torchvision import datasets, transforms
from utils import * # get the dataset
from pyhessian import hessian # Hessian computation
from pytorchcv.model_provider import get_model as ptcv_get_model # model

# This is a simple function, that will allow us to perturb the model paramters and get the result
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def getData(name='cifar10', train_bs=128, test_bs=1000):
    """
    Get the dataloader
    """
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    return train_loader, test_loader

# get the model 
model = ptcv_get_model("resnet20_cifar10", pretrained=True)
# change the model to eval mode to disable running stats upate
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()

# for illustrate, we only use one batch to do the tutorial
for inputs, targets in train_loader:
    break

# we use cuda to make the computation fast
model = model.cuda()
inputs, targets = inputs.cuda(), targets.cuda()

# create the hessian computation module
hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)



# Now let's compute the top eigenvalue. This only takes a few seconds.
top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
print(top_eigenvalues)

# Perturb the model
lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
n_perb = len(lams)
land = np.empty((n_perb, n_perb))

# create a copy of the model
model_perb = ptcv_get_model("resnet20_cifar10", pretrained=True)
model_perb.eval()
model_perb = model_perb.cuda()

for i in range(n_perb):
        for j in range(n_perb):
            model_perb = get_params(model, model_perb, top_eigenvector[0], lams[i])
            model_perb = get_params(model_perb, model_perb, top_eigenvector[1], lams[j])
            outputs = model_perb(inputs)
            land[i, j] = criterion(outputs, targets).item()
np.save('land_physical.npy', land)


# create the hessian computation module
# hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=True)
# # Now let's compute the top eigenvalue. This only takes a few seconds.
# top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
# print(top_eigenvalues)
# # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
# lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)

# loss_list = []

# # create a copy of the model
# model_perb = ptcv_get_model("resnet20_cifar10", pretrained=True)
# model_perb.eval()
# model_perb = model_perb.cuda()

# for lam in lams:
#     model_perb = get_params(model, model_perb, top_eigenvector[0], lam)
#     loss_list.append(criterion(model_perb(inputs), targets).item())

# np.save('lams_temp.npy', np.array(lams))
# np.save('loss_list_temp.npy', np.array(loss_list))
