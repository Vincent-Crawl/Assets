import torchvision
import torchvision.transforms as transforms

#transform_train和transform_test分别表示对训练集和测试集的预处理操作
#batch_size 一次迭代向神经网络传递多少图片
# shuffle 训练获取的数据集顺序打乱与否
# num_worker  子进程数量 以加速数据加载
# data数据集类型 ,data_dir数据集下载路径
def load_dataset(data, size, transform_train, transform_test, data_dir=None):
    if data_dir is None:
        data_dir = "../" + data
    if data == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "flower":
        trainset = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.Flowers102(root=data_dir, split="test", download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    elif data == "pets":
        trainset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=transform_train)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
        
        testset = torchvision.datasets.OxfordIIITPet(root=data_dir, split="test", download=True, transform=transform_test)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    
    return trainset, testset