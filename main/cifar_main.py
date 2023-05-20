import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cifar_model import ClientNet, ServerNet
from partymodel import ServeParty, ClientParty
from learner import VFLLearner

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
seed = 10


def set_seed():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(int(torch.empty((), dtype=torch.int64).random_().item()))


def run_experiment():
    data_dir = 'data/cifar10'
    output_dir = "summary_pic/cifar10/"
    n_local = [1, 1, 1, 1]
    delay_factor = [0, 0, 0]
    batch_size = 256
    epochs = 30
    shuffle = True
    pre_trained = 0  # pre_trained some mini-batches
    feature_selection = True  # do or not do
    after_rounds = 10  # after some rounds, update the feature selector
    feature_num = 1200  # the number of feature needed to be selected
    random = False  # select features randomly
    fixed = False  # select features dynamically or fixedly

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    div1 = 4  # 3*32*4
    div2 = 12  # 3*32*8  3 * 32 * 20

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    client1_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    client1_test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    client2_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    client2_test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    client3_train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    client3_test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)

    client1_train_dataset.data = train_dataset.data[:, :, :div1, :]
    print(client1_train_dataset.data.shape)
    client1_test_dataset.data = test_dataset.data[:, :, :div1, :]
    client2_train_dataset.data = train_dataset.data[:, :, div1:div2, :]
    print(client2_train_dataset.data.shape)
    client2_test_dataset.data = test_dataset.data[:, :, div1:div2, :]
    client3_train_dataset.data = train_dataset.data[:, :, div2:, :]
    print(client3_train_dataset.data.shape)
    client3_test_dataset.data = test_dataset.data[:, :, div2:, :]

    set_seed()
    g = torch.Generator()
    server_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    server_test_loader = DataLoader(test_dataset, batch_size=batch_size)

    set_seed()
    g = torch.Generator()
    client1_train_loader = DataLoader(client1_train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    client1_test_loader = DataLoader(client1_test_dataset, batch_size=batch_size)

    set_seed()
    g = torch.Generator()
    client2_train_loader = DataLoader(client2_train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    client2_test_loader = DataLoader(client2_test_dataset, batch_size=batch_size)

    set_seed()
    g = torch.Generator()
    client3_train_loader = DataLoader(client3_train_dataset, batch_size=batch_size, shuffle=shuffle, generator=g)
    client3_test_loader = DataLoader(client3_test_dataset, batch_size=batch_size)

    data_loader_list = [[server_train_loader, server_test_loader], [client1_train_loader, client1_test_loader],
                        [client2_train_loader, client2_test_loader], [client3_train_loader, client3_test_loader]]

    server_model = ServerNet(3).to(device)
    server_loss_func = nn.CrossEntropyLoss()
    server_optimizer = optim.Adam(server_model.parameters())

    client1_model = ClientNet(div1).to(device)
    client1_optimizer = optim.Adam(client1_model.parameters())

    client2_model = ClientNet(div2-div1).to(device)
    client2_optimizer = optim.Adam(client2_model.parameters())

    client3_model = ClientNet(32 - div2).to(device)
    client3_optimizer = optim.Adam(client3_model.parameters())

    server_party = ServeParty(model=server_model, loss_func=server_loss_func, optimizer=server_optimizer,
                              n_iter=n_local[0])
    client1_party = ClientParty(model=client1_model, optimizer=client1_optimizer, device=device, n_iter=n_local[1], random=random)
    client2_party = ClientParty(model=client2_model, optimizer=client2_optimizer, device=device, n_iter=n_local[2], random=random)
    client3_party = ClientParty(model=client3_model, optimizer=client3_optimizer, device=device, n_iter=n_local[3],
                                random=random)

    party_list = [server_party, client1_party, client2_party, client3_party]

    print("################################ Train Federated Models ############################")

    vfl_learner = VFLLearner(party_list, data_loader_list, epochs, delay_factor, output_dir, pre_trained, after_rounds,
                             feature_selection, feature_num, device, fixed)
    vfl_learner.start_learning()


if __name__ == '__main__':
    run_experiment()
