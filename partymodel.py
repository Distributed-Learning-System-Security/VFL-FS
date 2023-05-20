import torch


class ServeParty(object):
    def __init__(self, model, loss_func, optimizer, n_iter=1):
        super(ServeParty, self).__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.n_iter = n_iter
        
        self.h_dim_list = []
        self.parties_grad_list = []
        self.y = None
        self.batch_size = None
        self.h_input = None

    def set_batch(self, y):
        self.y = y
        self.batch_size = y.shape[0]

    def pull_parties_h(self, h_list):
        self.h_dim_list = [h.shape[1] for h in h_list]
        h_input = None
        for h in h_list:
            if h_input is None:
                h_input = h
            else:
                h_input = torch.cat([h_input, h], 1)
        h_input = h_input.detach()
        h_input.requires_grad = True
        self.h_input = h_input

    def compute_parties_grad(self):
        output = self.model(self.h_input)
        loss = self.loss_func(output, self.y)
        loss.backward()
        parties_grad = self.h_input.grad

        self.parties_grad_list = []
        start = 0
        for dim in self.h_dim_list:
            self.parties_grad_list.append(parties_grad[:, start:start+dim])
            start += dim
    
    def send_parties_grad(self):
        return self.parties_grad_list

    def local_update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def local_iterations(self):
        for i in range(self.n_iter-1):
            self.compute_parties_grad()
            self.local_update()

    def predict(self, h_list, y):
        batch_size = y.shape[0]
        self.pull_parties_h(h_list)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.h_input)
            loss = self.loss_func(output, y)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(y.view_as(pred)).sum().item()
            accuracy = correct/batch_size

        return loss, correct, accuracy

    def pre_train(self, target, parties_h_queue_list, parties_grad_queue_list):
        self.model.train()
        self.set_batch(target)
        h_list = []
        for q in parties_h_queue_list:
            h_list.append(q.get())
        self.pull_parties_h(h_list)
        self.compute_parties_grad()
        parties_grad_list = self.send_parties_grad()
        for i, q in enumerate(parties_grad_queue_list):
            q.put(parties_grad_list[i])
        self.local_update()
        self.local_iterations()


class ClientParty(object):
    def __init__(self, model, optimizer, device, n_iter=1, random=False):
        super(ClientParty, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.feature_selector = torch.ones_like(model.fs_layer.weights).to(device)
        self.random = random

        self.x = None
        self.h = None
        self.partial_grad = None
        self.batch_size = None

    def set_batch(self, x):
        x = torch.flatten(x, 1)
        self.x = x * self.feature_selector
        self.batch_size = x.shape[0]
    
    def compute_h(self):
        self.h = self.model(self.x)

    def send_h(self):
        return self.h

    def pull_grad(self, grad):
        self.partial_grad = grad

    def local_update(self):
        self.h.backward(self.partial_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def local_iterations(self):
        for i in range(self.n_iter-1):
            self.compute_h()
            self.local_update()

    def predict(self, x):
        predict_h = self.model(x)
        return predict_h

    def pre_train(self, data, h_queue, grad_queue):
        """
            pre-trained
        """
        self.model.train()
        self.set_batch(data)
        self.compute_h()
        h_queue.put(self.send_h())
        grad = grad_queue.get()
        self.pull_grad(grad)
        self.local_update()
        self.local_iterations()

    def update_fs(self, feature_num):
        """
            update feature selector
        """
        feature_len = self.x.shape[1]  # actual number
        if feature_len <= feature_num:
            return
        weights = self.model.fs_layer.weights
        self.feature_selector = torch.zeros_like(weights)
        if self.random is False:
            _, indices = torch.topk(weights, feature_num)
            self.feature_selector[indices] = 1  # update the feature selector
        else:
            _, indices = torch.topk(torch.randint_like(weights, 0, 1), feature_num)
            self.feature_selector[indices] = 1
