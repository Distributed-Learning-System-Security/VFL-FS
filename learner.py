import torch
import time
import random
import math
from threading import Thread
from queue import Queue
from torch.utils.tensorboard import SummaryWriter

# recording_period = math.ceil(recording * len(train_loader) / len(test_loader))
recording = 4


class VFLLearner(object):
    def __init__(self, party_list, data_loader_list, epochs, delay_factor, output_dir, pre_trained, after_rounds, feature_selection, feature_num, device, fixed):
        super(VFLLearner, self).__init__()
        self.server_party = party_list[0]
        self.server_data_loader = data_loader_list[0]
        self.client_party_list = party_list[1:]
        self.client_data_loader_list = data_loader_list[1:]
        self.epochs = epochs
        self.delay_factor = delay_factor
        self.output_dir = output_dir
        self.after_rounds = after_rounds
        self.pre_trained = pre_trained
        self.feature_selection = feature_selection
        self.feature_num = feature_num
        self.device = device
        self.fixed = fixed

        self.server_party_thread = None
        self.client_party_thread_list = []
        self.parties_num = len(party_list)

    def start_learning(self):
        parties_h_queue_list = []  # embedding: party->server
        parties_grad_queue_list = []  # gradient: server->party
        parties_predict_h_queue_list = []

        for i in range(self.parties_num-1):
            parties_h_queue_list.append(Queue())
            parties_grad_queue_list.append(Queue())
            parties_predict_h_queue_list.append(Queue())

        server_data = {
            'train_loader': self.server_data_loader[0],
            'test_loader': self.server_data_loader[1],
            'party': self.server_party,
            'epochs': self.epochs,
            'device': self.device,
            'output_dir': self.output_dir,
            'parties_h_queue_list': parties_h_queue_list,
            'parties_grad_queue_list': parties_grad_queue_list,
            'parties_predict_h_queue_list': parties_predict_h_queue_list,
            'pre_trained': self.pre_trained
            }
        self.server_party_thread = ServerPartyThread(server_data, 0)

        for id, party in enumerate(self.client_party_list):
            client_data = {
                'train_loader': self.client_data_loader_list[id][0],
                'test_loader': self.client_data_loader_list[id][1],
                'party': party,
                'epochs': self.epochs,
                'device': self.device,
                'delay_factor': self.delay_factor[id],
                'h_queue': parties_h_queue_list[id],
                'grad_queue': parties_grad_queue_list[id],
                'predict_h_queue': parties_predict_h_queue_list[id],
                'after_rounds': self.after_rounds,
                'pre_trained': self.pre_trained,
                'feature_selection': self.feature_selection,
                'feature_num': self.feature_num,
                'fixed': self.fixed
                }
            self.client_party_thread_list.append(ClientPartyThread(client_data, id+1))
        
        self.server_party_thread.start()
        for thread in self.client_party_thread_list:
            thread.start()


class ServerPartyThread(Thread):
    def __init__(self, data, thread_id):
        super(ServerPartyThread, self).__init__()
        self.data = data
        self.thread_id = thread_id

    def run(self):
        train_loader = self.data['train_loader']
        test_loader = self.data['test_loader']
        party = self.data['party']
        epochs = self.data['epochs']
        output_dir = self.data['output_dir']
        device = self.data['device']
        parties_h_queue_list = self.data['parties_h_queue_list']  # pull embedding
        parties_grad_queue_list = self.data['parties_grad_queue_list']  # push gradient
        parties_predict_h_queue_list = self.data['parties_predict_h_queue_list']
        pre_trained = self.data['pre_trained']
        
        recording_period = math.ceil(recording * len(train_loader) / len(test_loader))
        global_step = 0
        running_time = 0
        writer = SummaryWriter(log_dir=output_dir+time.strftime("%Y%m%d-%H%M"))

        # pre-trained
        for i in range(pre_trained):
            it = iter(train_loader)
            pre_trained_target = next(it)[1].to(device)
            party.pre_train(pre_trained_target, parties_h_queue_list, parties_grad_queue_list)

        for ep in range(epochs):
            print(f'server start epoch {ep}, batches={len(train_loader)}')
            it = iter(test_loader)
            for batch_idx, (_, target) in enumerate(train_loader):
                party.model.train()
                start_time = time.time()
                target = target.to(device)
                party.set_batch(target)
                print(f'server set batch {batch_idx}')

                h_list = []
                for q in parties_h_queue_list:
                    h_list.append(q.get())
                party.pull_parties_h(h_list)
                party.compute_parties_grad()
                parties_grad_list = party.send_parties_grad()
                for idx, q in enumerate(parties_grad_queue_list):
                    q.put(parties_grad_list[idx])
                print(f'server local update with batch {batch_idx}')
                party.local_update()
                party.local_iterations()

                end_time = time.time()
                spend_time = end_time - start_time
                running_time += spend_time
                global_step += 1
                if global_step % recording_period == 0:
                    party.model.eval()
                    predict_h_list = []
                    for q in parties_predict_h_queue_list:
                        predict_h_list.append(q.get())
                    predict_y = next(it)[1].to(device)
                    loss, correct, accuracy = party.predict(predict_h_list, predict_y)
                    writer.add_scalar("loss&step", loss.detach(), global_step)
                    writer.add_scalar("loss&time", loss.detach(), running_time * 1000)
                    writer.add_scalar("accuracy&step", accuracy, global_step)
                    writer.add_scalar("accuracy&time", accuracy, running_time*1000)
                    writer.add_scalar("running_time", running_time, global_step)
                    print(f'thread{self.thread_id}: server figure out loss={loss} correct={correct} accuracy={accuracy} spend_time={running_time}\n')
        writer.close()


class ClientPartyThread(Thread):
    def __init__(self, data, thread_id):
        super(ClientPartyThread, self).__init__()
        self.data = data
        self.thread_id = thread_id

    def run(self):
        train_loader = self.data['train_loader']
        test_loader = self.data['test_loader']
        party = self.data['party']
        epochs = self.data['epochs']
        device = self.data['device']
        h_queue = self.data['h_queue']  # push embedding
        grad_queue = self.data['grad_queue']  # pull gradient
        predict_h_queue = self.data['predict_h_queue']
        delay_factor = self.data['delay_factor']
        pre_trained = self.data['pre_trained']
        after_rounds = self.data['after_rounds']
        feature_selection = self.data['feature_selection']
        feature_num = self.data['feature_num']
        fixed = self.data['fixed']
        flag = True

        train_batches = len(train_loader)
        test_batches = len(test_loader)
        recording_period = math.ceil(recording * train_batches / test_batches)
        global_step = 0

        # pre-trained
        for i in range(pre_trained):
            it = iter(train_loader)
            pre_trained_data = next(it)[0].to(device)
            party.pre_train(pre_trained_data, h_queue, grad_queue)

        for ep in range(epochs):
            it = iter(test_loader)
            print(f'client{self.thread_id} start epoch {ep}, batches={len(train_loader)}')
            for batch_idx, (data, _) in enumerate(train_loader):
                party.model.train()
                data = data.to(device)
                # Tex iterations, update feature selector
                if (global_step % after_rounds == 9) and (feature_selection is True) and (flag is True):
                    party.update_fs(feature_num)
                    # if fixed, only select once
                    if fixed is True:
                        flag = False
                party.set_batch(data)
                print(f'client{self.thread_id} set batch {batch_idx}')

                party.compute_h()
                # simulate the delay of communication
                sleep_time = random.random() * delay_factor
                time.sleep(sleep_time)
                # print(f'client{self.thread_id} sleep {sleep_time}sec')
                h_queue.put(party.send_h())
                grad = grad_queue.get()
                print(f'client{self.thread_id} local update with batch {batch_idx}')
                party.pull_grad(grad)
                party.local_update()
                party.local_iterations()

                global_step += 1
                if global_step % recording_period == 0:
                    party.model.eval()
                    predict_x = next(it)[0].to(device)
                    predict_h_queue.put(party.predict(predict_x))
