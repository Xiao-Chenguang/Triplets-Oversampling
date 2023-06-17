import os
import torch
from torch import nn, optim
from .fl_tools import compute_updat, weight_sum, add_state
from copy import deepcopy
from .fl_metrics import Metrics


class Federation():
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        # define property of federation of learns
        self.dataloaders = dataloaders
        self.test_loader = test_loader
        self.global_model = model
        self.global_round = 0
        self.global_grad = None
        self.momentum_buffer = None
        self.args = args
        self.writer = self.__tb_writer__(args)

    def global_step(self):
        # 1. sample clients to be trained
        selected_clients = torch.randperm(self.args.num_clients)[:self.args.clients_per_round]

        # 2. broadcast model parameters to selected clients
        # 3. clients perform local training parallelly.
        # 4. clients feedback udpates to the server
        client_updates = self.client_fit(selected_clients)

        # 5. server aggregate the updates from clients into a new global model
        weights = torch.tensor([len(self.dataloaders[client].dataset) for client in selected_clients])
        weights = weights / weights.sum()
        self.server_agg(client_updates, weights)
        self.global_round += 1


    def train(self):
        # the main entrance for entire FL training process
        # perform num_epoch rounds of global training (global_step)
        # print the args
        # print(f'Start training Fed learners with follow settings:\n{self.args}')
        print(f'Start training Fed learners with follow settings:')
        for k in sorted(vars(self.args)):
            print(f'{k:>17} = {vars(self.args)[k]}')
        print('=' * 80)
        for epoch in range(self.args.global_epochs):
            self.global_step()
            if self.global_round % self.args.eval_frequency == 0:
                self.eval()
        self.writer.close()

    def client_fit(self, selected_clients):
        raise NotImplementedError

    def server_agg(self, updates, weights):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def __tb_writer__(self, args):
        path = f'./results/{args.task_name}_{args.dataset}_{args.ir}_{args.os}_{args.seed}.csv'
        if os.path.exists(path): raise FileExistsError(f'File {path} already exists!')
        with open(path, 'w') as f:
            f.write('dataset,ir,sampling,seed,epoch,acc,recall,precision,f1,gmean,auc,ap\n')
        head = f'{args.dataset},{args.ir},{args.os},{args.seed}'
        return path, head


class FedAvg(Federation):
    def __init__(self, dataloaders, test_loader, model, args) -> None:
        super().__init__(dataloaders, test_loader, model, args)

    def client_fit(self, selected_clients):
        updates = []
        loss = 0
        for i, client in enumerate(selected_clients):
            # print(f'train client: {i:2d}', end=' ')
            update, client_loss = train_client(self.global_model, self.dataloaders[client], self.args)
            updates.append(update)
            loss += client_loss
        loss /= len(selected_clients)
        print(f'Epoche [{self.global_round+1:4d}/{self.args.global_epochs}] >> train loss: {loss:.5f}', end=', ')
        return updates

    def server_agg(self, updates, weights):
        self.global_grad = weight_sum(updates, weights)

        self.global_model = add_state(self.global_model, self.global_grad, 1, 1)

    def eval(self):
        lossfn = nn.BCEWithLogitsLoss()
        self.global_model.eval()
        loss = 0
        test_size = len(self.test_loader.dataset)
        pred = []
        gtrue = []
        with torch.no_grad():
            for x, y in self.test_loader:
                batch_size = len(y)
                x, y = x.to(self.args.device), y.unsqueeze(1).to(self.args.device)
                logits = self.global_model(x)
                pred.append(logits.sigmoid())
                gtrue.append(y)
                batch_loss = lossfn(logits, y) * batch_size
                loss += batch_loss
        loss /= test_size
        pred = torch.concat(pred)
        gtrue = torch.concat(gtrue)
        metrics = Metrics(device=self.args.device)
        # acc, recall, precision, f1, gmean, auc, ap
        score = metrics(pred, gtrue)
        # log training
        with open(self.writer[0], 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(self.writer[1],self.global_round,*score))
        # self.writer.add_scalar('Loss/Test', loss, self.global_round)
        # for k, v in score.items():
        #     self.writer.add_scalar(k, v, self.global_round)
        print(f'test loss: {loss:.5f}, acc: {score[0]*100:5.2f}%')

    def train(self):
        # the main entrance for entire FL training process
        # perform num_epoch rounds of global training (global_step)
        # print the args
        # print(f'Start training Fed learners with follow settings:\n{self.args}')
        print(f'Start training Fed learners with follow settings:')
        for k in sorted(vars(self.args)):
            print(f'{k:>17} = {vars(self.args)[k]}')
        print('=' * 80)
        for epoch in range(self.args.global_epochs):
            self.global_step()
            if self.global_round % self.args.eval_frequency == 0:
                self.eval()

    def __tb_writer__(self, args):
        path = f'./results/{args.task_name}_{args.dataset}_{args.ir}_{args.os}_{args.seed}.csv'
        if os.path.exists(path): raise FileExistsError(f'File {path} already exists!')
        with open(path, 'w') as f:
            f.write('dataset,ir,sampling,seed,epoch,acc,recall,precision,f1,gmean,auc,ap\n')
        head = f'{args.dataset},{args.ir},{args.os},{args.seed}'
        return path, head


def train_client(global_model, dataloader, args):
    local_model = deepcopy(global_model)
    local_model.to(args.device)
    
    optimizer = optim.SGD(local_model.parameters(), lr=args.local_lr, momentum=args.local_momentum)
    lossfn = nn.BCEWithLogitsLoss()

    data_size = len(dataloader.dataset)

    loss = 0
    for step in range(args.local_epoch):
        for X, y in dataloader:
            optimizer.zero_grad()
            X, y = X.to(args.device), y.unsqueeze(1).to(args.device)
            logits = local_model(X)
            batch_loss = lossfn(logits, y)
            loss += batch_loss * len(y)
            batch_loss.backward()
            optimizer.step()
    loss /= data_size * args.local_epoch
    # print(f'training loss: {loss:6.4f}')
    
    update = compute_updat(local_model, global_model)
    return update, loss


