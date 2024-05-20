import torch
import torch.nn.functional as F
import json

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

class PytorchTrainer():
    def __init__(self, model=None, epochs=1, optimizer='ADAM', lr=1e-04, adam_beta1=0.9, adam_beta2=0.999, weight_decay=0, momentum=0.9, lr_scheduler=None, device='cpu', 
            save_dir='results', save_name='', log_interval=10, record_loss_every=5):

        self.model = model
        self.model.to(device)
        self.epochs = epochs
        if optimizer == 'ADAM':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=[adam_beta1, adam_beta2], weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.opt = optim
        self.lr_scheduler = lr_scheduler
        self.num_steps = 0
        self.device = device
        self.save_dir = save_dir
        self.save_name = save_name
        self.log_interval = log_interval
        self.record_loss_every = record_loss_every

        self.metrics = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    def train(self, data_loader, val_loader):
        
        self.model.train()
        
        for epoch in range(self.epochs):
            print(f"Now starting epoch: {epoch}")
            
            mean_epoch_loss = self._train_epoch(epoch, data_loader, val_loader)

            if self.model.training and epoch % self.record_loss_every == 1:
                json1 = json.dumps(self.metrics)
                f = open("./{}/{}_e{}_metrics.json".format(self.save_dir, self.save_name, epoch),"w")
                f.write(json1)
                f.close()

                torch.save({
                'model' : self.model.state_dict(),
                'optim' : self.opt.state_dict(), 
                #'lr_sch': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                }, './{}/{}_e{}_weights'.format(self.save_dir,self.save_name,epoch))

    def _train_epoch(self, epoch, data_loader, val_loader):

        for batch_idx, (data, label) in enumerate(data_loader):

            self.opt.zero_grad()

            data = data.to(self.device)
            label = label.to(self.device)

            output = self.model(data)

            loss = F.cross_entropy(output, label)

            loss.backward()
            self.opt.step()

            if batch_idx % self.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            self._eval_log(epoch, data_loader, val_loader)
            self.model.train()

    def _eval_log(self, epoch, data_loader, val_loader):
        
        self.model.eval()
        train_correct = 0
        train_loss = 0
        for data, label in data_loader:

            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                loss = F.cross_entropy(output, label)
                train_loss += loss.item()

                pred = get_likely_index(output)
                train_correct += number_of_correct(pred, label)

        self.metrics['train_loss'].append(train_loss/len(data_loader.dataset))
        self.metrics['train_acc'].append(train_correct/len(data_loader.dataset))
        
        val_correct = 0
        val_loss = 0
        for data, label in val_loader:

            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                loss = F.cross_entropy(output, label)
                val_loss += loss.item()

                pred = get_likely_index(output)
                val_correct += number_of_correct(pred, label)

        self.metrics['val_loss'].append(val_loss/len(val_loader.dataset))
        self.metrics['val_acc'].append(val_correct/len(val_loader.dataset))


        print(f"\nTest Epoch: {epoch}\tAccuracy: {val_correct}/{len(val_loader.dataset)} ({100. * val_correct / len(val_loader.dataset):.0f}%)\n")

        return None


