import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_layers()

    def init_layers(self):
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, h, c):
        x, (h_, c_) = self.lstm(x, (h, c))
        x = self.fc(x)
        return x, h_, c_


def sample(model,
           h,
           c,
           vocab_size,
           itoc,
           device,
           length: int = 200,
           seed: int = 0) -> str:
    x = F.one_hot(torch.tensor([seed], device=device), num_classes=vocab_size).float()
    ixes = []
    for t in range(length):
        x, h, c = model(x, h, c)
        probs = F.softmax(x, dim=1)
        ix = torch.multinomial(probs, 1).item()
        ixes.append(ix)
        x = F.one_hot(torch.tensor([ix], device=device), num_classes=vocab_size).float()
    text = ''.join(itoc[ix] for ix in ixes)
    return text


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = open('/tmp/input.txt', 'r').read()
    chars = list(set(''.join(data)))
    vocab_size = len(chars)
    seq_length = 25
    hidden_size = 100
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for c, i in ctoi.items()}
    print(f'Loaded {len(data)} characters of which {vocab_size} are unique')

    model = Net(vocab_size, hidden_size, vocab_size).to(device)
    model = torch.compile(model, mode='max-autotune')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-1)

    num_epochs = 100_000
    model.train()
    smooth_loss = -torch.log(torch.tensor(1. / vocab_size)).item() * seq_length
    h = torch.zeros((1, hidden_size), device=device)
    c = torch.zeros((1, hidden_size), device=device)
    p = 0
    for epoch in range(num_epochs):
        if p + seq_length + 1 >= len(data) or epoch == 0:
            h.zero_()
            c.zero_()
            p = 0
        inputs = [ctoi[ch] for ch in data[p:p + seq_length]]
        targets = [ctoi[ch] for ch in data[p + 1:p + seq_length + 1]]

        inputs = torch.tensor(inputs, device=device)
        inputs = F.one_hot(inputs, num_classes=vocab_size).float()
        targets = torch.tensor(targets, device=device)

        with torch.autocast(device):
            outputs, *_ = model(inputs, h, c)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p += seq_length
        smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: loss: {smooth_loss}')
        if epoch % 1000 == 0:
            print('=' * 80)
            model.eval()
            text = sample(model, h.clone(), c.clone(),
                          vocab_size, itoc, device,
                          torch.randint(150, 250, (1,)).item(),
                          torch.randint(0, vocab_size, (1,)).item())
            model.train()
            print(text)
            print('=' * 80)


if __name__ == '__main__':
    main()
