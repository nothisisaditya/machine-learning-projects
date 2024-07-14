from net import Net
from dataloader import Dataset

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# import wandb

import os
import time
import argparse

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--data', type=str, default='./data',
                        help='path to data')
    parser.add_argument('--batch-size', type=int, default=2**8,
                        help='batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=2, metavar='N',
                        help='number of workers for data loading (default: 2)')
    parser.add_argument('--epochs', type=int, default=90, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-p', '--print-freq', type=int, default=10,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-s', '--save-freq', type=int, default=10,
                        metavar='N', help='save frequency (default: 10)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--prof', action='store_true',
                        help='only run 10 iterations for profiling')
    parser.add_argument('--disable-dali', action='store_true',
                        help='not use DALI for data loading')
    return parser.parse_args()


def train(dataset, model, criterion, optimizer, device, disable_dali) -> tuple[float, float]:
    t = time.time()
    loss = 0.0

    for i, data in enumerate(dataset.train_loader):
        if disable_dali:
            images, labels = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
        else:
            images, labels = data[0]['data'], data[0]['label']
        optimizer.zero_grad()
        with torch.autocast(device, dtype=torch.bfloat16):
            output = model(images)
            loss = criterion(output, labels.view(-1).long())
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    return time.time() - t, loss.item()


def main():
    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Net(generator=torch.Generator().manual_seed(2**31 - 1), num_classes=10)
    model = model.to(device=device, memory_format=torch.channels_last)
    model = torch.compile(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    if not os.path.exists(os.path.join(args.data, 'models')):
        os.makedirs(os.path.join(args.data, 'models'))

    dataset = Dataset(data_dir=args.data,
                      batch_size=args.batch_size,
                      workers=args.workers,
                      cuda=device == 'cuda',
                      disable_dali=args.disable_dali)
    writer = SummaryWriter()
    # wandb.init(
    #     project='Benchmarking DALI',
    #     config={
    #         "learning_rate": 0.001,
    #         "architecture": "CNN",
    #         "dataset": "custom",
    #         "epochs": args.epochs
    #     }
    # )

    for epoch in range(args.start_epoch, args.epochs):
        t, loss = train(dataset, model, criterion, optimizer, device, args.disable_dali)

        if ((epoch - args.start_epoch) + 1) % args.print_freq == 0:
            print(f'Epoch {epoch} took {t:.2f}s, loss: {loss:.4f}')
            writer.add_scalar('Loss/train', loss, epoch)
            # wandb.log({'loss': loss})

        if ((epoch - args.start_epoch) + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.data, 'models', f'model_{epoch}.pt'))

    # wandb.finish()
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
