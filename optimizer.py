import torch.optim as optim

def make_optimizer(args, targets):
  optimizer = optim.AdamW(targets.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  return optimizer