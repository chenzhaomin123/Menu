USE_CUDA = True


cross_lr = 0.01
momentum = 0.95
weight_decay = 0.0005

center_lr = 0.5
epoch = 1000

batch_size = 48
shuffle = True
print_freq = 400

traindir = '/home/hx/data/train_test/train_chinese_food/'
mean = [0.3869, 0.5064, 0.6003]
std = [0.1335, 0.1243, 0.1291]

valdir = '/home/hx/data/train_test/val_chinese_food/'

num_classes = 172