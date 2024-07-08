import os
import logging

from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import segmentation_models_pytorch as smp

from nnmodule import PatchEmbed, Block, ResidualConnection, UpSampleBlock, DoubleConvolution, DownSampleBlock, Deconv, DoubleConv, VisionTransformer, load_custom_model
from models import UNet, UNETR
import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DIR = './Synapse/train_npz/'
LIST_DIR = './lists_Synapse/'
NUM_CLASSES = 9

if __name__ == '__main__':
    db_train = utils.Synapse_dataset(base_dir=TRAIN_DIR, list_dir=LIST_DIR, split="train",
                                transform=transforms.Compose(
                                [utils.RandomGenerator(output_size=[224, 224])]))

    trainloader = DataLoader(db_train, batch_size=24, shuffle=True, num_workers=1, pin_memory=True)
    ce_loss = CrossEntropyLoss()
    dice_loss = utils.DiceLoss(NUM_CLASSES)

    # Model and parameters
    model_name = 'UNet_4_dilation2'
    model = UNet(depth=4, input_size=[1], num_classes=9, dilation=2).to(DEVICE)
    unique_in_channel = True
    base_lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    max_epoch = 100
    max_iterations = max_epoch * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    # Training loop

    iter_num = 0

    for epoch_num in range(max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        
            # channel fix
            if unique_in_channel == False:
                image_batch = image_batch.expand(-1, 3, -1, -1)
        
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
        
        save_interval = 50 
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(model_name + '_epoch_' + str(epoch_num + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        
    save_mode_path = os.path.join(model_name + '_epoch_' + str(max_epoch) + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))