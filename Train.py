import numpy as np
import random
import argparse

import torch
from torch.utils.data import DataLoader

from utils_option import make_logger, parse
from utils_option import select_lossfn, select_optimizer
from dataset.Dataset_selector import DatasetSR
from models.model_selector import modelSR



def train(args):

    '''
    # ----------------------------------------
    # Seed & Settings
    # ----------------------------------------
    '''
    sr_mode = args['sr_mode']
    cuda = torch.cuda.is_available()

    logger_name = args['logger_name']
    logger_path = args['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)

    seed = args['seed']
    if seed is None:
        seed = random.randint(1, 100)
    logger.info('Random Seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info(f'cuda: {cuda}')
    logger.info(f'SR mode: {sr_mode}')

    args = args[sr_mode]
    scale = args['dataset']['train']['scale']
    batch_size = args['train']['batch_size']
    epoch = args['train']['epoch']

    logger.info(f'scale: {scale}, epoch:{epoch}, batch size:{batch_size}')


    '''
    # ----------------------------------------
    # DataLoader
    # ----------------------------------------
    '''
    Dataset_SR = DatasetSR(sr_mode)
    for phase, dataset_opt in args['dataset'].items():
        if phase == 'train':
            train_set = Dataset_SR(dataset_opt)
            train_dataloader = DataLoader(train_set,
                                          batch_size=batch_size, num_workers=dataset_opt['num_workers'],
                                          shuffle=True)
        elif phase == 'test':
            val_set = Dataset_SR(dataset_opt)
            val_dataloader = DataLoader(val_set,
                                        batch_size=batch_size, num_workers=dataset_opt['num_workers'],
                                        shuffle=True)

    logger.info(f'train_set: {len(train_set)}')
    logger.info(f'val_set: {len(val_set)}')


    '''
    # ----------------------------------------
    # Model & Settings
    # ----------------------------------------
    '''
    model = modelSR(sr_mode, scale)
    if cuda:
        model = model.cuda()
    loss_fn = select_lossfn(opt=args['train']['loss_fn'], reduction=args['train']['reduction'])
    optimizer = select_optimizer(opt=args['train']['optimizer'], lr=args['train']['learning_rate'], model=model)

    logger.info(f'loss_fn: {args["train"]["loss_fn"]}, reduction: {args["train"]["reduction"]}')
    logger.info(f'optimizer: {args["train"]["optimizer"]}, lr: {args["train"]["learning_rate"]}')


    '''
    # ----------------------
    # Training
    # ----------------------
    '''
    logger.info('-----> Start training...')
    for epoch_idx in range(epoch):
        logger.info(f'Epoch: {epoch_idx+1}/{epoch}')

        loss_idx = 0
        train_losses = 0
        model.train()

        for batch_idx, batch_data in enumerate(train_dataloader):

            train_batch = batch_data['in']
            label_batch = batch_data['hr']

            train_batch = train_batch[:, 0, :, :].unsqueeze(1)
            label_batch = label_batch[:, 0, :, :].unsqueeze(1)
            train_batch = train_batch.float()
            label_batch = label_batch.float()

            if cuda:
                train_batch, label_batch = train_batch.cuda(), label_batch.cuda()

            output_batch = model(train_batch, scale)
            train_loss = loss_fn(output_batch, label_batch)
            train_losses += train_loss
            loss_idx += 1

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_losses /= loss_idx
        logger.info(f', Avg_Train_Loss: {train_losses}')


        '''
        # --------------------------
        # Test
        # --------------------------
        '''

        loss_idx = 0
        val_losses = 0
        model.eval()

        for batch_idx, batch_data in enumerate(val_dataloader):

            test_batch = batch_data['in']
            label_batch = batch_data['hr']

            test_batch = test_batch[:, 0, :, :].unsqueeze(1)
            label_batch = label_batch[:, 0, :, :].unsqueeze(1)
            test_batch = test_batch.float()
            label_batch = label_batch.float()

            if cuda:
                test_batch, label_batch = test_batch.cuda(), label_batch.cuda()

            with torch.no_grad():
                output_batch = model(test_batch)
            val_loss = loss_fn(output_batch, label_batch)
            val_losses += val_loss
            loss_idx += 1

        val_losses /= loss_idx
        logger.info(f', Avg_Val_Loss: {val_losses}')


        '''
        # --------------------------
        # Save Checkpoint
        # --------------------------
        '''
        if (epoch_idx + 1) % args['train']['train_checkpoints'] == 0:
            model_path = f'model_zoo/{sr_mode}_check{epoch_idx + 1}_s{scale}e{epoch}b{batch_size}.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'| Saved Checkpoint at Epoch {epoch_idx + 1} to {model_path}')

    '''
    # --------------------------
    # Final Save Model Dict
    # --------------------------
    '''
    model_path = f'model_zoo/{sr_mode}_s{scale}e{epoch}b{batch_size}.pth'
    torch.save(model.state_dict(), model_path)

    logger.info(f'Saved State Dict in {model_path}')







def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args().opt)
    train(args)


if __name__ == '__main__':
    main()
