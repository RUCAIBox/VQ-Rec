import argparse
from logging import getLogger
from recbole.config import Config
from recbole.trainer.trainer import PretrainTrainer
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import init_seed, init_logger

from vqrec import VQRec
from data.dataset import PretrainVQRecDataset


def pretrain(dataset, **kwargs):
    # configurations initialization
    props = ['props/VQRec.yaml', 'props/pretrain.yaml']
    print(props)

    # configurations initialization
    config = Config(model=VQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = PretrainVQRecDataset(config)
    logger.info(dataset)

    pretrain_dataset = dataset.build()[0]
    pretrain_data = TrainDataLoader(config, pretrain_dataset, None, shuffle=True)

    # model loading and initialization
    model = VQRec(config, pretrain_data.dataset).to(config['device'])
    model.pq_codes = model.pq_codes.to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = PretrainTrainer(config, model)

    # model pre-training
    trainer.pretrain(pretrain_data, show_progress=True)

    return config['model'], config['dataset']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='FHCKM', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    model, dataset = pretrain(args.d)
