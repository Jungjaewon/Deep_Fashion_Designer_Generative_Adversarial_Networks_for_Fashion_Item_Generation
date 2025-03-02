import os, os.path as osp
import argparse
import yaml

def main(config, path):

    if path.split('config_')[1].startswith('ft_vis'):
        from ft_vis import FTVIS
        ft_vis = FTVIS(config)
        ft_vis.run()
    elif path.split('config_')[1].startswith('image'):
        from image_gen import IMAGEGEN
        image_gen = IMAGEGEN(config)
        image_gen.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='exp_config/config_0.yml', help='specifies config yaml file')

    params = parser.parse_args()

    assert osp.exists(params.config)
    config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
    main(config, osp.basename(params.config))