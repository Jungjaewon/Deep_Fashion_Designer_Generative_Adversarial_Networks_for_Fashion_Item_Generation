import argparse

from eval_part.eval_inception_score import Polyvore
from eval_part.eval_inception_score import inception_score

from eval_part.eval_fid import calculate_fid_given_paths

from eval_part.eval_lpips import calculate_lpips

# FID https://github.com/mseitzer/pytorch-fid
# Inception score https://github.com/sbarratt/inception-score-pytorch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, default='./image_result', help='a directory for evaluation')
    parser.add_argument('--orgin_img_dir', type=str, default='./test_img/test', help='a directory for GT')
    parser.add_argument('--i_score_net', type=str, default='inception_v3', choices=['vgg19_bn', 'inception_v3'])
    params = parser.parse_args()

    print(f'score_dir : {params.score_dir}')

    print("Calculating Inception Score on GT...")
    print(inception_score(Polyvore(params.orgin_img_dir, params.i_score_net)))

    print("Calculating Inception Score...")
    print(inception_score(Polyvore(params.score_dir, params.i_score_net)))

    print('Calculating FID...')
    print(calculate_fid_given_paths(fake_dir=params.score_dir, real_dir=params.orgin_img_dir))

    print('Calculating LPIPS...')
    print(calculate_lpips(fake_dir=params.score_dir, real_dir=params.orgin_img_dir))