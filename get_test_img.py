import os.path as osp
import os
import glob
import shutil
import cv2
import argparse

from tqdm import tqdm


# get val and test image
def get_test_image():

    os.makedirs(dst_dir, exist_ok=True)

    for mode in ['test', 'val']:
        print(f'{mode} is on processing!!')
        dst_img_dir = osp.join(dst_dir, mode)
        os.makedirs(dst_img_dir, exist_ok=True)

        for outfit_path in tqdm(glob.glob(osp.join(target_dir, mode, '*'))):

            outfit_id = osp.basename(outfit_path)
            for image_path in glob.glob(osp.join(target_dir, mode, outfit_id, '*.jpg')):

                img_name = osp.basename(image_path)
                if img_name not in ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']:
                    continue
                else:
                    img_name = f'{outfit_id}_{img_name}'
                    shutil.copy(src=image_path, dst=osp.join(dst_img_dir, img_name))

                    img = cv2.imread(osp.join(dst_img_dir, img_name))
                    img = cv2.resize(img, (256,256))
                    cv2.imwrite(osp.join(dst_img_dir, img_name), img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dst_dir', type=str, default='test_img', help='seed')
    parser.add_argument('--target_dir', type=str, default='outfitdata_set3_4598', help='new data directory')
    config = parser.parse_args()

    target_dir = config.target_dir
    dst_dir = config.dst_dir

    get_test_image()

    