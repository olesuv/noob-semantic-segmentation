from data_clean import DataClean
from model import UnetModel
from train import TrainModel

import utils


TRAIN_IMAGES_PATH = './data/train'
TRAIN_MASKS_PATH = './data/train-masks'
TEST_IMAGES_PATH = './data/test'
TEST_MASKS_PATH = './data/test-masks'

if __name__ == "__main__":
    train_dc = DataClean()
    train_dc.get_img_id_filename()
    train_dc.get_annotations()
    train_dc.merge_img_ann()
    train_dc.generate_masked_imgs()

    test_dc = DataClean()
    test_dc.get_img_id_filename(img_dir=TEST_IMAGES_PATH)
    test_dc.get_annotations(img_dir=TEST_IMAGES_PATH)
    test_dc.merge_img_ann()
    test_dc.generate_masked_imgs(
        img_dir=TEST_IMAGES_PATH, output_dir=TEST_MASKS_PATH)

    X_train, y_train = utils.load_images(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
    X_test, y_test = utils.load_images(TEST_IMAGES_PATH, TEST_IMAGES_PATH)

    unet_model = UnetModel()

    unet_train = TrainModel(X_train, y_train, unet_model.model)
    unet_train.train()
    unet_train.test_model(X_test, y_test)

    predicted_imgs = unet_train.model_predict(X_test)
    for i in range(5):
        utils.display_images(X_test[i], predicted_imgs[i], y_test[i])
