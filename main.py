from data_clean import DataClean
from model import UnetModel
from train import TrainModel
from benchmark import ModelBenchmark

import utils


TRAIN_IMAGES_PATH = './data/train'
TRAIN_MASKS_PATH = './data/train-masks'

TEST_IMAGES_PATH = './data/test'
TEST_MASKS_PATH = './data/test-masks'

VALID_IMAGES_PATH = './data/valid'
VALID_MASKS_PATH = './data/valid-masks'

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

    valid_dc = DataClean()
    valid_dc.get_img_id_filename(img_dir=VALID_IMAGES_PATH)
    valid_dc.get_annotations(img_dir=VALID_IMAGES_PATH)
    valid_dc.merge_img_ann()
    valid_dc.generate_masked_imgs(
        img_dir=VALID_IMAGES_PATH, output_dir=VALID_MASKS_PATH)

    X_train, y_train = utils.load_images(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
    X_test, y_test = utils.load_images(TEST_IMAGES_PATH, TEST_MASKS_PATH)
    X_val, y_val = utils.load_images(VALID_IMAGES_PATH, VALID_MASKS_PATH)

    unet_model = UnetModel()

    unet_train = TrainModel(X_train, y_train, unet_model.model, X_val, y_val)
    hist = unet_train.train()

    unet_benchmark = ModelBenchmark(
        trained_model=unet_train.model, train_hist=hist, X_test=X_test, y_test=y_test)
    unet_benchmark.plot_accuracy_val_accuracy()
    unet_benchmark.plot_dice_score_loss()

    unet_benchmark.test_model()
    unet_benchmark.plot_example_results()
