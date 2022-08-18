import numpy as np
import joblib
import imgaug

from src.training import load_images
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


sometimes = lambda aug: iaa.Sometimes(0.3, aug)
seq = iaa.Sequential(
    [
        sometimes(
            iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=imgaug.ALL, pad_cval=(0, 255))
        ),
        sometimes(
            iaa.OneOf(
                [
                    iaa.Dropout(
                        (0.01, 0.1), per_channel=0.5
                    ),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2
                    ),
                ]
            )
        ),
        # iaa.BlendAlphaSimplexNoise(iaa.OneOf([
        #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
        #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
        # ])),
        iaa.LinearContrast(
            (0.5, 2.0), per_channel=0.5
        ),  # improve or worsen the contrast
        sometimes(
            iaa.OneOf(
                [
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
                    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),
                ]
            )
        ),
        iaa.Fliplr(0.1),  # horizontally flip 10% of the images
        sometimes(
            iaa.GaussianBlur(sigma=(0, 3.0))
        ),  # blur images with a sigma of 0 to 3.0
        iaa.Resize((40, 40), interpolation=Image.Resampling.LANCZOS),
        # iaa.Grayscale(alpha=(0.0, 1.0)),
    ]
)

if __name__ == "__main__":

    for idx in range(7):

        accuracy = 0

        while accuracy < 0.88:

            images, exp_output = load_images("src/training/data")
            images_aug = seq(images=images)
            uniq_labels = sorted(list(set(exp_output)))

            train_input = []
            train_output = set()
            for i, image in enumerate(images_aug):
                train_input.append(image.flatten())
                label = uniq_labels[uniq_labels.index(exp_output[i])]
                train_output.add(label)

            outputs = list(sorted(train_output))

            # normalize pixels intensity
            train_input = np.array(train_input) / 255.0

            # convert outputs to one-hot vectors
            one_hot_output = np.zeros((len(images_aug), len(outputs)))

            for i, label in enumerate(exp_output):
                one_hot_output[i][outputs.index(label)] = 1

            X_train, X_test, y_train, y_test = train_test_split(
                train_input,
                one_hot_output,
                test_size=0.1,
            )

            mlp = MLPClassifier(
                hidden_layer_sizes=(
                    100,
                    50,
                ),
                max_iter=30000,
                alpha=0.1,
                activation="tanh",
                solver="lbfgs",
                # learning_rate="adaptive",
                # early_stopping=True,
                # verbose=True,
                # random_state=1,
                tol=1e-4,
                warm_start=True,
            )

            mlp.fit(X_train, y_train)

            accuracy = mlp.score(X_train, y_train)
            print("train data accuracy: {}".format(accuracy))

            accuracy = mlp.score(X_test, y_test)
            print("test data accuracy: {}".format(accuracy))

        file_name = "model_{}.joblib".format(idx)
        print("saved to {}".format(file_name))
        joblib.dump(mlp, file_name)
