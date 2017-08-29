from keras.applications import vgg16
from keras.applications import resnet50
from keras.applications import vgg19
from keras.applications import inception_v3
from sklearn.cross_validation import train_test_split

mods = [vgg16.VGG16,
        resnet50.ResNet50,
        vgg19.VGG19,
        inception_v3.InceptionV3]


class ModIterator(object):
    def __init__(self, mods, train_set, test_set, sample_size):
        self.mods = mods
        self.train_set = train_set
        self.test_set = test_set
        self.sample_size = sample_size

    def load_data(self):
        """Load data """

    def fit(self, mod, X_train, y_train):
        """Fit model"""

    def evaluate(self, mod, X_test, y_test):
        """Evaluate model on test set"""


def main():
    """main functionality"""
    for mod in mods:
        blah


if __name__ == '__main__':
    main()
