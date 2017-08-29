from keras.applications import vgg16
from keras.applications import resnet50
from keras.applications import vgg19
from keras.applications import inception_v3

mods = [vgg16.VGG16,
        resnet50.ResNet50,
        vgg19.VGG19,
        inception_v3.InceptionV3]

def main():
    """main functionality"""
    for mod in mods:
        blah


if __name__ == '__main__':
    main()
