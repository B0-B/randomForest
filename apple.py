

'''
A test of classifying apple quality - a dataset derived from 
https://www.kaggle.com/code/nelgiriyewithana/an-introduction-to-apple-quality-dataset/input
'''

from lib import *

if __name__ == '__main__':

    dataset = loadSetFromCsv('./apple_quality_cropped.csv')

    forest = Forest(dataset, seeds=5, depth=2)

    forest.grow(threads=1, verbose=True)

    sample = dataset.sample()[0]
    sample.show()
    print('prediction:\n', forest.classify(sample))