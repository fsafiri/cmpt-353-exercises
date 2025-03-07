import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb  
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import sys

# Representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 112, 0),
    'yellow': (255, 255, 0),
    'green': (0, 231, 0),
    'blue': (0, 0, 255),
    'purple': (185, 0, 185),
    'brown': (117, 60, 0),
    'pink': (255, 184, 184),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])

def plot_predictions(model, lum=67, resolution=300):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    
    X_grid = lab2rgb(lab_grid)

    
    y_grid = model.predict(X_grid.reshape((-1, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, -1)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def rgb_to_lab(X):
    """Convert an array of RGB values to LAB using skimage's rgb2lab."""
    return rgb2lab(X.reshape(-1, 1, 3)).reshape(-1, 3)

def main(infile):
    data = pd.read_csv('colour-data.csv')
    X = data[['R', 'G', 'B']].values / 255  
    y = data['Label'].values  

    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    
    model_rgb = GaussianNB()
    model_rgb.fit(X_train, y_train)
    y_pred_rgb = model_rgb.predict(X_test)
    rgb_accuracy = accuracy_score(y_test, y_pred_rgb)
    print(f'RGB accuracy: {rgb_accuracy }%')

    
    lab_pipeline = Pipeline([
        ('rgb_to_lab', FunctionTransformer(rgb_to_lab)),
        ('classifier', GaussianNB())
    ])

    
    lab_pipeline.fit(X_train, y_train)
    y_pred_lab = lab_pipeline.predict(X_test)
    lab_accuracy = accuracy_score(y_test, y_pred_lab)
    print(f'lab model accuracy: {lab_accuracy}%')

    
    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(lab_pipeline)
    plt.savefig('predictions_lab.png')

if __name__ == '__main__':
    main(sys.argv[1])
