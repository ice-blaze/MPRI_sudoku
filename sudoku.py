from classification.metrics import show_confusion_matrix, print_classification_report
from classification.svm import load_or_train
from image.cell_extraction import extract_cells, plot_extracted_cells
from PIL import Image
import numpy as np
from image.feature_extraction import extract_features
from skimage import io
from sklearn import cross_validation

# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 11
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

# Get trained classifier
# TOD.O
clf = load_or_train(True)

# Load sudoku image
# TOD.O: load the sudoku image as a gray level image
my_img = np.array(Image.open(im_path).convert('L'))


# Extract cells
# TO.DO
cells = extract_cells(my_img)

# Add data for each cell
# TOD.O: iterate over cells and append features to a list
features = []
for cell in cells:
	features.append(extract_features(cell))
	

# Classification
# TODO: use the classifier to predict on the list of feature
features_classe = []
for feature in features:
	features_classe.append(clf.predict(feature))


# Load solution to compare with, print metrics, and print confusion matrix
y_sudoku = np.loadtxt(ver_path).reshape(81)
error_count = 0
for i,val in enumerate(features_classe):
	if y_sudoku[i]!=val:
		error_count = error_count + 1

features_classe = np.array(features_classe)
# TODO: print classification report
#print_classification_report(y_true, y_pred, "classification report")
print_classification_report(y_sudoku, features_classe, "classification report")
# TODO: show confusion matrix
#show_confusion_matrix(y_true, y_predicted, "confusion matrix")
show_confusion_matrix(y_sudoku, features_classe, "confusion matrix")

# Print resulting sudoku
# TOD.O: print the resulting sudoku grid (use reshape() function to get a 9x9 grid print!
print np.reshape(features_classe,(9,9))


