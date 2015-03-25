from sklearn.metrics.metrics import confusion_matrix, classification_report
import pylab as pl


def show_confusion_matrix(y_true, y_pred, title=''):
	"""
	Plot (and print) a confusion matrix from y_true and y_predicted
	"""
	# TODO: show confusion matrix plot
	cm = confusion_matrix(y_true, y_pred)
	pl.matshow(cm)
	pl.title(title)
	pl.colorbar()
	pl.ylabel('True label')
	pl.xlabel('Predicted label')
	pl.show()


def print_classification_report(y_true, y_pred, title=''):
	"""
	Print a classification report
	"""

	# TODO: print classification report

	print(classification_report(y_true, y_pred))
