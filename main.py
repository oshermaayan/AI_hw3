import hw3_utils
import our_utils as utils
from classifier import *

from sklearn import tree

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit()
utils.evaluate()
