from joblib import dump
from joblib import load
import pandas as pd
import pickle
from models import Lasso_regression


with open('file_lasso.pickle','wb') as f:
    pickle.dump(Lasso_regression.lasso_model,f) # type: ignore

