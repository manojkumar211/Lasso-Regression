from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from best_values import lr_best_test
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures



        
    
# Lasso Regression Model:-


class Lassocv_regression:

    lr_best_train=[]
    lr_best_test=[]

    try:
        for i in range(0,20):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
            lr=LinearRegression()
            lr.fit(X_train,y_train)
            lr_train_pred=lr.predict(X_train)
            lr_test_pred=lr.predict(X_test)
            lr_best_train.append(lr.score(X_train,y_train))
            lr_best_test.append(lr.score(X_test,y_test))

    except Exception as e:
        raise Exception(f'Best RandomState Error in Lasso Regression :\n'+str(e))

  
    lasso_cv=LassoCV(alphas=None,max_iter=1000,cv=5)
    lasso_cv.fit(X_train,y_train)
    alpha_lasso_cv=lasso_cv.alpha_

    try:

        def __init__(self,lasso_cv,alpha_lasso_cv):

            self.lasso_cv=lasso_cv
            self.alpha_lasso_cv=alpha_lasso_cv

        def lasso_cv_regression(self):
            return self.lasso_cv
        def lasso_cv_alpha(self):
            return self.alpha_lasso_cv
        
    except Exception as e:
        raise Exception(f'Alpha Error in Lasso Regression :\n'+str(e))

class Lasso_regression(Lassocv_regression):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(lr_best_test))

    try:

        try:

            lasso_model=Lasso(Lassocv_regression.alpha_lasso_cv) # type: ignore
            lasso_model.fit(X_train,y_train)
            lasso_train_pred=lasso_model.predict(X_train)
            lasso_test_pred=lasso_model.predict(X_test)
            lasso_train_score=lasso_model.score(X_train,y_train)
            lasso_test_score=lasso_model.score(X_test,y_test)
            lasso_cross_val_score=cross_val_score(lasso_model,X,y,cv=5).mean()
            lasso_tr_mae=mean_absolute_error(y_train,lasso_train_pred)
            lasso_tr_mse=mean_squared_error(y_train,lasso_train_pred)
            lasso_tr_rmse=np.sqrt(mean_squared_error(y_train,lasso_train_pred))
            lasso_te_mae=mean_absolute_error(y_test,lasso_test_pred)
            lasso_te_mse=mean_squared_error(y_test,lasso_test_pred)
            lasso_te_rmse=np.sqrt(mean_squared_error(y_test,lasso_test_pred))

        except Exception as e:
            raise Exception(f'Error find in Lasso Regression :\n'+str(e))


        try:

            def __init__(self, lasso_cv, alpha_lasso_cv,lasso_model,lasso_train_pred,lasso_test_pred,lasso_train_score,lasso_test_score,lasso_cross_val_score,
                        lasso_tr_mae,lasso_tr_mse,lasso_tr_rmse,lasso_te_mae,lasso_te_mse,lasso_te_rmse,lr_best_train,lr_best_test):
                    
                try:
                
                    self.lasso_cv=lasso_cv
                    self.alpha_lasso_cv=alpha_lasso_cv
                    self.lasso_model=lasso_model
                    self.lasso_train_pred=lasso_train_pred
                    self.lasso_test_pred=lasso_test_pred
                    self.lasso_train_score=lasso_train_score
                    self.lasso_test_score=lasso_test_score
                    self.lasso_cross_val_score=lasso_cross_val_score
                    self.lasso_tr_mae=lasso_tr_mae
                    self.lasso_tr_mse=lasso_tr_mse
                    self.lasso_tr_rmse=lasso_tr_rmse
                    self.lasso_te_mae=lasso_te_mae
                    self.lasso_te_mse=lasso_te_mse
                    self.lasso_te_rmse=lasso_te_rmse
                    self.lr_best_train=lr_best_train
                    self.lr_best_test=lr_best_test

                except Exception as e:
                    raise Exception(f'Error find in Lasso Regression at Initiate level :\n'+str(e))

            try:

                def lasso_cv_regression(self):
                    return super().lasso_cv
                def lasso_cv_alpha(self):
                    return super().alpha_lasso_cv
                def lasso_model_regression(self):
                    return self.lasso_model
                def lasso_train_pred_regression(self):
                    return self.lasso_train_pred
                def lasso_test_pred_regression(self):
                    return self.lasso_test_pred
                def lasso_train_score_regression(self):
                    return self.lasso_train_score
                def lasso_test_score_regression(self):
                    return self.lasso_test_score
                def lasso_cross_val_score_regression(self):
                    return self.lasso_cross_val_score
                def lasso_train_mae_regression(self):
                    return self.lasso_tr_mae
                def lasso_train_mse_regression(self):
                    return self.lasso_tr_mse
                def lasso_train_rmse_regression(self):
                    return self.lasso_tr_rmse
                def lasso_test_mae_regression(self):
                    return self.lasso_te_mae
                def lasso_test_mse_regression(self):
                    return self.lasso_te_mse
                def lasso_test_rmse_regression(self):
                    return self.lasso_te_rmse
                def lr_best_train_poly(self):
                    return super().lr_best_train
                def lr_best_test_poly(self):
                    return super().lr_best_test
                
            except Exception as e:
                raise Exception(f'Error find in Lasso Regression at Defining level :\n'+str(e))
            
        except Exception as e:
            raise Exception(f'Error find in Lasso Regression at Inintiat and Defining level :\n'+str(e))

    except Exception as e:
        raise Exception(f'Total Error in Lasso Regression :\n'+str(e))

