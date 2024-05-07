import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE

#evaluation metrics
def eval_regressor(reg, Xtrain, Ytrain, Xtest, Ytest):
    score_train = reg.score(Xtrain, Ytrain)
    score_test = reg.score(Xtest, Ytest)
    mse_train = MSE(Ytrain, reg.predict(Xtrain))
    rmse_train = MSE(Ytrain, reg.predict(Xtrain), squared = False)
    mse_test = MSE(Ytest, reg.predict(Xtest))
    rmse_test = MSE(Ytest, reg.predict(Xtest), squared = False)

    print(f'r2_train: {score_train}')
    print(f'r2_test: {score_test}')
    print(f'mse_train: {mse_train}')
    print(f'mse_test: {mse_test}')
    print(f'rmse_train: {rmse_train}')
    print(f'rmse_test: {rmse_test}')
    return

#prediction plot
def prediction_plot(reg, Xtrain, Ytrain, Xtest, Ytest):
    plt.figure(figsize=(6,6),dpi=1000)#,facecolor="0.92")
    Ytrain_pre = reg.predict(Xtrain)
    Ytest_pre = reg.predict(Xtest)
    p2=plt.scatter(Ytest,Ytest_pre.reshape(-1,1),label='Test',marker='o',s=50,linewidths=1.2,color='w',edgecolors='#ce3c35')
    p1=plt.scatter(Ytrain, Ytrain_pre.reshape(-1,1), label='Training',s=50,marker='^',linewidths=1.2,color='w',edgecolors='#4258a1')
    plt.xlabel('Actual values', fontsize=22,weight='normal') 
    plt.ylabel('Predicted values', fontsize=22,weight='normal') 
    plt.tick_params(which='both', direction='in', length=5)
    plt.xticks(fontsize=20,weight='normal')
    plt.yticks(fontsize=20,weight='normal')
    m1=plt.legend(handles=[p1],fontsize=16,loc=(0.05,0.9),frameon=True)
    x = np.linspace(0,1.25)
    y = x
    y1 = 1.2 * x
    y2 = 0.8 * x
    plt.plot(x, y, color='black', linewidth=1.5, linestyle='--')
    plt.plot(x, y1, color='gray', linewidth=1.5, linestyle='--')
    plt.plot(x, y2, color='gray', linewidth=1.5, linestyle='--')   # linewidth 设置线的宽度， linesyyle设置线的形状
    plt.ylim(-0.05,1.25)
    plt.xlim(-0.05,1.25)
    ax = plt.gca().add_artist(m1)

    m2=plt.legend(handles=[p2],fontsize=16,loc=(0.62,0.3),frameon=True)
    plt.show()
    plt.rc('font',family='Arial',weight='normal')#,weight='bold'

def cross_regressor(reg, X, y):
    cv = KFold(n_splits=5,shuffle=True,random_state=1)
    validation_loss = cross_validate(reg,
                                     X, y,
                                     scoring=["r2","neg_mean_squared_error","neg_root_mean_squared_error"],
                                     cv=cv,
                                     return_train_score = True,
                                     verbose=False,
                                     n_jobs=6
                                    )
    cross_validate_r2_test = validation_loss['test_r2'].mean()
    cross_validate_r2_train = validation_loss['train_r2'].mean()
    cross_validate_mse_train = abs(validation_loss['train_neg_mean_squared_error']).mean()
    cross_validate_mse_test = abs(validation_loss['test_neg_mean_squared_error']).mean()
    cross_validate_rmse_train = abs(validation_loss['train_neg_root_mean_squared_error']).mean()
    cross_validate_rmse_test = abs(validation_loss['test_neg_root_mean_squared_error']).mean()
    print(f'cross_validate_r2_train: {cross_validate_r2_train}')
    print(f'cross_validate_r2_test: {cross_validate_r2_test}')
    print(f'cross_validate_mse_train: {cross_validate_mse_train}')
    print(f'cross_validate_mse_test: {cross_validate_mse_test}')
    print(f'cross_validate_rmse_train: {cross_validate_rmse_train}')
    print(f'cross_validate_rmse_test: {cross_validate_rmse_test}')
    return