import numpy as np
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
    x=np.random.rand(50)
    xma=np.column_stack((np.ones(50),x,x**2))
    #initial args [1,2,3]
    y=3*(x**2)+2*x+1+np.random.normal(0,0.01,(1,50))
    beta_hat=np.linalg.inv(xma.T@xma)@xma.T@y.T
    y_hat=xma@beta_hat
    print('OLS: ',beta_hat.T[0])
    rq=(y_hat.T@y_hat)/(y@y.T)
    print('The R Square is',rq[0][0])
    model=LinearRegression()
    x_data=xma
    y_data=y.reshape((-1,1))
    model.fit(x_data,y_data)
    b,c,a=model.coef_[0][1],model.coef_[0][2],model.intercept_[0]
    print('Estimated Coeffcients: a=',a,'b=',b,'c=',c)
