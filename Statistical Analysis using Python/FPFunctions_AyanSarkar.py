
# coding: utf-8
"""
Created on Sun Nov 25 2018
This module constains function to create bar graph, scatter plot & produce plots for a regression anlysis.
It also contians function to perform tTest for Equal Variances & Anova Test
@author: Ayan Sarkar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, DataFrame
from scipy import stats
from statsmodels.formula.api import ols
import plotly.plotly as py            #using package to draw pie chart which was not discussed in class
import plotly.graph_objs as go        #using package to add style to the pie chart which was not discussed in class


#Function for drawing bar charts
def Bargraph(title,idx,value,ylabel,xlabel,figName,barColor):
    
    plt.suptitle(title,fontsize='12')
    index = idx
    plt.bar(index, value,color=barColor)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    plt.savefig(figName, bbox_inches='tight')

#Function for drawing scatter plots
def ScatterPlot(title,xvalue,yvalue,figName,plotColor):
    
    x1 = xvalue
    y1 = yvalue
    plt.scatter(x1,y1,color=plotColor)
    plt.title(title+str(round(x1.corr(y1),2)))
    plt.savefig(figName, bbox_inches='tight')
    
#Function to multiple regression analysis plots
def my_multreg(model, ydata, xlabel, ylabel, HLstart, HLEnd, actvspredplot=True, residplot=True):
    r2adj = round(model.rsquared_adj,2)
    p_val = round(model.f_pvalue,4)
    coefs = model.params
    coefsindex = coefs.index 
    regeq = round(coefs[0],3) 
    cnt = 1
    for i in coefs[1:]:
        regeq=f"{regeq} + {round(i,3)} {coefsindex[cnt]}"
        cnt = cnt + 1
    if actvspredplot==True:
        #Scatterplot for Multiple Regression - y vs predicted y
        predict_y = model.predict()
        plt.scatter(ydata,predict_y)
        minSls=np.array(ydata).min()
        maxSls=np.array(ydata).max()
        diag = np.arange(minSls,maxSls,(maxSls-minSls)/50)
        plt.scatter(diag,diag,color='red',label='perfect prediction')
        plt.suptitle(regeq)
        plt.title(f' with adjR2: {r2adj}, F p-val {p_val}',size=10)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')
        plt.show()
    if residplot==True:
        #Scatterplot residuals 'errors' vs predicted values
        resid = model.resid
        predict_y = model.predict()
        plt.scatter(predict_y, resid)
        plt.suptitle(regeq)
        plt.hlines(0,HLstart,HLEnd) #horizontal line at 0 error
        plt.ylabel('Residuals')
        plt.xlabel(ylabel)
        plt.show()
    return r2adj, p_val, regeq   

#Function to draw Pie Charts
def Piechart(PlotTitle,Plotlabel,PieVal1,PiTitle1,PieVal2,PiTitle2,PieVal3,PiTitle3):
    fig = {
        'data': [
            {
                'labels': Plotlabel,
                'values': PieVal1,
                'type': 'pie',
                'name': PiTitle1,
                'title':PiTitle1,
                'domain': {'x': [0, .48],
                           'y': [0, .49]},
                'hoverinfo':'label+percent+name',
                'textinfo':'value'
            },
            {
                'labels': Plotlabel,
                'values': PieVal2,
                'type': 'pie',
                'name': PiTitle2,
                'title':PiTitle2,
                'domain': {'x': [.52, 1],
                           'y': [0, .49]},
                'hoverinfo':'label+percent+name',
                'textinfo':'value'
    
            },
            {
                'labels': Plotlabel,
                'values': PieVal3,
                'type': 'pie',
                'name': PiTitle3,
                'title':PiTitle3,
                'domain': {'x': [.24, .72],
                           'y': [.51, 1]},
                'hoverinfo':'label+percent+name',
                'textinfo':'value'
            }
        ],
        'layout': {'title': PlotTitle ,
                   'showlegend': True}
    }
    
    return py.iplot(fig, filename='pie_chart_subplots')

#Function to perform t-test of Equal Variances for 2 sets of values
def tTestEqlVar(tValue1,tValue2,tTitle,rConclusion,frConclusion,yLabel,figName):
    #t test for equal variances
    alpha = .05
    tvar, p_valvar = stats.bartlett(tValue1,tValue2)
    
    
    print(tTitle)
    print(f"The t test statistic is {round(tvar,3)} and the p-value is {round(p_valvar,4)}")
    if p_valvar < alpha:
        print(rConclusion)
        tEqVar=False
        ttype='Welch (unequal variances) Two-Sample t test'
    else:
        print(frConclusion)
        tEqVar=True
        ttype='Two-Sample t test (assuming equal variances)'
        
    # Create the boxplot
    y=[tValue1,tValue2]
    plt.boxplot(y)
    plt.title(f't: {round(tvar,3)}, p-val: {round(p_valvar,4)}',size=10)
    plt.suptitle(ttype,size=10)
    plt.xticks(range(1,3),[f"4 Bed rooms: {round(tValue1.mean(),2)}",
                               f"5 Bed rooms: {round(tValue2.mean(),2)}"])
    plt.ylabel(yLabel)
    plt.savefig(figName, bbox_inches='tight')
    plt.show()

#Function to perform Annova-test for multiple sets of values
def AnovaTest(tValue1,tValue2,tValue3,tValue4,tTitle,xlabel,yLabel,figName):
    alpha = .05
    f, p_val = stats.f_oneway(tValue1,tValue2,tValue3,tValue4)
    print(tTitle)
    print(f"The F test statistic is {round(f,3)} and the p-value is {round(p_val,4)}")
    if p_val < alpha:
        print("Conclusion: Reject Ho: At least one group mean is different")
        ANOVAtype = "ANOVA: At least one group mean different"
    else:
        print("Conclusion: Fail to Reject Ho: We can't reject that the means are the same")
        ANOVAtype = "ANOVA: Group Means are the same"
        
    # Create the boxplot
    y=[tValue1,tValue2,tValue3,tValue4]
    plt.boxplot(y)
    plt.title(f'F: {round(f,3)}, p-val: {round(p_val,4)}',size=10)
    plt.suptitle(ANOVAtype,size=10)
    plt.xticks(range(1,5),[f"2 BR: {round(tValue1.mean(),2)}",
                               f"3 BR: {round(tValue2.mean(),2)}",
                               f"4 BR: {round(tValue3.mean(),2)}",
                               f"5 BR: {round(tValue4.mean(),2)}"])
    plt.xlabel(xlabel) #From below
    plt.ylabel(yLabel)
    plt.savefig(figName, bbox_inches='tight')
    plt.show()
    
