import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

flag = False

def graph(data, predictions, ticker): 
    data['Predictions'] = predictions
    data['Buy'] = data.Predictions == 1
    axes = data.plot(x='RelativeDate', y='Close', kind='line')
    data[data.Buy].plot(x='RelativeDate', y='Close', kind='scatter', ax = axes, c='green', marker='^', label='Buy')    
    data[~data.Buy].plot(x='RelativeDate', y='Close', kind='scatter', ax = axes, c='red', marker='v', label='Sell')    
    plt.legend(loc='lower right');
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.title(ticker+'Stock',fontsize=20)
    
    plt.show()