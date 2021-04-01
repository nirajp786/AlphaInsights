
import matplotlib.pyplot as plt

flag = False

def graph(data, predictions, ticker): 
    data['Predictions'] = predictions
    data['Buy'] = data.Predictions == 1
      
    axes = data.plot(x='Date', y='Close', kind='line')
    data[data.Buy].plot(x='Date', y='Close', kind='scatter', ax = axes, c='green', marker='^', label='Buy')    
    #data[~data.Buy].plot(x='RelativeDate', y='Close', kind='scatter', ax = axes, c='red', marker='v', label='Sell')    
    plt.legend(loc='lower right');
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.title(ticker+' Stock',fontsize=20)
    
    plt.show()