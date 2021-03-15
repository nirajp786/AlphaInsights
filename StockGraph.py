import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def graph(ticker): 
    APPL_STOCK = pd.read_csv(ticker+'.csv') #maybe we can reduce reading the file here again? - redundancy
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Value',fontsize=20)
    plt.title(ticker+'Stock',fontsize=20)

    def animate(i):
        data = APPL_STOCK.iloc[:i]
        p = sns.lineplot(x=data.index, y=data['Close'], data=data, color="r")
        p.tick_params(labelsize=17)
        plt.setp(p.lines,linewidth=3)
        return data

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=1000, repeat=False, interval=10)
    plt.show()