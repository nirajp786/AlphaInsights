
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Note: Use %matplotlib qt for animation


def graph(data, predictions, ticker): 
    data['Predictions'] = predictions
    data['Buy'] = data.Predictions == 1
      
    axes = data.plot(x='Date', y='Close', kind='line')
    data[data.Buy].plot(x='Date', y='Close', kind='scatter', ax = axes, c='green', marker='^', label='Buy')    
    #data[~data.Buy].plot(x='RelativeDate', y='Close', kind='scatter', ax = axes, c='red', marker='v', label='Sell')    
    plt.legend(loc='lower right');
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Price',fontsize=20)
    plt.title(ticker+' Stock',fontsize=20)
    
    plt.show()
    
    
    
def simulate(data, predictions, ticker):
    data['Predictions'] = predictions
    data['Buy'] = data.Predictions == 1
    
    buy_points = []            
    for i in range(len(data)):
        if (data.iloc[i][11]):
            buy_points.append(data.iloc[i][4])
        else:
            buy_points.append(None)
    data.insert(len(data.columns), "BuyPoints", buy_points)
    print(data.head())
    
    # set a figure of size (10,6):
    fig = plt.figure(figsize=(10,6))
    # set subplot grid parameters (1x1 grid, 1st subplot):
    ax1 = fig.add_subplot(1,1,1)
    # add limits on the x axis defined by the sample period (0 is the first observation, -1 the final observation):
    ax1.axis(xmin = data.Date[0], xmax = data.Date[len(data) - 1])
    # add limits on the y-axis defined by minimum and maximum of the respective series, incorporate some additional room:
    ax1.axis(ymin= (data['Close'].min()-0.1),ymax=(data['Close'].max()+0.1))
    
    plt.xlim([data.Date[0], data.Date[len(data) - 1]])
    plt.autoscale(False)
    
    # define the function animate, which has the input argument of i:
    def animate(i):
        #   set the variable data to contain 0 to the (i+1)th row:
        df =  data.iloc[:int(i+1)]  #select data range
        #   initialise xp as an empty list:
        xp = []
        #   initialise yp as an empty list:
        yp = []
        #   initialise zp as an empty list:
        zp = []
  
        #   set the variable lines as equal to the variable data:
        lines = df
        
        #   for a line in lines:
        for line in lines:
            #     x is equal to the index (time domain):
            xp = df['Date']
            #     y is equal to the 'Close' column
            yp = df['Close']
            #     z is equal to the 'BuyPoints' column
            zp = df['BuyPoints']

        #   clear ax(1):
        ax1.clear()
        
        #   plot stock chart:
        ax1.plot(xp, yp)
        #   plot buy points:   
        ax1.scatter(xp, zp, c='green', marker='^') 
        
        #   provide a label for the x-axis:
        plt.xlabel('Date',fontsize=12)
        #   provide a label for the y-axis:  
        plt.ylabel('Price',fontsize=12)
        #   provide a plot title:   
        plt.title(ticker+' Stock',fontsize=20)

    
    # call Matplotlib animation.Funcanimation, providing the input arguments of fig, animate, the number of frames and an interval:
    ani = animation.FuncAnimation(fig, animate, frames = len(data), interval=100) 
    
    # Use the 'ffmpeg' writer:
    Writer = animation.writers['ffmpeg']
    # Set the frames per second and bitrate of the video:
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # save the animation to the predefined output directory:
    ani.save('animation_video.mp4', writer=writer)