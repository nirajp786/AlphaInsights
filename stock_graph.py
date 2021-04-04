import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Note: Use %matplotlib qt for animation

# ----------------------------------------------------------------------------
def graph(data, predictions, ticker, classifier):
    """
    graph creates a 2 dimensional plot with time on the X axis and Price on 
    the Y axis. A price chart for the given stock and time period is generated
    with "BUY" signals highlighted as green arrows.
    --------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :param predictions: an array of predictions
    :param ticker: the name of the stock 
    :param classifier: the name of the classifier used to make predictions
    :return: None 
    """     
    # Add 'Buy' column to identify plotting points
    data_copy = data.copy()
    data_copy['Predictions'] = predictions
    data_copy['Buy'] = data_copy.Predictions == 1

    # Plot the axes
    axes = data_copy.plot(x='Date', y='Close', kind='line')

    # Plote the "buy" points
    data_copy[data_copy.Buy].plot(x='Date', y='Close', kind='scatter', ax = axes, c='green', marker='^', label='Buy')

    # Optional: Plot non-buy points
    #data_copy[~data_copy.Buy].plot(x='RelativeDate', y='Close', kind='scatter', ax = axes, c='red', marker='v', label='Sell')

    # Add other chart elements
    plt.legend(loc='lower right');
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Price',fontsize=20)
    plt.title(ticker+' Stock: ' + classifier,fontsize=20)

    plt.show()

# ----------------------------------------------------------------------------
def simulate(data, predictions, ticker):
    """
    simulate creates an animated plot with time on the X axis and Price on 
    the Y axis. A price chart for the given stock and time period is generated
    sequentially with "BUY" signals highlighted as green arrows. Results are 
    stored as an mpeg video file.
    --------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :param predictions: an array of predictions
    :param ticker: the name of the stock 
    :return: None (The animated graph is generated and saved)
    """     
    data_copy = data.copy()

    # Add 'Buy' column to identify plotting points
    data_copy['Predictions'] = predictions

    # Create "Buy Points" column in the dataframe (points where stock is purchased)
    buy_points = []
    for index, row in data_copy.iterrows():
        if (row['Predictions']):
            buy_points.append(row['Close'])
        else:
            buy_points.append(None)
    data_copy.insert(len(data_copy.columns), "BuyPoints", buy_points)

    # set a figure of size (10,6):
    fig = plt.figure(figsize=(10,6))
    # set subplot grid parameters (1x1 grid, 1st subplot):
    ax1 = fig.add_subplot(1,1,1)
    # add limits on the x axis defined by the sample period (0 is the first observation, -1 the final observation):
    ax1.axis(xmin = data_copy.Date[0], xmax = data_copy.Date[len(data_copy) - 1])
    # add limits on the y-axis defined by minimum and maximum of the respective series, incorporate some additional room:
    ax1.axis(ymin= (data_copy['Close'].min()-0.1),ymax=(data_copy['Close'].max()+0.1))

    # define the function animate, which has the input argument of i:
    def animate(i):
        '''
        animate(i) is a lamda function called by animation.FuncAnimation. 
        ----------------------------------------------------------------------
        :param i: An automatically generated, increasing value that is used 
                  to identify frames in the animation
        :return: None
        '''    
        #   set the variable df to contain 0 to the (i+1)th row:
        df =  data_copy.iloc[:int(i+1)]  #select data range
        
        #   initialize variables
        xp = []
        yp = []
        zp = []
        lines = df

        # Populate the arrays:
        for line in lines:
            xp = df['Date']
            yp = df['Close']
            zp = df['BuyPoints']

        # Clear the old plot before redrawing it
        ax1.clear()

        # Plot the chart
        ax1.plot(xp, yp)
        # Plot the 'buy' signals
        ax1.scatter(xp, zp, c='green', marker='^')

        # Generate labels
        plt.xlabel('Date',fontsize=12)
        plt.ylabel('Price',fontsize=12)
        plt.title(ticker+' Stock',fontsize=20)

    # Call animation.FuncAnimation to animate the chart
    ani = animation.FuncAnimation(fig, animate, frames = len(data_copy), interval=100)

    # Identify a writer (for saving the animation)
    Writer = animation.writers['ffmpeg']
    # Set the frames per second and bitrate of the video:
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # Save the animation to the predefined output directory:
    ani.save('animation_video.mp4', writer=writer)
