# AlphaInsights

# Abstract 

Investors, speculators, and scientists have spent years studying the movement of stock prices. Stocks are securities that represent the ownership of a portion of a corporation. Supply and demand determine the price of a stock and these parameters are, in turn, affected by numerous other factors. Investors who can correctly weigh these factors and determine price movement stand to profit. Making such predictions, however, is a difficult task due to the sheer number of elements involved in determining price movement. There are many who believe reliable predictions are impossible due the randomness of the markets (Random Walk Theory). 

We believe that modern artificial intelligence (AI) and machine learning techniques can be used to predict price movement and identify profitable entry and exit points for stock trades. We will design an AI agent that will use historical data to model price movement. It will then monitor new incoming data and “buy” stock at points that it identifies as profitable based on the model. 

In our initial phase of development we will create a simple reflex agent. The environment will be limited to two variables - time and price. We will train the agent using historical price data for a single stock. A k-nearest neighbors algorithm will be used to identify “buy signals” based on predicted price. 

In subsequent iterations we will expand the model and the complexity of our agent in one or more ways. We will consider expanding the number of features used in making predictions; additional factors such as transaction volume and moving averages may be integrated into the model. We will also consider using different algorithms such as a linear regression model, support vector machine, or neural network to enhance the model’s predictive power. 
 
In the future we would also like to explore methods in which our agent could self-evaluate so that it might independently evolve its model and buying strategy as market conditions changed. We could, for example, implement a genetic algorithm to mix and match elements of various strategies in order to find the best one. 

We see this as an interesting and complex problem that will expand not only our understanding of AI and agents, but also our knowledge of finance, economics, and probability. 

## Glossary
* The kNN algorthim: a classfication method that uses existing data and categorizes it to make a prediction
* Indicators: factors used to interpret stocks to forecast market moves
* Data Mining: process of using different data analysis tool to see patterns and relationship within that data
* Classification: predicting which category something falls under 
* Regression: predicting what value a variable will have.
