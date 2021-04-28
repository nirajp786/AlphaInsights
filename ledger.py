import pandas as pd
from datetime import timedelta

SEPERATOR = '=========================================='

# ----------------------------------------------------------------------------
def determine_pl(buy_price, price_change, bid_size):
    """
    determine_pl calcultes the profit or loss when $1000.00 worth of a given
    stock is bought at a given price and sold after the price changes by 
    a given amount.
    --------------------------------------------------------------------------
    :param buy_price: the purchase price of the stock
    :param price_change: the price change from when the stock was bought to
                         when it is sold
    :return: The number of shares purchased and the profit or loss
    """     
    shares_bought = bid_size / buy_price
    profit_loss = shares_bought * price_change
    return shares_bought, profit_loss

# ----------------------------------------------------------------------------
def run_ledger(data, predictions, ticker):
    """
    run_ledger runs a hypothetical simulation of what would occur if a given 
    stock was purchased each time the decision function gave a buy signal.
    A dataframe is generated with the following format:
    INDEX | DATE | BALANCE | WIN/LOSS | SHARES | BUY $ | SELL $ | PROFIT/LOSS 
    The total number of transactions and the total profit / loss is reported.
    --------------------------------------------------------------------------
    :param data: a dataframe containing stock data
    :param predictions: an array of predictions
    :param ticker: the name of the stock 
    :return: Dataframe containing the information as described above
    """        
    # Define the starting account balance
    original_balance = 25000
    
    # Define the amount to spend on each transaction
    bid_size = 1000
    
    data['Predictions'] = predictions
    balance = original_balance
    empty_ledger = {'Date': [data.iloc[0][0] - timedelta(days = 1)],
                    'Balance': [balance],
                    'WL': ['-'],
                    'Shares' : [0],
                    'BuyPrice': [0],
                    'SellPrice': [0],
                    'ProfitLoss': [0]}
    transaction_history = pd.DataFrame(empty_ledger)

    for index, row in data.iterrows():
        if (row['Predictions']):
            n_shares, profit_loss = determine_pl(row['Close'], row['TenDayChange'], bid_size)
            previous_balance = balance
            balance = balance + profit_loss
            # Mark W (win) if balance increases, and mark L (loss) otherwise
            if (balance > previous_balance):
                outcome = 'W'
            else:
                outcome = 'L'
            new_row = {'Date': row['Date'],
                       'Balance': balance,
                       'WL': outcome,
                       'Shares': n_shares,
                       'BuyPrice': row['Close'],
                       'SellPrice': row['Close'] + row['TenDayChange'],
                       'ProfitLoss': profit_loss}
            transaction_history = transaction_history.append(new_row, ignore_index = True)
    
    print("SIMULATION USING BEST MODEL")
    print("------------------------------------")    
    print(transaction_history.head(25))
    print(SEPERATOR)
    print("Stock Symbol: ", ticker.upper())
    print("Number of Transactions: ", len(transaction_history))
    print("Total Profit / Loss = {:.2f}".format(balance - original_balance))
    print(SEPERATOR)

    return transaction_history

