import pandas as pd
from datetime import timedelta


def determine_pl(buy_price, price_change):
    shares_bought = 1000 / buy_price
    profit_loss = shares_bought * price_change
    return shares_bought, profit_loss
    
def run_ledger(data, predictions, ticker):
    data['Predictions'] = predictions
    original_balance = 25000
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
            n_shares, profit_loss = determine_pl(row['Close'], row['TenDayChange'])
            
            previous_balance = balance
            balance = balance + profit_loss
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
            
    print(transaction_history)
    print()
    print("Stock: ", ticker.upper())
    print("Number of Transactions: ", len(transaction_history))
    print("Total Profit / Loss = ", balance - original_balance)
    
    return transaction_history