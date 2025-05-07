import pandas as pd
import datetime

def generate_transactions_data(filename="transactions.csv"):
    """
    Generates sample transaction data and saves it to a CSV file.
    """
    data = [
        {
            "date": datetime.date(2024, 1, 15),
            "description": "Grocery shopping",
            "amount": -50.00,
            "category": "Food",
            "account": "Checking Account",
        },
        {
            "date": datetime.date(2024, 1, 20),
            "description": "Salary deposit",
            "amount": 2000.00,
            "category": "Income",
            "account": "Checking Account",
        },
        {
            "date": datetime.date(2024, 1, 25),
            "description": "Rent payment",
            "amount": -1000.00,
            "category": "Housing",
            "account": "Checking Account",
        },
        {
            "date": datetime.date(2024, 2, 1),
            "description": "Dinner with friends",
            "amount": -30.00,
            "category": "Food",
            "account": "Credit Card",
        },
        {
            "date": datetime.date(2024, 2, 5),
            "description": "Bonus payment",
            "amount": 500.00,
            "category": "Income",
            "account": "Savings Account",
        },
        {
            "date": datetime.date(2024, 2, 10),
            "description": "Utilities bill",
            "amount": -75.00,
            "category": "Utilities",
            "account": "Checking Account",
        },
        {
            "date": datetime.date(2024, 2, 12),
            "description": "Valentine's Day Gift",
            "amount": -100,
            "category": "Gifts",
            "account": "Credit Card"
        },
        {
            "date": datetime.date(2024, 3, 1),
            "description": "Monthly Subscription",
            "amount": -15.99,
            "category": "Subscriptions",
            "account": "Checking Account"
        },
        {
            "date": datetime.date(2024, 3, 5),
            "description": "Car Insurance",
            "amount": -120.00,
            "category": "Transportation",
            "account": "Checking Account"
        },
        {
            "date": datetime.date(2024, 3, 10),
            "description": "Concert Tickets",
            "amount": -80.00,
            "category": "Entertainment",
            "account": "Credit Card"
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated sample transaction data and saved to {filename}")


def generate_budgets_data(filename="budgets.csv"):
    """
    Generates sample budget data and saves it to a CSV file.
    """
    data = [
        {
            "category": "Food",
            "amount": 300.00,
            "start_date": datetime.date(2024, 1, 1),
            "end_date": datetime.date(2024, 3, 31),
        },
        {
            "category": "Housing",
            "amount": 1000.00,
            "start_date": datetime.date(2024, 1, 1),
            "end_date": datetime.date(2024, 3, 31),
        },
        {
            "category": "Utilities",
            "amount": 150.00,
            "start_date": datetime.date(2024, 1, 1),
            "end_date": datetime.date(2024, 3, 31),
        },
        {
            "category": "Transportation",
            "amount": 200,
            "start_date": datetime.date(2024, 1, 1),
            "end_date": datetime.date(2024, 3, 31)
        },
        {
            "category": "Entertainment",
            "amount": 100,
            "start_date": datetime.date(2024, 1, 1),
            "end_date": datetime.date(2024, 3, 31)
        },
    ]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated sample budget data and saved to {filename}")


def generate_accounts_data(filename="accounts.csv"):
    """
    Generates sample financial account data and saves it to a CSV file.
    """
    data = [
        {
            "name": "Checking Account",
            "account_type": "checking",
            "balance": 1500.00,
            "currency": "USD",
        },
        {
            "name": "Savings Account",
            "account_type": "savings",
            "balance": 5000.00,
            "currency": "USD",
        },
        {
            "name": "Credit Card",
            "account_type": "credit",
            "balance": -500.00,  # Example of a credit card balance
            "currency": "USD",
        },
        {
            "name": "Investment Account",
            "account_type": "investment",
            "balance": 10000.00,
            "currency": "USD"
        },
        {
            "name": "Cash Account",
            "account_type": "cash",
            "balance": 200,
            "currency": "USD"
        }
    ]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated sample account data and saved to {filename}")

if __name__ == "__main__":
    generate_transactions_data()
    generate_budgets_data()
    generate_accounts_data()
