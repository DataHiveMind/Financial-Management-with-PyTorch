import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Core Data Structures ---

@dataclass
class Transaction:
    """
    Represents a single financial transaction.
    """
    date: datetime.date
    description: str
    amount: float
    category: str  # Added category
    account: str   # Added account

    def __post_init__(self):
        """
        Validation and data cleaning.
        """
        if not isinstance(self.date, datetime.date):
            raise TypeError("date must be a datetime.date object")
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.amount, (int, float)):
            raise TypeError("amount must be a number")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if not isinstance(self.account, str):
            raise TypeError("account must be a string")
        if self.amount == 0:
            raise ValueError("amount cannot be zero")
        self.description = self.description.strip()
        self.category = self.category.strip()
        self.account = self.account.strip()

    def __repr__(self):
        """
        Improved string representation for easier debugging and logging.  Includes account.
        """
        return f"Transaction(date={self.date}, description='{self.description}', amount={self.amount:.2f}, category='{self.category}', account='{self.account}')"


@dataclass
class Budget:
    """
    Represents a budget for a specific category.
    """
    category: str
    amount: float
    start_date: datetime.date
    end_date: datetime.date

    def __post_init__(self):
        """
        Validation
        """
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if not isinstance(self.amount, (int, float)):
            raise TypeError("amount must be a number")
        if not isinstance(self.start_date, datetime.date):
            raise TypeError("start_date must be a datetime.date object")
        if not isinstance(self.end_date, datetime.date):
            raise TypeError("end_date must be a datetime.date object")
        if self.amount < 0:
            raise ValueError("budget amount cannot be negative")
        if self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        self.category = self.category.strip()

    def __repr__(self):
        return f"Budget(category='{self.category}', amount={self.amount:.2f}, start_date={self.start_date}, end_date={self.end_date})"



@dataclass
class FinancialAccount:
    """
    Represents a financial account (e.g., checking, savings, credit card).
    """
    name: str
    account_type: str  # e.g., 'checking', 'savings', 'credit'
    balance: float = 0.0  # Initial balance, defaults to 0
    currency: str = "USD" # added currency

    def __post_init__(self):
        """
        Validation and data cleaning.
        """
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not isinstance(self.account_type, str):
            raise TypeError("account_type must be a string")
        if not isinstance(self.balance, (int, float)):
            raise TypeError("balance must be a number")
        if not isinstance(self.currency, str):
            raise TypeError("currency must be a string")
        self.name = self.name.strip()
        self.account_type = self.account_type.strip()
        self.currency = self.currency.strip().upper() # Ensure currency is uppercase

        # Basic account type validation (can be expanded)
        valid_account_types = ['checking', 'savings', 'credit', 'investment', 'cash']
        if self.account_type.lower() not in valid_account_types:
            raise ValueError(f"Invalid account type: {self.account_type}.  Must be one of {valid_account_types}")
        if len(self.currency) != 3:
            raise ValueError(f"Invalid currency code: {self.currency}. Must be a 3-letter currency code (e.g., USD, EUR, GBP)")

    def __repr__(self):
        return f"FinancialAccount(name='{self.name}', account_type='{self.account_type}', balance={self.balance:.2f}, currency='{self.currency}')"

# --- Helper Functions ---

def load_transactions_from_csv(filepath: str) -> List[Transaction]:
    """
    Loads transactions from a CSV file.  Handles more date formats and empty files.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}.  Returning an empty list of transactions.")
        return []
    except pd.errors.EmptyDataError:
        print(f"File is empty: {filepath}. Returning an empty list of transactions.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}.  Returning an empty list of transactions.")
        return []

    # Check for essential columns
    required_columns = ['date', 'description', 'amount', 'category', 'account']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV file.  Returning an empty list of transactions.")
            return []

    transactions = []
    for _, row in df.iterrows():
        try:
            # Handle different date formats
            date_str = row['date']
            if isinstance(date_str, str):
                try:
                    date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    try:
                        date = datetime.datetime.strptime(date_str, '%m/%d/%Y').date()
                    except ValueError:
                        date = datetime.datetime.strptime(date_str, '%Y/%m/%d').date()
            elif isinstance(date_str, datetime.date):
                date = date_str
            else:
                raise ValueError(f"Invalid date format: {date_str}")
            description = str(row['description'])
            amount = float(row['amount'])
            category = str(row['category'])
            account = str(row['account'])
            transaction = Transaction(date, description, amount, category, account)
            transactions.append(transaction)
        except Exception as e:
            print(f"Skipping invalid transaction: {row}. Error: {e}")
            continue  # Skip invalid rows

    return transactions



def save_transactions_to_csv(filepath: str, transactions: List[Transaction]):
    """
    Saves transactions to a CSV file.  Includes the 'account' field.
    """
    if not transactions:
        # Create an empty DataFrame and save it to CSV.
        pd.DataFrame(columns=['date', 'description', 'amount', 'category', 'account']).to_csv(filepath, index=False)
        return

    df = pd.DataFrame([
        {
            'date': t.date,
            'description': t.description,
            'amount': t.amount,
            'category': t.category,
            'account': t.account  # Include the account
        } for t in transactions
    ])
    df.to_csv(filepath, index=False)



def load_budgets_from_csv(filepath: str) -> List[Budget]:
    """
    Loads budgets from a CSV file. Handles file errors.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Returning an empty list of budgets.")
        return []
    except pd.errors.EmptyDataError:
        print(f"File is empty: {filepath}. Returning an empty list of budgets.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}. Returning an empty list of budgets.")
        return []

    # Check for essential columns
    required_columns = ['category', 'amount', 'start_date', 'end_date']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV file.  Returning an empty list.")
            return []
    budgets = []
    for _, row in df.iterrows():
        try:
            start_date = datetime.datetime.strptime(row['start_date'], '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(row['end_date'], '%Y-%m-%d').date()
            budget = Budget(
                category=str(row['category']),
                amount=float(row['amount']),
                start_date=start_date,
                end_date=end_date
            )
            budgets.append(budget)
        except Exception as e:
            print(f"Skipping invalid budget: {row}. Error: {e}")
    return budgets



def save_budgets_to_csv(filepath: str, budgets: List[Budget]):
    """
    Saves budgets to a CSV file.  Handles empty budget lists.
    """
    if not budgets:
        pd.DataFrame(columns=['category', 'amount', 'start_date', 'end_date']).to_csv(filepath, index=False)
        return

    df = pd.DataFrame([
        {
            'category': b.category,
            'amount': b.amount,
            'start_date': b.start_date,
            'end_date': b.end_date
        } for b in budgets
    ])
    df.to_csv(filepath, index=False)


def load_accounts_from_csv(filepath: str) -> List[FinancialAccount]:
    """Loads financial accounts from a CSV file."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File not found: {filepath}. Returning an empty list of accounts.")
        return []
    except pd.errors.EmptyDataError:
        print(f"File is empty: {filepath}. Returning an empty list of accounts.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}. Returning an empty list of accounts.")
        return []

    required_columns = ['name', 'account_type', 'balance', 'currency']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing column '{col}' in CSV file. Returning an empty list.")
            return []

    accounts = []
    for _, row in df.iterrows():
        try:
            account = FinancialAccount(
                name=str(row['name']),
                account_type=str(row['account_type']),
                balance=float(row['balance']),
                currency=str(row['currency'])
            )
            accounts.append(account)
        except Exception as e:
            print(f"Skipping invalid account: {row}. Error: {e}")
    return accounts



def save_accounts_to_csv(filepath: str, accounts: List[FinancialAccount]):
    """Saves financial accounts to a CSV file."""
    if not accounts:
        pd.DataFrame(columns=['name', 'account_type', 'balance', 'currency']).to_csv(filepath, index=False)
        return
    df = pd.DataFrame([
        {
            'name': account.name,
            'account_type': account.account_type,
            'balance': account.balance,
            'currency': account.currency
        } for account in accounts
    ])
    df.to_csv(filepath, index=False)



# --- Core Functions ---

def add_transaction(transactions: List[Transaction],
                    date: datetime.date,
                    description: str,
                    amount: float,
                    category: str,
                    account: str) -> List[Transaction]:
    """
    Adds a new transaction to the list.
    """
    try:
        transaction = Transaction(date, description, amount, category, account)
        transactions.append(transaction)
        return transactions
    except Exception as e:
        print(f"Error adding transaction: {e}")
        return transactions # Return original list on error

def get_transactions_by_date(transactions: List[Transaction], date: datetime.date) -> List[Transaction]:
    """
    Retrieves all transactions for a specific date.
    """
    return [t for t in transactions if t.date == date]



def get_transactions_by_category(transactions: List[Transaction], category: str) -> List[Transaction]:
    """
    Retrieves all transactions for a specific category.
    """
    return [t for t in transactions if t.category.lower() == category.lower()]

def get_transactions_by_account(transactions: List[Transaction], account: str) -> List[Transaction]:
    """
    Retrieves all transactions for a specific account.
    """
    return [t for t in transactions if t.account.lower() == account.lower()]


def calculate_total_expenses(transactions: List[Transaction]) -> float:
    """
    Calculates the total expenses from the list of transactions.
    """
    return sum(t.amount for t in transactions if t.amount < 0)



def calculate_total_income(transactions: List[Transaction]) -> float:
    """
    Calculates the total income from the list of transactions.
    """
    return sum(t.amount for t in transactions if t.amount > 0)



def generate_financial_summary(transactions: List[Transaction]) -> Dict[str, float]:
    """
    Generates a summary of income and expenses.
    """
    income = calculate_total_income(transactions)
    expenses = calculate_total_expenses(transactions)
    return {
        'income': income,
        'expenses': expenses,
        'net_balance': income + expenses
    }



def add_budget(budgets: List[Budget], category: str, amount: float, start_date: datetime.date, end_date: datetime.date) -> List[Budget]:
    """Adds a new budget to the list of budgets."""
    try:
        budget = Budget(category, amount, start_date, end_date)
        budgets.append(budget)
        return budgets
    except Exception as e:
        print(f"Error adding budget: {e}")
        return budgets # Return original list

def get_budget_by_category(budgets: List[Budget], category: str) -> Union[Budget, None]:
    """Retrieves a budget by its category."""
    for budget in budgets:
        if budget.category.lower() == category.lower():
            return budget
    return None



def update_budget(budgets: List[Budget], category: str, new_amount: float, new_start_date: datetime.date, new_end_date: datetime.date) -> List[Budget]:
    """
    Updates an existing budget.
    """
    for i, budget in enumerate(budgets):
        if budget.category.lower() == category.lower():
            try:
                updated_budget = Budget(category, new_amount, new_start_date, new_end_date)
                budgets[i] = updated_budget # Update in place
                return budgets
            except Exception as e:
                print(f"Error updating budget: {e}")
                return budgets # Return original
    print(f"Budget with category '{category}' not found.")
    return budgets # return original



def delete_budget(budgets: List[Budget], category: str) -> List[Budget]:
    """
    Deletes a budget from the list.
    """
    for i, budget in enumerate(budgets):
        if budget.category.lower() == category.lower():
            del budgets[i]
            return budgets
    print(f"Budget with category '{category}' not found.")
    return budgets # return original



def track_budget_vs_actual(transactions: List[Transaction], budgets: List[Budget]) -> Dict[str, Dict[str, float]]:
    """
    Tracks spending against budgets for each category.
    """
    category_totals = {}
    for transaction in transactions:
        category = transaction.category.lower()
        if transaction.amount < 0:  # Only track expenses
            if category not in category_totals:
                category_totals[category] = 0
            category_totals[category] += abs(transaction.amount)

    budget_vs_actual = {}
    for budget in budgets:
        category = budget.category.lower()
        actual_spending = category_totals.get(category, 0)
        budget_vs_actual[category] = {
            'budgeted': budget.amount,
            'actual': actual_spending,
            'variance': budget.amount - actual_spending
        }
    return budget_vs_actual



def add_account(accounts: List[FinancialAccount], name: str, account_type: str, balance: float, currency: str) -> List[FinancialAccount]:
    """Adds a new financial account."""
    try:
        account = FinancialAccount(name, account_type, balance, currency)
        accounts.append(account)
        return accounts
    except Exception as e:
        print(f"Error adding account: {e}")
        return accounts

def get_account_by_name(accounts: List[FinancialAccount], name: str) -> Union[FinancialAccount, None]:
    """Retrieves a financial account by its name."""
    for account in accounts:
        if account.name.lower() == name.lower():
            return account
    return None



def update_account(accounts: List[FinancialAccount], name: str, new_account_type: str, new_balance: float, new_currency: str) -> List[FinancialAccount]:
    """Updates an existing financial account."""
    for i, account in enumerate(accounts):
        if account.name.lower() == name.lower():
            try:
                updated_account = FinancialAccount(name, new_account_type, new_balance, new_currency)
                accounts[i] = updated_account
                return accounts
            except Exception as e:
                print(f"Error updating account: {e}")
                return accounts
    print(f"Account with name '{name}' not found.")
    return accounts



def delete_account(accounts: List[FinancialAccount], name: str) -> List[FinancialAccount]:
    """Deletes a financial account."""
    for i, account in enumerate(accounts):
        if account.name.lower() == name.lower():
            del accounts[i]
            return accounts
    print(f"Account with name '{name}' not found.")
    return accounts



def transfer_funds(accounts: List[FinancialAccount], from_account_name: str, to_account_name: str, amount: float, transaction_date: datetime.date, description: str = "Funds Transfer") -> List[FinancialAccount]:
    """Transfers funds between two accounts."""
    if amount <= 0:
        print("Transfer amount must be positive.")
        return accounts

    from_account = get_account_by_name(accounts, from_account_name)
    to_account = get_account_by_name(accounts, to_account_name)

    if not from_account or not to_account:
        print("One or both accounts not found.")
        return accounts

    if from_account.balance < amount:
        print("Insufficient funds for transfer.")
        return accounts

    if from_account.currency != to_account.currency:
        print("Currency mismatch.  Cannot transfer funds between accounts with different currencies.")
        return accounts


    from_account.balance -= amount
    to_account.balance += amount
    #add transaction
    transactions.append(Transaction(transaction_date, description + " to " + to_account_name, -amount, "Transfer", from_account_name))
    transactions.append(Transaction(transaction_date, description + " from " + from_account_name, amount, "Transfer", to_account_name))
    return accounts



# --- Visualization Functions ---

def visualize_transactions_by_category(transactions: List[Transaction], title: str = "Transactions by Category"):
    """
    Visualizes transactions by category using a bar chart.
    """
    if not transactions:
        print("No transactions to visualize.")
        return

    category_totals = {}
    for t in transactions:
        category = t.category.lower()
        if category not in category_totals:
            category_totals[category] = 0
        category_totals[category] += abs(t.amount)  # Use absolute value for consistent display

    if not category_totals:
        print("No transactions to visualize for the given categories.")
        return

    categories = list(category_totals.keys())
    totals = list(category_totals.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=totals)
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Total Amount")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()



def visualize_budget_vs_actual(budget_vs_actual: Dict[str, Dict[str, float]], title="Budget vs. Actual Spending"):
    """
    Visualizes budget vs. actual spending using a bar chart.
    """
    if not budget_vs_actual:
        print("No budget data to visualize.")
        return

    categories = list(budget_vs_actual.keys())
    budgeted_amounts = [data['budgeted'] for data in budget_vs_actual.values()]
    actual_amounts = [data['actual'] for data in budget_vs_actual.values()]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, budgeted_amounts, width, label='Budgeted')
    plt.bar(x + width/2, actual_amounts, width, label='Actual')
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Amount")
    plt.xticks(x, categories, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()



def visualize_account_balances(accounts: List[FinancialAccount], title="Account Balances"):
    """
    Visualizes account balances using a bar chart.
    """
    if not accounts:
        print("No accounts to visualize.")
        return

    account_names = [account.name for account in accounts]
    balances = [account.balance for account in accounts]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=account_names, y=balances)
    plt.title(title)
    plt.xlabel("Account Name")
    plt.ylabel("Balance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()



# --- PyTorch Model (Optional for Future Use) ---

class FinanceModel(nn.Module):
    """
    A simple neural network for predicting future financial values (e.g., expenses, income).
    This is a placeholder for future functionality and is not used in the current version.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(FinanceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(model: nn.Module,
                transactions: List[Transaction],
                input_size: int,
                hidden_size: int,
                output_size: int,
                learning_rate: float = 0.01,
                epochs: int = 100) -> None:
    """
    Trains a PyTorch model on transaction data.  This is a placeholder.
    """
    # Prepare data (this is a simplified example - real data prep would be more complex)
    # Example: Predict next day's expense based on previous transactions
    data = []
    for i in range(len(transactions) - 1):
        input_data = [
            transactions[i].amount,
            transactions[i].date.toordinal(),
        ]  # Example features: amount and date
        output_data = [transactions[i + 1].amount]  # Predict next amount
        data.append((torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)))

    if not data:
        print("Not enough data to train the model.")
        return

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters, lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for inputs, targets in data:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# --- Main Function ---

def main():
    """
    Main function to run the financial management application.
    """
    transactions_file = 'transactions.csv'
    budgets_file = 'budgets.csv'
    accounts_file = 'accounts.csv' # Added accounts file

    # Load data
    transactions = load_transactions_from_csv(transactions_file)
    budgets = load_budgets_from_csv(budgets_file)
    accounts = load_accounts_from_csv(accounts_file) # Load accounts

    # Initialize PyTorch model (placeholder - adjust input/output sizes as needed)
    input_size = 2  # Example: amount and date
    hidden_size = 8
    output_size = 1  # Predicting amount
    model = FinanceModel(input_size, hidden_size, output_size).to(device)

    # Main loop
    while True:
        print("\n--- Financial Management Menu ---")
        print("1. Add Transaction")
        print("2. View Transactions by Date")
        print("3. View Transactions by Category")
        print("4. View Transactions by Account") # Added
        print("5. Calculate Total Expenses")
        print("6. Calculate Total Income")
        print("7. Generate Financial Summary")
        print("8. Add Budget")
        print("9. View Budget by Category")
        print("10. Update Budget")
        print("11. Delete Budget")
        print("12. Track Budget vs. Actual")
        print("13. Visualize Transactions by Category")
        print("14. Visualize Budget vs. Actual")
        print("15. Add Financial Account") # Added
        print("16. View Account by Name") # Added
        print("17. Update Account") # Added
        print("18. Delete Account") # Added
        print("19. Visualize Account Balances") # Added
        print("20. Transfer Funds") # Added
        print("21. Train Model (Placeholder)")  # Added
        print("0. Exit")

        choice = input("Enter your choice: ")

        try:
            if choice == '1':
                date_str = input("Enter transaction date (YYYY-MM-DD): ")
                date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                description = input("Enter transaction description: ")
                amount = float(input("Enter transaction amount: "))
                category = input("Enter transaction category: ")
                account = input("Enter account name: ")
                transactions = add_transaction(transactions, date, description, amount, category, account)
                print("Transaction added.")

            elif choice == '2':
                date_str = input("Enter date to view transactions (YYYY-MM-DD): ")
                date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                transactions_on_date = get_transactions_by_date(transactions, date)
                if transactions_on_date:
                    print("\nTransactions on", date, ":")
                    for t in transactions_on_date:
                        print(t)
                else:
                    print("No transactions found for that date.")

            elif choice == '3':
                category = input("Enter category to view transactions: ")
                transactions_in_category = get_transactions_by_category(transactions, category)
                if transactions_in_category:
                    print(f"\nTransactions in category '{category}':")
                    for t in transactions_in_category:
                        print(t)
                else:
                    print("No transactions found for that category.")

            elif choice == '4':
                account = input("Enter account name to view transactions: ")
                transactions_in_account = get_transactions_by_account(transactions, account)
                if transactions_in_account:
                    print(f"\nTransactions in account '{account}':")
                    for t in transactions_in_account:
                        print(t)
                else:
                    print("No transactions found for that account.")

            elif choice == '5':
                total_expenses = calculate_total_expenses(transactions)
                print(f"Total Expenses: {total_expenses:.2f}")

            elif choice == '6':
                total_income = calculate_total_income(transactions)
                print(f"Total Income: {total_income:.2f}")

            elif choice == '7':
                summary = generate_financial_summary(transactions)
                print("\nFinancial Summary:")
                print(f"Income: {summary['income']:.2f}")
                print(f"Expenses: {summary['expenses']:.2f}")
                print(f"Net Balance: {summary['net_balance']:.2f}")

            elif choice == '8':
                category = input("Enter budget category: ")
                amount = float(input("Enter budget amount: "))
                start_date_str = input("Enter budget start date (YYYY-MM-DD): ")
                start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
                end_date_str = input("Enter budget end date (YYYY-MM-DD): ")
                end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
                budgets = add_budget(budgets, category, amount, start_date, end_date)
                print("Budget added.")

            elif choice == '9':
                category = input("Enter category to view budget: ")
                budget = get_budget_by_category(budgets, category)
                if budget:
                    print("\nBudget for", category, ":")
                    print(budget)
                else:
                    print("No budget found for that category.")

            elif choice == '10':
                category = input("Enter category of budget to update: ")
                new_amount = float(input("Enter new budget amount: "))
                new_start_date_str = input("Enter new budget start date (YYYY-MM-DD): ")
                new_start_date = datetime.datetime.strptime(new_start_date_str, '%Y-%m-%d').date()
                new_end_date_str = input("Enter new budget end date (YYYY-MM-DD): ")
                new_end_date = datetime.datetime.strptime(new_end_date_str, '%Y-%m-%d').date()

                budgets = update_budget(budgets, category, new_amount, new_start_date, new_end_date)
                print("Budget updated.")

            elif choice == '11':
                category = input("Enter category of budget to delete: ")
                budgets = delete_budget(budgets, category)
                print("Budget deleted.")

            elif choice == '12':
                budget_vs_actual = track_budget_vs_actual(transactions, budgets)
                if budget_vs_actual:
                    print("\nBudget vs. Actual Spending:")
                    for category, data in budget_vs_actual.items():
                        print(f"\nCategory: {category}")
                        print(f"  Budgeted: {data['budgeted']:.2f}")
                        print(f"  Actual: {data['actual']:.2f}")
                        print(f"  Variance: {data['variance']:.2f}")
                else:
                    print("No budget data available.")

            elif choice == '13':
                visualize_transactions_by_category(transactions)

            elif choice == '14':
                budget_vs_actual = track_budget_vs_actual(transactions, budgets)
                visualize_budget_vs_actual(budget_vs_actual)

            elif choice == '15':
                name = input("Enter account name: ")
                account_type = input("Enter account type (e.g., checking, savings, credit): ")
                balance = float(input("Enter initial balance: "))
                currency = input("Enter currency (e.g., USD, EUR, GBP): ")
                accounts = add_account(accounts, name, account_type, balance, currency)
                print("Account added.")

            elif choice == '16':
                name = input("Enter account name to view: ")
                account = get_account_by_name(accounts, name)
                if account:
                    print("\nAccount Details:")
                    print(account)
                else:
                    print("Account not found.")

            elif choice == '17':
                name = input("Enter account name to update: ")
                new_account_type = input("Enter new account type: ")
                new_balance = float(input("Enter new balance: "))
                new_currency = input("Enter new currency: ")
                accounts = update_account(accounts, name, new_account_type, new_balance, new_currency)
                print("Account updated.")

            elif choice == '18':
                name = input("Enter account name to delete: ")
                accounts = delete_account(accounts, name)
                print("Account deleted.")

            elif choice == '19':
                visualize_account_balances(accounts)

            elif choice == '20':
                from_account_name = input("Enter name of account to transfer from: ")
                to_account_name = input("Enter name of account to transfer to: ")
                amount = float(input("Enter amount to transfer: "))
                date_str = input("Enter transfer date (YYYY-MM-DD): ")
                transfer_date = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                accounts = transfer_funds(accounts, from_account_name, to_account_name, amount, transfer_date)
                print("Funds transferred.")

            elif choice == '21':
                train_model(model, transactions, input_size, hidden_size, output_size) #train the model
                print("Trained")

            elif choice == '0':
                print("Exiting application. Saving data...")
                save_transactions_to_csv(transactions_file, transactions)
                save_budgets_to_csv(budgets_file, budgets)
                save_accounts_to_csv(accounts_file, accounts) # Save accounts
                break

            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
