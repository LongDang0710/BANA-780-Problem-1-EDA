import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset from the specified file path
file_path = r"C:\Users\lilbu\Desktop\BANA 780 Problem 1 EDA\Superstore.csv"

# Load the CSV file
superstore_data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Inspect the data
print(superstore_data.head())  # Display the first few rows of the dataset
print(superstore_data.info())  # Show data types and non-null counts
print(superstore_data.describe(include='all'))  # Summary statistics for numerical and categorical columns

# Convert 'Order Date' to datetime format
superstore_data['Order Date'] = pd.to_datetime(superstore_data['Order Date'], dayfirst=True)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# 1. Total Sales, Profit, and Quantity Sold
total_sales = superstore_data['Sales'].sum()
total_profit = superstore_data['Profit'].sum()
total_quantity = superstore_data['Quantity'].sum()
print(f"Total Sales: ${total_sales:.2f}, Total Profit: ${total_profit:.2f}, Total Quantity Sold: {total_quantity}")

# Distribution Visualization
plt.figure(figsize=(12, 6))
sns.histplot(superstore_data['Sales'], bins=50, kde=True, color='blue', label='Sales', alpha=0.6)
sns.histplot(superstore_data['Profit'], bins=50, kde=True, color='green', label='Profit', alpha=0.6)
sns.histplot(superstore_data['Discount'], bins=50, kde=True, color='red', label='Discount', alpha=0.6)
plt.title('Distribution of Sales, Profit, and Discounts', fontsize=14)
plt.xlabel('Values', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.show()

# 2. Top Products by Sales and Profit
top_products_by_sales = superstore_data.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_products_by_profit = superstore_data.groupby('Product Name')['Profit'].sum().sort_values(ascending=False).head(10)

# Top Products by Sales Visualization
top_sales_products = top_products_by_sales.sort_values(ascending=True)
plt.figure(figsize=(10, 6))
top_sales_products.plot(kind='barh', color='orange')
plt.title('Top 10 Products by Sales', fontsize=14)
plt.xlabel('Sales', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Top Products by Profit Visualization
top_profit_products = top_products_by_profit.sort_values(ascending=True)
plt.figure(figsize=(10, 6))
top_profit_products.plot(kind='barh', color='purple')
plt.title('Top 10 Products by Profit', fontsize=14)
plt.xlabel('Profit', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 3. Most and Least Profitable Categories and Sub-Categories
category_profit = superstore_data.groupby('Category')['Profit'].sum().sort_values()
subcategory_profit = superstore_data.groupby('Sub-Category')['Profit'].sum().sort_values()

# Category Profit Visualization
plt.figure(figsize=(10, 6))
category_profit.plot(kind='bar', color='skyblue')
plt.title('Profit by Category', fontsize=14)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Sub-Category Profit Visualization
plt.figure(figsize=(12, 6))
subcategory_profit.plot(kind='bar', color='lightgreen')
plt.title('Profit by Sub-Category', fontsize=14)
plt.xlabel('Sub-Category', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Sales and Profit by Region
region_sales_profit = superstore_data.groupby('Region')[['Sales', 'Profit']].sum()

# Region Sales and Profit Visualization
region_sales_profit.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightgreen'])
plt.title('Sales and Profit by Region', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['Sales', 'Profit'])
plt.show()

# 5. Monthly Sales and Profit Trends
monthly_sales_profit = superstore_data.groupby(superstore_data['Order Date'].dt.to_period('M'))[['Sales', 'Profit']].sum()

# Monthly Sales and Profit Visualization
monthly_sales_profit.plot(figsize=(12, 6))
plt.title('Monthly Sales and Profit Trends', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.legend(['Sales', 'Profit'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pie Chart: Distribution of Customer Segments
plt.figure(figsize=(8, 8))
superstore_data['Segment'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    colors=['skyblue', 'orange', 'lightgreen'],
    labels=['Consumer', 'Corporate', 'Home Office'],
)
plt.title('Customer Segment Distribution', fontsize=14)
plt.ylabel('')  # Remove y-axis label
plt.show()

# Pie Chart: Distribution of Product Categories
plt.figure(figsize=(8, 8))
superstore_data['Category'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=140,
    colors=['coral', 'gold', 'lightblue'],
    labels=['Office Supplies', 'Furniture', 'Technology'],
)
plt.title('Product Category Distribution', fontsize=14)
plt.ylabel('')  # Remove y-axis label
plt.show()

# 1. Impact of Discounts on Profit Margins
discount_profit_correlation = superstore_data[['Discount', 'Profit']].corr().iloc[0, 1]

# Visualization: Scatterplot of Discount vs. Profit
plt.figure(figsize=(10, 6))
sns.scatterplot(data=superstore_data, x='Discount', y='Profit', alpha=0.5)
plt.title(f'Impact of Discount on Profit (Correlation: {discount_profit_correlation:.2f})', fontsize=14)
plt.xlabel('Discount', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.grid(alpha=0.7)
plt.show()

# 2. Correlation Analysis: Sales, Quantity, Discount, and Profit
correlation_matrix = superstore_data[['Sales', 'Quantity', 'Discount', 'Profit']].corr()

# Visualization: Heatmap of Correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix: Sales, Quantity, Discount, and Profit', fontsize=14)
plt.show()

# 3. Regional Analysis: Profitability
region_profitability = superstore_data.groupby('Region')['Profit'].sum().sort_values()

# Visualization: Profit by Region
plt.figure(figsize=(10, 6))
region_profitability.plot(kind='bar', color='teal')
plt.title('Profit by Region', fontsize=14)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 4. Customer Segment Analysis: Average Order Value and Frequency
segment_analysis = superstore_data.groupby('Segment').agg(
    Average_Order_Value=('Sales', 'mean'),
    Order_Frequency=('Order ID', 'count')
)

# Visualization: Segment Analysis
segment_analysis.plot(kind='bar', figsize=(10, 6), color=['orange', 'blue'])
plt.title('Customer Segment Analysis: Average Order Value and Frequency', fontsize=14)
plt.xlabel('Segment', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(rotation=0)
plt.legend(['Average Order Value', 'Order Frequency'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Aggregate only numeric columns by month for time-series modeling
monthly_data = superstore_data.groupby(superstore_data['Order Date'].dt.to_period('M')).sum(numeric_only=True)
monthly_data.index = monthly_data.index.to_timestamp()  # Convert PeriodIndex to Timestamp

# Splitting the data into training and testing sets (80% train, 20% test)
train_size = int(len(monthly_data) * 0.8)
train_data = monthly_data.iloc[:train_size]
test_data = monthly_data.iloc[train_size:]

# Build and fit the model for Sales
sales_model = ExponentialSmoothing(
    train_data['Sales'],
    seasonal='additive',
    seasonal_periods=12,
    trend='additive'
).fit()

# Forecast sales for the test period
sales_forecast = sales_model.forecast(len(test_data))

# Build and fit the model for Profit
profit_model = ExponentialSmoothing(
    train_data['Profit'],
    seasonal='additive',
    seasonal_periods=12,
    trend='additive'
).fit()

# Forecast profit for the test period
profit_forecast = profit_model.forecast(len(test_data))

# Extend the forecast horizon for future predictions
forecast_horizon = 12  # Predict 12 months into the future
extended_sales_forecast = sales_model.forecast(forecast_horizon)
extended_profit_forecast = profit_model.forecast(forecast_horizon)

# Create a datetime index for the extended forecast
forecast_index = pd.date_range(start=test_data.index[-1] + pd.offsets.MonthBegin(),
                               periods=forecast_horizon, freq='MS')

# Visualize the extended Sales Forecast
plt.figure(figsize=(14, 6))
plt.plot(train_data.index, train_data['Sales'], label='Training Sales', color='blue')
plt.plot(test_data.index, test_data['Sales'], label='Actual Sales', color='green')
plt.plot(test_data.index, sales_forecast, label='Forecasted Sales (Validation)', color='orange', linestyle='dashed')
plt.plot(forecast_index, extended_sales_forecast, label='Extended Forecasted Sales', color='red', linestyle='dotted')
plt.title('Sales Forecasting with Extended Predictions', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend()
plt.grid(alpha=0.7)
plt.show()

# Visualize the extended Profit Forecast
plt.figure(figsize=(14, 6))
plt.plot(train_data.index, train_data['Profit'], label='Training Profit', color='blue')
plt.plot(test_data.index, test_data['Profit'], label='Actual Profit', color='green')
plt.plot(test_data.index, profit_forecast, label='Forecasted Profit (Validation)', color='orange', linestyle='dashed')
plt.plot(forecast_index, extended_profit_forecast, label='Extended Forecasted Profit', color='red', linestyle='dotted')
plt.title('Profit Forecasting with Extended Predictions', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit', fontsize=12)
plt.legend()
plt.grid(alpha=0.7)
plt.show()

# **1. Discount Optimization**
# Evaluate the effect of discounts on profits
discount_analysis = superstore_data.groupby('Discount').agg(
    Avg_Profit=('Profit', 'mean'),
    Total_Sales=('Sales', 'sum'),
    Count=('Order ID', 'count')
)

# Visualization: Impact of Discount on Profitability
plt.figure(figsize=(12, 6))
plt.plot(discount_analysis.index, discount_analysis['Avg_Profit'], marker='o', color='blue', label='Average Profit')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Break-even Profit')
plt.title('Impact of Discount on Profit', fontsize=14)
plt.xlabel('Discount (%)', fontsize=12)
plt.ylabel('Average Profit', fontsize=12)
plt.legend()
plt.grid(alpha=0.7)
plt.show()

# **3. Shipping Optimization**
# Evaluate profitability across different shipping modes
shipping_analysis = superstore_data.groupby('Ship Mode').agg(
    Total_Profit=('Profit', 'sum'),
    Avg_Profit=('Profit', 'mean'),
    Total_Sales=('Sales', 'sum')
)

# Visualization: Shipping Modes Profitability
plt.figure(figsize=(10, 6))
shipping_analysis['Total_Profit'].plot(kind='bar', color='purple', alpha=0.8)
plt.title('Profit by Shipping Mode', fontsize=14)
plt.xlabel('Shipping Mode', fontsize=12)
plt.ylabel('Total Profit', fontsize=12)
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.show()

# **4. Inventory Management**
# Analyze underperforming products and suggest actions
low_profit_products = superstore_data.groupby('Product Name')['Profit'].sum().sort_values().head(10)
low_sales_products = superstore_data.groupby('Product Name')['Sales'].sum().sort_values().head(10)

# Visualization: Low-Profit Products
plt.figure(figsize=(12, 6))
low_profit_products.plot(kind='barh', color='orange', alpha=0.8)
plt.title('Low-Profit Products', fontsize=14)
plt.xlabel('Profit', fontsize=12)
plt.ylabel('Product Name', fontsize=12)
plt.grid(axis='x', alpha=0.7)
plt.tight_layout()
plt.show()

# Display actionable insights in tabular format
print("Discount Analysis:")
print(discount_analysis)
print("\nShipping Mode Analysis:")
print(shipping_analysis)
print("\nTop Products by Sales:")
print(top_products_by_sales)
print("\nLow-Profit Products:")
print(low_profit_products)
