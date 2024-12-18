#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

url = 'C:\\Users\\DELL\\Downloads\\online_retail_II.csv'
data = pd.read_csv(url,encoding='unicode_escape')

# 1. The first few rows of the dataset to understand its structure
print("Initial Data Preview:")
print(data.head())


# In[2]:


# 2. Handle Missing Data
# Identify missing values in each column
missing_data = data.isnull().sum()
missing_percentage = (missing_data / len(data)) * 100
print("\nMissing Data Overview:")
print(missing_percentage)

# If missing data is less than 5%, impute or drop rows; if more, we might drop the column
# Dropping rows with missing 'CustomerID' as it is critical for analysis
data = data.dropna(subset=['Customer ID'])
data['Price'] = data['Price'].fillna(data['Price'].median())

# 3. Detect and Handle Outliers
from scipy import stats

# Z-score method for detecting outliers in numerical columns
numerical_columns = ['Quantity', 'Price']
z_scores = np.abs(stats.zscore(data[numerical_columns]))

# Set threshold for outliers
outlier_threshold = 3
outliers = (z_scores > outlier_threshold).all(axis=1)

# Remove rows with outliers
data_cleaned = data[~outliers]

# 4. Ensure Data Type Consistency
# Convert 'InvoiceDate' to datetime type
data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'], errors='coerce')

# Check for correct data types
print("\nData Types after cleaning:")
print(data_cleaned.dtypes)

# 5. Feature Engineering
# Create a new 'TotalSales' column by multiplying Quantity and UnitPrice
data_cleaned['TotalSales'] = data_cleaned['Quantity'] * data_cleaned['Price']

# Extract temporal features from 'InvoiceDate'
data_cleaned['Year'] = data_cleaned['InvoiceDate'].dt.year
data_cleaned['Month'] = data_cleaned['InvoiceDate'].dt.month
data_cleaned['Day'] = data_cleaned['InvoiceDate'].dt.day

# Check the new features
print("\nData with new features:")
print(data_cleaned[['TotalSales', 'Year', 'Month', 'Day']].head())

# 6. Final Cleanup - Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()



# In[3]:


customers_df = data[['Customer ID', 'Country']].drop_duplicates().dropna()
customers_df.to_csv('cleaned_customers.csv', index=False)

# Step 2: Create Products Table
# For the Products table, we’ll use 'StockCode', 'Description', and 'Price'
products_df = data[['StockCode', 'Description', 'Price']].drop_duplicates().dropna()
products_df = products_df.rename(columns={'StockCode': 'ProductID', 'Description': 'ProductName'})
products_df.to_csv('cleaned_products.csv', index=False)

# Step 3: Create Sales Table
# For the Sales table, we’ll use 'Customer ID', 'StockCode' (as ProductID), 'Quantity', and 'InvoiceDate'
sales_df = data[['Invoice', 'Customer ID', 'StockCode', 'Quantity', 'Price', 'InvoiceDate']].copy()
sales_df['TotalSales'] = sales_df['Quantity'] * sales_df['Price']  # Calculate total sales
sales_df = sales_df.rename(columns={'StockCode': 'ProductID'})
sales_df = sales_df.dropna()
sales_df.to_csv('cleaned_sales.csv', index=False)

# Now, we have the cleaned data saved in separate CSV files:
# 'cleaned_customers.csv', 'cleaned_products.csv', 'cleaned_sales.csv'
print("CSV files have been created: cleaned_customers.csv, cleaned_products.csv, cleaned_sales.csv")


# In[6]:


import sqlite3
import pandas as pd

# Step 1: Create a SQLite Database and Tables
def create_database():
    # Create a the SQLite database 
    conn = sqlite3.connect('retail_data.db')
    cursor = conn.cursor()

    # Step 1: Create Customers Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Customers (
        CustomerID TEXT PRIMARY KEY,
        Country TEXT
    );
    """)

    # Step 2: Create Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Products (
        ProductID TEXT PRIMARY KEY,
        ProductName TEXT,
        Price REAL
    );
    """)

    # Step 3: Create Sales Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Sales (
        SaleID INTEGER PRIMARY KEY AUTOINCREMENT,
        CustomerID TEXT,
        ProductID TEXT,
        Quantity INTEGER,
        TotalSales REAL,
        InvoiceDate TEXT,
        FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID),
        FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
    );
    """)

    # Commit the changes to the database
    conn.commit()

    # Closing the connection for now
    conn.close()

    print("Database and tables have been created successfully.")

# Step 2: Load and Clean Data
def load_and_clean_data():
    # Load the original data
    data = pd.read_csv('C:\\Users\\DELL\\Downloads\\online_retail_II.csv')
    # Create Customers Table Data
    customers_df = data[['Customer ID', 'Country']].drop_duplicates().dropna()
    customers_df.to_csv('cleaned_customers.csv', index=False)

    # Create Products Table Data
    products_df = data[['StockCode', 'Description', 'Price']].drop_duplicates().dropna()
    products_df = products_df.rename(columns={'StockCode': 'ProductID', 'Description': 'ProductName'})
    products_df.to_csv('cleaned_products.csv', index=False)

    # Create Sales Table Data
    sales_df = data[['Invoice', 'Customer ID', 'StockCode', 'Quantity', 'Price', 'InvoiceDate']].copy()
    sales_df['TotalSales'] = sales_df['Quantity'] * sales_df['Price']  # Calculate total sales
    sales_df = sales_df.rename(columns={'StockCode': 'ProductID'})
    sales_df = sales_df.dropna()
    sales_df.to_csv('cleaned_sales.csv', index=False)

    print("CSV files have been created: cleaned_customers.csv, cleaned_products.csv, cleaned_sales.csv")

# Step 3: Insert Data into SQLite Database
def insert_data():
    # Load the cleaned CSV files
    customers_df = pd.read_csv('cleaned_customers.csv')
    products_df = pd.read_csv('cleaned_products.csv')
    sales_df = pd.read_csv('cleaned_sales.csv')

    # Connect to the SQLite database
    conn = sqlite3.connect('retail_data.db')
    cursor = conn.cursor()

    # Step 1: Insert data into Customers table
    for _, row in customers_df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO Customers (CustomerID, Country) VALUES (?, ?);
        """, (row['Customer ID'], row['Country']))

    # Step 2: Insert data into Products table
    for _, row in products_df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO Products (ProductID, ProductName, Price) VALUES (?, ?, ?);
        """, (row['ProductID'], row['ProductName'], row['Price']))

    # Step 3: Insert data into Sales table
    for _, row in sales_df.iterrows():
        cursor.execute("""
            INSERT INTO Sales (CustomerID, ProductID, Quantity, TotalSales, InvoiceDate)
            VALUES (?, ?, ?, ?, ?);
        """, (row['Customer ID'], row['ProductID'], row['Quantity'], row['TotalSales'], row['InvoiceDate']))

    # Commit the changes
    conn.commit()

    # Close the connection
    conn.close()

    print("Data has been successfully inserted into the database.")

# Step 4: Verify the Data Insertion
def verify_data():
    # Reconnect to the database and check if the data was inserted correctly
    conn = sqlite3.connect('retail_data.db')
    cursor = conn.cursor()

    # Verify Customers Table
    cursor.execute("SELECT * FROM Customers LIMIT 5;")
    print("Customers Table Sample:")
    print(cursor.fetchall())

    # Verify Products Table
    cursor.execute("SELECT * FROM Products LIMIT 5;")
    print("Products Table Sample:")
    print(cursor.fetchall())

    # Verify Sales Table
    cursor.execute("SELECT * FROM Sales LIMIT 5;")
    print("Sales Table Sample:")
    print(cursor.fetchall())

    # Close the connection
    conn.close()

def main():
    create_database()
    load_and_clean_data()
    insert_data()
    verify_data()
if __name__ == "__main__":
    main()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
# Connect to SQLite Database
conn = sqlite3.connect('retail_data.db')

# Create a cursor object
cursor = conn.cursor()

# SQL Queries to retrieve data
customers_query = "SELECT * FROM Customers;"
products_query = "SELECT * FROM Products;"
sales_query = "SELECT * FROM Sales;"

# Fetch data into pandas DataFrame
customers_df = pd.read_sql(customers_query, conn)
products_df = pd.read_sql(products_query, conn)
sales_df = pd.read_sql(sales_query, conn)

# Close the connection
conn.close()

# Preview data
print(customers_df.head())
print(products_df.head())
print(sales_df.head())

# Data Cleaning
# Convert InvoiceDate to datetime and drop rows with missing values
sales_df['InvoiceDate'] = pd.to_datetime(sales_df['InvoiceDate'])
sales_df.dropna(subset=['CustomerID', 'ProductID', 'Quantity', 'TotalSales'], inplace=True)

# Univariate Analysis - Distribution of Total Sales
plt.figure(figsize=(10, 6))
sns.histplot(sales_df['TotalSales'], bins=50, kde=True, color='blue')
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()

# Univariate Analysis - Boxplot for Quantity
plt.figure(figsize=(10, 6))
sns.boxplot(x=sales_df['Quantity'])
plt.title('Boxplot of Quantity')
plt.xlabel('Quantity')
plt.show()

# Bivariate Analysis - Sales by Country
country_sales = sales_df.groupby('CustomerID')['TotalSales'].sum().reset_index()
country_sales = country_sales.merge(customers_df[['CustomerID', 'Country']], on='CustomerID', how='left')

country_sales_summary = country_sales.groupby('Country')['TotalSales'].sum().sort_values(ascending=False)
country_sales_summary.head(10).plot(kind='bar', figsize=(12, 6), color='green')
plt.title('Total Sales by Country')
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Bivariate Analysis - Scatter Plot: Quantity vs Total Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=sales_df['Quantity'], y=sales_df['TotalSales'], color='purple')
plt.title('Quantity vs Total Sales')
plt.xlabel('Quantity')
plt.ylabel('Total Sales')
plt.show()

# Multivariate Analysis - Correlation Heatmap
corr_matrix = sales_df[['Quantity', 'TotalSales']].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, square=True)
plt.title('Correlation Heatmap')
plt.show()
sales_with_products = sales_df.merge(products_df[['ProductID', 'ProductName']], on='ProductID', how='left')
top_products = sales_with_products['ProductName'].value_counts().head(10).index
sales_subset = sales_with_products[sales_with_products['ProductName'].isin(top_products)]

plt.figure(figsize=(8, 8))  # Set a fixed size
sns.pairplot(sales_subset[['Quantity', 'TotalSales', 'ProductName']], hue='ProductName', palette='Set1', plot_kws={'alpha': 0.6})
plt.show()

# Statistical Analysis - Basic Statistics
mean_sales = sales_df['TotalSales'].mean()
median_sales = sales_df['TotalSales'].median()
std_sales = sales_df['TotalSales'].std()

print(f"Mean Sales: {mean_sales:.2f}")
print(f"Median Sales: {median_sales:.2f}")
print(f"Standard Deviation of Sales: {std_sales:.2f}")

# Correlation Coefficients - Quantity vs Total Sales
correlation = sales_df[['Quantity', 'TotalSales']].corr()
print("Correlation between Quantity and Total Sales:")
print(correlation)


# In[8]:


sales_df = pd.read_csv('cleaned_sales.csv')
# Check the first few rows and columns to ensure proper data loading
print(sales_df.head())


# In[10]:


# t-test for comparing sales between two products
# Example: t-test for comparing sales between two products
product1_sales = sales_df[sales_df['ProductID'] == 'A']['TotalSales']
product2_sales = sales_df[sales_df['ProductID'] == 'B']['TotalSales']

t_stat, p_value = stats.ttest_ind(product1_sales, product2_sales)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis - there is a significant difference between the two products.")
else:
    print("Fail to reject the null hypothesis - no significant difference between the two products.")


# In[11]:


# Create a contingency table for chi-square test (assuming 'ProductCategory' and 'CustomerID' exist)
contingency_table = pd.crosstab(sales_df['ProductID'], sales_df['Customer ID'])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-squared Statistic: {chi2}")
print(f"P-value: {p_val}")

# Interpretation
if p_val < 0.05:
    print("Reject the null hypothesis - there is a significant association between the variables.")
else:
    print("Fail to reject the null hypothesis - no significant association.")


# In[12]:


import statsmodels.api as sm
# Assuming 'Quantity' and 'Price' are predictors for 'TotalSales'
X = sales_df[['Quantity', 'Price']]  # Features
y = sales_df['TotalSales']  # Target variable

# Add constant for intercept
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()

# View regression results
print(model.summary())


# In[13]:


from sklearn.model_selection import train_test_split
# Prepare data for scikit-learn model
X = sales_df[['Quantity', 'Price']]  # Features
y = sales_df['TotalSales']  # Target variable
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data for scikit-learn model
X = sales_df[['Quantity', 'Price']]  # Features
y = sales_df['TotalSales']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[9]:


import seaborn as sns
from sklearn.cluster import KMeans
# Clustering based on 'TotalSales' and 'Quantity'
X = sales_df[['TotalSales', 'Quantity']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # 3 clusters, you can adjust based on your needs
sales_df['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
sns.scatterplot(x='Quantity', y='TotalSales', hue='Cluster', data=sales_df, palette='viridis')
plt.title('K-Means Clustering: Sales vs Quantity')
plt.show()

# Print the cluster centers
print("Cluster Centers:", kmeans.cluster_centers_)


# In[14]:


pip install --upgrade streamlit


# In[12]:


pip install pandas plotly seaborn


# In[14]:


streamlit_code ='''
import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Function to fetch data from SQLite database
def get_data(query):
    with sqlite3.connect("retail_data.db") as conn:
        return pd.read_sql_query(query, conn)

# Streamlit Setup (in Jupyter, use st.set_page_config() for layout setup)
st.set_page_config(page_title="Retail Dashboard", layout="wide")

# Load data
st.sidebar.header("Data Selection")
data_type = st.sidebar.radio("Select Data Type:", ("Sales", "Customers", "Products"))

if data_type == "Sales":
    data = get_data("SELECT * FROM Sales")
elif data_type == "Customers":
    data = get_data("SELECT * FROM Customers")
else:
    data = get_data("SELECT * FROM Products")

# Display Data
st.title(f"{data_type} Dashboard")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader(f"{data_type} Data")
    st.dataframe(data)

# Check the columns to avoid KeyError
st.write("Columns in the data:", data.columns)

# Basic Analysis
st.sidebar.header("Analysis Options")
analyze = st.sidebar.checkbox("Perform Analysis")

if analyze:
    if data_type == "Sales":
        st.subheader("Sales Analysis")
        total_sales = data['TotalSales'].sum()
        avg_sales = data['TotalSales'].mean()
        st.metric("Total Sales", f"${total_sales:,.2f}")
        st.metric("Average Sales", f"${avg_sales:,.2f}")

        # Sales by InvoiceDate (instead of Date or OrderDate)
        if 'InvoiceDate' in data.columns:
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
            daily_sales = data.groupby(data['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
            fig = px.line(daily_sales, x='InvoiceDate', y='TotalSales', title="Daily Sales Trend")
            st.plotly_chart(fig)
        else:
            st.warning("InvoiceDate column is not available in the Sales data.")

    elif data_type == "Customers":
        st.subheader("Customer Distribution by Country")

        # Group by 'Country' and count the number of unique 'CustomerID' for each country
        customer_distribution = data.groupby('Country')['CustomerID'].nunique().reset_index()
        customer_distribution.columns = ['Country', 'CustomerCount']

        # Plot the distribution
        fig = px.bar(customer_distribution, x='Country', y='CustomerCount', title="Number of Customers by Country")
        st.plotly_chart(fig)

    elif data_type == "Products":
        st.subheader("Product Analysis")
        # Assuming you want to calculate total sales by ProductID using the 'Sales' data
        sales_data = get_data("SELECT * FROM Sales")  # Assuming Sales data has ProductID and Quantity
        product_sales = sales_data.groupby('ProductID').agg({'TotalSales': 'sum'}).reset_index()
        product_sales = product_sales.sort_values(by='TotalSales', ascending=False).head(10)

        # Merge with Products data for ProductName
        products = get_data("SELECT * FROM Products")
        product_sales = pd.merge(product_sales, products[['ProductID', 'ProductName']], on='ProductID', how='left')

        fig = px.bar(product_sales, x='ProductName', y='TotalSales', title="Top 10 Most Sold Products")
        st.plotly_chart(fig, key="top_products")

# Advanced Visualizations
st.sidebar.header("Visualization Options")
visualize = st.sidebar.checkbox("Enable Visualizations")

if visualize:
    st.subheader("Custom Visualizations")
    if data_type == "Sales":
        st.markdown("*Sales Distribution by Product*")
        product_sales = data.groupby('ProductID')['TotalSales'].sum().reset_index()
        fig = px.pie(product_sales, values='TotalSales', names='ProductID', title="Sales by Product")
        st.plotly_chart(fig)

    elif data_type == "Customers":
        st.markdown("**Customer Distribution by Country (Pie Chart)**")

    # Group by 'Country' and count the number of unique 'CustomerID' for each country
        customer_distribution = data.groupby('Country')['CustomerID'].nunique().reset_index()
        customer_distribution.columns = ['Country', 'CustomerCount']

    # Plot the pie chart
        fig = px.pie(
            customer_distribution,
            names='Country',
            values='CustomerCount',
            title="Customer Distribution by Country",
            labels={'Country': 'Country', 'CustomerCount': 'Number of Customers'},
            hole=0.4  # Optional: Creates a donut chart effect
        )

    # Customize appearance
        fig.update_traces(textinfo='percent+label', textfont_size=14)
        fig.update_layout(
            title_font_size=20,
            showlegend=True
        )

    # Display the pie chart in Streamlit
        st.plotly_chart(fig, key="customer_distribution_piechart")



    elif data_type == "Products":
        st.markdown("*Product Sales Distribution*")
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(x=data['Price'])
        st.pyplot(fig)

st.sidebar.info("Use the options above to customize your dashboard.")'''
with open("dashboard.py", "w") as file:
    file.write(streamlit_code)
print("Streamlit code has been saved to 'dashboard.py'")

