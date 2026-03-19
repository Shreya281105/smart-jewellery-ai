import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# --- Config ---
N = 300

order_sources    = ['Website', 'Instagram', 'Facebook', 'Marketplace']
payment_methods  = ['UPI', 'Credit Card', 'Debit Card', 'COD', 'Net Banking']
return_statuses  = ['Not Returned', 'Returned']
genders          = ['Female', 'Male']
age_groups       = ['18-25', '26-35', '36-45', '46-60']
locations        = ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata']
customer_types   = ['New', 'Returning']

categories       = ['Necklace', 'Earrings', 'Ring', 'Bracelet', 'Bangle', 'Pendant', 'Anklet']
materials        = ['Gold', 'Silver', 'Diamond', 'Platinum', 'Rose Gold', 'Kundan']
occasions        = ['Wedding', 'Anniversary', 'Festival', 'Casual', 'Birthday', 'Valentine']
seasons          = ['Summer', 'Winter', 'Monsoon', 'Festive']

# Price ranges by material
price_map = {
    'Gold': (5000, 50000),
    'Silver': (500, 8000),
    'Diamond': (20000, 200000),
    'Platinum': (15000, 100000),
    'Rose Gold': (4000, 40000),
    'Kundan': (2000, 20000),
}

start_date = datetime(2023, 1, 1)

rows = []
for i in range(1, N + 1):
    material   = random.choice(materials)
    category   = random.choice(categories)
    occasion   = random.choice(occasions)
    season     = random.choice(seasons)
    price_low, price_high = price_map[material]
    price      = round(random.uniform(price_low, price_high), 2)
    qty        = random.randint(1, 4)
    discount   = random.choice([0, 5, 10, 15, 20, 25])
    total      = round(price * qty * (1 - discount / 100), 2)
    profit_pct = round(random.uniform(10, 45), 2)
    weight     = round(random.uniform(2, 50), 2)
    delivery   = random.randint(2, 10)
    rating     = round(random.uniform(2.5, 5.0), 1)
    ret_status = np.random.choice(return_statuses, p=[0.82, 0.18])
    order_date = start_date + timedelta(days=random.randint(0, 729))
    cust_type  = random.choice(customer_types)
    gender     = random.choice(genders)
    age        = random.choice(age_groups)
    location   = random.choice(locations)

    rows.append({
        'OrderID':        f'ORD{i:04d}',
        'OrderDate':      order_date.strftime('%Y-%m-%d'),
        'OrderSource':    random.choice(order_sources),
        'PaymentMethod':  random.choice(payment_methods),
        'DeliveryTime':   delivery,
        'ReturnStatus':   ret_status,
        'CustomerID':     f'CUST{random.randint(1, 150):04d}',
        'CustomerName':   f'Customer_{random.randint(1, 150)}',
        'Gender':         gender,
        'AgeGroup':       age,
        'Location':       location,
        'CustomerType':   cust_type,
        'ProductID':      f'PROD{random.randint(1, 80):04d}',
        'ProductName':    f'{material} {category}',
        'Category':       category,
        'Material':       material,
        'Weight':         weight,
        'Price':          price,
        'Quantity':       qty,
        'TotalAmount':    total,
        'Discount':       discount,
        'FeedbackRating': rating,
        'ProfitMargin':   profit_pct,
        'Season':         season,
        'Occasion':       occasion,
    })

df = pd.DataFrame(rows)

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'jewellery_sales.csv')
df.to_csv(save_path, index=False)
print(f"Saved to: {save_path}")
print(f"Dataset created: {len(df)} rows")
print(df.head())
