# here we are calculating the Interquartile Range (IQR) 
# for the 'res_price' column in the dataset to identify potential outliers. 

import pandas as pd 

dataset = pd.read_csv("data/greece_listings.csv")

Q1 = dataset['res_price'].quantile(0.25)
Q3 = dataset['res_price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

print(f"Unique addresses: {dataset['res_address'].nunique()}")

freq = dataset['res_address'].value_counts() # Counting how many times each address appears so.. 
dataset['neighborhood'] = dataset['res_address'].where( #..if it appears 10 or more i keep the name..
    dataset['res_address'].isin(freq[freq >= 20].index), other='Other' #..if it appears less i replace it with "Other"
)
print(dataset['neighborhood'].value_counts())
print(dataset['neighborhood'].nunique())
print(freq)