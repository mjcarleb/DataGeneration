"""
Goals:
(1) Generate "real" firm trades  (10,000)
(2) Generate "real" market trades (10,000)
(3) Generate "synthetic historical" breaks - with break assignment labels for training a DT model (1,000,000)
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd

np.random.seed(42)
data_dir = "data/"

#################################################
# Create operational employee table
#################################################
ops_names = ["mary", "suzy", "tyrell",
             "hasan", "vicky", "asad",
             "monique", "gretchen", "trieu",
             "greta", "mia", "dante",
             "kareem", "amal", "anthony",
             "suresh", "paula", "tiger",
             "justin", "mirka", "julian",
             "bianca", "sabine", "myles",
             "mona", "bryce", "darnell",
             "yui", "riku", "haruto",
             "akari", "sakura", "william",
             "ema", "mei", "david",
             ]

ops_emails = [f"{n}@email.com" for n in ops_names]

ops_titles = ["manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              "manager", "analyst", "analyst",
              ]

ops_departments = ["sett_jam", "sett_jam", "sett_jam",
                   "ca_lime", "ca_lime", "ca_lime",
                   "lend_pie", "lend_pie", "lend_pie",
                   "sett_jam", "sett_jam", "sett_jam",
                   "ca_lime", "ca_lime", "ca_lime",
                   "lend_pie", "lend_pie", "lend_pie",
                   "sett_jam", "sett_jam", "sett_jam",
                   "ca_lime", "ca_lime", "ca_lime",
                   "lend_pie", "lend_pie", "lend_pie",
                   "sett_jam", "sett_jam", "sett_jam",
                   "ca_lime", "ca_lime", "ca_lime",
                   "lend_pie", "lend_pie", "lend_pie",
                   ]

ops_regions = ["us", "us", "us",
               "us", "us", "us",
               "us", "us", "us",
               "uk", "uk", "uk",
               "uk", "uk", "uk",
               "uk", "uk", "uk",
               "eu", "eu", "eu",
               "eu", "eu", "eu",
               "eu", "eu", "eu",
               "jpn", "jpn", "jpn",
               "jpn", "jpn", "jpn",
               "jpn", "jpn", "jpn",
               ]


ddf = dd.from_pandas(pd.DataFrame({"ops_names":ops_names,
                                   "ops_emails": ops_emails,
                                   "ops_titles": ops_titles,
                                   "ops_departments": ops_departments,
                                   "ops_regions":  ops_regions}),
                     npartitions=2)

ddf.to_parquet(path=f"{data_dir}ops_department")

df = pd.read_parquet(path=f"{data_dir}ops_department")

df.to_csv(f"{data_dir}test.csv")

def create_trans_ref_dist(n_trans):

    trans_ref_dist = [f"000000000{i}"[-10:] for i in range(n_trans)]
    return trans_ref_dist

def create_account_dist(n_trans):

    n_accounts = 250
    start_account = 0
    accounts = ["00000"+f"{int(acct)}"[-6:] for acct in range(start_account, start_account+n_accounts)]
    prob = [80/50/100 if i < 50 else 20/200/100 for i in range(n_accounts)]
    accounts_dist = [np.random.choice(a=accounts, p=prob) for i in range(n_trans)]
    return accounts_dist

def create_security_id_dist(n_trans):

    n_security_ids = 50
    security_ids = []
    for i in range(n_security_ids):
        alpha = np.random.choice(["AA", "BB", "CC", "WW", "ZZ"])
        numeric = f"000000{i}"[-7:]
        security_ids.append(f"{alpha}-{numeric}")
    prob = [80/10/100 if i < 10 else 20/40/100 for i in range(n_security_ids)]
    security_ids_dist = [np.random.choice(a=security_ids, p=prob) for i in range(n_trans)]
    return security_ids_dist

def create_quantity_dist(n_trans):

    quantity_dist = []
    for i in range(n_trans):
        p = np.random.sample()

        if p < 0.1:
            q = np.random.choice([20, 40, 60, 80, 100])

        elif p < 0.25:
            q = np.random.choice([100, 200, 300, 400, 500, 600, 700, 800, 900])

        elif p < 0.5:
            q = np.random.choice([1000, 1500, 2000, 2500, 5000, 6000, 7000, 7500, 8000, 9000])

        elif p < 0.9:
            q = np.random.choice([10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 55000, 60000, 65000, 70000,
                                  75000, 80000, 90000])
        elif p < 1.0:
            q = np.random.choice([100000, 150000, 200000, 250000])

        quantity_dist.append(f"00000000000{q}"[-12:])

    return quantity_dist


#################################################
# Create distributions of each field in trade file
#################################################
n_trans = 10000

create_trans_ref_dist(n_trans)
create_account_dist(n_trans)
create_security_id_dist(n_trans)
create_quantity_dist(n_trans)

"""
assume trade date: May 9, 2022
our party id:  1234 or 5678
Real firm trades:
(1) Agree on attributes of trade table
    - tran_ref (e.g., 10 digit number)
    - account_number (e.g., 6 digit)
    - security_id (e.g., 2 alpha + 7 numeric)
    - quantity (only positive, integer)
    - price (all positive, xx.xx double)
    - price_currency (see below...assume same as amount currency)
    - tran_type ("deliver free", "receive free", "deliver vs. payment", "receive vs. payment")
    - amount (0 for free, > 0 for payment)
    - amount_currency ("USD", "EUR", "GBP", "YEN")
    - counter_party (4 digits)
    - participant_id (4 digits:  us(1234 or 5678), eu(1235), uk(), jpn())  - 8 particpants)
    - market (DTC, EC, CREST, JASDEC") - us implies US, CREST can GBP or Euro
    - sd (YYYY-MM-DD)
    - actual_sd (YYYY-MM-DD)
    - source_system ("sett-jam", "ca-lime", "lend-pie")
    - source_system_ref s+9     c+7     l+8
    - trade_state:  verified, unverified, pending, settled, cancelled
    - userid:  s+     c+    l+   [system userid 5 digits.......or 5 letters for people....."s_39878" or "mcarl"]
    
    
(2) For each attribute, define "distribution" valid values
    - tran_ref (e.g., 10 digit number)   all unique
    - account_number (e.g., 6 digit)    250 account, top 50 get 80%
    - security_id (e.g., 2 alpha + 7 numeric)  50, top 10 get 80%......2 as sanctioned 
    - security_sactioned_ TRUE FALSE
    - quantity (only positive, integer)    100 100,000 random uniform
    - price (all positive, xx.xx double)........imagine a starting price for security & make price vary by up to 2%
    - price_currency (see below...assume same as amount currency)
    - tran_type ("deliver free", "receive free", "deliver vs. payment", "receive vs. payment") [10,10,40,40]
    - amount (0 for free, > 0 for payment)    20 / 80
    - amount_currency ("USD", "EUR", "GBP")   60, 20, 20
    - counter_party (4 digits)    10 per market......even participation  (excluding 2 more:  1234, 5678)
    - participant_id (4 digits:  1234 or 5678)   50/50   mandatory
    - sd (YYYY-MM-DD)   + 2 days May 11
    - actual_sd (YYYY-MM-DD)  + 2 days    May 11 100%
    - source_system ("sett-jam", "ca-lime", "lend-pie")   1/3, 1/3, 1/3
    - source_system_ref s+9     c+7     l+8    all unique
    - trade_state:  verified, unverified, pending, settled, cancelled   all settled
    - userid:  s+     c+    l+   [system userid 5 digits.......or 5 letters for people....."s_39878" or "mcarl"]....1+3 per system

"""

"""
Real market trades:
(1) For each attribute, define "distribution" valid values
    - tran_ref (e.g., 10 digit number)   all unique
    - account_number (e.g., 6 digit)    250 account, top 50 get 80%
    - security_id (e.g., 2 alpha + 7 numeric)  50, top 10 get 80%
    - quantity (only positive, integer)    100 100,000 random uniform
    - price (all positive, xx.xx double)........imagine a starting price for security & make price vary by up to 2%
    - price_currency (see below...assume same as amount currency)
    - tran_type ("deliver free", "receive free", "deliver vs. payment", "receive vs. payment") [10,10,40,40]
    - amount (0 for free, > 0 for payment)    20 / 80
    - amount_currency ("USD", "EUR", "GBP")   60, 20, 20
    - counter_party (4 digits)    10......even participation  (excluding 2 more:  1234, 5678)
    - participant_id (4 digits:  1234 or 5678)   50/50   mandatory
    - sd (YYYY-MM-DD)   + 2 days May 11
    - actual_sd (YYYY-MM-DD)  + 2 days    May 11 100%
    - source_system ("sett-jam", "ca-lime", "lend-pie")   1/3, 1/3, 1/3
    - source_system_ref s+9     c+7     l+8    all unique
    - trade_state:  verified, unverified, pending, settled, cancelled   all settled
    - userid:  s+     c+    l+   [system userid 5 digits.......or 5 letters for people....."s_39878" or "mcarl"]....1+3 per system] 90% system ids
    
    SOURCES OF NOISE:
    - Start with file above
    - Grab 0.5% and delete them (i.e., remove 1/2 1% of firm trades) = so, market trades = 9950 market trades
    - Grab 0.5^ of the firm trade and delete = so, I will have only 9950 firm trades
"""

"""
Synthetic firm trades:
(1) For each attribute, define "distribution" valid values
    - Add feature break_id
    - Add feature / label = ops_break_assignment
    - Features for deciding assignment
        - source system
        - userid
        - transaction_type
        - currency
        - market
    - Step 1:  You could tell real rules to generate the LAST ASSIGNMENT LABEL
        - Anyone of 12
        - 100% = source system + region
        - any where the price implicitly on trade is < $1, then LAST ASSIGNMENT LABEL COMPLIANCE
        - sanctioned security = SANCTIONS COMPLIANCE
        
    Step 2:  Add noise to 5% of trades
        - mix up currency and market
        - people work on trades outside their market
        - people working on systems outside their normal systems
        
    


36 poeple, 12 team (4 regions x 3 activities)
US - SETTLEMENT TEAM:
- suzy_johnson
- sam_patterson
- jane_doe (manager)

EUR - SETTLEMENT TEAM:
- jay_smith
- suzy_johnson
- sam_patterson
- jane_doe (manager)

UK - SETTLEMENT TEAM:
- jay_smith
- suzy_johnson
- sam_patterson
- jane_doe (manager)

JPN - SETTLEMENT TEAM:
- jay_smith
- suzy_johnson
- sam_patterson
- jane_doe (manager)

GLOBAL COMPLIANCE MANAGER = MARY, Bill = sanction

"""