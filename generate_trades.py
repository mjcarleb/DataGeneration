"""
Goals:
(1) Generate "real" firm trades  (10,000)
(2) Generate "real" market trades (10,000)
(3) Generate "synthetic historical" breaks - with break assignment labels for training a DT model (1,000,000)
"""

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