from copy import deepcopy as copy

DB_NAME = 'deep_credit' # 딥크레딧 DB

### Secure information
DB_INFO = dict(host='165.246.34.133',
               port=3307,
               user='deepcredit',
               password='value!0328')

DB_INFO_deep_credit = copy(DB_INFO)
DB_INFO_deep_credit['db'] = DB_NAME