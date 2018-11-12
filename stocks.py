import pymysql.cursors
import pprint as pp
import json
with open('config.json', encoding='utf-8') as data_file:
   config = json.loads(data_file.read())
# Connect to the database
connection = pymysql.connect(host=config['host'],
                             user=config['user'],
                             password=config['password'],
                             db=config['db'],
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
def getAllStockNames():
    sql =  "SELECT DISTINCT(Stock) FROM `DowJonesComponentsDowIndexComp` "
    with connection.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    return result
def selectData(sql):
    print(sql)
    with connection.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()
    return result
stock_names = getAllStockNames()
# pp.pprint(stock_names)
for stock in stock_names:
    print(stock['Stock'])
    sql =  "SELECT *, '(Open+High+Low+Close)/4' as 'OHLC Avg', '(High+Low+Close)/3' as 'HLC Avg', "
    sql += "'(Close-Open)/(Open)' as 'Daily Return', "
    sql += "'AVG(Volume)' as 'Volume Avg', AVG(Value) as 'Value Avg', AVG(NumTicks) as 'NumTicks Avg', "
    sql += "STDDEV_SAMP(Open), STDDEV_SAMP(High), STDDEV_SAMP(Low), STDDEV_SAMP(Close) "
    sql += "FROM `DowJonesComponentsDowIndexComp` "
    sql += "GROUP BY 'Date'"
# sql =  "SELECT * FROM `DowJonesComponentsDowIndexComp` "
# sql += "WHERE DAY(Date) = '10' AND MONTH(Date) = '10' AND YEAR(Date) = '2017'"
result = selectData(sql)
pp.pprint(result)
print(len(result))
