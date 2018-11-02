import pymysql.cursors
import pprint as pp

# Connect to the database
connection = pymysql.connect(host='cs2.mwsu.edu',
                             user='stock_user',
                             password='WeMg7jic5ysKzQNo',
                             db='ml_stock',
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
pp.pprint(stock_names)

for stock in stock_names:
    print(stock['Stock'])

# sql =  "SELECT * FROM `DowJonesComponentsDowIndexComp` "
# sql += "WHERE DAY(Date) = '10' AND MONTH(Date) = '10' AND YEAR(Date) = '2017'"
# result = selectData(sql)



# pp.pprint(result)
# print(len(result))