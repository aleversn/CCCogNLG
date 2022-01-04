# %%
import pymssql

class SQLManager:
    def __init__(self, databaseName, username, password, host='.'):
        self.host = host
        self.username = username
        self.password = password
        self.databaseName = databaseName
        self._GetConnect()
        
    def _GetConnect(self):
        self.conn = pymssql.connect(
            self.host, self.username, self.password, self.databaseName)
        cur = self.conn.cursor()
        if not cur:
            raise(NameError, "连接数据库失败")
        else:
            return cur

    def ExecQuery(self, sql):
        cur = self._GetConnect()
        cur.execute(sql)
        data = cur.fetchall()

        self.conn.close()
        return data

    def ExecNonQuery(self, sql):
        cur = self._GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()

    def ReadList(self, listName):
        l = self.ExecQuery('select * from {}'.format(listName))
        return l

    def ReadListFilter(self, listName, filter):
        l = self.ExecQuery(
            'select * from {} where {};'.format(listName, filter))
        return l

    def ValidUser(self, listName, col_id, id, col_pwd, pwd):
        l = self.ExecQuery(
            "SELECT * FROM {} WHERE {} = '{}' AND {} = '{}';".format(listName, col_id, id, col_pwd, pwd))
        return len(l) > 0

    def Exists(self, listName, col_name, value):
        l = self.ExecQuery(
            "SELECT * FROM {} WHERE {} = {};".format(listName, col_name, value))
        return len(l) > 0

    def ExistsFilter(self, listName, filter):
        l = self.ExecQuery(
            "SELECT * FROM {} WHERE {}".format(listName, filter))
        return len(l) > 0

    def Insert(self, listName, values):
        if(len(values) % 2 != 0):
            return False
        half = int(len(values) / 2)
        colName = ''
        colValue = ''
        for i in range(half):
            if i != half - 1:
                colName = colName + values[i] + ','
            else:
                colName = colName + values[i]
        for i in range(half, len(values)):
            if i != len(values) - 1:
                colValue = colValue + "'" + values[i] + "',"
            else:
                colValue = colValue + "'" + values[i] + "'"

        cmd = "INSERT INTO {} ({}) VALUES ({});".format(
            listName, colName, colValue)
        try:
            self.ExecNonQuery(cmd)
            return True
        except:
            return False
        finally:
            self.conn.close()

    def Inserts(self, listName, cols, vals):
        if(len(vals) % len(cols) != 0):
            return False
        colName = ''
        colValue = ''
        sumValue = ''
        for i in range(len(cols)):
            if i != len(cols) - 1:
                colName = colName + cols[i] + ','
            else:
                colName = colName + cols[i]

        groups = int(len(vals) / len(cols))
        for i in range(groups):
            colValue = ''
            for j in range(len(cols)):
                if j != len(cols) - 1:
                    colValue = colValue + "'" + vals[i * len(cols) + j] + "',"
                else:
                    colValue = colValue + "'" + vals[i * len(cols) + j] + "'"
            if(sumValue == ''):
                sumValue = "({})".format(colValue)
            else:
                sumValue += ",({})".format(colValue)

        cmd = "INSERT INTO {} ({}) VALUES {};".format(
            listName, colName, sumValue)
        self.ExecNonQuery(cmd)
        return True
        try:
            self.ExecNonQuery(cmd)
            return True
        except:
            return False
        finally:
            self.conn.close()

    def Update(self, listName, filter, values):
        if(len(values) % 2 != 0):
            return False
        half = int(len(values) / 2)
        final = ''
        for i in range(half):
            if i != half - 1:
                final = final + \
                    " {} = '{}',".format(values[i], values[i + half])
            else:
                final = final + \
                    " {} = '{}'".format(values[i], values[i + half])

        cmd = "UPDATE {} SET {} WHERE {};".format(listName, final, filter)
        try:
            self.ExecNonQuery(cmd)
            return True
        except:
            return False
        finally:
            self.conn.close()

    def Delete(self, listName, filter):
        cmd = "DELETE FROM {} WHERE {};".format(listName, filter)
        try:
            self.ExecNonQuery(cmd)
            return True
        except:
            return False
        finally:
            self.conn.close()

    def GetCol(self, listName, colName, filter):
        l = self.ExecQuery("SELECT {} FROM {} WHERE {};".format(
            colName, listName, filter))
        return l

    def GetRow(self, listName, colName, colValue):
        l = self.ExecQuery(
            "SELECT * FROM {} WHERE {} = '{}';".format(listName, colName, colValue))
        return l

    def Get(self, sqlCommand):
        l = self.ExecQuery('{}'.format(sqlCommand))
        return l

    def Set(self, sqlCommand):
        cmd = '{}'.format(sqlCommand)
        try:
            self.ExecNonQuery(cmd)
            return True
        except:
            return False
        finally:
            self.conn.close()
