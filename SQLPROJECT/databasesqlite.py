import sqlite3

connection = sqlite3.connect("student.db")

cursor=connection.cursor()

table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

cursor.execute('''Insert Into STUDENT values("Zidan","AI ENGINEERING","A",85)''')
cursor.execute('''Insert Into STUDENT values("Harsh","AI ENGINEERING","B",89)''')
cursor.execute('''Insert Into STUDENT values("Arshad","MUTHBAAZI","A",95)''')
cursor.execute('''Insert Into STUDENT values("Farhan","CHAMDI","C",100)''')
cursor.execute('''Insert Into STUDENT values("Ashif","FULL STACK DEV","D",90)''')

print("The inserted records are")

data = cursor.execute("""Select * from STUDENT""")
for row in data:
    print(row)

connection.commit()
connection.close()