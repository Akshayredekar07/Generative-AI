import sqlite3

# connect to sqllite3 
connection = sqlite3.connect("student_data.db")

# create a cursor object to insert data and create table 
cursor = connection.cursor()

# create a table
table_info="""
create table Student(
name varchar(60),
class varchar(20),
section varchar(25),
marks int
)
"""

cursor.execute(table_info)


# insert records
cursor.execute("""
INSERT INTO Student (name, class, section, marks) VALUES
('Amit Sharma', '10', 'A', 85),
('Sneha Patel', '10', 'B', 78),
('Rahul Gupta', '9', 'A', 92),
('Pooja Verma', '11', 'C', 88),
('Vikas Kumar', '12', 'A', 91),
('Anjali Singh', '9', 'B', 73),
('Rohan Malhotra', '11', 'A', 84),
('Kiran Joshi', '10', 'C', 79),
('Arjun Reddy', '12', 'B', 87),
('Snehal Joshi', '9', 'C', 90),
('Manish Yadav', '11', 'A', 76),
('Sonal Mehta', '10', 'B', 82),
('Deepak Kumar', '12', 'C', 85),
('Neha Sharma', '9', 'A', 88),
('Vikram Singh', '11', 'B', 80);
""")


# dispaly all recods
data=cursor.execute("""SELECT * FROM Student""")

for index, row in enumerate(data):
    print(f"{index}: {row}")

# Commit the changes
connection.commit()
connection.close()
