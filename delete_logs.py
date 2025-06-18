import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect("log.db")  # replace with your actual DB filename
cursor = conn.cursor()

# Delete all records from 'logs' table
cursor.execute("DELETE FROM access_log")

# Commit changes and close connection
conn.commit()
conn.close()

print("All logs cleared.")
