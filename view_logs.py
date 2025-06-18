import sqlite3
import csv

# Connect to database
conn = sqlite3.connect('log.db')
cursor = conn.cursor()

# Fetch all rows
cursor.execute("SELECT * FROM access_log ORDER BY timestamp DESC")
rows = cursor.fetchall()

# Display logs
print("\n--- Access Log ---")
for row in rows:
    print(f"ID: {row[0]} | Time: {row[1]} | Name: {row[2]} | Plate: {row[3]}")

# Export to CSV
with open("access_log.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID", "Timestamp", "Name", "Plate"])
    writer.writerows(rows)

print("\n✅ Logs exported to access_log.csv")

# Close connection
conn.close()
