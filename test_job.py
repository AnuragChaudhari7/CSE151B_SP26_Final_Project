import time
import os
from datetime import datetime

output_dir = "/home/ddeleena/private/CSE151B_SP26_Final_Project/data" #TODO: Change directory
output_file = os.path.join(output_dir, "test_background_job.txt")

os.makedirs(output_dir, exist_ok=True)
print(f"Starting 4-minute test job. Writing to {output_file}...")

#Runs for 4 mins
for i in range(1, 5):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Saved data at time {current_time} (Minute {i}/4)\n"
    
    # append (don't overwrite data)
    with open(output_file, "a") as f:
        f.write(message)
        
    print(f"Logged: {message.strip()}")
    time.sleep(60)

print("Test job complete!")

# command: nohup python -u test_job.py > test_job.log 2>&1 &