# Import the simulator functions
from BrainControlledCraneSimulator import add_command_to_queue, main

# Commands from your ML model
ml_predictions = ['hoist', 'raise_boom', 'hoist', 'lower_boom', 'Q']
for command in ml_predictions:
    add_command_to_queue(command)


# Start the simulator - it will execute all queued commands
# The cell will block until you close the window (ESC or Q)
print("Starting crane simulator...")
main()

print("Simulator closed")