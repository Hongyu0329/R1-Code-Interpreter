from collections import deque

# Define the initial state
start_number = 0
start_light = 'red'
target_number = 8

# Define the operations for each button
def press_A(number, light):
    return number + 2, 'green' if light == 'red' else 'red'

def press_B(number, light):
    if light == 'green':
        return number + 1, 'red'
    return None

def press_C(number, light):
    if light == 'green':
        return number * 2, 'red'
    return None

# BFS to find the shortest sequence
queue = deque([(start_number, start_light, [])])
visited = set()

while queue:
    number, light, path = queue.popleft()
    
    if number == target_number:
        print(' → '.join(path))
        break
    
    if (number, light) in visited:
        continue
    visited.add((number, light))
    
    # Try pressing each button
    new_state = press_A(number, light)
    if new_state:
        queue.append((new_state[0], new_state[1], path + ['A']))
    
    new_state = press_B(number, light)
    if new_state:
        queue.append((new_state[0], new_state[1], path + ['B']))
    
    new_state = press_C(number, light)
    if new_state:
        queue.append((new_state[0], new_state[1], path + ['C']))