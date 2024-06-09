import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue
import heapq
import time
from pyswip import Prolog


NUM_AGENTS = 2

def create_maze(dim):
    maze = np.ones((dim * 2 + 1, dim * 2 + 1))
    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0
    stack = [(x, y)]
    np.random.seed()

    while stack:
        x, y = stack[-1]
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2 * nx + 1, 2 * ny + 1] == 1:
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    for _ in range(int(dim * 1.5)):
        x, y = random.randint(0, dim - 1), random.randint(0, dim - 1)
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and maze[2 * x + 1, 2 * y + 1] == 0 and maze[2 * nx + 1, 2 * ny + 1] == 0:
                if maze[2 * x + 1 + dx // 2, 2 * y + 1 + dy // 2] == 1:
                    maze[2 * x + 1 + dx // 2, 2 * y + 1 + dy // 2] = 0
                    break

    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze

def find_path_bfs(maze, start, end, agent_paths):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = Queue()
    queue.put((start, [start]))
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True

    while not queue.empty():
        node, path = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                agent_paths.append(path + [next_node])
                return path + [next_node]
            if 0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1] and \
                    maze[next_node] != 1 and not visited[next_node]:
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))
    return []

def find_path_dfs(maze, start, end, agent_paths):
    stack = [(start, [start])]
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True

    while stack:
        node, path = stack.pop()
        if node == end:
            agent_paths.append(path + [end])
            return path + [end]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (node[0] + dx, node[1] + dy)
            if 0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1] and \
                    maze[next_node] != 1 and not visited[next_node]:
                visited[next_node] = True
                stack.append((next_node, path + [next_node]))
    return []

def find_path_greedy(maze, start, end, agent_paths):
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(start, end), start, [start]))
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True

    while priority_queue:
        _, node, path = heapq.heappop(priority_queue)
        if node == end:
            agent_paths.append(path + [end])
            return path + [end]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (node[0] + dx, node[1] + dy)
            if 0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1] and \
                    maze[next_node] != 1 and not visited[next_node]:
                visited[next_node] = True
                heapq.heappush(priority_queue, (heuristic(next_node, end), next_node, path + [next_node]))
    return []

def find_path_astar(maze, start, end, agent_paths):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start, [start]))
    cost_so_far = {start: 0}
    visited = np.zeros_like(maze, dtype=bool)

    while priority_queue:
        _, current, path = heapq.heappop(priority_queue)
        if current == end:
            agent_paths.append(path + [end])
            return path + [end]
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (current[0] + dx, current[1] + dy)
            new_cost = cost_so_far[current] + 1
            if 0 <= next_node[0] < maze.shape[0] and 0 <= next_node[1] < maze.shape[1] and maze[next_node] != 1:
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + heuristic(next_node, end)
                    heapq.heappush(priority_queue, (priority, next_node, path + [next_node]))
                    visited[next_node] = True

    return []

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])  

def draw_maze(maze, paths=None, title=""):
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')

    if paths:
        for i, path in enumerate(paths):
            if path:
                x_coords = [x[1] for x in path]
                y_coords = [y[0] for y in path]
                ax.plot(x_coords, y_coords, color='C'+str(i), linewidth=2, label=f"Agent {i+1}")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.show()

def plot_times(times, algorithms):
    fig, ax = plt.subplots()
    ax.barh(algorithms, times, color=['gold', 'blue', 'green', 'red'])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Time taken by each algorithm')
    plt.show()

def run_algorithm(algorithm, maze, agents):
    paths = []
    times = []
    agent_paths = []
    for start, end in agents:
        start_time = time.time()
        path = algorithm(maze, start, end, agent_paths)
        end_time = time.time()
        paths.append(path)
        times.append(end_time - start_time)
    return agent_paths, times

def logic_reasoning(agent_paths):
    prolog = Prolog()

    prolog.assertz("adjacent(X, Y) :- nextto(X, Y).")
    prolog.assertz("adjacent(X, Y) :- nextto(Y, X).")
    prolog.assertz("nextto(1, 2).")
    prolog.assertz("nextto(2, 3).")
    prolog.assertz("nextto(3, 4).")
    prolog.assertz("nextto(4, 5).")

   
    for i, path in enumerate(agent_paths):
        print(f"Agent {i + 1} path: {path}")
        for node in path:
            query = f"adjacent({node[0]}, {node[1]})."
            result = list(prolog.query(query))
            if result:
                print(f"Agent {i + 1}: Moving from {node[0]} to {node[1]} based on Prolog rule.")

if __name__ == "__main__":
    dim = int(input("Enter the dimensions for the maze: "))
    maze = create_maze(dim)
    
    print("Select an algorithm to visualize the path:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Greedy Best-First Search")
    print("4. A* Search")
    print("5. All of the above")
    choice = int(input("Enter the number of your choice: "))

    agents = [
        ((1, 1), (maze.shape[0] - 2, maze.shape[1] - 2)),
        ((maze.shape[0] - 2, maze.shape[1] - 2), (1, 1))
    ]

    if choice == 1:
        agent_paths, times = run_algorithm(find_path_bfs, maze, agents)
        draw_maze(maze, agent_paths, "Breadth-First Search (BFS)")
        plot_times(times, ["Agent 1 (Start -> End)", "Agent 2 (End -> Start)"])
    elif choice == 2:
        agent_paths, times = run_algorithm(find_path_dfs, maze, agents)
        draw_maze(maze, agent_paths, "Depth-First Search (DFS)")
        plot_times(times, ["Agent 1 (Start -> End)", "Agent 2 (End -> Start)"])
    elif choice == 3:
        agent_paths, times = run_algorithm(find_path_greedy, maze, agents)
        draw_maze(maze, agent_paths, "Greedy Best-First Search")
        plot_times(times, ["Agent 1 (Start -> End)", "Agent 2 (End -> Start)"])
    elif choice == 4:
        agent_paths, times = run_algorithm(find_path_astar, maze, agents)
        draw_maze(maze, agent_paths, "A* Search")
        plot_times(times, ["Agent 1 (Start -> End)", "Agent 2 (End -> Start)"])
    elif choice == 5:
        bfs_paths, bfs_times = run_algorithm(find_path_bfs, maze, agents)
        draw_maze(maze, bfs_paths, "Breadth-First Search (BFS)")

        dfs_paths, dfs_times = run_algorithm(find_path_dfs, maze, agents)
        draw_maze(maze, dfs_paths, "Depth-First Search (DFS)")

        greedy_paths, greedy_times = run_algorithm(find_path_greedy, maze, agents)
        draw_maze(maze, greedy_paths, "Greedy Best-First Search")

        astar_paths, astar_times = run_algorithm(find_path_astar, maze, agents)
        draw_maze(maze, astar_paths, "A* Search")

        times = [
            sum(bfs_times) / len(bfs_times),
            sum(dfs_times) / len(dfs_times),
            sum(greedy_times) / len(greedy_times),
            sum(astar_times) / len(astar_times)
        ]
        algorithms = ["Breadth-First Search (BFS)", "Depth-First Search (DFS)", "Greedy Best-First Search", "A* Search"]
        plot_times(times, algorithms)

        best_algorithm = algorithms[np.argmin(times)]
        print(f"The best algorithm is {best_algorithm} with an average time of {min(times):.4f} seconds.")
    else:
        print("Invalid choice!")

