import heapq
# Задание 1
def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = find_all_paths(graph, node, end, path)
            for p in new_paths:
                paths.append(p)
    return paths

graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': []
}
paths = find_all_paths(graph, 'A', 'B')
print("Длины всех путей из A в B:", [len(path) - 1 for path in paths])


# Задание 2
def dijkstra(graph, start, end):
    queue, visited, paths = [(0, start, [])], set(), {}
    while queue:
        (cost, current, path) = heapq.heappop(queue)
        if current not in visited:
            visited.add(current)
            paths[current] = (cost, path + [current])
            if current == end:
                return paths
            for neighbor, c in graph[current].items():
                heapq.heappush(queue, (cost + c, neighbor, path + [current]))


graph = {
    'A': {'B': 1, 'C': 3},
    'B': {'A': 1, 'D': 2},
    'C': {'A': 3, 'D': 1},
    'D': {}
}
shortest_paths = dijkstra(graph, 'A', 'D')
print("Кратчайший путь из A в D:", shortest_paths['D'][1])


# Задание 3
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        for j, weight in graph[i].items():
            dist[i][j] = weight

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

graph = {
    0: {1: 3, 2: 7},
    1: {0: 3, 2: 1},
    2: {0: 7, 1: 1}
}
shortest_paths = floyd_warshall(graph)
print("Кратчайшие пути между всеми парами вершин:", shortest_paths)




# Задание 4
def graph_diameter(graph):
    n = len(graph)
    diameter = 0

    for i in range(n):
        for j in range(n):
            if i != j and graph[i][j] != float('inf'):
                diameter = max(diameter, graph[i][j])

    return diameter

graph = [
    [0, 2, 4, float('inf')],
    [2, 0, 1, 5],
    [4, 1, 0, 3],
    [float('inf'), 5, 3, 0]
]
diameter = graph_diameter(graph)
print("Диаметр графа:", diameter)






# Задание 5
def min_spanning_tree(graph):
    n = len(graph)
    visited = set()
    min_spanning_tree_edges = []

    start_node = list(graph.keys())[0]
    priority_queue = [(0, start_node)]

    while priority_queue:
        weight, current_node = heapq.heappop(priority_queue)

        if current_node not in visited:
            visited.add(current_node)

            for neighbor, edge_weight in graph[current_node].items():
                if neighbor not in visited:
                    heapq.heappush(priority_queue, (edge_weight, neighbor))
                    min_spanning_tree_edges.append((current_node, neighbor, edge_weight))

    return min_spanning_tree_edges

graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'D': 5},
    'C': {'A': 4, 'D': 1},
    'D': {'B': 5, 'C': 1}
}
min_spanning_tree_edges = min_spanning_tree(graph)
print("Каркас минимального веса:", min_spanning_tree_edges)





# Задание 6
def complement(graph):
    complement_graph = {node: set() for node in graph}
    all_nodes = set(graph.keys())

    for node in graph:
        neighbors = set(graph[node])
        complement_graph[node] = all_nodes - neighbors - {node}

    return complement_graph


graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D'},
    'C': {'A', 'D'},
    'D': {'B', 'C'}
}
complement_graph = complement(graph)
print("Дополнение графа:", complement_graph)




# Задание 7
def is_complete(graph):
    all_nodes = set(graph.keys())

    for node in graph:
        neighbors = set(graph[node])
        if neighbors != all_nodes - {node}:
            return False

    return True

complete_graph = {
    'A': {'B', 'C'},
    'B': {'A', 'C'},
    'C': {'A', 'B'}
}
incomplete_graph = {
    'A': {'B', 'C'},
    'B': {'A'}
}
print("Граф complete_graph полный:", is_complete(complete_graph))
print("Граф incomplete_graph полный:", is_complete(incomplete_graph))






# Задание 8
def find_sources(graph):
    sources = set(graph.keys())

    for node in graph:
        for neighbors in graph.values():
            if node in neighbors:
                sources.discard(node)

    return sources

graph = {
    'A': {'B', 'C'},
    'B': {'C'},
    'C': {}
}
sources = find_sources(graph)
print("Источники графа:", sources)





# Задание 9
def unreachable_nodes(graph, start):
    visited = set()
    stack = [start]

    while stack:
        node = stack.pop()
        visited.add(node)
        stack.extend(neigh for neigh in graph[node] if neigh not in visited)

    return set(graph.keys()) - visited

graph = {
    'A': {'B', 'C'},
    'B': {'C'},
    'C': {}
}
unreachable = unreachable_nodes(graph, 'A')
print("Недостижимые из A вершины:", unreachable)






# Задача 10
def paths_through_vertex(graph, start, end, through_vertex):
    paths = []

    def dfs(current, path):
        if current == end:
            if through_vertex in path:
                paths.append(path)
            return
        for neighbor in graph[current]:
            if neighbor not in path:
                dfs(neighbor, path + [neighbor])

    dfs(start, [start])
    return paths

graph = {
    'A': {'B', 'C'},
    'B': {'D'},
    'C': {'D'},
    'D': {}
}
through_paths = paths_through_vertex(graph, 'A', 'D', 'B')
print("Пути из A в D через B:", through_paths)



# Задача 11
def find_paths_with_length(graph, start, end, length):
    paths = []

    def dfs(current, path, current_length):
        if current == end and current_length > length:
            paths.append(path)
            return
        for neighbor in graph[current]:
            if neighbor not in path:
                dfs(neighbor, path + [neighbor], current_length + 1)

    dfs(start, [start], 0)
    return paths

graph = {
    'A': {'B', 'C'},
    'B': {'D'},
    'C': {'D'},
    'D': {}
}
specified_length = 2
paths = find_paths_with_length(graph, 'A', 'D', specified_length)
print(f"Пути из A в D длиной больше {specified_length}: {paths}")




# Задача 12
def max_length_path(graph, start):
    max_length = 0
    max_path = []

    def dfs(current, path):
        nonlocal max_length, max_path
        if len(path) > max_length:
            max_length = len(path)
            max_path = path.copy()
        for neighbor in graph[current]:
            if neighbor not in path:
                dfs(neighbor, path + [neighbor])

    dfs(start, [start])
    return max_path

graph = {
    'A': {'B', 'C'},
    'B': {'D'},
    'C': {'D'},
    'D': {}
}
max_path = max_length_path(graph, 'A')
print("Путь максимальной длины из A:", max_path)



# Задача 13
def max_length_path_in_graph(graph):
    max_length = 0
    max_path = []

    for start_node in graph:
        path = max_length_path(graph, start_node)
        if len(path) > max_length:
            max_length = len(path)
            max_path = path

    return max_path

graph = {
    'A': {'B', 'C'},
    'B': {'D'},
    'C': {'D'},
    'D': {}
}
max_length_path = max_length_path_in_graph(graph)
print("Путь максимальной длины в графе:", max_length_path)





# Задача 14
def graph_median(graph):
    n = len(graph)
    all_paths = []

    for start_node in graph:
        for end_node in graph:
            if start_node != end_node:
                all_paths.extend(find_all_paths(graph, start_node, end_node))

    median = None
    min_sum_distance = float('inf')

    for node in graph:
        total_distance = sum(len(path) - 1 for path in all_paths if node in path)
        if total_distance < min_sum_distance:
            min_sum_distance = total_distance
            median = node

    return median

graph = {
    'A': {'B', 'C'},
    'B': {'D'},
    'C': {'D'},
    'D': {}
}
median = graph_median(graph)
print("Медиана графа:", median)




# Задача 15
def is_connected(graph):
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    start_node = next(iter(graph))
    dfs(start_node)

    return len(visited) == len(graph)

connected_graph = {
    'A': {'B', 'C'},
    'B': {'A', 'C'},
    'C': {'A', 'B'}
}
disconnected_graph = {
    'A': {'B'},
    'B': {'A'},
    'C': {}
}
print("Граф connected_graph связный:", is_connected(connected_graph))
print("Граф disconnected_graph связный:", is_connected(disconnected_graph))





# Задача 16
def is_tree(graph):
    visited = set()
    stack = [next(iter(graph))]

    while stack:
        node = stack.pop()
        visited.add(node)
        neighbors = graph[node]
        stack.extend(neighbor for neighbor in neighbors if neighbor not in visited)

    return len(visited) == len(graph) and all(len(graph[node]) == 2 for node in graph if node not in visited)

tree_graph = {
    'A': {'B', 'C'},
    'B': {'A'},
    'C': {'A'}
}
non_tree_graph = {
    'A': {'B', 'C'},
    'B': {'A', 'C'},
    'C': {'A', 'B'}
}
print("Граф tree_graph является деревом:", is_tree(tree_graph))
print("Граф non_tree_graph является деревом:", is_tree(non_tree_graph))









