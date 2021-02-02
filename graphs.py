
def find_leaves(graph):
    ''' return the keys that have empty values '''
    # return [k for k, v in graph.iteritems() if len(v) == 0]
    return [k for k, v in graph.items() if len(v) == 0]

def find_roots(graph):
    '''return the roots of the graph that are not pointed by the values'''
    # roots = graph.keys()
    roots = list(graph)
    # for onodes in graph.itervalues():
    for onodes in graph.values():
        [roots.remove(j) for j in onodes if j in roots]
    return roots

def recursive_dfs(graph, start, path=[]):
    '''recursive depth first search from start'''
    path = path + [start]
    for node in graph[start]:
        if not node in path:
            path = recursive_dfs(graph, node, path)
    return path

#       0
#       |
#       1
#      / \
#     2   3
#    / \   \
#   4   5   6
#  / \
# 7   8
#      \
#       9
G2_TEST = {0: [1,],
           1: [2, 3],
           2: [4, 5],
           3: [6,],
           4: [7, 8],
           5: [],
           6: [],
           7: [],
           8: [9],
           9: []}

def _condensate(graph, start, path=[], cache=[]):
    path = path + [start]

    if len(cache) == 0:
        cache = [[]]
    cache[-1] += [start]

    if len(graph[start]) > 1 or \
       len(graph[start]) == 0:
        cache.append([])

    for node in graph[start]:
        if not node in path:
            path, cache = _condensate(graph, node, path, cache)

    return path, cache

def condensate(graph, start):
    return _condensate(graph, start)[1][:-1]

#        0
#      /   \
#     1     2
#    / \   / \
#   3   4 5   6
#  / \
# 7   8
#    / \
#   9   10
G_TEST = {0: [1, 2],
          1: [3, 4],
          2: [5, 6],
          3: [7, 8],
          4: [],
          5: [],
          6: [],
          7: [], 8: [9, 10], 9: [], 10: []}


def all_paths(graph, start):
    schedule = recursive_dfs(graph, start)

    all_paths = []
    path = [schedule[0]]
    for node in schedule[1:]:

        # remove fiducial arcs
        while node not in graph[path[-1]]:
            path = path[:-1]

        # add node
        path.append(node)

        # if leaf, add path and remove from list
        if len(graph[node]) == 0:
            all_paths.append(path)
            path = path[:-1]

    return all_paths
