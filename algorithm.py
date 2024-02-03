'DFS와 BFS'
# 첫째 줄에 정점의 개수 N(1 ≤ N ≤ 1,000), 간선의 개수 M(1 ≤ M ≤ 10,000), 탐색을 시작할 정점의 번호 V가 주어진다. 
# 다음 M개의 줄에는 간선이 연결하는 두 정점의 번호가 주어진다. 어떤 두 정점 사이에 여러 개의 간선이 있을 수 있다. 입력으로 주어지는 간선은 양방향이다.
# from collections import deque
# import sys
# input = sys.stdin.readline

# N,M,V =map(int, input().split())
# graph = [[] for _ in range(N + 1)]
# visited1 = [False] * (N+1)
# visited2 = [False] * (N+1)

# for i in range(M):
#     a, b = map(int, input().split())
#     graph[a].append(b)
#     graph[b].append(a)
# def dfs(n):
#     visited1[n] = True
#     print(n, end=' ')
#     for i in graph[n]:
#         if not visited1[i]:
#             dfs(i)

# def bfs(v):
#     q = deque([v])
#     visited2[v] = True
#     while q:
#         v = q.popleft()
#         print(v, end=' ')
#         for i in graph[v]:
#             if not visited2[i]:
#                 q.append(i)
#                 visited2[i] = True
    
# dfs(V)
# print()
# bfs(V)
# from collections import deque

# # 정점, 간선수, 시작점을 입력받음
# N, M, V = map(int, input().split())
# graph = [[] for _ in range(N + 1)]
# visited1 = [False] * (N + 1)
# visited2 = [False] * (N + 1)

# for i in range(M):
#     a, b = map(int, input().split())
#     graph[a].append(b)
#     graph[b].append(a)

# # 작은 순서대로 정렬
# for i in range(1, N + 1):
#     graph[i].sort()

# def dfs(n):
#     visited1[n] = True
#     print(n, end=' ')
#     for i in graph[n]:
#         if not visited1[i]:
#             dfs(i)

# def bfs(v):
#     q = deque([v])
#     visited2[v] = True
#     while q:
#         v = q.popleft()
#         print(v, end=' ')
#         for i in graph[v]:
#             if not visited2[i]:
#                 q.append(i)
#                 visited2[i] = True

# dfs(V)
# print()
# bfs(V)


# from collections import deque
# import sys
# sys.setrecursionlimit(10**6)
# input = sys.stdin.readline

# N, M = map(int, input().split())
# graph = [[] for _ in range(N + 1)]
# visited = [False] * (N + 1)
# cnt = 0

# for i in range(M):
#     a, b = map(int, input().split())
#     graph[a].append(b)
#     graph[b].append(a)

# def dfs(v):
#     visited[v] = True
#     for i in graph[v]:
#         if not visited[i]:
#             dfs(i)
# for i in range(1, N+1):
#     if not visited[i]:
#         dfs(i)
#         cnt += 1
# print(cnt)

# import sys
# sys.setrecursionlimit(10**6)
# input = sys.stdin.readline

# # dfs 함수
# def dfs(graph, v, visited):
#     visited[v] = True
#     for i in graph[v]:
#         if not visited[i]:
#             dfs(graph, i, visited)

# n, m = map(int, input().split()) # 정점의 개수, 간선의 개수
# graph = [[] for _ in range(n+1)]
# for i in range(m):
#     u, v = map(int, input().split())
#     graph[u].append(v)
#     graph[v].append(u)

# count = 0 # 연결 노드의 수
# visited = [False] * (n+1)
# for i in range(1, n+1):
#     if not visited[i]:
#         dfs(graph, i, visited)
#         count += 1 # dfs 한 번 끝날 때마다 count+1

# print(count)
'1476번 파이썬'
# e, s, m = map(int,input().split())
# E, S, M, Y = 1,1,1,1
# while True:
#     if e == E and s == S and m == M:
#         break
#     E += 1
#     S += 1
#     M += 1
#     Y += 1
#     if E >= 16:
#         E -= 15
#     if S >= 29:
#         S -= 28
#     if M >= 20:
#         M -= 19
# print(Y)

'리모콘 문제'
# +,-로 이동하는 횟수
# target = int(input())
# ans = abs(100 - target)
# M = int(input())
# if M:
#     broken = set(input().split())
# else:
#     broken = set()
# for num in range(1000001):
#     for N in range(num):
#         if N in broken:
#             break
#         else:
#             ans = min(ans, len(str(num)) + abs(num - target))
# print(ans)

'쉬운 계단수 문제'

# 0일때 0 출력
# dp[1] = 9 .....0
# dp[2] = 17 .....1
# dp[3] = 29 .....2
# dp[4] = 39 .....3
# 9 -> 17 -> 29
# 1일때 9 출력
# 2일때 17가지
# 12, 21,23, 32,34, 43,44,45, 56,54, 65,67, 76,78, 87,89, 98
# 3일 때 121,123, 212,234,232, 321,323,343,345, 456,454,432,434, 545,543,567,565, 656,654,676,678, 789,787,765,767, 878,876, 987,989 -> 29가지

# import sys
# n = int(sys.stdin.readline())

# dp = [[0] * 10 for _ in range(n+1)]

# for i in range(1, 10):
#     dp[1][i] = 1

# for i in range(2, n+1):
#     for j in range(10):
#         if j == 0:
#             dp[i][j] = dp[i-1][1]
#         elif 1 <= j <= 8:
#             dp[i][j] = dp[i-1][j-1] + dp[i-1][j+1]
#         else:
#             dp[i][j] = dp[i-1][8]
# print(sum(dp[n] )% 1000000000)
        
# 0일 때 0 
# 1~8일 때 0 or 2
# 9일 때 8 1가지

# dp[0] : 0 0 0 0 0 0 0 0 0 0
# dp[1] : 0 1 1 1 1 1 1 1 1 1
# dp[2] : 1 1 2 2 2 2 2 2 2 1
# dp[3] : 1 3 3 4 4 4 4 4 3 2
# dp[3][3] = dp[2][2] + dp[2][4] -> dp[i][j] = dp[i-1][j-1] + dp[i-1][j+1]
# dp[i][j] = dp[i-1][8]
# dp[i][j] = dp[i-1][1]
# 111

# # n = 1 일때 경우 초기화
#     1 0
# 1 : 1 0 --------------> 1
# 2 : 1 0 --------------> 10
# 3 : 2 0 --------------> 101, 100
# 4 : 3 0 --------------> 1010, 1001,1000
# 5 : 5 0 --------------> 10101, 10010, 10001, 10100, 10000
# # n > 2
# n = int(input())
# dp = [0] * (n+1)
# dp[1]=1
# dp[2]=1

# for i in range(3, n+1):
#     dp[i] = dp[i-1] + dp[i-2]

# print(dp[n])

'이분 그래프 어렵다..'
# 아직 방문하지 않았다면
# 현재 노드와 다른 색을 전달
# 방문 했는데 색이 같아? group 그러면 False

# import sys
# input = sys.stdin.readline
# def dfs(v, visited, graph, group):
#     visited[v] = group
#     for i in graph[v]:
#         if visited[i] == 0: # 아직 방문하지 않았다면
#             result = dfs(i, visited, graph, -group)
#             if not result:
#                 return False
#         else:
#             visited[i] == group
#             return False
# K = int(input())
# for i in range(K):
#     V, E = map(int, input().split())
#     graph = [[] for _ in range(V+1)]
#     visited = [0] * (V+1)
#     for _ in range(E):
#         a, b = map(int, input().split())
#         graph[a].append(b)
#         graph[b].append(a)

#     for i in range(1, V+1):
#         if visited[i] == 0:
#             result = (dfs(i, visited, graph, 1))
#             if not result:
#                 break
#     print("YES" if result else "NO")
'테트로미노 구글링을 통해서 알게되었음.'
# import sys
# input = sys.stdin.readline
# N, M = map(int, input().split())
# move = [(0,1), (0,-1), (1,0), (-1,0)]
# board =[list(map(int, input().split())) for _ in range(N)]
# visited = [[False] * M for _ in range(N)]

# maxValue = 0

# # 제외한 모양들 최대값 계산
# def DFS(x, y, dsum, cnt):
#     global maxValue
#     # 모양이 완성되었을 때 최대값 계산
#     if cnt == 4:
#         maxValue = max(maxValue, dsum)
#         return
#     for n in range(4):
#         nx = x + move[n][0]
#         ny = y + move[n][1]
#         if 0 <= nx < N and 0 <= ny < M and not visited[nx][ny]:
#             visited[nx][ny] = True
#             DFS(nx, ny, dsum + board[nx][ny], cnt + 1)
#             visited[nx][ny] = False
# def excel(x,y):
#     global maxValue
#     for n in range(4):
#         tmp = board[x][y]
#         for k in range(3):
#             t = 
'N과 M 1'
# import sys
# input = sys.stdin.readline
# N, M = map(int, input().split())
# arr = set([i for i in range(1, N + 1)])

# for i in arr:
#     print(' '.join(map(str,i)))

# from itertools import permutations

# n, m = map(int, input().split())
# nums = [i for i in range(1, n+1)]

# p = list(permutations(nums, m))

# for i in p:
#     print(' '.join(map(str,i)))
'N과 M 3'
# n,m= map(int,input().split())
 
# s = []
 
# def dfs(n):
#     if len(s)==m:
#         print(' '.join(map(str,s)))
#         return
    
#     for i in range(n,n+1):
#         s.append(i)
#         dfs(i)
#         s.pop()
# dfs(1)

# n,m = map(int, input().split())
 
# s = []
 
# def dfs(start):
#     if len(s)==m:
#         print(' '.join(map(str,s)))
#         return
    
#     for i in range(start, n+1):
#         s.append(i)
#         dfs(i)
#         s.pop()
    
# dfs(1)

'N과 M 4'
# from itertools import combinations
# n, m = map(int, input().split())
# nums = [i for i in range(1, n+1)]

# p = list(combinations(nums, m))
# for i in p:
#     print(' '.join(map(str,i)))
