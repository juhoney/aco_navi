import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Ant:
    def __init__(self, num_locations):
        self.num_locations = num_locations
        self.tour = []
        self.total_cost = 0
        self.total_time = 0
        self.visited = [False] * num_locations
    def visit_location(self, location, cost, time):  # 지나간 노드 표시 및 거리 합산
        self.tour.append(location)
        self.total_cost += cost
        self.total_time += time
        self.visited[location] = True

    def clear(self):  # 초기화
        self.tour = []
        self.total_cost = 0
        self.total_time = 0
        self.visited = [False] * self.num_locations

class ACO:
    def __init__(self, num_ants, num_iterations, alpha, beta, evaporation_rate, num_locations, cost_matrix, trafficjam_matrix):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.num_locations = num_locations
        self.cost_matrix = cost_matrix
        self.trafficjam_matrix = trafficjam_matrix
        self.pheromone_matrix = np.ones((num_locations, num_locations)) * 0.01
        self.ants = [Ant(num_locations) for _ in range(num_ants)]
        self.initialize_pheromone_matrix()

    def initialize_pheromone_matrix(self):  # 페로몬 행렬 선언
        for i in range(self.num_locations):
            for j in range(self.num_locations):
                if i != j:
                    self.pheromone_matrix[i][j] = (1.0 / self.cost_matrix[i][j]) * (1.0 / self.trafficjam_matrix[i][j])

    def run(self, start_location, end_location):
        best_tour = None
        best_cost = float('inf')
        best_time = float('inf')

        for iteration in range(self.num_iterations):  # iteration: 반복횟수
            for ant in self.ants:
                ant.clear()
                ant.visit_location(start_location, 0, 0)

                while not ant.visited[end_location]:
                    current_location = ant.tour[-1]
                    next_location = self.select_next_location(ant, current_location)
                    ant.visit_location(next_location, self.cost_matrix[current_location][next_location], self.trafficjam_matrix[current_location][next_location])

                # Complete the tour by returning to the start location
                ant.visit_location(start_location, self.cost_matrix[ant.tour[-1]][start_location], self.trafficjam_matrix[ant.tour[-1]][start_location])

                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_tour = ant.tour[:]
                    best_time = ant.total_time

            self.update_pheromones()

        return best_tour, best_cost, best_time

    def select_next_location(self, ant, current_location):
        probabilities = []
        pheromone = self.pheromone_matrix[current_location]
        cost = self.cost_matrix[current_location]

        for location in range(self.num_locations):
            if not ant.visited[location]:
                prob = (pheromone[location] ** self.alpha) * ((1.0 / cost[location]) ** self.beta)
                probabilities.append(prob)
            else:
                probabilities.append(0)

        total = sum(probabilities)
        if total == 0:
            return random.choice([location for location in range(self.num_locations) if not ant.visited[location]])

        probabilities = [prob / total for prob in probabilities]
        next_location = np.random.choice(range(self.num_locations), p=probabilities)
        return next_location

    def update_pheromones(self):
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        for ant in self.ants:
            for i in range(len(ant.tour) - 1):
                from_location = ant.tour[i]
                to_location = ant.tour[i + 1]
                self.pheromone_matrix[from_location][to_location] += 1.0 / ant.total_cost

def create_cost_matrix(num_locations):
    np.random.seed(0)
    cost_matrix = np.random.randint(1, 50, size=(num_locations, num_locations))
    cost_matrix = np.minimum(cost_matrix, cost_matrix.T)  # 대칭 행렬로 만듦
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix

def create_trafficjam_matrix(num_locations):
    np.random.seed(1)
    trafficjam_matrix = np.random.randint(1, 100, size=(num_locations, num_locations))
    np.fill_diagonal(trafficjam_matrix, 1)
    return trafficjam_matrix

def plot_graph_with_edges(cost_matrix, route):
    num_locations = len(cost_matrix)
    G = nx.Graph()

    for i in range(num_locations):
        G.add_node(i)

    for i in range(num_locations):
        for j in range(i + 1, num_locations):
            if cost_matrix[i][j] != 0:
                G.add_edge(i, j, weight=cost_matrix[i][j])

    pos = nx.spring_layout(G)  # 노드의 위치 결정

    plt.figure(figsize=(45, 30))  # 그림 크기 조정

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)  # 노드 표시

    # 각 엣지에 대해 비용 행렬의 값으로 가중치 설정
    edge_labels = {(i, j): cost_matrix[i][j] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    nx.draw_networkx_edges(G, pos, edgelist=[(route[i], route[i + 1]) for i in range(len(route) - 1)], width=2, alpha=0.5, edge_color='red')  # 경로 표시
    nx.draw_networkx_edges(G, pos, edgelist=[(route[-1], route[0])], width=2, alpha=0.5, edge_color='red')  # 경로 표시

    plt.title('Optimal Route')
    plt.show()

def main():
    num_locations = 20
    num_ants = 100
    num_iterations = 200
    alpha = 1.0
    beta = 2.0
    evaporation_rate = 0.2

    cost_matrix = create_cost_matrix(num_locations)
    trafficjam_matrix = create_trafficjam_matrix(num_locations)
    print("Cost Matrix:")
    print(cost_matrix)
    print("Traffic Jam Matrix:")
    print(trafficjam_matrix)

    aco = ACO(num_ants, num_iterations, alpha, beta, evaporation_rate, num_locations, cost_matrix, trafficjam_matrix)

    # 입력 받은 출발지와 도착지
    start_location = int(input("출발지를 입력하세요 (0부터 {0}까지의 정수): ".format(num_locations-1)))
    end_location = int(input("도착지를 입력하세요 (0부터 {0}까지의 정수): ".format(num_locations-1)))

    best_tour, best_cost, best_time = aco.run(start_location, end_location)

    print(f"출발지: {start_location}, 도착지: {end_location}")
    print(f"최적 경로: {best_tour}")
    print(f"최적 경로의 총 거리: {best_cost}")
    print(f"총 걸린 시간: {best_time}")

    plot_graph_with_edges(cost_matrix, best_tour)

if __name__ == "__main__":
    main()
