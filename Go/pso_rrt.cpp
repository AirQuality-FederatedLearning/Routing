#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <map>
#include <set>
#include <random>
#include <json/json.h>

// Constants for geographic boundaries (example for Coimbatore)
const double MinLat = 10.90;
const double MaxLat = 11.13;
const double MinLon = 76.86;
const double MaxLon = 77.04;

// Struct for a Node (intersection)
struct Node {
    int64_t id;
    double lat;
    double lon;
};

// Struct for a RoadGraph containing nodes and edges
struct RoadGraph {
    std::map<int64_t, Node> nodes;
    std::map<int64_t, std::vector<int64_t>> edges;
};

// PSO + RRT Configuration settings
struct PSOConfig {
    int swarm_size = 20;          // Number of particles
    int num_iterations = 50;      // Number of iterations
    float w = 0.5;                // Inertia weight
    float c1 = 1.5;               // Personal attraction coefficient
    float c2 = 1.5;               // Global attraction coefficient
    int num_vehicles = 5;
    double max_route_distance = 15000;  // Maximum allowed route distance (meters)
    int max_route_steps = 20;     // Max number of steps per route
    float explore_prob = 0.3;     // Exploration probability
    float mutation_rate = 0.2;    // Route mutation probability
};

// Struct for the routes
struct Route {
    std::vector<std::vector<int64_t>> vehicles;  // Routes for each vehicle
    RoadGraph* graph;
    PSOConfig config;
    double fitness;

    // Random walk to generate a route
    std::vector<int64_t> randomRoute() {
        std::vector<int64_t> route;
        double total_dist = 0.0;
        std::vector<int64_t> nodes_list;

        // Collect all node IDs
        for (const auto& node : graph->nodes) {
            nodes_list.push_back(node.first);
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, nodes_list.size() - 1);

        int64_t current = nodes_list[dis(gen)];
        route.push_back(current);

        while (total_dist < config.max_route_distance) {
            const auto& neighbors = graph->edges[current];
            if (neighbors.empty()) break;
            int64_t next_node = neighbors[dis(gen)];
            route.push_back(next_node);
            total_dist += haversine(graph->nodes[current], graph->nodes[next_node]);
            current = next_node;
        }

        return route;
    }

    // Haversine formula to calculate distance between two nodes
    double haversine(const Node& a, const Node& b) {
        const double R = 6371e3;  // Earth's radius in meters
        double φ1 = a.lat * M_PI / 180;
        double φ2 = b.lat * M_PI / 180;
        double Δφ = (b.lat - a.lat) * M_PI / 180;
        double Δλ = (b.lon - a.lon) * M_PI / 180;

        double a1 = std::sin(Δφ / 2) * std::sin(Δφ / 2) + std::cos(φ1) * std::cos(φ2) * std::sin(Δλ / 2) * std::sin(Δλ / 2);
        double c = 2 * std::atan2(std::sqrt(a1), std::sqrt(1 - a1));
        return R * c;
    }

    // Calculate fitness based on route lengths and penalties for edge violations
    double fitnessFunction(const std::vector<std::vector<int64_t>>& solution) {
        double total_length = 0.0;
        std::map<std::pair<int64_t, int64_t>, int> edge_usage;

        for (const auto& route : solution) {
            double route_length = 0.0;
            for (size_t i = 0; i < route.size() - 1; ++i) {
                int64_t u = route[i];
                int64_t v = route[i + 1];
                route_length += haversine(graph->nodes[u], graph->nodes[v]);
                edge_usage[{u, v}]++;
                edge_usage[{v, u}]++;
            }
            total_length += route_length;
        }

        double penalty = 0.0;
        for (const auto& edge : edge_usage) {
            if (edge.second > 3) {
                penalty += config.mutation_rate * (edge.second - 3) * haversine(graph->nodes[edge.first.first], graph->nodes[edge.first.second]);
            }
        }

        return total_length - penalty;
    }

    // Build a route using random node approach
    std::vector<int64_t> buildRRTRoute(int max_route_steps) {
        std::vector<int64_t> route;
        int64_t current = route.empty() ? randomRoute()[0] : route.back();
        route.push_back(current);
        int steps = 0;
        while (steps < max_route_steps) {
            const auto& neighbors = graph->edges[current];
            if (neighbors.empty()) break;
            int64_t rand_node = neighbors[rand() % neighbors.size()];

            try {
                std::vector<int64_t> shortest_path = shortestPath(current, rand_node);
                route.insert(route.end(), shortest_path.begin(), shortest_path.end());
                steps += shortest_path.size();
            } catch (...) {
                continue;
            }
        }

        return route;
    }

    // Shortest path between two nodes
    std::vector<int64_t> shortestPath(int64_t start, int64_t end) {
        // Implement shortest path logic (Dijkstra, A*, etc.)
        std::vector<int64_t> path;
        // For simplicity, assuming a direct path is available
        path.push_back(start);
        path.push_back(end);
        return path;
    }

    // Initialize swarm for PSO
    std::vector<std::vector<std::vector<int64_t>>> initializeSwarm() {
        std::vector<std::vector<std::vector<int64_t>>> swarm;
        for (int i = 0; i < config.swarm_size; ++i) {
            std::vector<std::vector<int64_t>> solution;
            for (int j = 0; j < config.num_vehicles; ++j) {
                solution.push_back(buildRRTRoute(config.max_route_steps));
            }
            swarm.push_back(solution);
        }
        return swarm;
    }

    // Update velocity of particles
    std::vector<std::vector<int64_t>> updateVelocity(const std::vector<std::vector<int64_t>>& current_solution,
                                                     const std::vector<std::vector<int64_t>>& personal_best,
                                                     const std::vector<std::vector<int64_t>>& global_best) {
        // Implement velocity update using personal and global best solutions
        std::vector<std::vector<int64_t>> new_solution = current_solution; // Placeholder for update logic
        return new_solution;
    }

    // Mutate the solution
    void mutate(std::vector<std::vector<int64_t>>& solution) {
        for (auto& route : solution) {
            if (rand() % 100 < (config.mutation_rate * 100)) {
                route = buildRRTRoute(config.max_route_steps);
            }
        }
    }

    // Run PSO + RRT to optimize the route
    void run() {
        auto swarm = initializeSwarm();
        std::vector<std::vector<int64_t>> global_best;
        double global_best_fitness = -std::numeric_limits<double>::infinity();

        for (int iteration = 0; iteration < config.num_iterations; ++iteration) {
            for (auto& solution : swarm) {
                double fitness = fitnessFunction(solution);
                if (fitness > global_best_fitness) {
                    global_best_fitness = fitness;
                    global_best = solution;
                }
            }

            // Update velocity and positions of particles in the swarm
            for (auto& solution : swarm) {
                mutate(solution);  // Apply mutation
                solution = updateVelocity(solution, global_best, global_best);
            }
        }

        saveGeoJSON(global_best, "optimized_route.geojson");
    }

    // Convert solution to GeoJSON
    void saveGeoJSON(const std::vector<std::vector<int64_t>>& solution, const std::string& filename) {
        Json::Value root;
        root["type"] = "FeatureCollection";

        for (size_t i = 0; i < solution.size(); ++i) {
            Json::Value feature;
            feature["type"] = "Feature";
            feature["properties"]["vehicle"] = static_cast<int>(i + 1);

            Json::Value geometry;
            geometry["type"] = "MultiLineString";
            for (const auto& route : solution) {
                Json::Value coordinates;
                for (const auto& node_id : route) {
                    const Node& node = graph->nodes[node_id];
                    Json::Value coord;
                    coord.append(node.lon);
                    coord.append(node.lat);
                    coordinates.append(coord);
                }
                geometry["coordinates"].append(coordinates);
            }

            feature["geometry"] = geometry;
            root["features"].append(feature);
        }

        std::ofstream file(filename);
        file << root;
        file.close();
    }
};

int main() {
    srand(time(0));

    // Load road network data (this is a placeholder for loading from OSM data)
    RoadGraph road_graph;
    // road_graph = loadRoadData(...);

    // Create a Route instance and run the optimization
    PSOConfig config;
    Route optimizer(&road_graph, config);
    optimizer.run();

    std::cout << "Optimization completed, GeoJSON file generated!" << std::endl;

    return 0;
}
