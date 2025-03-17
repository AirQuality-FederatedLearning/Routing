package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/MaxHalford/eaopt"
	"github.com/qedus/osmpbf"
)

// --------------------------------------------------
// CONFIGURATION
// --------------------------------------------------

type Config struct {
	PopulationSize   int
	Generations      int
	MutationRate     float64
	CrossoverRate    float64
	NumVehicles      int
	MaxLat, MinLat   float64
	MaxLon, MinLon   float64
	BalanceWeight    float64
	OverusePenalty   float64
}

var config = Config{
	PopulationSize: 50,
	Generations:    100,
	MutationRate:   0.3,
	CrossoverRate:  0.7,
	NumVehicles:    5,
	MaxLat:         11.13,
	MinLat:         10.90,
	MaxLon:         77.04,
	MinLon:         76.86,
	BalanceWeight:  0.1,
	OverusePenalty: 10000.0,
}

// --------------------------------------------------
// ROAD GRAPH
// --------------------------------------------------

type Node struct {
	ID  int64
	Lat float64
	Lon float64
}

type RoadGraph struct {
	Nodes map[int64]Node
	Edges map[int64][]int64
}

// --------------------------------------------------
// SOLUTION (Genome)
// --------------------------------------------------

type Solution struct {
	Permutation []int64
	Graph       *RoadGraph
	StationID   int64
	Fitness     float64
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1) Load OSM data
	roadGraph := loadOSMData("southern-zone-latest.osm.pbf")

	// 2) Find nearest node to some station lat/lon
	stationLat, stationLon := 11.001516, 76.966536
	stationID := findNearestNode(roadGraph, stationLat, stationLon)
	fmt.Println("Chosen station node:", stationID)

	// 3) Configure the GA
	gaConfig := eaopt.GAConfig{
		NPops:         uint(runtime.NumCPU()),
		PopSize:       uint(config.PopulationSize),
		NGenerations:  uint(config.Generations),
		HofSize:       1,
		ParallelEval:  true,
		Model: eaopt.ModGenerational{
			Selector:  eaopt.SelTournament{NContestants: 3},
			MutRate:   config.MutationRate,
			CrossRate: config.CrossoverRate,
		},
	}

	ga, err := gaConfig.NewGA()
	if err != nil {
		log.Fatal(err)
	}

	ga.Callback = func(g *eaopt.GA) {
		best := g.HallOfFame[0]
		fmt.Printf("Generation %d/%d, Best Fitness = %.4f\n",
			g.Generations, config.Generations, best.Fitness)
	}

	// 4) Run the GA Minimization
	fmt.Println("Starting optimization...")
	start := time.Now()
	err = ga.Minimize(func(rng *rand.Rand) eaopt.Genome {
		return NewRandomSolution(roadGraph, stationID)
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Completed in %v\n", time.Since(start))

	best := ga.HallOfFame[0].Genome.(*Solution)
	fmt.Printf("Best Fitness: %.4f\n", best.Fitness)

	// 5) Save routes to GeoJSON
	saveGeoJSON(best, "routes.geojson")
	fmt.Println("Routes saved to routes.geojson")
}

// --------------------------------------------------
// OSM LOADING
// --------------------------------------------------

func loadOSMData(filename string) *RoadGraph {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	graph := &RoadGraph{
		Nodes: make(map[int64]Node),
		Edges: make(map[int64][]int64),
	}

	decoder := osmpbf.NewDecoder(file)
	if err := decoder.Start(runtime.GOMAXPROCS(-1)); err != nil {
		log.Fatal(err)
	}

	for {
		v, err := decoder.Decode()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		switch obj := v.(type) {
		case *osmpbf.Node:
			if obj.Lat >= config.MinLat && obj.Lat <= config.MaxLat &&
				obj.Lon >= config.MinLon && obj.Lon <= config.MaxLon {
				graph.Nodes[obj.ID] = Node{
					ID:  obj.ID,
					Lat: obj.Lat,
					Lon: obj.Lon,
				}
			}
		case *osmpbf.Way:
			if isRoadWay(obj) {
				addWayToGraph(graph, obj)
			}
		}
	}
	return graph
}

// Decide if a way is suitable for driving
func isRoadWay(way *osmpbf.Way) bool {
	for k, v := range way.Tags {
		if k == "highway" && v != "footway" && v != "path" {
			return true
		}
	}
	return false
}

// Add edges to the graph for each pair of consecutive nodes in the way
func addWayToGraph(graph *RoadGraph, way *osmpbf.Way) {
	for i := 0; i < len(way.NodeIDs)-1; i++ {
		a := way.NodeIDs[i]
		b := way.NodeIDs[i+1]

		_, okA := graph.Nodes[a]
		_, okB := graph.Nodes[b]

		// Only add edge if both nodes are within our area
		if okA && okB {
			graph.Edges[a] = append(graph.Edges[a], b)
			graph.Edges[b] = append(graph.Edges[b], a)
		}
	}
}

// --------------------------------------------------
// STATION HELPER
// --------------------------------------------------

func findNearestNode(graph *RoadGraph, lat, lon float64) int64 {
	var bestID int64
	minDist := math.MaxFloat64
	for id, node := range graph.Nodes {
		d := haversine(lat, lon, node.Lat, node.Lon)
		if d < minDist {
			minDist = d
			bestID = id
		}
	}
	return bestID
}

// --------------------------------------------------
// GENOME IMPLEMENTATION
// --------------------------------------------------

// NewRandomSolution creates a random solution (shuffled permutation of all nodes except stationID).
func NewRandomSolution(g *RoadGraph, stationID int64) *Solution {
	var nodes []int64
	for id := range g.Nodes {
		if id != stationID {
			nodes = append(nodes, id)
		}
	}

	rand.Shuffle(len(nodes), func(i, j int) {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	})

	return &Solution{
		Permutation: nodes,
		Graph:       g,
		StationID:   stationID,
		Fitness:     0.0,
	}
}

// Evaluate calculates the total distance + penalties (imbalance, edge overuse).
func (s *Solution) Evaluate() (float64, error) {
	segments := splitPermutation(s.Permutation, config.NumVehicles)

	// We'll calculate route distances and edge usage
	distances := make([]float64, config.NumVehicles)
	edgeUsage := make(map[[2]int64]int)

	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, seg := range segments {
		wg.Add(1)
		go func(idx int, route []int64) {
			defer wg.Done()

			vehicleDist, usedEdges := computeRouteDistanceAndEdges(s.Graph, s.StationID, route, s.StationID)
			mu.Lock()
			distances[idx] = vehicleDist
			for _, e := range usedEdges {
				if e[0] > e[1] {
					e[0], e[1] = e[1], e[0]
				}
				edgeUsage[[2]int64{e[0], e[1]}]++
			}
			mu.Unlock()
		}(i, seg)
	}
	wg.Wait()

	// Total distance
	totalDistance := 0.0
	for _, d := range distances {
		totalDistance += d
	}

	// Imbalance penalty (std dev)
	meanDist := totalDistance / float64(config.NumVehicles)
	var variance float64
	for _, d := range distances {
		diff := d - meanDist
		variance += diff * diff
	}
	stdDev := math.Sqrt(variance / float64(config.NumVehicles))
	imbalancePenalty := config.BalanceWeight * stdDev

	// Edge overuse penalty
	overusePenalty := 0.0
	for _, usage := range edgeUsage {
		if usage > 3 {
			overusePenalty += config.OverusePenalty * float64(usage-3)
		}
	}

	s.Fitness = totalDistance + imbalancePenalty + overusePenalty
	return s.Fitness, nil
}

// Crossover performs single-point crossover on the permutation.
func (s *Solution) Crossover(other eaopt.Genome, rng *rand.Rand) {
	o := other.(*Solution)
	if rng.Float64() > config.CrossoverRate {
		return
	}
	if len(s.Permutation) < 2 {
		return
	}
	point := rng.Intn(len(s.Permutation) - 1)
	child1 := append([]int64(nil), s.Permutation[:point]...)
	child2 := append([]int64(nil), o.Permutation[:point]...)

	// Add remainder from other, skipping duplicates
	for _, v := range o.Permutation {
		if !contains(child1, v) {
			child1 = append(child1, v)
		}
	}
	for _, v := range s.Permutation {
		if !contains(child2, v) {
			child2 = append(child2, v)
		}
	}
	s.Permutation = child1
	o.Permutation = child2
}

// Mutate swaps two random positions in the permutation.
func (s *Solution) Mutate(rng *rand.Rand) {
	if rng.Float64() > config.MutationRate {
		return
	}
	if len(s.Permutation) < 2 {
		return
	}
	i, j := rng.Intn(len(s.Permutation)), rng.Intn(len(s.Permutation))
	s.Permutation[i], s.Permutation[j] = s.Permutation[j], s.Permutation[i]
}

// Clone creates a deep copy of the solution.
func (s *Solution) Clone() eaopt.Genome {
	clone := &Solution{
		Permutation: make([]int64, len(s.Permutation)),
		Graph:       s.Graph,
		StationID:   s.StationID,
		Fitness:     s.Fitness,
	}
	copy(clone.Permutation, s.Permutation)
	return clone
}

// --------------------------------------------------
// HELPER FUNCTIONS
// --------------------------------------------------

// Splits a permutation into n nearly equal segments
func splitPermutation(perm []int64, n int) [][]int64 {
	length := len(perm)
	segments := make([][]int64, n)
	base := length / n
	extra := length % n

	start := 0
	for i := 0; i < n; i++ {
		size := base
		if i < extra {
			size++
		}
		end := start + size
		if end > length {
			end = length
		}
		segments[i] = perm[start:end]
		start = end
	}
	return segments
}

// computeRouteDistanceAndEdges returns the total distance and edges used
// in the route [start -> mid... -> end]
func computeRouteDistanceAndEdges(g *RoadGraph, start int64, mid []int64, end int64) (float64, [][2]int64) {
	routeNodes := make([]int64, 0, len(mid)+2)
	routeNodes = append(routeNodes, start)
	routeNodes = append(routeNodes, mid...)
	routeNodes = append(routeNodes, end)

	var dist float64
	var edges [][2]int64

	for i := 0; i < len(routeNodes)-1; i++ {
		a, b := routeNodes[i], routeNodes[i+1]

		ndA, okA := g.Nodes[a]
		ndB, okB := g.Nodes[b]
		if !okA || !okB {
			continue
		}
		d := haversine(ndA.Lat, ndA.Lon, ndB.Lat, ndB.Lon)
		dist += d
		edges = append(edges, [2]int64{a, b})
	}
	return dist, edges
}

// Haversine distance in meters
func haversine(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371e3
	dLat := (lat2 - lat1) * (math.Pi / 180.0)
	dLon := (lon2 - lon1) * (math.Pi / 180.0)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*(math.Pi/180.0))*math.Cos(lat2*(math.Pi/180.0))*math.Sin(dLon/2)*math.Sin(dLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

func contains(arr []int64, v int64) bool {
	for _, x := range arr {
		if x == v {
			return true
		}
	}
	return false
}

// --------------------------------------------------
// GEOJSON OUTPUT
// --------------------------------------------------

func saveGeoJSON(sol *Solution, filename string) {
	segments := splitPermutation(sol.Permutation, config.NumVehicles)

	var features []map[string]interface{}
	for vid, seg := range segments {
		fullRoute := append([]int64{sol.StationID}, seg...)
		fullRoute = append(fullRoute, sol.StationID)

		var coords [][]float64
		for _, nid := range fullRoute {
			if nd, ok := sol.Graph.Nodes[nid]; ok {
				coords = append(coords, []float64{nd.Lon, nd.Lat})
			}
		}

		feature := map[string]interface{}{
			"type": "Feature",
			"geometry": map[string]interface{}{
				"type":        "LineString",
				"coordinates": coords,
			},
			"properties": map[string]interface{}{
				"vehicle": vid + 1,
			},
		}
		features = append(features, feature)
	}

	featureCollection := map[string]interface{}{
		"type":     "FeatureCollection",
		"features": features,
	}

	data, err := json.MarshalIndent(featureCollection, "", "  ")
	if err != nil {
		log.Fatal("Failed to marshal GeoJSON:", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Fatal("Failed to write GeoJSON:", err)
	}
	fmt.Printf("GeoJSON saved to %s\n", filename)
}
