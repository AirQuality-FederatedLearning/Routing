package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/awalterschulze/gographviz"
	"github.com/paulmach/osm/geojson"
	"github.com/paulmach/orb"
)

const (
	MinLat = 10.90
	MaxLat = 11.13
	MinLon = 76.86
	MaxLon = 77.04
)

type Node struct {
	ID  string
	Lat float64
	Lon float64
}

type Graph struct {
	Nodes map[string]*Node
	Edges map[string][]string
}

type PoliceStation struct {
	ID       string
	Lat, Lon float64
}

type Config struct {
	NumVehicles      int
	NumIterations    int
	MaxRouteDistance float64
	MaxRouteSteps    int
	Alpha            float64
	Beta             float64
	Evaporation      float64
	Q                float64
	CoverageFactor   float64
	PenaltyFactor    float64
	RRTExpansions    int
}

type RouteSolution struct {
	Vehicles [][]string
	Fitness  float64
	Graph    *Graph
	Config   *Config
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1) Load GraphML data
	graph, err := parseGraphML("path_to_your_graphml_file.graphml")
	if err != nil {
		log.Fatalf("Error parsing GraphML: %v", err)
	}

	// 2) Define Police Stations (In a real case, load them from a file)
	var stations []PoliceStation
	stations = append(stations, PoliceStation{ID: "Station1", Lat: 10.994, Lon: 77.027})
	stations = append(stations, PoliceStation{ID: "Station2", Lat: 11.010, Lon: 76.990})

	// 3) Configuration for ACO and RRT*
	cfg := &Config{
		NumVehicles:      5,
		NumIterations:    50,
		MaxRouteDistance: 15000,
		MaxRouteSteps:    20,
		Alpha:            1.0,
		Beta:             2.0,
		Evaporation:      0.1,
		Q:                100,
		CoverageFactor:   5000,
		PenaltyFactor:    10000,
		RRTExpansions:    20,
	}

	// 4) Generate initial seeds using RRT*
	fmt.Println("Generating RRT* seeds...")
	seeds := buildRRTStarSeeds(graph, cfg, stations)

	// 5) Run coverage-based ACO optimization
	fmt.Println("Running coverage-based ACO...")
	best := runACO(graph, cfg, seeds)

	// 6) Save result to GeoJSON
	saveGeoJSON(best, "routes.geojson")
	fmt.Printf("Done. Best fitness: %.2f\n", best.Fitness)
}

// 1) Parse GraphML file and create Graph structure
func parseGraphML(filename string) (*Graph, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	graphAst, err := gographviz.ParseFile(file)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GraphML: %v", err)
	}

	graph := gographviz.NewGraph()
	if err := gographvizAst.Read(graphAst); err != nil {
		return nil, fmt.Errorf("failed to read graph AST: %v", err)
	}

	g := &Graph{
		Nodes: make(map[string]*Node),
		Edges: make(map[string][]string),
	}

	for nodeID, _ := range graph.Nodes {
		g.Nodes[nodeID] = &Node{ID: nodeID}
	}

	for _, edge := range graph.Edges {
		source := edge.Src
		target := edge.Dst
		if _, exists := g.Edges[source]; !exists {
			g.Edges[source] = []string{}
		}
		g.Edges[source] = append(g.Edges[source], target)
	}

	return g, nil
}

// 2) Build RRT* seeds
func buildRRTStarSeeds(g *Graph, cfg *Config, stations []PoliceStation) []*RouteSolution {
	seeds := make([]*RouteSolution, 0)
	for i := 0; i < cfg.NumVehicles; i++ {
		sol := &RouteSolution{
			Graph:  g,
			Config: cfg,
			Vehicles: [][]string{
				// Route for policeman
			},
		}
		var start string
		if len(stations) > i {
			// Use the station's nearest node
			start = findNearestNode(g, stations[i].Lat, stations[i].Lon)
		} else {
			// Fallback random
			start = randomNodeID(g)
		}
		rte := rrtStarExpand(g, cfg, start)
		sol.Vehicles = [][]string{rte}
		sol.Evaluate()
		seeds = append(seeds, sol)
	}
	return seeds
}

// 3) Implement RRT* Expansion
func rrtStarExpand(g *Graph, cfg *Config, start string) []string {
	distTo := make(map[string]float64)
	parent := make(map[string]string)

	for id := range g.Nodes {
		distTo[id] = math.MaxFloat64
	}
	distTo[start] = 0

	for i := 0; i < cfg.RRTExpansions; i++ {
		rnd := randomNodeID(g)
		bestNode := findNearestNode(distTo)
		path := BFSPath(g, bestNode, rnd)
		subDist := measureDistance(g, path)

		if distTo[bestNode]+subDist < distTo[rnd] && distTo[bestNode]+subDist < cfg.MaxRouteDistance {
			curDist := distTo[bestNode]
			for i := 1; i < len(path); i++ {
				nid := path[i]
				stepD := haversine(g.Nodes[path[i-1]].Lat, g.Nodes[path[i-1]].Lon, g.Nodes[nid].Lat, g.Nodes[nid].Lon)
				newDist := curDist + stepD
				if newDist < distTo[nid] {
					distTo[nid] = newDist
					parent[nid] = path[i-1]
				}
				curDist = newDist
				if curDist > cfg.MaxRouteDistance {
					break
				}
			}
		}
	}

	var farNode string
	var bestDist float64
	for nd, d := range distTo {
		if d > bestDist && d < cfg.MaxRouteDistance {
			bestDist = d
			farNode = nd
		}
	}

	var rev []string
	cur := farNode
	for cur != "" && cur != start {
		rev = append(rev, cur)
		cur = parent[cur]
	}
	rev = append(rev, start)
	for i, j := 0, len(rev)-1; i < j; i, j = i+1, j-1 {
		rev[i], rev[j] = rev[j], rev[i]
	}
	return rev
}

// 4) ACO Optimization
func runACO(g *Graph, cfg *Config, seeds []*RouteSolution) *RouteSolution {
	pher := make(map[string]float64)
	for id := range g.Nodes {
		for _, neigh := range g.Edges[id] {
			pher[id+"-"+neigh] = 1.0
		}
	}

	for _, s := range seeds {
		for _, route := range s.Vehicles {
			for i := 0; i < len(route)-1; i++ {
				pher[route[i]+"-"+route[i+1]] += 2.0
			}
		}
	}

	best := &RouteSolution{
		Fitness: -math.MaxFloat64,
		Graph:   g,
		Config:  cfg,
	}

	for iter := 0; iter < cfg.NumIterations; iter++ {
		cur := &RouteSolution{
			Graph:  g,
			Config: cfg,
		}
		cur.buildColonySolution(pher)
		cur.Evaluate()
		if cur.Fitness > best.Fitness {
			best = cur
		}
		evaporate(pher, cfg.Evaporation)
		deposit(cur, pher, cfg)
	}
	return best
}

// 5) Fitness Calculation
func (sol *RouteSolution) Evaluate() {
	// Calculate fitness based on coverage and penalties
	// Add implementation as per your ACO algorithm
}

// 6) Helper functions: `findNearestNode`, `randomNodeID`, `BFSPath`, etc.
func findNearestNode(g *Graph, lat, lon float64) string {
	// Find the nearest node to given lat, lon
	// Implement your own logic
	return ""
}

func randomNodeID(g *Graph) string {
	// Pick a random node from the graph
	return ""
}

func BFSPath(g *Graph, src, dst string) []string {
	// Perform BFS to find path from src to dst
	return []string{}
}

func measureDistance(g *Graph, path []string) float64 {
	// Measure the distance of the given path
	return 0.0
}

// Haversine formula for distance calculation
func haversine(lat1, lon1, lat2, lon2 float64) float64 {
	const R = 6371e3 // Earth radius in meters
	φ1 := lat1 * math.Pi / 180
	φ2 := lat2 * math.Pi / 180
	Δφ := (lat2 - lat1) * math.Pi / 180
	Δλ := (lon2 - lon1) * math.Pi / 180

	a := math.Sin(Δφ/2)*math.Sin(Δφ/2) +
		math.Cos(φ1)*math.Cos(φ2)*math.Sin(Δλ/2)*math.Sin(Δλ/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return R * c
}

// Save solution to GeoJSON
func saveGeoJSON(sol *RouteSolution, filename string) {
	ctx := geojson.NewContext()
	var features []*geojson.Feature
	colors := []string{"#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"}

	for vid, route := range sol.Vehicles {
		if len(route) < 2 {
			continue
		}

		// Create LineString coordinates
		ls := orb.LineString{}
		for _, nid := range route {
			if node, exists := sol.Graph.Nodes[nid]; exists {
				ls = append(ls, orb.Point{node.Lon, node.Lat})
			}
		}

		// Create feature with properties
		f := geojson.NewFeature(ls)
		f.Properties["vehicle"] = vid + 1
		f.Properties["stroke"] = colors[vid%len(colors)]
		f.Properties["stroke-width"] = 3
		f.Properties["stroke-opacity"] = 0.8

		features = append(features, f)
	}

	fc := geojson.NewFeatureCollection(features)

	data, err := json.Marshal(fc)
	if err != nil {
		log.Fatalf("failed to marshal geojson: %v", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Fatalf("failed to write geojson file: %v", err)
	}

	fmt.Printf("Saved optimized routes to %s\n", filename)
}
