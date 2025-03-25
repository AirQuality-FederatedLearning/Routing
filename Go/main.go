package main

import (
	"encoding/json"
	"encoding/xml"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/tealeg/xlsx"
)

// ========== GraphML structures ==========

type GraphML struct {
	XMLName xml.Name `xml:"graphml"`
	Keys    []Key    `xml:"key"`
	Graph   Graph    `xml:"graph"`
}

type Key struct {
	ID   string `xml:"id,attr"`
	For  string `xml:"for,attr"`
	Attr string `xml:"attr.name,attr"`
	Type string `xml:"attr.type,attr"`
}

type Graph struct {
	XMLName xml.Name `xml:"graph"`
	Edges   []Edge   `xml:"edge"`
	Nodes   []Node   `xml:"node"`
}

type Node struct {
	ID   string `xml:"id,attr"`
	Data []Data `xml:"data"`
}

type Edge struct {
	Source string `xml:"source,attr"`
	Target string `xml:"target,attr"`
	Data   []Data `xml:"data"`
}

type Data struct {
	Key   string `xml:"key,attr"`
	Value string `xml:",chardata"`
}

// ========== Internal Graph Representation ==========

type GraphNode struct {
	ID       string
	X, Y     float64
	OutEdges []string
}

type GraphEdge struct {
	From     string
	To       string
	Length   float64
	Geometry [][]float64 // Store coordinates for the road segment
}

type RoadGraph struct {
	Nodes map[string]*GraphNode
	Edges map[string]map[string]*GraphEdge
}

// ========== PSO Structures ==========

type Route []string
type Solution []Route

type Particle struct {
	Position Solution
	BestPos  Solution
	BestFit  float64
}

type PSO struct {
	Graph         *RoadGraph
	Config        PSOConfig
	Swarm         []Particle
	GlobalBest    Solution
	GlobalBestFit float64
	PoliceNodes   []string
}

type PSOConfig struct {
	NumVehicles      int     // Number of vehicles (routes) to generate in each solution
	SwarmSize        int     // Number of particles in the swarm
	Iterations       int     // Number of PSO iterations
	W                float64 // Not used much here, but commonly part of velocity update
	C1, C2           float64 // Not used strictly in "velocity," but we do a simplified approach
	CoverageFactor   float64 // Factor that weighs how valuable coverage is
	OverlapPenalty   float64 // Penalty factor for overlapping coverage too many times
	MaxRouteDistance float64 // Maximum route distance for a single route
	MutationRate     float64 // Probability of route mutation
}

// ========== GeoJSON Structures ==========

type GeoJSONFeature struct {
	Type       string                 `json:"type"`
	Geometry   GeoJSONGeometry        `json:"geometry"`
	Properties map[string]interface{} `json:"properties"`
}

type GeoJSONGeometry struct {
	Type        string        `json:"type"`
	Coordinates [][][]float64 `json:"coordinates"`
}

type GeoJSON struct {
	Type     string           `json:"type"`
	Features []GeoJSONFeature `json:"features"`
}

// ========== MAIN ==========

func main() {
	// Use a random seed for randomization in the PSO
	rand.Seed(time.Now().UnixNano())

	// ------- Modify these filenames/paths as needed -------
	graphFile := "Coimbatore_India_network.graphml"
	excelFile := "C:\\Users\\DELL\\Documents\\Amrita\\4th year\\ArcGis\\coimbatore_police.xlsx"
	outputGeoJSON := "routes.geojson"
	// ------------------------------------------------------

	// 1. Load the road network from GraphML
	graph, err := loadGraphML(graphFile)
	if err != nil {
		panic(err)
	}

	// 2. Load the police stations from Excel => nearest nodes
	policeNodes, err := getPoliceNodesFromExcel(excelFile, graph)
	if err != nil {
		panic(err)
	}

	// 3. Set up the PSO parameters
	config := PSOConfig{
		NumVehicles:      25,
		SwarmSize:        20,
		Iterations:       250,
		W:                0.5,
		C1:               1.5,
		C2:               1.5,
		CoverageFactor:   5000,
		OverlapPenalty:   10000,
		MaxRouteDistance: 15000,
		MutationRate:     0.2,
	}

	// 4. Create and run the PSO
	pso := NewPSO(graph, config, policeNodes)
	bestSolution, bestFit := pso.Run()
	fmt.Printf("Final best fitness: %.2f\n", bestFit)

	// 5. Convert the best solution to GeoJSON and write it out
	geojson := solutionToGeoJSON(pso.Graph, bestSolution)
	file, _ := json.MarshalIndent(geojson, "", "  ")
	_ = os.WriteFile(outputGeoJSON, file, 0644)

	fmt.Println("Routes exported to:", outputGeoJSON)
}

// ========== Load Graph from GraphML ==========

// loadGraphML reads a GraphML file and populates our RoadGraph structure.
func loadGraphML(filename string) (*RoadGraph, error) {
	file, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var graphml GraphML
	if err := xml.Unmarshal(file, &graphml); err != nil {
		return nil, err
	}

	roadGraph := &RoadGraph{
		Nodes: make(map[string]*GraphNode),
		Edges: make(map[string]map[string]*GraphEdge),
	}

	// Identify which key IDs correspond to x, y, and length
	var xKey, yKey, lengthKey, geomKey string
	for _, key := range graphml.Keys {
		switch key.Attr {
		case "x":
			xKey = key.ID
		case "y":
			yKey = key.ID
		case "length":
			lengthKey = key.ID
		case "geometry":
			geomKey = key.ID // If your GraphML uses this attribute name
		}
	}

	// Process nodes (read X, Y from node data)
	for _, node := range graphml.Graph.Nodes {
		var x, y float64
		for _, data := range node.Data {
			switch data.Key {
			case xKey:
				x, _ = strconv.ParseFloat(data.Value, 64)
			case yKey:
				y, _ = strconv.ParseFloat(data.Value, 64)
			}
		}
		roadGraph.Nodes[node.ID] = &GraphNode{
			ID:       node.ID,
			X:        x,
			Y:        y,
			OutEdges: []string{},
		}
	}

	// Process edges
	for _, edge := range graphml.Graph.Edges {
		var length float64
		var geometry [][]float64 // actual geometry along the edge

		for _, data := range edge.Data {
			// Length
			if data.Key == lengthKey {
				length, _ = strconv.ParseFloat(data.Value, 64)
			}
			// Geometry
			if data.Key == geomKey || data.Key == "geometry" {
				// We expect geometry data possibly like: "lon1,lat1 lon2,lat2 lon3,lat3 ..."
				points := strings.Split(strings.TrimSpace(data.Value), " ")
				for _, point := range points {
					coords := strings.Split(point, ",")
					if len(coords) == 2 {
						lon, errLon := strconv.ParseFloat(coords[0], 64)
						lat, errLat := strconv.ParseFloat(coords[1], 64)
						if errLon == nil && errLat == nil {
							geometry = append(geometry, []float64{lon, lat})
						}
					}
				}
			}
		}

		// Create adjacency if not present
		if _, ok := roadGraph.Edges[edge.Source]; !ok {
			roadGraph.Edges[edge.Source] = make(map[string]*GraphEdge)
		}
		// Store the edge
		roadGraph.Edges[edge.Source][edge.Target] = &GraphEdge{
			From:     edge.Source,
			To:       edge.Target,
			Length:   length,
			Geometry: geometry,
		}
		// Append to the node's out-edges
		roadGraph.Nodes[edge.Source].OutEdges = append(
			roadGraph.Nodes[edge.Source].OutEdges,
			edge.Target,
		)
	}

	return roadGraph, nil
}

// ========== Load Police Stations ==========

// getPoliceNodesFromExcel reads the geometry from the first cell in each row,
// extracts lat/lon from a string like "POINT (lon lat)", then finds the nearest
// node in our graph by Haversine distance. Returns the node IDs for each station.
func getPoliceNodesFromExcel(filename string, graph *RoadGraph) ([]string, error) {
	xlFile, err := xlsx.OpenFile(filename)
	if err != nil {
		return nil, err
	}
	var policeNodes []string
	if len(xlFile.Sheets) == 0 {
		return nil, fmt.Errorf("no sheets in Excel file")
	}
	sheet := xlFile.Sheets[0]
	// Assume first row is header, so start from row 1
	for i, row := range sheet.Rows {
		if i == 0 {
			continue // skip header
		}
		if len(row.Cells) == 0 {
			continue
		}

		geomCell := row.Cells[0]
		pointStr := strings.TrimPrefix(geomCell.String(), "POINT (")
		pointStr = strings.TrimSuffix(pointStr, ")")
		coords := strings.Split(pointStr, " ")
		if len(coords) != 2 {
			continue
		}

		lon, errLon := strconv.ParseFloat(coords[0], 64)
		lat, errLat := strconv.ParseFloat(coords[1], 64)
		if errLon != nil || errLat != nil {
			continue
		}

		// Find nearest node by Haversine distance
		nearestID := ""
		minDist := math.MaxFloat64
		for _, node := range graph.Nodes {
			dist := haversine(lon, lat, node.X, node.Y)
			if dist < minDist {
				minDist = dist
				nearestID = node.ID
			}
		}
		if nearestID != "" {
			policeNodes = append(policeNodes, nearestID)
		}
	}

	return policeNodes, nil
}

// ========== Haversine Distance ==========

// haversine computes the distance in meters between (lon1, lat1) and (lon2, lat2).
func haversine(lon1, lat1, lon2, lat2 float64) float64 {
	const R = 6371000 // Earth radius in meters
	φ1 := lat1 * math.Pi / 180
	φ2 := lat2 * math.Pi / 180
	Δφ := (lat2 - lat1) * math.Pi / 180
	Δλ := (lon2 - lon1) * math.Pi / 180

	a := math.Sin(Δφ/2)*math.Sin(Δφ/2) +
		math.Cos(φ1)*math.Cos(φ2)*
			math.Sin(Δλ/2)*math.Sin(Δλ/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	return R * c
}

// ========== PSO Logic ==========

// NewPSO initializes a PSO object with the given graph, config, and police station node IDs.
func NewPSO(graph *RoadGraph, config PSOConfig, policeNodes []string) *PSO {
	return &PSO{
		Graph:       graph,
		Config:      config,
		PoliceNodes: policeNodes,
	}
}

// Run performs the PSO process: initialize swarm, iterative improvement, track global best.
func (pso *PSO) Run() (Solution, float64) {
	pso.initializeSwarm()

	for iter := 0; iter < pso.Config.Iterations; iter++ {
		for i := range pso.Swarm {
			// "velocity update" – we produce a new solution based on the particle’s current/best and global best
			newPos := pso.velocityUpdate(pso.Swarm[i].Position)
			// Then mutate it
			mutated := pso.mutate(newPos)
			// Evaluate fitness
			newFit := pso.fitness(mutated)

			// If the new solution is better than this particle’s best, update
			if newFit > pso.Swarm[i].BestFit {
				pso.Swarm[i].Position = mutated
				pso.Swarm[i].BestPos = mutated
				pso.Swarm[i].BestFit = newFit

				// Also update global best if needed
				if newFit > pso.GlobalBestFit {
					pso.GlobalBest = mutated
					pso.GlobalBestFit = newFit
				}
			}
		}

		fmt.Printf("Iteration %3d, Best Fit: %.2f\n", iter, pso.GlobalBestFit)
	}

	return pso.GlobalBest, pso.GlobalBestFit
}

// initializeSwarm creates random solutions (particles) and determines the initial best.
func (pso *PSO) initializeSwarm() {
	pso.Swarm = make([]Particle, pso.Config.SwarmSize)
	pso.GlobalBestFit = -math.MaxFloat64

	for i := range pso.Swarm {
		sol := pso.generateSolution() // random
		fit := pso.fitness(sol)
		pso.Swarm[i] = Particle{
			Position: sol,
			BestPos:  sol,
			BestFit:  fit,
		}
		if fit > pso.GlobalBestFit {
			pso.GlobalBest = sol
			pso.GlobalBestFit = fit
		}
	}
}

// generateSolution creates a random solution with pso.Config.NumVehicles routes.
func (pso *PSO) generateSolution() Solution {
	sol := make(Solution, pso.Config.NumVehicles)
	for i := range sol {
		sol[i] = pso.generateRoute()
	}
	return sol
}

// generateRoute picks a random police-node start, then extends a route until we hit MaxRouteDistance
// or no more adjacency.
func (pso *PSO) generateRoute() Route {
	if len(pso.PoliceNodes) == 0 {
		return nil
	}
	current := pso.PoliceNodes[rand.Intn(len(pso.PoliceNodes))]
	route := Route{current}
	distance := 0.0

	for {
		neighbors := pso.Graph.Nodes[current].OutEdges
		if len(neighbors) == 0 {
			break
		}
		next := neighbors[rand.Intn(len(neighbors))]
		edge, exists := pso.Graph.Edges[current][next]
		if !exists {
			break
		}
		if distance+edge.Length > pso.Config.MaxRouteDistance {
			break
		}
		route = append(route, next)
		distance += edge.Length
		current = next
	}
	return route
}

// fitness calculates a coverage score minus overlap penalties.
func (pso *PSO) fitness(sol Solution) float64 {
	edgeCounts := make(map[string]int)
	coveredEdges := make(map[string]bool)

	// Count coverage and also track overlaps
	for _, route := range sol {
		for i := 0; i < len(route)-1; i++ {
			from, to := route[i], route[i+1]
			if _, exists := pso.Graph.Edges[from][to]; exists {
				key := fmt.Sprintf("%s-%s", from, to)
				edgeCounts[key]++
				coveredEdges[key] = true
			}
		}
	}

	// Score coverage
	coverage := 0.0
	for edgeKey := range coveredEdges {
		parts := strings.Split(edgeKey, "-")
		if edgeData, exists := pso.Graph.Edges[parts[0]][parts[1]]; exists {
			coverage += edgeData.Length * pso.Config.CoverageFactor
		}
	}

	// Apply penalty if edges are covered more than 3 times
	penalty := 0.0
	for edgeKey, count := range edgeCounts {
		if count > 3 {
			parts := strings.Split(edgeKey, "-")
			if edgeData, exists := pso.Graph.Edges[parts[0]][parts[1]]; exists {
				penalty += edgeData.Length * pso.Config.OverlapPenalty * float64(count-3)
			}
		}
	}

	return coverage - penalty
}

// velocityUpdate uses a simplified approach: picks from current, random best from swarm, or global best.
func (pso *PSO) velocityUpdate(current Solution) Solution {
	newSol := make(Solution, len(current))
	for i := range newSol {
		r := rand.Float64()
		switch {
		case r < 0.34:
			// keep same route
			newSol[i] = current[i]
		case r < 0.67:
			// pick a random particle's best route
			if len(pso.Swarm) > 0 {
				rp := pso.Swarm[rand.Intn(len(pso.Swarm))].BestPos
				newSol[i] = rp[i]
			} else {
				newSol[i] = current[i]
			}
		default:
			// use the global best route
			newSol[i] = pso.GlobalBest[i]
		}
	}
	return newSol
}

// mutate randomly modifies segments of routes
func (pso *PSO) mutate(sol Solution) Solution {
	mutated := make(Solution, len(sol))
	copy(mutated, sol)

	for i := range mutated {
		if rand.Float64() < pso.Config.MutationRate && len(mutated[i]) > 3 {
			start := rand.Intn(len(mutated[i]) - 2)
			end := start + 1 + rand.Intn(len(mutated[i])-start-1)
			if end >= len(mutated[i]) {
				end = len(mutated[i]) - 1
			}
			// Generate a new path segment between mutated[i][start] and mutated[i][end]
			newSegment := pso.generateRouteSegment(mutated[i][start], mutated[i][end])
			// Rebuild the route
			updatedRoute := append([]string{}, mutated[i][:start]...)
			updatedRoute = append(updatedRoute, newSegment...)
			if end < len(mutated[i])-1 {
				updatedRoute = append(updatedRoute, mutated[i][end+1:]...)
			}
			mutated[i] = updatedRoute
		}
	}
	return mutated
}

// generateRouteSegment attempts a short path from start to end by random walking up to 20 steps.
func (pso *PSO) generateRouteSegment(start, end string) Route {
	segment := Route{start}
	current := start
	steps := 0
	maxSteps := 20

	for current != end && steps < maxSteps {
		neighbors := pso.Graph.Nodes[current].OutEdges
		if len(neighbors) == 0 {
			break
		}
		next := neighbors[rand.Intn(len(neighbors))]
		segment = append(segment, next)
		current = next
		steps++
		// If we randomly found the end, done
		if current == end {
			break
		}
	}
	// If we never reached end, forcibly append it
	if current != end {
		segment = append(segment, end)
	}
	return segment
}

// ========== Convert the final solution to GeoJSON ==========

// solutionToGeoJSON turns each route (vehicle) into a MultiLineString feature.
// If an edge has geometry, we use it. Otherwise, we do a quick fallback line from node coords.
func solutionToGeoJSON(graph *RoadGraph, solution Solution) *GeoJSON {
	// Some arbitrary colors so each route can be visually distinct in GIS
	colors := []string{"#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"}

	var features []GeoJSONFeature

	for vid, route := range solution {
		if len(route) < 2 {
			continue
		}

		var coordinates [][][]float64 // array of lines, each line is array of coords
		for i := 0; i < len(route)-1; i++ {
			from := route[i]
			to := route[i+1]

			edge, exists := graph.Edges[from][to]
			if exists && len(edge.Geometry) > 0 {
				// Use stored geometry
				lineCoords := make([][]float64, len(edge.Geometry))
				for j, coord := range edge.Geometry {
					lineCoords[j] = []float64{coord[0], coord[1]}
				}
				coordinates = append(coordinates, lineCoords)
			} else {
				// Fallback to direct node coords if geometry missing
				fromNode := graph.Nodes[from]
				toNode := graph.Nodes[to]
				if fromNode != nil && toNode != nil {
					coordinates = append(coordinates, [][]float64{
						{fromNode.X, fromNode.Y},
						{toNode.X, toNode.Y},
					})
				}
			}
		}

		if len(coordinates) == 0 {
			continue
		}

		feature := GeoJSONFeature{
			Type: "Feature",
			Geometry: GeoJSONGeometry{
				Type:        "MultiLineString",
				Coordinates: coordinates,
			},
			Properties: map[string]interface{}{
				"vehicle":      vid + 1,
				"stroke":       colors[vid%len(colors)],
				"stroke-width": 3,
			},
		}
		features = append(features, feature)
	}

	return &GeoJSON{
		Type:     "FeatureCollection",
		Features: features,
	}
}
