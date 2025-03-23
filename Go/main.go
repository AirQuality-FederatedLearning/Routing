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

// GraphML structures
type GraphML struct {
	XMLName xml.Name `xml:"graphml"`
	Keys    []Key    `xml:"key"`
	Graph   Graph    `xml:"graph"`
}

type Key struct {
	ID    string `xml:"id,attr"`
	For   string `xml:"for,attr"`
	Attr  string `xml:"attr.name,attr"`
	Type  string `xml:"attr.type,attr"`
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

// Graph representation
type GraphNode struct {
	ID       string
	X, Y     float64
	OutEdges []string
}

type GraphEdge struct {
	From, To string
	Length   float64
}

type RoadGraph struct {
	Nodes map[string]*GraphNode
	Edges map[string]map[string]*GraphEdge
}

// PSO structures
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
	NumVehicles      int
	SwarmSize        int
	Iterations       int
	W, C1, C2       float64
	CoverageFactor   float64
	OverlapPenalty   float64
	MaxRouteDistance float64
	MutationRate     float64
}

// GeoJSON structures
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

func main() {
	rand.Seed(time.Now().UnixNano())

	// Load road network
	graph, err := loadGraphML("Coimbatore_India_major.graphml")
	if err != nil {
		panic(err)
	}

	// Load police stations from Excel
	policeNodes, err := getPoliceNodesFromExcel("C:\\Users\\DELL\\Documents\\Amrita\\4th year\\ArcGis\\coimbatore_police.xlsx", graph)
	if err != nil {
		panic(err)
	}

	config := PSOConfig{
		NumVehicles:      50,
		SwarmSize:        20,
		Iterations:       50,
		W:                0.5,
		C1:               1.5,
		C2:               1.5,
		CoverageFactor:   5000,
		OverlapPenalty:   10000,
		MaxRouteDistance: 15000,
		MutationRate:     0.2,
	}

	pso := NewPSO(graph, config, policeNodes)
	bestSolution, bestFit := pso.Run()

	fmt.Printf("Best fitness: %.2f\n", bestFit)

	geojson := solutionToGeoJSON(pso.Graph, bestSolution)
	file, _ := json.MarshalIndent(geojson, "", "  ")
	_ = os.WriteFile("routes.geojson", file, 0644)
}

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

	// Find coordinate keys
	var xKey, yKey, lengthKey string
	for _, key := range graphml.Keys {
		switch key.Attr {
		case "x":
			xKey = key.ID
		case "y":
			yKey = key.ID
		case "length":
			lengthKey = key.ID
		}
	}

	// Process nodes
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
		for _, data := range edge.Data {
			if data.Key == lengthKey {
				length, _ = strconv.ParseFloat(data.Value, 64)
			}
		}

		if _, ok := roadGraph.Edges[edge.Source]; !ok {
			roadGraph.Edges[edge.Source] = make(map[string]*GraphEdge)
		}
		roadGraph.Edges[edge.Source][edge.Target] = &GraphEdge{
			From:   edge.Source,
			To:     edge.Target,
			Length: length,
		}
		roadGraph.Nodes[edge.Source].OutEdges = append(
			roadGraph.Nodes[edge.Source].OutEdges,
			edge.Target,
		)
	}

	return roadGraph, nil
}

func getPoliceNodesFromExcel(filename string, graph *RoadGraph) ([]string, error) {
	xlFile, err := xlsx.OpenFile(filename)
	if err != nil {
		return nil, err
	}

	var policeNodes []string
	sheet := xlFile.Sheets[0]

	for _, row := range sheet.Rows[1:] { // Skip header
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

		lon, _ := strconv.ParseFloat(coords[0], 64)
		lat, _ := strconv.ParseFloat(coords[1], 64)

		// Find nearest node
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

func NewPSO(graph *RoadGraph, config PSOConfig, policeNodes []string) *PSO {
	return &PSO{
		Graph:       graph,
		Config:      config,
		PoliceNodes: policeNodes,
	}
}

func (pso *PSO) Run() (Solution, float64) {
	pso.initializeSwarm()

	for iter := 0; iter < pso.Config.Iterations; iter++ {
		for i := range pso.Swarm {
			newPos := pso.velocityUpdate(pso.Swarm[i].Position)
			mutated := pso.mutate(newPos)
			newFit := pso.fitness(mutated)

			if newFit > pso.Swarm[i].BestFit {
				pso.Swarm[i].Position = mutated
				pso.Swarm[i].BestPos = mutated
				pso.Swarm[i].BestFit = newFit

				if newFit > pso.GlobalBestFit {
					pso.GlobalBest = mutated
					pso.GlobalBestFit = newFit
				}
			}
		}
		fmt.Printf("Iteration %d, Best Fit: %.2f\n", iter, pso.GlobalBestFit)
	}

	return pso.GlobalBest, pso.GlobalBestFit
}

func (pso *PSO) initializeSwarm() {
	pso.Swarm = make([]Particle, pso.Config.SwarmSize)
	pso.GlobalBestFit = -math.MaxFloat64

	for i := range pso.Swarm {
		sol := pso.generateSolution()
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

func (pso *PSO) generateSolution() Solution {
	sol := make(Solution, pso.Config.NumVehicles)
	for i := range sol {
		sol[i] = pso.generateRoute()
	}
	return sol
}

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

func (pso *PSO) fitness(sol Solution) float64 {
	edgeCounts := make(map[string]int)
	coveredEdges := make(map[string]bool)

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

	coverage := 0.0
	for edge := range coveredEdges {
		parts := strings.Split(edge, "-")
		if edgeData, exists := pso.Graph.Edges[parts[0]][parts[1]]; exists {
			coverage += edgeData.Length * pso.Config.CoverageFactor
		}
	}

	penalty := 0.0
	for edge, count := range edgeCounts {
		if count > 3 {
			parts := strings.Split(edge, "-")
			if edgeData, exists := pso.Graph.Edges[parts[0]][parts[1]]; exists {
				penalty += edgeData.Length * pso.Config.OverlapPenalty * float64(count-3)
			}
		}
	}

	return coverage - penalty
}

func (pso *PSO) velocityUpdate(current Solution) Solution {
	newSol := make(Solution, len(current))
	for i := range newSol {
		r := rand.Float64()
		switch {
		case r < 0.34:
			newSol[i] = current[i]
		case r < 0.67:
			if len(pso.Swarm) > 0 {
				randomParticle := pso.Swarm[rand.Intn(len(pso.Swarm))].BestPos
				newSol[i] = randomParticle[i]
			}
		default:
			newSol[i] = pso.GlobalBest[i]
		}
	}
	return newSol
}

func (pso *PSO) mutate(sol Solution) Solution {
	mutated := make(Solution, len(sol))
	copy(mutated, sol)

	for i := range mutated {
		if rand.Float64() < pso.Config.MutationRate && len(mutated[i]) > 3 {
			start := rand.Intn(len(mutated[i]) - 2)
			if start < 0 {
				start = 0
			}
			end := start + 1 + rand.Intn(len(mutated[i])-start-1)
			if end >= len(mutated[i]) {
				end = len(mutated[i]) - 1
			}

			newSegment := pso.generateRouteSegment(mutated[i][start], mutated[i][end])
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

func (pso *PSO) generateRouteSegment(start, end string) Route {
	segment := Route{start}
	current := start
	steps := 0

	for current != end && steps < 20 {
		neighbors := pso.Graph.Nodes[current].OutEdges
		if len(neighbors) == 0 {
			break
		}

		next := neighbors[rand.Intn(len(neighbors))]
		segment = append(segment, next)
		current = next
		steps++

		if current == end {
			break
		}
	}

	if current != end {
		segment = append(segment, end)
	}

	return segment
}

func solutionToGeoJSON(graph *RoadGraph, solution Solution) *GeoJSON {
	colors := []string{"#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231"}
	features := make([]GeoJSONFeature, 0)

	for vid, route := range solution {
		if len(route) < 2 {
			continue
		}

		var coords [][][]float64
		for i := 0; i < len(route)-1; i++ {
			fromNode := graph.Nodes[route[i]]
			toNode := graph.Nodes[route[i+1]]
			if fromNode == nil || toNode == nil {
				continue
			}
			coords = append(coords, [][]float64{
				{fromNode.X, fromNode.Y},
				{toNode.X, toNode.Y},
			})
		}

		if len(coords) == 0 {
			continue
		}

		feature := GeoJSONFeature{
			Type: "Feature",
			Geometry: GeoJSONGeometry{
				Type:        "MultiLineString",
				Coordinates: coords,
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