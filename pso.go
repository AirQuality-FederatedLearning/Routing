package main

import (
    "encoding/json"
    "encoding/xml"
    "fmt"
    "io/ioutil"
    "math"
    "math/rand"
    "os"
    "path/filepath"
    "time"
)

// --------------------------------------------------------------------------------------
// DATA STRUCTURES
// --------------------------------------------------------------------------------------

type GraphML struct {
    XMLName xml.Name `xml:"graphml"`
    Graph   Graph    `xml:"graph"`
}

type Graph struct {
    Nodes []Node `xml:"node"`
    Edges []Edge `xml:"edge"`
}

type Node struct {
    ID    string  `xml:"id,attr"`
    Lat   float64 `xml:"data[key='d4'],string"`
    Lon   float64 `xml:"data[key='d3'],string"`
    Edges []*Edge
}

type Edge struct {
    ID     string  `xml:"id,attr"`
    Source string  `xml:"source,attr"`
    Target string  `xml:"target,attr"`
    Length float64 `xml:"data[key='d9'],string"`
}

// GeoJSON output
type GeoJSON struct {
    Type     string    `json:"type"`
    Features []Feature `json:"features"`
}

type Feature struct {
    Type       string     `json:"type"`
    Geometry   Geometry   `json:"geometry"`
    Properties Properties `json:"properties"`
}

type Geometry struct {
    Type        string      `json:"type"`
    Coordinates [][]float64 `json:"coordinates"`
}

type Properties struct {
    Vehicle int    `json:"vehicle"`
    Color   string `json:"color"`
}

// PSO CONFIG
type PSOConfig struct {
    SwarmSize         int     // number of particles
    Iterations        int     // number of iterations
    Inertia           float64 // inertia weight
    Cognitive         float64 // c1
    Social            float64 // c2
    MaxRoadUsage      int
    MaxRouteDistance  float64
    EnableDistanceCap bool
    CoverageRadius    float64
}

// Particle: a solution that contains routes for each policeman
type Particle struct {
    Positions [][]string // positions for each policeman = list of node IDs
    Vel       [][]string // not a typical numeric velocity, so we approximate route “mutation”
    BestPos   [][]string // best position found so far
    BestCost  float64    // cost of best position
    Cost      float64
}

// PSO is the global manager
type PSO struct {
    Config    PSOConfig
    Graph     map[string]*Node
    Edges     map[string]*Edge
    AQIHotspots map[string]float64

    Swarm     []*Particle
    GlobalBestPos   [][]string
    GlobalBestCost  float64
}

// --------------------------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------------------------

func main() {
    graphmlFile := "major_road_cache\\Coimbatore_India_major.graphml"
    outputFolder := "solution_cache"
    outputFile := filepath.Join(outputFolder, "pso_solution.geojson")

    pso := &PSO{
        Config: PSOConfig{
            SwarmSize:         20,
            Iterations:        30,
            Inertia:           0.7,
            Cognitive:         1.4,
            Social:            1.4,
            MaxRoadUsage:      2,
            MaxRouteDistance:  20000,
            EnableDistanceCap: true,
            CoverageRadius:    500.0,
        },
        AQIHotspots: make(map[string]float64),
    }

    // Load
    if err := pso.LoadGraphML(graphmlFile); err != nil {
        fmt.Println("Error loading graph:", err)
        return
    }

    // Optionally mark random AQI hotspots
    pso.InitializeHotspots()

    // Initialize swarm
    pso.InitializeSwarm(5) // 5 policeman
    // Solve
    fmt.Println("Starting PSO optimization...")
    pso.Optimize()
    fmt.Println("PSO optimization finished.")

    // Build a final route from best solution
    pso.SaveGeoJSON(outputFile)
    fmt.Println("PSO solution saved to:", outputFile)
}

// --------------------------------------------------------------------------------------
// LOAD & INITIALIZE
// --------------------------------------------------------------------------------------

func (p *PSO) LoadGraphML(filename string) error {
    fmt.Println("Loading graph from:", filename)
    xmlFile, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer xmlFile.Close()

    data, _ := ioutil.ReadAll(xmlFile)
    var gml GraphML
    if err := xml.Unmarshal(data, &gml); err != nil {
        return err
    }

    p.Graph = make(map[string]*Node)
    for _, node := range gml.Graph.Nodes {
        n := node
        p.Graph[node.ID] = &n
    }

    p.Edges = make(map[string]*Edge)
    for _, edge := range gml.Graph.Edges {
        e := edge
        p.Edges[e.ID] = &e
        if sourceNode, exists := p.Graph[e.Source]; exists {
            sourceNode.Edges = append(sourceNode.Edges, &e)
        }
    }

    fmt.Printf("Graph loaded: %d nodes, %d edges\n", len(p.Graph), len(p.Edges))
    return nil
}

func (p *PSO) InitializeHotspots() {
    rand.Seed(time.Now().UnixNano())
    for _, node := range p.Graph {
        if rand.Float64() < 0.10 {
            p.AQIHotspots[node.ID] = 1.0 + 4.0*rand.Float64()
        }
    }
}

func (p *PSO) InitializeSwarm(numPolicemen int) {
    // We'll define each swarm particle as random routes for each policeman
    rand.Seed(time.Now().UnixNano())

    // gather feasible nodes
    var feasibleNodes []string
    for id, n := range p.Graph {
        if len(n.Edges) > 0 {
            feasibleNodes = append(feasibleNodes, id)
        }
    }

    p.Swarm = make([]*Particle, p.Config.SwarmSize)
    p.GlobalBestCost = math.MaxFloat64

    for i := 0; i < p.Config.SwarmSize; i++ {
        // For each policeman, pick a random route (some random length)
        routes := make([][]string, numPolicemen)
        for j := 0; j < numPolicemen; j++ {
            pathLen := 10 + rand.Intn(5) // random path length ~ 10-14
            route := make([]string, pathLen)
            for k := 0; k < pathLen; k++ {
                route[k] = feasibleNodes[rand.Intn(len(feasibleNodes))]
            }
            routes[j] = route
        }

        part := &Particle{
            Positions: routes,
            Vel:       nil, // We'll store some structure if we want to do advanced route "velocity"
        }
        part.Cost = p.EvaluateCost(part.Positions)
        part.BestPos = copy2DStringSlice(part.Positions)
        part.BestCost = part.Cost

        if part.Cost < p.GlobalBestCost {
            p.GlobalBestCost = part.Cost
            p.GlobalBestPos = copy2DStringSlice(part.Positions)
        }
        p.Swarm[i] = part
    }
}

// --------------------------------------------------------------------------------------
// PSO CORE
// --------------------------------------------------------------------------------------

func (p *PSO) Optimize() {
    for iter := 1; iter <= p.Config.Iterations; iter++ {
        fmt.Printf("PSO Iteration %d / %d\n", iter, p.Config.Iterations)

        for i := 0; i < len(p.Swarm); i++ {
            part := p.Swarm[i]

            // 1) Update route based on "velocity" – simplified approach
            // We'll do a random re-route mutation approach
            newPos := p.RandomRouteMutation(part.Positions, 0.3)

            newCost := p.EvaluateCost(newPos)
            // If better than personal best
            if newCost < part.BestCost {
                part.BestCost = newCost
                part.BestPos = copy2DStringSlice(newPos)
            }
            // If better than global best
            if newCost < p.GlobalBestCost {
                p.GlobalBestCost = newCost
                p.GlobalBestPos = copy2DStringSlice(newPos)
            }

            // In typical PSO, you do inertia + cognitive + social velocity updates,
            // but here we approximate by direct random route changes that bias
            // toward personal & global best.

            // We'll do a second random "cognitive" shift
            cogChance := 0.2
            if rand.Float64() < cogChance {
                newPos2 := p.MergeRoutes(newPos, part.BestPos)
                cost2 := p.EvaluateCost(newPos2)
                if cost2 < newCost {
                    newPos = newPos2
                    newCost = cost2
                }
            }

            // "Social" shift toward global best
            socChance := 0.2
            if rand.Float64() < socChance {
                newPos3 := p.MergeRoutes(newPos, p.GlobalBestPos)
                cost3 := p.EvaluateCost(newPos3)
                if cost3 < newCost {
                    newPos = newPos3
                    newCost = cost3
                }
            }

            // commit
            part.Positions = newPos
            part.Cost = newCost
            if part.Cost < part.BestCost {
                part.BestPos = copy2DStringSlice(newPos)
                part.BestCost = newCost
            }
            if newCost < p.GlobalBestCost {
                p.GlobalBestCost = newCost
                p.GlobalBestPos = copy2DStringSlice(newPos)
            }
        }
    }
}

// Return cost of a set of policeman routes
func (p *PSO) EvaluateCost(routes [][]string) float64 {
    // We'll measure coverage, route length, over-usage, plus a hotspot incentive

    totalDistance := 0.0
    edgeUsage := make(map[string]int)
    coverageReward := 0.0
    coverageMap := make(map[string]bool)

    for _, route := range routes {
        routeDist := 0.0
        for i := 0; i < len(route)-1; i++ {
            fromID := route[i]
            toID := route[i+1]
            dist := p.distanceBetween(fromID, toID)
            routeDist += dist

            // track usage
            edgeKey := fmt.Sprintf("%s->%s", fromID, toID)
            edgeUsage[edgeKey]++

            // coverage – if policeman is within p.Config.CoverageRadius of a node, mark covered
            // but to keep it simpler, we only do coverage of the route nodes themselves
            coverageMap[fromID] = true
            coverageMap[toID] = true
        }
        if p.Config.EnableDistanceCap && routeDist > p.Config.MaxRouteDistance {
            // big penalty for route that exceeds distance
            routeDist *= 2
        }
        totalDistance += routeDist
    }

    // Over usage penalty
    overusePenalty := 0.0
    for _, c := range edgeUsage {
        if c > p.Config.MaxRoadUsage {
            overusePenalty += float64(c - p.Config.MaxRoadUsage) * 1000
        }
    }

    // Hotspot coverage: each route node that matches a hotspot yields a negative cost
    // (i.e. a reward) to encourage route passing. We'll do a simple approach
    hotspotReward := 0.0
    for nodeID := range coverageMap {
        if val, ok := p.AQIHotspots[nodeID]; ok {
            // reduce cost by val * 100
            hotspotReward += val * 100.0
        }
    }

    // final cost
    cost := totalDistance + overusePenalty - hotspotReward
    return cost
}

// A random route mutation "velocity" for demonstration
func (p *PSO) RandomRouteMutation(old [][]string, mutationProb float64) [][]string {
    newRoutes := copy2DStringSlice(old)
    for i := 0; i < len(newRoutes); i++ {
        for j := 0; j < len(newRoutes[i]); j++ {
            if rand.Float64() < mutationProb {
                // pick a random node
                newRoutes[i][j] = p.pickFeasibleNode()
            }
        }
    }
    return newRoutes
}

// Merge two sets of routes with a random approach
func (p *PSO) MergeRoutes(a, b [][]string) [][]string {
    out := copy2DStringSlice(a)
    for i := 0; i < len(a) && i < len(b); i++ {
        if rand.Float64() < 0.5 {
            out[i] = copyStringSlice(b[i])
        }
    }
    return out
}

func (p *PSO) pickFeasibleNode() string {
    // pick random from graph
    idx := rand.Intn(len(p.Graph))
    var key string
    c := 0
    for k := range p.Graph {
        if c == idx {
            key = k
            break
        }
        c++
    }
    return key
}

// distanceBetween node IDs
func (p *PSO) distanceBetween(id1, id2 string) float64 {
    n1 := p.Graph[id1]
    n2 := p.Graph[id2]
    if n1 == nil || n2 == nil {
        return 100000 // large penalty
    }
    return Haversine(n1.Lat, n1.Lon, n2.Lat, n2.Lon)
}

// Haversine in meters
func Haversine(lat1, lon1, lat2, lon2 float64) float64 {
    const R = 6371e3
    φ1 := lat1 * math.Pi / 180
    φ2 := lat2 * math.Pi / 180
    Δφ := (lat2 - lat1) * math.Pi / 180
    Δλ := (lon2 - lon1) * math.Pi / 180

    a := math.Sin(Δφ/2)*math.Sin(Δφ/2) +
        math.Cos(φ1)*math.Cos(φ2)*math.Sin(Δλ/2)*math.Sin(Δλ/2)
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
    return R * c
}

// --------------------------------------------------------------------------------------
// SAVE RESULT
// --------------------------------------------------------------------------------------

// Convert the best global route set to a GeoJSON
func (p *PSO) SaveGeoJSON(filename string) {
    // p.GlobalBestPos has dimension [policeman][sequenceOfNodes]
    // We'll convert each policeman’s route into a line
    features := make([]Feature, 0)
    colors := []string{"#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"}

    for i, route := range p.GlobalBestPos {
        coords := make([][]float64, 0)
        for _, nodeID := range route {
            n := p.Graph[nodeID]
            if n != nil {
                coords = append(coords, []float64{n.Lon, n.Lat})
            }
        }
        if len(coords) < 2 {
            continue
        }
        feature := Feature{
            Type: "Feature",
            Geometry: Geometry{
                Type:        "LineString",
                Coordinates: coords,
            },
            Properties: Properties{
                Vehicle: i + 1,
                Color:   colors[i%len(colors)],
            },
        }
        features = append(features, feature)
    }

    geoJSON := GeoJSON{
        Type:     "FeatureCollection",
        Features: features,
    }
    data, err := json.MarshalIndent(geoJSON, "", "  ")
    if err != nil {
        fmt.Println("Error marshalling geojson:", err)
        return
    }
    if err := ioutil.WriteFile(filename, data, 0644); err != nil {
        fmt.Println("Error writing geojson:", err)
    }
}

// --------------------------------------------------------------------------------------
// UTILS
// --------------------------------------------------------------------------------------
func copy2DStringSlice(src [][]string) [][]string {
    out := make([][]string, len(src))
    for i := range src {
        out[i] = copyStringSlice(src[i])
    }
    return out
}

func copyStringSlice(s []string) []string {
    out := make([]string, len(s))
    copy(out, s)
    return out
}
