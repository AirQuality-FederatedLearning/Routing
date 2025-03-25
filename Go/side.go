package main

import (
    "encoding/json"
    "encoding/xml"
    "fmt"
    "github.com/tealeg/xlsx"
    "math"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "time"
)

// -----------------------------------------
// 1) GraphML Structures + RoadGraph
// -----------------------------------------

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

// Internal graph representation
type RoadGraph struct {
    Nodes        map[string]*GraphNode
    Edges        map[string]map[string]*GraphEdge
    EdgeNeighbors map[string][]string // For AQI "radius": adjacency among edges
}

type GraphNode struct {
    ID       string
    X, Y     float64
    OutEdges []string
}

type GraphEdge struct {
    From     string
    To       string
    Length   float64
    Geometry [][]float64
}

// -----------------------------------------
// 2) PSO Structures
// -----------------------------------------

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

// Extended config with “aqi radius” adjacency coverage + “divergence”
type PSOConfig struct {
    NumVehicles      int
    SwarmSize        int
    Iterations       int
    W, C1, C2        float64

    CoverageFactor   float64
    OverlapPenalty   float64
    MaxRouteDistance float64
    MutationRate     float64

    // "AQI radius" is handled by EdgeNeighbors. We won't store that here,
    // but we do store route divergence parameters if desired:
    DivergenceThreshold float64 // Dist in meters for start nodes
    DivergencePenalty   float64 // Penalty if routes start too close
}

// -----------------------------------------
// 3) GeoJSON Structures
// -----------------------------------------

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

// -----------------------------------------
// MAIN
// -----------------------------------------
func main() {
    rand.Seed(time.Now().UnixNano())

    // Adjust these file paths if needed
    graphFile := "Coimbatore_India_network.graphml"
    excelFile := "C:\\Users\\DELL\\Documents\\Amrita\\4th year\\ArcGis\\coimbatore_police.xlsx"
    outputGeoJSON := "routes.geojson"

    // 1. Load the road network from GraphML
    graph, err := loadGraphML(graphFile)
    if err != nil {
        panic(err)
    }

    // 2. Build an “edge neighbors” adjacency for the “AQI coverage radius”
    buildEdgeNeighbors(graph)

    // 3. Load the police stations from Excel => nearest nodes
    policeNodes, err := getPoliceNodesFromExcel(excelFile, graph)
    if err != nil {
        panic(err)
    }

    // 4. Set up PSO parameters
    config := PSOConfig{
        NumVehicles:         25,
        SwarmSize:           20,
        Iterations:          250,
        W:                   0.5,
        C1:                  1.5,
        C2:                  1.5,
        CoverageFactor:      5000,
        OverlapPenalty:      10000,
        MaxRouteDistance:    15000,
        MutationRate:        0.2,
        DivergenceThreshold: 10000,  // e.g., 2 km
        DivergencePenalty:   8000,  // penalty if 2 routes start < 2km
    }

    // 5. Create and run the PSO
    pso := NewPSO(graph, config, policeNodes)
    bestSolution, bestFit := pso.Run()
    fmt.Printf("Final best fitness: %.2f\n", bestFit)

    // 6. Convert the best solution to GeoJSON and write it out
    geojson := solutionToGeoJSON(pso.Graph, bestSolution)
    file, _ := json.MarshalIndent(geojson, "", "  ")
    _ = os.WriteFile(outputGeoJSON, file, 0644)

    fmt.Println("Routes exported to:", outputGeoJSON)
}

// -----------------------------------------
// 4) Load Graph + Build EdgeNeighbors
// -----------------------------------------

func loadGraphML(filename string) (*RoadGraph, error) {
    file, err := os.ReadFile(filename)
    if err != nil {
        return nil, err
    }
    var graphml GraphML
    if err := xml.Unmarshal(file, &graphml); err != nil {
        return nil, err
    }

    rg := &RoadGraph{
        Nodes: make(map[string]*GraphNode),
        Edges: make(map[string]map[string]*GraphEdge),
        // EdgeNeighbors to be filled after we read everything
        EdgeNeighbors: make(map[string][]string),
    }

    var xKey, yKey, lengthKey, geomKey string
    for _, k := range graphml.Keys {
        switch k.Attr {
        case "x":
            xKey = k.ID
        case "y":
            yKey = k.ID
        case "length":
            lengthKey = k.ID
        case "geometry":
            geomKey = k.ID
        }
    }

    // Process nodes
    for _, n := range graphml.Graph.Nodes {
        var x, y float64
        for _, d := range n.Data {
            switch d.Key {
            case xKey:
                x, _ = strconv.ParseFloat(d.Value, 64)
            case yKey:
                y, _ = strconv.ParseFloat(d.Value, 64)
            }
        }
        rg.Nodes[n.ID] = &GraphNode{
            ID:       n.ID,
            X:        x,
            Y:        y,
            OutEdges: []string{},
        }
    }

    // Process edges
    for _, e := range graphml.Graph.Edges {
        var length float64
        var geometry [][]float64
        for _, d := range e.Data {
            if d.Key == lengthKey {
                length, _ = strconv.ParseFloat(d.Value, 64)
            }
            if d.Key == geomKey || d.Key == "geometry" {
                pts := strings.Fields(strings.TrimSpace(d.Value))
                for _, pt := range pts {
                    coords := strings.Split(pt, ",")
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
        if _, ok := rg.Edges[e.Source]; !ok {
            rg.Edges[e.Source] = make(map[string]*GraphEdge)
        }
        rg.Edges[e.Source][e.Target] = &GraphEdge{
            From:     e.Source,
            To:       e.Target,
            Length:   length,
            Geometry: geometry,
        }
        rg.Nodes[e.Source].OutEdges = append(rg.Nodes[e.Source].OutEdges, e.Target)
    }

    return rg, nil
}

// buildEdgeNeighbors populates rg.EdgeNeighbors so that if an edge is used,
// edges that share a node with it are considered “nearby” and get covered automatically.
func buildEdgeNeighbors(rg *RoadGraph) {
    // We'll assign each directed edge a “key” = "From-To"
    // Then for each node, gather all edges that start or end at that node.
    // All those edges become neighbors of each other.
    edgeMap := make(map[string]float64) // key->length
    for from, m := range rg.Edges {
        for to, ed := range m {
            k := from + "-" + to
            edgeMap[k] = ed.Length
            rg.EdgeNeighbors[k] = []string{} // initialize
        }
    }
    // for each node, get all edges that start from it or end at it
    // but in a directed sense, "start from" is just from->node
    // "end at" is another from->node. We'll do it by scanning the adjacency
    for nodeID, nodeObj := range rg.Nodes {
        // gather all edges that leave this node
        outEdges := nodeObj.OutEdges
        var edgeKeys []string
        for _, neighbor := range outEdges {
            k := nodeID + "-" + neighbor
            if _, ok := edgeMap[k]; ok {
                edgeKeys = append(edgeKeys, k)
            }
        }
        // also gather edges that come into this node (someone else->nodeID)
        // We can find them by scanning reverse adjacency or scanning entire map
        // For simplicity, let's do a quick pass in rg.Edges
        for f, toMap := range rg.Edges {
            if f == nodeID {
                continue
            }
            if _, ok := toMap[nodeID]; ok {
                k := f + "-" + nodeID
                if _, ok2 := edgeMap[k]; ok2 {
                    edgeKeys = append(edgeKeys, k)
                }
            }
        }

        // Now all edgeKeys share the node nodeID => they are neighbors
        // link them pairwise
        for _, e1 := range edgeKeys {
            for _, e2 := range edgeKeys {
                if e1 != e2 {
                    rg.EdgeNeighbors[e1] = append(rg.EdgeNeighbors[e1], e2)
                }
            }
        }
    }
}

// -----------------------------------------
// 5) Load Police Station => Node IDs
// -----------------------------------------

func getPoliceNodesFromExcel(filename string, rg *RoadGraph) ([]string, error) {
    xf, err := xlsx.OpenFile(filename)
    if err != nil {
        return nil, err
    }
    if len(xf.Sheets) == 0 {
        return nil, fmt.Errorf("excel has no sheets")
    }
    sheet := xf.Sheets[0]
    var police []string

    for i, row := range sheet.Rows {
        if i == 0 {
            continue
        }
        if len(row.Cells) == 0 {
            continue
        }
        val := row.Cells[0].String() // e.g. "POINT (lon lat)"
        part := strings.TrimPrefix(val, "POINT (")
        part = strings.TrimSuffix(part, ")")
        coords := strings.Split(part, " ")
        if len(coords) != 2 {
            continue
        }
        lon, errLon := strconv.ParseFloat(coords[0], 64)
        lat, errLat := strconv.ParseFloat(coords[1], 64)
        if errLon != nil || errLat != nil {
            continue
        }
        // find nearest node
        bestID := ""
        bestDist := math.MaxFloat64
        for _, nd := range rg.Nodes {
            d := haversine(lon, lat, nd.X, nd.Y)
            if d < bestDist {
                bestDist = d
                bestID = nd.ID
            }
        }
        if bestID != "" {
            police = append(police, bestID)
        }
    }
    return police, nil
}

// haversine => distance in meters
func haversine(lon1, lat1, lon2, lat2 float64) float64 {
    const R = 6371000
    φ1 := lat1 * math.Pi / 180
    φ2 := lat2 * math.Pi / 180
    dφ := (lat2 - lat1) * math.Pi / 180
    dλ := (lon2 - lon1) * math.Pi / 180

    a := math.Sin(dφ/2)*math.Sin(dφ/2) +
        math.Cos(φ1)*math.Cos(φ2)*math.Sin(dλ/2)*math.Sin(dλ/2)
    c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
    return R * c
}

// -----------------------------------------
// 6) PSO Implementation (with constraints)
// -----------------------------------------

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
            // velocity update
            candidate := pso.velocityUpdate(pso.Swarm[i].Position)
            // mutate
            mutated := pso.mutate(candidate)
            // evaluate
            newFit := pso.fitness(mutated)

            // if improved, accept
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
        fmt.Printf("Iteration %3d, Best Fit: %.2f\n", iter, pso.GlobalBestFit)
    }
    return pso.GlobalBest, pso.GlobalBestFit
}

func (pso *PSO) initializeSwarm() {
    pso.Swarm = make([]Particle, pso.Config.SwarmSize)
    pso.GlobalBestFit = -math.MaxFloat64

    for i := 0; i < pso.Config.SwarmSize; i++ {
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

// generateSolution => random routes from police nodes
func (pso *PSO) generateSolution() Solution {
    s := make(Solution, pso.Config.NumVehicles)
    for i := 0; i < pso.Config.NumVehicles; i++ {
        s[i] = pso.generateRoute()
    }
    return s
}

// generateRoute => pick random police node, walk until max dist or no adjacency
func (pso *PSO) generateRoute() Route {
    if len(pso.PoliceNodes) == 0 {
        // fallback: pick any node
        if len(pso.Graph.Nodes) == 0 {
            return nil
        }
        var firstID string
        for nid := range pso.Graph.Nodes {
            firstID = nid
            break
        }
        return Route{firstID}
    }
    start := pso.PoliceNodes[rand.Intn(len(pso.PoliceNodes))]
    route := Route{start}
    distSoFar := 0.0
    current := start

    for {
        outEdges := pso.Graph.Nodes[current].OutEdges
        if len(outEdges) == 0 {
            break
        }
        nextNode := outEdges[rand.Intn(len(outEdges))]
        e, ok := pso.Graph.Edges[current][nextNode]
        if !ok {
            break
        }
        if distSoFar+e.Length > pso.Config.MaxRouteDistance {
            break
        }
        route = append(route, nextNode)
        distSoFar += e.Length
        current = nextNode
    }

    return route
}

// velocity update => pick from current, personal best from random particle, or global best
func (pso *PSO) velocityUpdate(cur Solution) Solution {
    newSol := make(Solution, len(cur))
    for i := 0; i < len(cur); i++ {
        r := rand.Float64()
        switch {
        case r < 0.34:
            // keep same
            newSol[i] = cur[i]
        case r < 0.67:
            // pick random swarm best
            rp := pso.Swarm[rand.Intn(len(pso.Swarm))].BestPos
            newSol[i] = rp[i]
        default:
            // global best
            newSol[i] = pso.GlobalBest[i]
        }
    }
    return newSol
}

// mutate => randomly rebuild a segment
func (pso *PSO) mutate(sol Solution) Solution {
    mutated := make(Solution, len(sol))
    copy(mutated, sol)

    for i := range mutated {
        if rand.Float64() < pso.Config.MutationRate && len(mutated[i]) > 3 {
            rStart := rand.Intn(len(mutated[i]) - 2)
            rEnd := rStart + 1 + rand.Intn(len(mutated[i])-rStart-1)
            if rEnd >= len(mutated[i]) {
                rEnd = len(mutated[i]) - 1
            }
            seg := pso.generateRouteSegment(mutated[i][rStart], mutated[i][rEnd])
            newRoute := append([]string{}, mutated[i][:rStart]...)
            newRoute = append(newRoute, seg...)
            if rEnd < len(mutated[i]) - 1 {
                newRoute = append(newRoute, mutated[i][rEnd+1:]...)
            }
            mutated[i] = newRoute
        }
    }
    return mutated
}

// generateRouteSegment => random walk up to some steps from start->end
func (pso *PSO) generateRouteSegment(start, end string) Route {
    seg := Route{start}
    current := start
    maxSteps := 20
    steps := 0

    for current != end && steps < maxSteps {
        outEdges := pso.Graph.Nodes[current].OutEdges
        if len(outEdges) == 0 {
            break
        }
        nextNode := outEdges[rand.Intn(len(outEdges))]
        seg = append(seg, nextNode)
        current = nextNode
        steps++
        if current == end {
            break
        }
    }
    // forcibly append end if never reached
    if current != end {
        seg = append(seg, end)
    }
    return seg
}

// fitness => coverage with AQI adjacency, minus overlap, plus divergence penalty if needed
func (pso *PSO) fitness(sol Solution) float64 {
    // 1) Mark edges used by each route
    // 2) coverage => first time we use an edge that isn't "adjacent" to an already covered edge
    // 3) overlap => penalty if an edge is used more than 3 times
    // 4) route divergence => penalty if start nodes are too close

    // edge usage
    edgeCounts := make(map[string]int)
    covered := make(map[string]bool) // edges that are already covered (including their neighbors)
    coverageVal := 0.0

    // gather start nodes for divergence
    var startCoords [][]float64

    for _, route := range sol {
        // record the start node’s coords for divergence
        if len(route) > 0 {
            st := route[0]
            if stNode, ok := pso.Graph.Nodes[st]; ok {
                startCoords = append(startCoords, []float64{stNode.X, stNode.Y})
            } else {
                startCoords = append(startCoords, []float64{0, 0})
            }
        } else {
            startCoords = append(startCoords, []float64{0, 0})
        }

        // walk edges
        for i := 0; i < len(route)-1; i++ {
            from, to := route[i], route[i+1]
            if _, exists := pso.Graph.Edges[from][to]; exists {
                ekey := from + "-" + to
                edgeCounts[ekey]++
                // If not already “covered,” we gain coverage
                if !covered[ekey] {
                    // add coverage
                    edgeObj := pso.Graph.Edges[from][to]
                    coverageVal += edgeObj.Length * pso.Config.CoverageFactor
                    // Mark it & neighbors as covered
                    covered[ekey] = true
                    for _, nb := range pso.Graph.EdgeNeighbors[ekey] {
                        covered[nb] = true
                    }
                }
            }
        }
    }

    // Overlap penalty
    penaltyVal := 0.0
    for ekey, count := range edgeCounts {
        if count > 3 {
            // each edge beyond 3 gets penalized
            parts := strings.Split(ekey, "-")
            if len(parts) == 2 {
                if eObj, ok := pso.Graph.Edges[parts[0]][parts[1]]; ok {
                    penaltyVal += eObj.Length * pso.Config.OverlapPenalty * float64(count-3)
                }
            }
        }
    }

    // Route divergence penalty: if start nodes are too close, apply penalty
    // Compare each pair among startCoords
    divPenalty := 0.0
    for i := 0; i < len(startCoords); i++ {
        for j := i + 1; j < len(startCoords); j++ {
            d := haversine(startCoords[i][0], startCoords[i][1], startCoords[j][0], startCoords[j][1])
            if d < pso.Config.DivergenceThreshold {
                // they started too close
                divPenalty += pso.Config.DivergencePenalty
            }
        }
    }

    return coverageVal - penaltyVal - divPenalty
}

// -----------------------------------------
// 7) Convert solution => GeoJSON
// -----------------------------------------

func solutionToGeoJSON(graph *RoadGraph, sol Solution) *GeoJSON {
    colors := []string{
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8",
        "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    }

    var features []GeoJSONFeature
    for vid, route := range sol {
        if len(route) < 2 {
            continue
        }

        var lines [][][]float64
        for i := 0; i < len(route)-1; i++ {
            from, to := route[i], route[i+1]
            e, ok := graph.Edges[from][to]
            if !ok {
                // check reverse
                if rev, ok2 := graph.Edges[to][from]; ok2 {
                    e = rev
                    ok = true
                }
            }
            if ok && e != nil {
                if len(e.Geometry) > 0 {
                    lineCoords := make([][]float64, len(e.Geometry))
                    for j, pt := range e.Geometry {
                        lineCoords[j] = []float64{pt[0], pt[1]}
                    }
                    lines = append(lines, lineCoords)
                } else {
                    // fallback
                    fn, fok := graph.Nodes[from]
                    tn, tok := graph.Nodes[to]
                    if fok && tok {
                        lines = append(lines, [][]float64{
                            {fn.X, fn.Y},
                            {tn.X, tn.Y},
                        })
                    }
                }
            } else {
                // fallback if no edge
                fn, fok := graph.Nodes[from]
                tn, tok := graph.Nodes[to]
                if fok && tok {
                    lines = append(lines, [][]float64{
                        {fn.X, fn.Y},
                        {tn.X, tn.Y},
                    })
                }
            }
        }

        if len(lines) == 0 {
            continue
        }

        feat := GeoJSONFeature{
            Type: "Feature",
            Geometry: GeoJSONGeometry{
                Type:        "MultiLineString",
                Coordinates: lines,
            },
            Properties: map[string]interface{}{
                "vehicle":      vid + 1,
                "stroke":       colors[vid%len(colors)],
                "stroke-width": 3,
            },
        }
        features = append(features, feat)
    }

    return &GeoJSON{
        Type:     "FeatureCollection",
        Features: features,
    }
}
