"""
Semantic-Aware VNF Placement - Python Example
Load OWL ontology, perform reasoning, and execute SPARQL queries
Compatible with the paper: "Semantic-Aware VNF Placement" (Phan et al.)
"""

from owlready2 import get_ontology, sync_reasoner, default_world
import rdflib
from rdflib import Graph
import pandas as pd

# ====================== 1. LOAD ONTOLOGY ======================
print("Loading nfv-infra.ttl ...")
onto = get_ontology("nfv-infra.ttl").load(format="turtle")
print(f"Ontology loaded successfully!")
print(f"   Classes: {len(list(onto.classes()))}")
print(f"   Individuals: {len(list(onto.individuals()))}")

# ====================== 2. RUN REASONING (HermiT) ======================
print("\n🧠 Running HermiT reasoner (OWL 2 EL profile)...")
with onto:
    sync_reasoner(infer_property_values=True, infer_data_property_values=True)
print("Reasoning completed!")

# Show inferred IncompatibleNode
incompatible = list(onto.IncompatibleNode.instances())
print(f"   Inferred Incompatible Nodes: {len(incompatible)}")
for node in incompatible[:5]:  # show first 5
    print(f"     → {node}")

# ====================== 3. SPARQL QUERY - Candidate Extraction (exactly as in paper) ======================
print("\nExecuting SPARQL query (Listing 1 in paper)...")

# SPARQL query giống hệt Listing 1 trong bài báo
sparql_query = """
PREFIX : <http://example.org/nfv-infra#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?node ?cpu ?latency ?hasAccel
WHERE {
    ?node rdf:type :EdgeServer .
    ?node :hasAvailableCPU ?cpu .
    FILTER (?cpu >= 4)
    ?node :hasLatency ?latency .
    ?node :hasHardwareAccel ?hasAccel .
    ?node :locatedIn ?zone .
    ?zone rdf:type :LowLatencyZone .
}
ORDER BY DESC(?cpu)
"""

results = default_world.sparql(sparql_query)

print("✅ Feasible Edge Candidates (sorted by CPU):")
data = []
for row in results:
    node = str(row[0]).split("#")[-1]
    cpu = int(row[1])
    latency = int(row[2])
    has_accel = bool(row[3])
    data.append([node, cpu, latency, has_accel])

df = pd.DataFrame(data, columns=["Node", "CPU", "Latency (ms)", "Has Hardware Accel"])
print(df)

# ====================== 4. EXAMPLE: Get all candidate nodes for a specific VNF ======================
print("\nExample: Candidates for FirewallVNF (LowLatency + HardwareAccel)")
v = onto.FirewallVNF
candidates = list(default_world.sparql("""
    PREFIX : <http://example.org/nfv-infra#>
    SELECT ?node WHERE {
        ?node rdf:type :Server .
        ?node :hasAvailableCPU ?cpu FILTER (?cpu >= 4)
        ?node :hasHardwareAccel true
        ?node :hasLatency ?lat FILTER (?lat <= 5)
    }
"""))

print(f"   → Found {len(candidates)} compatible nodes for {v}")
for c in candidates[:6]:
    print("     ", str(c[0]).split("#")[-1])

# ====================== 5. EXPORT TO NetworkX (for simulation integration) ======================
print("\n🌐 Converting ontology to NetworkX graph (ready for VNF placement simulation)...")
import networkx as nx
G = nx.DiGraph()

for node in onto.Server.instances():
    node_name = str(node).split("#")[-1]
    G.add_node(node_name, 
               cpu=node.hasAvailableCPU[0] if node.hasAvailableCPU else 0,
               latency=node.hasLatency[0] if node.hasLatency else 999,
               hardware_accel=node.hasHardwareAccel[0] if node.hasHardwareAccel else False)

print(f"   NetworkX graph created with {G.number_of_nodes()} nodes")
print("   Ready to integrate with PuLP ILP or Genetic Algorithm!")

# ====================== 6. SAVE RESULTS (optional) ======================
df.to_csv("candidate_nodes.csv", index=False)
print("\nSaved candidate_nodes.csv")
print("\nAll done! You can now use this ontology in your VNF placement simulation.")