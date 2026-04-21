"""
Semantic-Aware VNF Placement Simulation
- Load RDF ontology
- Semantic filtering (SPARQL + reasoning)
- ILP solver (PuLP) - optimal
- Genetic Algorithm (DEAP) - near-optimal & fast
Author: Duc-Thien Phan (based on the paper)
"""

import networkx as nx
from owlready2 import get_ontology, sync_reasoner, default_world
import pulp
import random
from deap import base, creator, tools, algorithms
import numpy as np
import time

# ====================== 1. LOAD ONTOLOGY & REASONING ======================
print("Loading ontology...")
onto = get_ontology("nfv-infra.owl").load()
with onto:
    sync_reasoner()
print("Reasoning completed!")

# Get all servers
servers = [str(s).split("#")[-1] for s in onto.Server.instances()]
print(f"Total servers: {len(servers)}")

# ====================== 2. SEMANTIC FILTERING (SPARQL) ======================
def get_feasible_nodes(vnf_requirements):
    """Semantic pre-filtering using SPARQL"""
    query = """
    PREFIX : <http://example.org/nfv-infra#>
    SELECT ?node WHERE {
        ?node rdf:type :Server .
        ?node :hasAvailableCPU ?cpu . FILTER(?cpu >= %d)
        ?node :hasHardwareAccel ?accel . FILTER(?accel = %s)
        ?node :hasLatency ?lat . FILTER(?lat <= %d)
    }
    """ % (vnf_requirements["cpu"], "true" if vnf_requirements["hw_accel"] else "false", vnf_requirements["latency"])
    
    results = default_world.sparql(query)
    return [str(row[0]).split("#")[-1] for row in results]

# Example SFC: Firewall -> LoadBalancer
sfc = [
    {"id": "Firewall", "cpu": 4, "hw_accel": True, "latency": 5},
    {"id": "LoadBalancer", "cpu": 8, "hw_accel": True, "latency": 3}
]

candidate_dict = {vnf["id"]: get_feasible_nodes(vnf) for vnf in sfc}
print("Semantic candidates:", {k: len(v) for k, v in candidate_dict.items()})

# ====================== 3. INTEGER LINEAR PROGRAMMING (ILP) with PuLP ======================
def solve_ilp(sfc, candidates):
    prob = pulp.LpProblem("Semantic_VNF_Placement_ILP", pulp.LpMinimize)
    
    # Variables: x[v][n] = 1 if VNF v placed on node n
    x = {}
    for i, vnf in enumerate(sfc):
        for node in candidates[vnf["id"]]:
            x[(i, node)] = pulp.LpVariable(f"x_{i}_{node}", cat="Binary")
    
    # Objective: minimize total latency (simplified - in real use real latency matrix)
    prob += pulp.lpSum(x[(i, node)] * 1 for i, node in x)  # placeholder
    
    # Capacity constraints (simplified)
    for node in servers:
        prob += pulp.lpSum(x[(i, node)] for i, node in x if node in candidates[sfc[i]["id"]]) * sfc[i]["cpu"] <= 16  # example
    
    # Each VNF placed exactly once
    for i, vnf in enumerate(sfc):
        prob += pulp.lpSum(x[(i, node)] for node in candidates[vnf["id"]]) == 1
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    placement = {}
    for (i, node), var in x.items():
        if var.value() > 0.5:
            placement[sfc[i]["id"]] = node
    return placement, pulp.value(prob.objective)

start = time.time()
ilp_placement, ilp_latency = solve_ilp(sfc, candidate_dict)
print(f"ILP solved in {time.time()-start:.4f}s → Placement: {ilp_placement}")

# ====================== 4. GENETIC ALGORITHM (DEAP) ======================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def evaluate(individual):
    # individual = [node_for_vnf0, node_for_vnf1, ...]
    latency = 0  # placeholder
    return (latency,)

toolbox = base.Toolbox()
toolbox.register("attr_node", lambda: random.choice(servers))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_node, n=len(sfc))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, verbose=False)

ga_placement = {sfc[i]["id"]: ind[i] for i, ind in enumerate(pop[0])}
print(f"GA best placement: {ga_placement}")

print("\n🎉 Simulation completed!")
print("ILP (optimal)  :", ilp_placement)
print("GA (near-opt)  :", ga_placement)