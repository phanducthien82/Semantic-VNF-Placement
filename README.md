# Semantic-Aware VNF Placement Ontology

**nfv-infra.owl** – OWL 2 EL ontology for RDF-based VNF placement in 5G/6G networks.

**Authors**: Duc-Thien Phan, KenLi Li, Nhuan D. To  
**Paper**: Semantic-Aware VNF Placement: Bridging RDF Knowledge Graphs with Hybrid Optimization for 5G/6G Networks

## How to use
```bash
pip install owlready2
python -c "
from owlready2 import get_ontology
onto = get_ontology('nfv-infra.owl').load()
print('Loaded', len(list(onto.classes())), 'classes')
"