"""
PII Graph Builder using NetworkX for entity relationship analysis.
"""
import json
import logging
from typing import Dict, List, Any, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIGraphBuilder:
    """
    Builds and analyzes entity graphs to detect PII clusters and re-identification risks.
    
    This class uses NetworkX to:
    - Build entity co-occurrence graphs
    - Detect clusters of related PII
    - Calculate centrality measures
    - Identify high-risk entity combinations
    """
    
    def __init__(self, proximity_window: int = 50):
        """
        Initialize the PII Graph Builder.
        
        Args:
            proximity_window: Window size for co-occurrence detection
        """
        self.proximity_window = proximity_window
        self.graph = nx.Graph()
        self.entity_metadata = {}
        self.risk_weights = self._initialize_risk_weights()
    
    def _initialize_risk_weights(self) -> Dict[str, float]:
        """Initialize risk weights for different entity types."""
        return {
            'ssn': 1.0,
            'credit_card': 0.9,
            'name': 0.8,
            'phone': 0.7,
            'email': 0.6,
            'address': 0.5,
            'zip': 0.4,
            'organization': 0.3,
            'location': 0.2
        }
    
    def build_graph(self, ner_results: Dict[str, Any], 
                   proximity_results: Dict[str, Any] = None) -> nx.Graph:
        """
        Build entity co-occurrence graph from NER and proximity results.
        
        Args:
            ner_results: Results from NER detector
            proximity_results: Results from proximity analyzer (optional)
            
        Returns:
            NetworkX graph with entities as nodes and co-occurrences as edges
        """
        self.graph.clear()
        self.entity_metadata.clear()
        
        # Group entities by row for co-occurrence analysis
        entities_by_row = defaultdict(list)
        
        # Add entities from NER results
        for detection in ner_results['detections']:
            row = detection['row']
            entity = detection['entity']
            
            # Create unique node ID
            node_id = f"{row}_{entity['start']}_{entity['end']}_{entity['text']}"
            
            # Add node with metadata
            self.graph.add_node(node_id, **{
                'text': entity['text'],
                'type': entity['type'],
                'row': row,
                'column': detection['column'],
                'start': entity['start'],
                'end': entity['end'],
                'confidence': entity['confidence'],
                'method': entity['method']
            })
            
            self.entity_metadata[node_id] = entity
            entities_by_row[row].append(node_id)
        
        # Add inferred entities from proximity analysis
        if proximity_results:
            for analysis in proximity_results['proximity_analyses']:
                row = analysis['row']
                
                for inferred_entity in analysis['inferred_entities']:
                    node_id = f"inf_{row}_{inferred_entity['start']}_{inferred_entity['end']}_{inferred_entity['text']}"
                    
                    self.graph.add_node(node_id, **{
                        'text': inferred_entity['text'],
                        'type': inferred_entity['type'],
                        'row': row,
                        'start': inferred_entity['start'],
                        'end': inferred_entity['end'],
                        'confidence': inferred_entity['confidence'],
                        'method': 'proximity_inference',
                        'risk_level': inferred_entity.get('risk_level', 'medium')
                    })
                    
                    entities_by_row[row].append(node_id)
        
        # Add edges for co-occurring entities
        self._add_cooccurrence_edges(entities_by_row)
        
        # Add edges based on proximity relations
        if proximity_results:
            self._add_proximity_edges(proximity_results)
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _add_cooccurrence_edges(self, entities_by_row: Dict[int, List[str]]) -> None:
        """Add edges between entities that co-occur in the same row."""
        for row, node_ids in entities_by_row.items():
            # Add edges between all pairs of entities in the same row
            for i, node1 in enumerate(node_ids):
                for node2 in node_ids[i+1:]:
                    # Calculate edge weight based on entity types
                    weight = self._calculate_edge_weight(node1, node2)
                    self.graph.add_edge(node1, node2, weight=weight, relation_type='cooccurrence')
    
    def _add_proximity_edges(self, proximity_results: Dict[str, Any]) -> None:
        """Add edges based on proximity relations."""
        if not proximity_results or 'proximity_analyses' not in proximity_results:
            return
            
        for analysis in proximity_results['proximity_analyses']:
            if 'row' not in analysis:
                continue
                
            row = analysis['row']
            
            for relation in analysis.get('proximity_relations', []):
                try:
                    # Validate relation structure
                    if not isinstance(relation, dict) or 'entity1' not in relation or 'entity2' not in relation:
                        logger.warning(f"Invalid relation structure: {relation}")
                        continue
                    
                    # Find corresponding nodes in the graph
                    node1 = self._find_node_by_entity(row, relation['entity1'])
                    node2 = self._find_node_by_entity(row, relation['entity2'])
                    
                    if node1 and node2:
                        # Calculate weight based on distance and risk level
                        weight = self._calculate_proximity_weight(relation)
                        self.graph.add_edge(node1, node2, 
                                          weight=weight, 
                                          relation_type='proximity',
                                          distance=relation.get('distance', 0))
                except Exception as e:
                    logger.warning(f"Error processing proximity relation: {e}")
                    continue
    
    def _find_node_by_entity(self, row: int, entity: Dict[str, Any]) -> str:
        """Find graph node corresponding to an entity."""
        for node_id, data in self.graph.nodes(data=True):
            if (data['row'] == row and 
                data['start'] == entity['start'] and 
                data['end'] == entity['end'] and
                data['text'] == entity['text']):
                return node_id
        return None
    
    def _calculate_edge_weight(self, node1: str, node2: str) -> float:
        """Calculate edge weight based on entity types and risk levels."""
        data1 = self.graph.nodes[node1]
        data2 = self.graph.nodes[node2]
        
        type1 = data1['type']
        type2 = data2['type']
        
        # Base weight from risk weights
        weight1 = self.risk_weights.get(type1, 0.1)
        weight2 = self.risk_weights.get(type2, 0.1)
        
        # Combine weights (geometric mean)
        combined_weight = (weight1 * weight2) ** 0.5
        
        # Boost for high-risk combinations
        high_risk_combinations = [
            ('name', 'ssn'), ('name', 'phone'), ('name', 'email'),
            ('address', 'name'), ('zip', 'org')
        ]
        
        if ((type1, type2) in high_risk_combinations or 
            (type2, type1) in high_risk_combinations):
            combined_weight *= 1.5
        
        return combined_weight
    
    def _calculate_proximity_weight(self, relation: Dict[str, Any]) -> float:
        """Calculate weight for proximity-based edges."""
        distance = relation['distance']
        max_distance = self.proximity_window
        
        # Weight decreases with distance
        distance_factor = max(0.1, 1.0 - (distance / max_distance))
        
        # Get base weight from entity types
        entity1_type = relation['entity1']['type']
        entity2_type = relation['entity2']['type']
        
        weight1 = self.risk_weights.get(entity1_type, 0.1)
        weight2 = self.risk_weights.get(entity2_type, 0.1)
        
        return (weight1 * weight2) ** 0.5 * distance_factor
    
    def analyze_graph(self) -> Dict[str, Any]:
        """
        Analyze the built graph for clusters and risk patterns.
        
        Returns:
            Graph analysis results including clusters, centrality, and risk metrics
        """
        if self.graph.number_of_nodes() == 0:
            return {'error': 'No entities found in graph'}
        
        analysis = {
            'graph_stats': self._get_graph_statistics(),
            'connected_components': self._analyze_connected_components(),
            'centrality_measures': self._calculate_centrality_measures(),
            'risk_clusters': self._identify_risk_clusters(),
            'reidentification_risk': self._assess_reidentification_risk()
        }
        
        return analysis
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics."""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'is_connected': nx.is_connected(self.graph)
        }
    
    def _analyze_connected_components(self) -> List[Dict[str, Any]]:
        """Analyze connected components in the graph."""
        components = list(nx.connected_components(self.graph))
        
        component_analysis = []
        for i, component in enumerate(components):
            subgraph = self.graph.subgraph(component)
            
            # Get entity types in this component
            entity_types = [self.graph.nodes[node]['type'] for node in component]
            type_counts = Counter(entity_types)
            
            # Calculate component risk score
            risk_score = sum(self.risk_weights.get(entity_type, 0.1) 
                           for entity_type in entity_types) / len(entity_types)
            
            component_analysis.append({
                'component_id': i,
                'size': len(component),
                'entity_types': dict(type_counts),
                'risk_score': risk_score,
                'nodes': list(component)
            })
        
        # Sort by risk score (highest first)
        component_analysis.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return component_analysis
    
    def _calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures for nodes."""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        # Get top nodes for each measure
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'degree_centrality': dict(top_degree),
            'betweenness_centrality': dict(top_betweenness),
            'closeness_centrality': dict(top_closeness)
        }
    
    def _identify_risk_clusters(self) -> List[Dict[str, Any]]:
        """Identify high-risk clusters of entities."""
        risk_clusters = []
        
        # Find cliques (complete subgraphs)
        cliques = list(nx.find_cliques(self.graph))
        
        for clique in cliques:
            if len(clique) >= 2:  # Only consider cliques with 2+ nodes
                # Calculate cluster risk
                entity_types = [self.graph.nodes[node]['type'] for node in clique]
                unique_types = set(entity_types)
                
                # Risk increases with number of unique entity types
                risk_score = len(unique_types) * 0.2
                
                # Boost for high-risk entity types
                high_risk_types = {'ssn', 'credit_card', 'name'}
                if any(t in unique_types for t in high_risk_types):
                    risk_score += 0.5
                
                risk_clusters.append({
                    'nodes': clique,
                    'size': len(clique),
                    'entity_types': list(unique_types),
                    'risk_score': risk_score,
                    'cluster_type': 'clique'
                })
        
        # Sort by risk score
        risk_clusters.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return risk_clusters
    
    def _assess_reidentification_risk(self) -> Dict[str, Any]:
        """Assess overall re-identification risk."""
        if self.graph.number_of_nodes() == 0:
            return {'risk_level': 'low', 'score': 0.0}
        
        # Calculate risk factors
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        
        # Risk increases with graph connectivity
        connectivity_risk = min(total_edges / (total_nodes * (total_nodes - 1) / 2), 1.0)
        
        # Risk from high-value entities
        high_risk_entities = sum(1 for node, data in self.graph.nodes(data=True)
                                if data['type'] in ['ssn', 'credit_card', 'name'])
        entity_risk = high_risk_entities / total_nodes
        
        # Risk from large connected components
        components = list(nx.connected_components(self.graph))
        largest_component_size = max(len(comp) for comp in components) if components else 0
        component_risk = largest_component_size / total_nodes
        
        # Overall risk score
        overall_risk_score = (connectivity_risk * 0.4 + entity_risk * 0.4 + component_risk * 0.2)
        
        # Determine risk level
        if overall_risk_score > 0.7:
            risk_level = 'high'
        elif overall_risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'score': overall_risk_score,
            'factors': {
                'connectivity_risk': connectivity_risk,
                'entity_risk': entity_risk,
                'component_risk': component_risk
            }
        }
    
    def visualize_graph(self, output_path: str = None, max_nodes: int = 50) -> str:
        """
        Visualize the entity graph.
        
        Args:
            output_path: Path to save the visualization
            max_nodes: Maximum number of nodes to display
            
        Returns:
            Path to the saved visualization
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("No nodes to visualize")
            return None
        
        # Limit nodes for visualization
        if self.graph.number_of_nodes() > max_nodes:
            # Get top nodes by degree centrality
            degree_centrality = nx.degree_centrality(self.graph)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_show = [node for node, _ in top_nodes]
            subgraph = self.graph.subgraph(nodes_to_show)
        else:
            subgraph = self.graph
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Color nodes by entity type
        node_colors = []
        type_colors = {
            'name': 'red',
            'email': 'blue',
            'phone': 'green',
            'ssn': 'orange',
            'address': 'purple',
            'credit_card': 'brown',
            'organization': 'pink',
            'location': 'gray',
            'zip': 'yellow'
        }
        
        for node in subgraph.nodes():
            entity_type = subgraph.nodes[node]['type']
            node_colors.append(type_colors.get(entity_type, 'lightblue'))
        
        # Draw the graph
        nx.draw(subgraph, pos, 
                node_color=node_colors,
                node_size=300,
                font_size=8,
                font_weight='bold',
                with_labels=True,
                edge_color='gray',
                alpha=0.7)
        
        # Create legend
        legend_elements = [mpatches.Patch(color=color, label=entity_type.title())
                          for entity_type, color in type_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title('PII Entity Relationship Graph')
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            output_path = '/Users/ynaung/MSDS/DE/llm_security/hw4/pii_graph_visualization.png'
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graph visualization saved to {output_path}")
        return output_path
    
    def export_graph_data(self, output_path: str = None) -> str:
        """
        Export graph data to JSON format.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        if output_path is None:
            output_path = '/Users/ynaung/MSDS/DE/llm_security/hw4/pii_graph_data.json'
        
        # Convert graph to JSON-serializable format
        graph_data = {
            'nodes': [
                {
                    'id': node,
                    'data': data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'data': data
                }
                for edge, data in self.graph.edges(data=True)
            ],
            'metadata': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'proximity_window': self.proximity_window
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"Graph data exported to {output_path}")
        return output_path


if __name__ == "__main__":
    # Test the graph builder
    from ner_detector import PIINERDetector, create_sample_data
    from proximity_analyzer import ProximityAnalyzer
    
    # Create sample data and run NER
    sample_file = create_sample_data()
    ner_detector = PIINERDetector()
    ner_results = ner_detector.process_csv(sample_file)
    
    # Run proximity analysis
    proximity_analyzer = ProximityAnalyzer()
    proximity_results = proximity_analyzer.process_csv_proximity(sample_file, ner_results)
    
    # Build and analyze graph
    graph_builder = PIIGraphBuilder()
    graph = graph_builder.build_graph(ner_results, proximity_results)
    analysis = graph_builder.analyze_graph()
    
    print("Graph Analysis Results:")
    print(f"Graph Stats: {analysis['graph_stats']}")
    print(f"Connected Components: {len(analysis['connected_components'])}")
    print(f"Risk Clusters: {len(analysis['risk_clusters'])}")
    print(f"Re-identification Risk: {analysis['reidentification_risk']}")
    
    # Visualize graph
    viz_path = graph_builder.visualize_graph()
    print(f"Graph visualization saved to: {viz_path}")
    
    # Export graph data
    export_path = graph_builder.export_graph_data()
    print(f"Graph data exported to: {export_path}")
