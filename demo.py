#!/usr/bin/env python3
"""
Demo script for PII Detection Agent.
"""
import os
import json
from pii_agent import PIIDetectionAgent
from ner_detector import create_sample_data


def create_demo_data():
    """Create a more comprehensive demo dataset."""
    demo_data = """name,email,phone,address,ssn,credit_card,notes
John Doe,john.doe@email.com,555-123-4567,123 Main St New York NY 10001,123-45-6789,4532-1234-5678-9012,Works at ABC Corp
Jane Smith,jane.smith@company.org,(555) 987-6543,456 Oak Ave Los Angeles CA 90210,987-65-4321,5555-4444-3333-2222,Medical record: Patient ID 12345
Bob Johnson,bob@test.net,555.111.2222,789 Pine Rd Chicago IL 60601,456-78-9012,4111-1111-1111-1111,IP address: 192.168.1.100
Alice Brown,alice.brown@example.com,555 333 4444,321 Elm St Houston TX 77001,789-01-2345,6011-1111-1111-1111,Salary: $75,000
Charlie Wilson,charlie.wilson@demo.org,555-555-5555,654 Maple Dr Phoenix AZ 85001,321-54-9876,3000-0000-0000-0004,Employee ID: EMP001
Diana Lee,diana.lee@sample.com,555-777-8888,987 Cedar Ln Seattle WA 98101,654-32-1098,2223-0000-0000-0000,Patient: Diagnosis confidential
Frank Miller,frank.miller@example.net,555-999-0000,147 Birch St Denver CO 80201,147-85-2963,4000-0000-0000-0002,Account balance: $12,500
Grace Taylor,grace.taylor@demo.com,555-222-3333,258 Spruce Ave Miami FL 33101,258-74-1963,5555-5555-5555-4444,Insurance policy: POL123456"""
    
    demo_file = "/Users/ynaung/MSDS/DE/llm_security/hw4/demo_data.csv"
    with open(demo_file, 'w') as f:
        f.write(demo_data)
    
    return demo_file


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_results_summary(results):
    """Print a summary of the results."""
    print_section("PII DETECTION RESULTS SUMMARY")
    
    if not results['success']:
        print(f"❌ Processing failed: {results['error']}")
        return
    
    print("✅ Processing completed successfully!")
    print(f"📊 Report: {results['report_path']}")
    print(f"🔒 Masked CSV: {results['masked_csv_path']}")
    print(f"📈 Graph Visualization: {results['graph_visualization']}")
    print(f"📋 Graph Data: {results['graph_data']}")
    
    # Load and display report summary
    if os.path.exists(results['report_path']):
        with open(results['report_path'], 'r') as f:
            report = json.load(f)
        
        print_section("DETECTION SUMMARY")
        
        # NER Summary
        ner_summary = report['ner_summary']
        print(f"🔍 Total PII Entities Detected: {ner_summary['total_detections']}")
        print("📋 Entity Types Found:")
        for entity_type, count in ner_summary['detection_types'].items():
            print(f"   • {entity_type.title()}: {count}")
        
        # Proximity Summary
        prox_summary = report['proximity_summary']
        print(f"\n⚠️  Risk Assessment:")
        print(f"   • High Risk Rows: {prox_summary['high_risk_rows']}")
        print(f"   • Medium Risk Rows: {prox_summary['medium_risk_rows']}")
        print(f"   • Low Risk Rows: {prox_summary['low_risk_rows']}")
        
        # Graph Summary
        graph_summary = report['graph_summary']
        graph_stats = graph_summary['graph_stats']
        print(f"\n🕸️  Graph Analysis:")
        print(f"   • Total Nodes: {graph_stats['nodes']}")
        print(f"   • Total Edges: {graph_stats['edges']}")
        print(f"   • Graph Density: {graph_stats['density']:.3f}")
        
        reid_risk = graph_summary['reidentification_risk']
        risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        print(f"   • Re-identification Risk: {risk_emoji.get(reid_risk['risk_level'], '⚪')} {reid_risk['risk_level'].upper()}")
        
        # Recommendations
        print_section("SECURITY RECOMMENDATIONS")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")


def demonstrate_individual_components():
    """Demonstrate individual components."""
    print_section("INDIVIDUAL COMPONENT DEMONSTRATION")
    
    from ner_detector import PIINERDetector
    from proximity_analyzer import ProximityAnalyzer
    from graph_builder import PIIGraphBuilder
    
    # Create demo data
    demo_file = create_demo_data()
    
    try:
        # 1. NER Detection
        print("🔍 Running NER Detection...")
        ner_detector = PIINERDetector()
        ner_results = ner_detector.process_csv(demo_file)
        summary = ner_detector.get_detection_summary(ner_results)
        print(f"   Found {summary['total_detections']} entities across {len(summary['columns_affected'])} columns")
        
        # 2. Proximity Analysis
        print("📏 Running Proximity Analysis...")
        proximity_analyzer = ProximityAnalyzer()
        proximity_results = proximity_analyzer.process_csv_proximity(demo_file, ner_results)
        high_risk = sum(1 for analysis in proximity_results['proximity_analyses'] 
                       if analysis['overall_risk'] == 'high')
        print(f"   Identified {high_risk} high-risk rows")
        
        # 3. Graph Analysis
        print("🕸️  Building Entity Graph...")
        graph_builder = PIIGraphBuilder()
        graph = graph_builder.build_graph(ner_results, proximity_results)
        analysis = graph_builder.analyze_graph()
        print(f"   Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        print(f"   Re-identification risk: {analysis['reidentification_risk']['risk_level']}")
        
        # 4. Visualization
        print("📊 Creating Graph Visualization...")
        viz_path = graph_builder.visualize_graph()
        print(f"   Visualization saved to: {viz_path}")
        
    finally:
        # Clean up demo file
        if os.path.exists(demo_file):
            os.remove(demo_file)


def main():
    """Main demo function."""
    print_section("PII DETECTION AGENT DEMO")
    print("This demo showcases the complete PII detection pipeline")
    print("using NER, Proximity Analysis, and Graph Theory.")
    
    # Create demo data
    demo_file = create_demo_data()
    
    try:
    # Initialize agent
    print("\n🤖 Initializing PII Detection Agent...")
    agent = PIIDetectionAgent()
    
    # Check if OpenAI is configured
    if agent.llm is None:
        print("⚠️  OpenAI API key not configured. LLM features will be limited.")
        print("   Set OPENAI_API_KEY in .env.local for enhanced analysis.")
    else:
        print("✅ OpenAI integration ready")
        
        # Process the demo CSV
        print("📁 Processing demo CSV file...")
        results = agent.process_csv(demo_file)
        
        # Display results
        print_results_summary(results)
        
        # Demonstrate individual components
        demonstrate_individual_components()
        
        print_section("DEMO COMPLETED")
        print("✅ All components demonstrated successfully!")
        print("📁 Check the output directory for generated files:")
        print("   • pii_detection_report.json - Comprehensive analysis report")
        print("   • masked_data.csv - Original data with PII masked")
        print("   • pii_graph_visualization.png - Entity relationship graph")
        print("   • pii_graph_data.json - Raw graph data")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo file
        if os.path.exists(demo_file):
            os.remove(demo_file)


if __name__ == "__main__":
    main()
