"""
PII Detection Agent using LangGraph for orchestration.
"""
import json
import logging
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import pandas as pd
from pathlib import Path

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# OpenAI imports
from langchain_openai import ChatOpenAI

# Local imports
from ner_detector import PIINERDetector
from proximity_analyzer import ProximityAnalyzer
from graph_builder import PIIGraphBuilder
from config import OPENAI_API_KEY, MASK_CHAR, DEFAULT_PROXIMITY_WINDOW

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIAgentState(TypedDict):
    """State for the PII detection agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    csv_path: str
    output_dir: str
    ner_results: Optional[Dict[str, Any]]
    proximity_results: Optional[Dict[str, Any]]
    graph_results: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    masked_csv_path: Optional[str]
    error: Optional[str]


class PIIDetectionAgent:
    """
    Main PII Detection Agent using LangGraph for orchestration.
    
    This agent coordinates the entire PII detection pipeline:
    1. NER Detection
    2. Proximity Analysis
    3. Graph Analysis
    4. Report Generation
    5. CSV Masking
    """
    
    def __init__(self, api_key: str = None, proximity_window: int = DEFAULT_PROXIMITY_WINDOW):
        """
        Initialize the PII Detection Agent.
        
        Args:
            api_key: OpenAI API key
            proximity_window: Window size for proximity analysis
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.proximity_window = proximity_window
        
        # Initialize components
        self.ner_detector = PIINERDetector()
        self.proximity_analyzer = ProximityAnalyzer(window_size=proximity_window)
        self.graph_builder = PIIGraphBuilder(proximity_window=proximity_window)
        
        # Initialize LLM
        self._setup_llm()
        
        # Build the agent graph
        self.agent = self._build_agent()
    
    def _setup_llm(self) -> None:
        """Setup the OpenAI LLM."""
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM features will be limited.")
            self.llm = None
            return
        
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=self.api_key,
                temperature=0.1
            )
            logger.info("Successfully initialized OpenAI LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
    
    def _build_agent(self) -> StateGraph:
        """Build the LangGraph agent."""
        workflow = StateGraph(PIIAgentState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("run_ner", self._run_ner_detection)
        workflow.add_node("run_proximity", self._run_proximity_analysis)
        workflow.add_node("run_graph_analysis", self._run_graph_analysis)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("mask_csv", self._mask_csv)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("validate_input")
        
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue,
            {
                "continue": "run_ner",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "run_ner",
            self._should_continue,
            {
                "continue": "run_proximity",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "run_proximity",
            self._should_continue,
            {
                "continue": "run_graph_analysis",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "run_graph_analysis",
            self._should_continue,
            {
                "continue": "generate_report",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_report",
            self._should_continue,
            {
                "continue": "mask_csv",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("mask_csv", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _validate_input(self, state: PIIAgentState) -> PIIAgentState:
        """Validate input CSV file."""
        try:
            csv_path = state["csv_path"]
            
            # Check if file exists
            if not os.path.exists(csv_path):
                state["error"] = f"CSV file not found: {csv_path}"
                return state
            
            # Check file size
            file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            if file_size_mb > 100:  # 100MB limit
                state["error"] = f"File too large: {file_size_mb:.2f}MB (max 100MB)"
                return state
            
            # Try to read CSV
            df = pd.read_csv(csv_path)
            if df.empty:
                state["error"] = "CSV file is empty"
                return state
            
            # Create output directory
            output_dir = state.get("output_dir", os.path.dirname(csv_path))
            os.makedirs(output_dir, exist_ok=True)
            state["output_dir"] = output_dir
            
            logger.info(f"Successfully validated CSV: {len(df)} rows, {len(df.columns)} columns")
            state["messages"].append(AIMessage(content=f"Validated CSV file with {len(df)} rows"))
            
        except Exception as e:
            state["error"] = f"Input validation failed: {str(e)}"
            logger.error(f"Input validation error: {e}")
        
        return state
    
    def _run_ner_detection(self, state: PIIAgentState) -> PIIAgentState:
        """Run NER detection on the CSV file."""
        try:
            csv_path = state["csv_path"]
            
            logger.info("Starting NER detection...")
            ner_results = self.ner_detector.process_csv(csv_path)
            
            # Get summary
            summary = self.ner_detector.get_detection_summary(ner_results)
            
            state["ner_results"] = ner_results
            state["messages"].append(AIMessage(
                content=f"NER detection completed: {summary['total_detections']} entities found"
            ))
            
            logger.info(f"NER detection completed: {summary['total_detections']} entities")
            
        except Exception as e:
            state["error"] = f"NER detection failed: {str(e)}"
            logger.error(f"NER detection error: {e}")
        
        return state
    
    def _run_proximity_analysis(self, state: PIIAgentState) -> PIIAgentState:
        """Run proximity analysis."""
        try:
            csv_path = state["csv_path"]
            ner_results = state["ner_results"]
            
            logger.info("Starting proximity analysis...")
            proximity_results = self.proximity_analyzer.process_csv_proximity(csv_path, ner_results)
            
            # Calculate overall risk
            overall_risks = [analysis['overall_risk'] for analysis in proximity_results['proximity_analyses']]
            high_risk_count = sum(1 for risk in overall_risks if risk == 'high')
            
            state["proximity_results"] = proximity_results
            state["messages"].append(AIMessage(
                content=f"Proximity analysis completed: {high_risk_count} high-risk rows identified"
            ))
            
            logger.info(f"Proximity analysis completed: {high_risk_count} high-risk rows")
            
        except Exception as e:
            state["error"] = f"Proximity analysis failed: {str(e)}"
            logger.error(f"Proximity analysis error: {e}")
        
        return state
    
    def _run_graph_analysis(self, state: PIIAgentState) -> PIIAgentState:
        """Run graph analysis."""
        try:
            ner_results = state["ner_results"]
            proximity_results = state["proximity_results"]
            
            logger.info("Starting graph analysis...")
            
            # Build graph
            graph = self.graph_builder.build_graph(ner_results, proximity_results)
            
            # Analyze graph
            graph_analysis = self.graph_builder.analyze_graph()
            
            # Create visualization (with error handling)
            output_dir = state["output_dir"]
            viz_path = os.path.join(output_dir, "pii_graph_visualization.png")
            try:
                self.graph_builder.visualize_graph(viz_path)
            except Exception as viz_error:
                logger.warning(f"Graph visualization failed: {viz_error}")
                viz_path = None
            
            # Export graph data (with error handling)
            graph_data_path = os.path.join(output_dir, "pii_graph_data.json")
            try:
                self.graph_builder.export_graph_data(graph_data_path)
            except Exception as export_error:
                logger.warning(f"Graph data export failed: {export_error}")
                graph_data_path = None
            
            graph_results = {
                "analysis": graph_analysis,
                "visualization_path": viz_path,
                "graph_data_path": graph_data_path,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
            
            state["graph_results"] = graph_results
            state["messages"].append(AIMessage(
                content=f"Graph analysis completed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
            ))
            
            logger.info(f"Graph analysis completed: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Graph analysis error: {e}")
            # Don't fail the entire pipeline, just log the error and continue
            state["graph_results"] = {
                "analysis": {"error": str(e)},
                "visualization_path": None,
                "graph_data_path": None,
                "graph_stats": {"nodes": 0, "edges": 0}
            }
            state["messages"].append(AIMessage(
                content=f"Graph analysis encountered issues but pipeline continued"
            ))
        
        return state
    
    def _generate_report(self, state: PIIAgentState) -> PIIAgentState:
        """Generate final PII detection report."""
        try:
            ner_results = state["ner_results"]
            proximity_results = state["proximity_results"]
            graph_results = state["graph_results"]
            
            logger.info("Generating final report...")
            
            # Compile comprehensive report
            report = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "input_file": state["csv_path"],
                    "proximity_window": self.proximity_window
                },
                "ner_summary": self.ner_detector.get_detection_summary(ner_results),
                "proximity_summary": {
                    "total_analyses": len(proximity_results['proximity_analyses']),
                    "high_risk_rows": sum(1 for analysis in proximity_results['proximity_analyses'] 
                                        if analysis.get('overall_risk') == 'high'),
                    "medium_risk_rows": sum(1 for analysis in proximity_results['proximity_analyses'] 
                                          if analysis.get('overall_risk') == 'medium'),
                    "low_risk_rows": sum(1 for analysis in proximity_results['proximity_analyses'] 
                                        if analysis.get('overall_risk') == 'low')
                },
                "graph_summary": graph_results.get("analysis", {}),
                "detailed_results": {
                    "ner_detections": ner_results["detections"],
                    "proximity_analyses": proximity_results["proximity_analyses"]
                },
                "recommendations": self._generate_recommendations(ner_results, proximity_results, graph_results)
            }
            
            # Save report
            output_dir = state["output_dir"]
            report_path = os.path.join(output_dir, "pii_detection_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            state["final_report"] = report
            state["messages"].append(AIMessage(
                content=f"Report generated and saved to {report_path}"
            ))
            
            logger.info(f"Final report saved to {report_path}")
            
        except Exception as e:
            state["error"] = f"Report generation failed: {str(e)}"
            logger.error(f"Report generation error: {e}")
        
        return state
    
    def _mask_csv(self, state: PIIAgentState) -> PIIAgentState:
        """Create masked version of the CSV file."""
        try:
            csv_path = state["csv_path"]
            ner_results = state["ner_results"]
            output_dir = state["output_dir"]
            
            logger.info("Creating masked CSV...")
            
            # Read original CSV
            df = pd.read_csv(csv_path)
            
            # Group detections by row and column
            detections_by_location = {}
            for detection in ner_results["detections"]:
                row = detection["row"]
                col = detection["column"]
                key = (row, col)
                
                if key not in detections_by_location:
                    detections_by_location[key] = []
                detections_by_location[key].append(detection["entity"])
            
            # Mask detected PII
            masked_df = df.copy()
            for (row, col), entities in detections_by_location.items():
                if row < len(masked_df) and col in masked_df.columns:
                    original_text = str(masked_df.iloc[row][col])
                    masked_text = original_text
                    
                    # Sort entities by position (descending) to avoid index shifting
                    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
                    
                    for entity in sorted_entities:
                        start = entity['start']
                        end = entity['end']
                        masked_text = masked_text[:start] + MASK_CHAR + masked_text[end:]
                    
                    masked_df.iloc[row, masked_df.columns.get_loc(col)] = masked_text
            
            # Apply comprehensive masking using regex patterns
            masked_df = self._apply_comprehensive_masking(masked_df)
            
            # Save masked CSV
            masked_csv_path = os.path.join(output_dir, "masked_data.csv")
            masked_df.to_csv(masked_csv_path, index=False)
            
            state["masked_csv_path"] = masked_csv_path
            state["messages"].append(AIMessage(
                content=f"Masked CSV saved to {masked_csv_path}"
            ))
            
            logger.info(f"Masked CSV saved to {masked_csv_path}")
            
        except Exception as e:
            state["error"] = f"CSV masking failed: {str(e)}"
            logger.error(f"CSV masking error: {e}")
        
        return state
    
    def _apply_comprehensive_masking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive masking using regex patterns to catch missed PII."""
        import re
        
        masked_df = df.copy()
        
        # Define comprehensive regex patterns for PII
        patterns = {
            'ssn': re.compile(r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b|\b(?:[0-9]{4}[-.\s]?){3}[0-9]{4}\b'),
            'phone': re.compile(r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|(?:\+?[0-9]{1,3}[-.\s]?)?\(?([0-9]{2,4})\)?[-.\s]?([0-9]{2,4})[-.\s]?([0-9]{2,4})'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ip_address': re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            'date_of_birth': re.compile(r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            'name_pattern': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b')
        }
        
        # Apply masking to all string columns
        for col in masked_df.columns:
            if masked_df[col].dtype == 'object':  # String columns
                for idx, value in masked_df[col].items():
                    if pd.notna(value):
                        text = str(value)
                        masked_text = text
                        
                        # Apply each pattern
                        for pattern_name, pattern in patterns.items():
                            if pattern_name == 'name_pattern':
                                # For names, be more conservative - only mask if it looks like a full name
                                matches = pattern.findall(masked_text)
                                for match in matches:
                                    if len(match.split()) == 2:  # First Last format
                                        masked_text = masked_text.replace(match, MASK_CHAR)
                            else:
                                # For other patterns, replace with mask
                                masked_text = pattern.sub(MASK_CHAR, masked_text)
                        
                        masked_df.at[idx, col] = masked_text
        
        return masked_df
    
    def _handle_error(self, state: PIIAgentState) -> PIIAgentState:
        """Handle errors in the pipeline."""
        error_msg = state.get("error", "Unknown error occurred")
        logger.error(f"Pipeline error: {error_msg}")
        
        state["messages"].append(AIMessage(
            content=f"Error: {error_msg}"
        ))
        
        return state
    
    def _should_continue(self, state: PIIAgentState) -> str:
        """Determine if the pipeline should continue or handle error."""
        return "error" if state.get("error") else "continue"
    
    def _generate_recommendations(self, ner_results: Dict[str, Any], 
                                proximity_results: Dict[str, Any], 
                                graph_results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on analysis results."""
        recommendations = []
        
        # NER-based recommendations
        ner_summary = self.ner_detector.get_detection_summary(ner_results)
        if ner_summary['total_detections'] > 0:
            recommendations.append(
                f"Found {ner_summary['total_detections']} PII entities. "
                "Consider implementing data minimization practices."
            )
        
        # Proximity-based recommendations
        high_risk_rows = sum(1 for analysis in proximity_results['proximity_analyses'] 
                           if analysis['overall_risk'] == 'high')
        if high_risk_rows > 0:
            recommendations.append(
                f"Identified {high_risk_rows} high-risk rows with multiple PII types in proximity. "
                "These pose significant re-identification risks."
            )
        
        # Graph-based recommendations
        if "analysis" in graph_results and "reidentification_risk" in graph_results["analysis"]:
            reid_risk = graph_results["analysis"]["reidentification_risk"]
            if reid_risk.get("risk_level") == "high":
                recommendations.append(
                    "High re-identification risk detected due to entity clustering. "
                    "Consider data anonymization or pseudonymization techniques."
                )
        
        # General recommendations
        recommendations.extend([
            "Implement access controls and audit logging for PII data.",
            "Consider using differential privacy techniques for sensitive datasets.",
            "Regular PII audits should be conducted to maintain compliance."
        ])
        
        return recommendations
    
    def process_csv(self, csv_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process a CSV file through the complete PII detection pipeline.
        
        Args:
            csv_path: Path to the input CSV file
            output_dir: Directory to save outputs (default: same as input file)
            
        Returns:
            Dictionary containing all results and file paths
        """
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        
        # Initialize state
        initial_state = PIIAgentState(
            messages=[HumanMessage(content=f"Process CSV file: {csv_path}")],
            csv_path=csv_path,
            output_dir=output_dir,
            ner_results=None,
            proximity_results=None,
            graph_results=None,
            final_report=None,
            masked_csv_path=None,
            error=None
        )
        
        # Run the agent
        logger.info(f"Starting PII detection pipeline for {csv_path}")
        final_state = self.agent.invoke(initial_state)
        
        # Return results
        return {
            "success": final_state.get("error") is None,
            "error": final_state.get("error"),
            "report_path": os.path.join(output_dir, "pii_detection_report.json") if final_state.get("final_report") else None,
            "masked_csv_path": final_state.get("masked_csv_path"),
            "graph_visualization": final_state.get("graph_results", {}).get("visualization_path"),
            "graph_data": final_state.get("graph_results", {}).get("graph_data_path"),
            "messages": [msg.content for msg in final_state.get("messages", [])]
        }


def main():
    """Main function for testing the PII Detection Agent."""
    # Create sample data
    from ner_detector import create_sample_data
    sample_file = create_sample_data()
    
    # Initialize agent
    agent = PIIDetectionAgent()
    
    # Process the sample CSV
    results = agent.process_csv(sample_file)
    
    print("PII Detection Agent Results:")
    print(f"Success: {results['success']}")
    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print(f"Report: {results['report_path']}")
        print(f"Masked CSV: {results['masked_csv_path']}")
        print(f"Graph Visualization: {results['graph_visualization']}")
        print(f"Graph Data: {results['graph_data']}")
    
    print("\nMessages:")
    for msg in results['messages']:
        print(f"- {msg}")


if __name__ == "__main__":
    main()
