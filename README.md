# PII Detection Agent

A comprehensive PII (Personally Identifiable Information) detection system that uses Named Entity Recognition (NER), Proximity Analysis, and Graph Theory to identify and assess privacy risks in structured data.

## 📚 Documentation

For complete documentation, setup instructions, and troubleshooting, see **[DOCUMENTATION.md](DOCUMENTATION.md)**.

## Features

- **NER-based Detection**: Uses spaCy for extracting PII entities (names, emails, phones, SSNs, addresses, etc.)
- **Proximity Analysis**: Identifies non-obvious PII through context-based inference
- **Graph Theory**: Builds entity relationship graphs to detect clusters and re-identification risks
- **OpenAI Integration**: Uses OpenAI's GPT models for enhanced analysis
- **Comprehensive Reporting**: Generates detailed JSON reports with risk assessments
- **Complete PII Masking**: Creates masked versions of CSV files with no data leakage
- **Web Interface**: Modern TypeScript frontend with drag-and-drop CSV upload
- **Production Ready**: Comprehensive error handling, logging, and security measures

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd /Users/ynaung/MSDS/DE/llm_security/hw4
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

4. Set up environment variables (optional):
```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Usage

```python
from pii_agent import PIIDetectionAgent

# Initialize the agent
agent = PIIDetectionAgent()

# Process a CSV file
results = agent.process_csv("your_data.csv")

# Check results
if results['success']:
    print(f"Report saved to: {results['report_path']}")
    print(f"Masked CSV saved to: {results['masked_csv_path']}")
    print(f"Graph visualization: {results['graph_visualization']}")
else:
    print(f"Error: {results['error']}")
```

### Advanced Usage

```python
from pii_agent import PIIDetectionAgent
from ner_detector import PIINERDetector
from proximity_analyzer import ProximityAnalyzer
from graph_builder import PIIGraphBuilder

# Initialize components individually
ner_detector = PIINERDetector()
proximity_analyzer = ProximityAnalyzer(window_size=100)
graph_builder = PIIGraphBuilder(proximity_window=100)

# Process CSV with NER
ner_results = ner_detector.process_csv("data.csv")

# Run proximity analysis
proximity_results = proximity_analyzer.process_csv_proximity("data.csv", ner_results)

# Build and analyze graph
graph = graph_builder.build_graph(ner_results, proximity_results)
analysis = graph_builder.analyze_graph()

# Visualize results
graph_builder.visualize_graph("graph.png")
```

### Command Line Usage

```python
# Run the main script with sample data
python pii_agent.py

# Run unit tests
python test_pii_detection.py
```

## Input/Output

### Input
- **CSV files** with arbitrary columns containing structured/unstructured text
- Configurable proximity window size
- Optional OpenAI API key for enhanced analysis

### Output
- **JSON Report** (`pii_detection_report.json`): Comprehensive analysis results
- **Masked CSV** (`masked_data.csv`): Original data with PII replaced by `***`
- **Graph Visualization** (`pii_graph_visualization.png`): Entity relationship graph
- **Graph Data** (`pii_graph_data.json`): Raw graph data for further analysis

## Report Structure

The generated JSON report contains:

```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "input_file": "data.csv",
    "proximity_window": 50
  },
  "ner_summary": {
    "total_detections": 25,
    "detection_types": {
      "name": 5,
      "email": 5,
      "phone": 5,
      "ssn": 3,
      "address": 7
    },
    "columns_affected": ["name", "email", "phone"],
    "rows_affected": [0, 1, 2, 3, 4]
  },
  "proximity_summary": {
    "total_analyses": 10,
    "high_risk_rows": 2,
    "medium_risk_rows": 3,
    "low_risk_rows": 5
  },
  "graph_summary": {
    "graph_stats": {
      "nodes": 25,
      "edges": 15,
      "density": 0.05,
      "average_clustering": 0.3
    },
    "connected_components": [...],
    "risk_clusters": [...],
    "reidentification_risk": {
      "risk_level": "medium",
      "score": 0.6
    }
  },
  "recommendations": [
    "Found 25 PII entities. Consider implementing data minimization practices.",
    "Identified 2 high-risk rows with multiple PII types in proximity.",
    "Implement access controls and audit logging for PII data."
  ]
}
```

## Architecture

The system consists of four main components:

### 1. NER Detector (`ner_detector.py`)
- Uses spaCy for named entity recognition
- Implements regex patterns for structured data (SSN, credit cards, etc.)
- Returns structured JSON with entity metadata

### 2. Proximity Analyzer (`proximity_analyzer.py`)
- Analyzes entity proximity within configurable windows
- Identifies high-risk entity combinations
- Infers additional PII from context

### 3. Graph Builder (`graph_builder.py`)
- Builds entity co-occurrence graphs using NetworkX
- Detects clusters and connected components
- Calculates centrality measures and risk scores
- Generates visualizations

### 4. PII Agent (`pii_agent.py`)
- Orchestrates the entire pipeline using LangGraph
- Handles error management and logging
- Generates comprehensive reports
- Creates masked CSV files

## Configuration

Key configuration options in `config.py`:

```python
DEFAULT_PROXIMITY_WINDOW = 50  # characters
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
MASK_CHAR = "***"
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 1000  # rows per chunk for large files
```

## Testing

Run the comprehensive test suite:

```bash
python test_pii_detection.py
```

The test suite includes:
- Unit tests for each component
- Integration tests for the complete pipeline
- Performance tests with larger datasets
- Error handling tests

## Security Considerations

### Implemented Security Measures:
- Input validation and sanitization
- File size limits to prevent DoS attacks
- Safe file handling with proper cleanup
- Error handling without information leakage

### Recommendations:
- Run in isolated environments for sensitive data
- Implement proper access controls
- Use encrypted storage for reports
- Regular security audits

## Performance

The system is optimized for:
- **Memory efficiency**: Processes large files in chunks
- **Speed**: Uses efficient algorithms and data structures
- **Scalability**: Handles datasets with thousands of rows

Performance benchmarks (on sample hardware):
- 100 rows: ~2-5 seconds
- 1000 rows: ~20-50 seconds
- 10000 rows: ~3-10 minutes

## Limitations

- Requires spaCy English model for NER
- Limited to English text processing
- Graph visualization may be slow for very large datasets
- LLM features require OpenAI API key

## Contributing

1. Follow PEP8 coding standards
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure security best practices

## License

This project is for educational and research purposes. Please ensure compliance with data protection regulations when processing real PII data.

## Support

For issues or questions:
1. Check the test suite for usage examples
2. Review the configuration options
3. Ensure all dependencies are properly installed
4. Verify input data format and size limits
