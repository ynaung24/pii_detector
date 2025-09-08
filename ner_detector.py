"""
PII NER Detector using spaCy for entity extraction.
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import spacy
from spacy.tokens import Doc
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIINERDetector:
    """
    PII detection using spaCy NER and custom regex patterns.
    
    Detects various PII entities including:
    - Names (PERSON)
    - Email addresses
    - Phone numbers
    - SSNs
    - Addresses
    - Credit card numbers
    - IP addresses
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the PII NER Detector.
        
        Args:
            model_name: spaCy model name to use for NER
        """
        self.model_name = model_name
        self.nlp = None
        self._load_model()
        self._compile_regex_patterns()
    
    def _load_model(self) -> None:
        """Load the spaCy model with error handling."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Successfully loaded spaCy model: {self.model_name}")
        except OSError as e:
            logger.error(f"Failed to load spaCy model {self.model_name}: {e}")
            logger.info("Please install the model using: python -m spacy download en_core_web_sm")
            raise
    
    def _compile_regex_patterns(self) -> None:
        """Compile regex patterns for additional PII detection."""
        self.patterns = {
            'email': re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            'phone': re.compile(
                r'(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})|(?:\+?[0-9]{1,3}[-.\s]?)?\(?([0-9]{2,4})\)?[-.\s]?([0-9]{2,4})[-.\s]?([0-9]{2,4})',
                re.IGNORECASE
            ),
            'ssn': re.compile(
                r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b'
            ),
            'credit_card': re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'
            ),
            'ip_address': re.compile(
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ),
            'zip_code': re.compile(
                r'\b\d{5}(?:-\d{4})?\b'
            ),
            'date_of_birth': re.compile(
                r'\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12][0-9]|3[01])[-/](?:19|20)\d{2}\b'
            ),
            'name': re.compile(
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            )
        }
    
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII entities in text using spaCy NER and regex patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities with metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        entities = []
        
        # Use spaCy NER for named entities
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.9,  # spaCy doesn't provide confidence scores
                    'method': 'spacy_ner',
                    'type': self._map_spacy_label(ent.label_)
                })
        
        # Use regex patterns for structured data
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'text': match.group(),
                    'label': pattern_name.upper(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,  # Regex patterns have high confidence
                    'method': 'regex',
                    'type': pattern_name
                })
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x['start'])
        
        return entities
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy labels to PII types."""
        mapping = {
            'PERSON': 'name',
            'ORG': 'organization',
            'GPE': 'location',
            'LOC': 'location'
        }
        return mapping.get(label, label.lower())
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on position and text."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity['start'], entity['end'], entity['text'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def process_csv(self, csv_path: str, text_columns: List[str] = None) -> Dict[str, Any]:
        """
        Process a CSV file and detect PII in specified columns.
        
        Args:
            csv_path: Path to the CSV file
            text_columns: List of column names to analyze (None for all string columns)
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Determine columns to analyze
            if text_columns is None:
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            results = {
                'file_path': csv_path,
                'total_rows': len(df),
                'analyzed_columns': text_columns,
                'detections': []
            }
            
            # Process each row and column
            for idx, row in df.iterrows():
                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col])
                        entities = self.detect_entities(text)
                        
                        for entity in entities:
                            results['detections'].append({
                                'row': idx,
                                'column': col,
                                'entity': entity
                            })
            
            logger.info(f"Detected {len(results['detections'])} PII entities")
            return results
            
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_path}: {e}")
            raise
    
    def get_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of detection results.
        
        Args:
            results: Results from process_csv method
            
        Returns:
            Summary statistics
        """
        summary = {
            'total_detections': len(results['detections']),
            'detection_types': {},
            'columns_affected': set(),
            'rows_affected': set()
        }
        
        for detection in results['detections']:
            entity_type = detection['entity']['type']
            summary['detection_types'][entity_type] = summary['detection_types'].get(entity_type, 0) + 1
            summary['columns_affected'].add(detection['column'])
            summary['rows_affected'].add(detection['row'])
        
        summary['columns_affected'] = list(summary['columns_affected'])
        summary['rows_affected'] = list(summary['rows_affected'])
        
        return summary


def create_sample_data() -> str:
    """Create sample CSV data for testing."""
    sample_data = """name,email,phone,address,notes
John Doe,john.doe@email.com,555-123-4567,123 Main St New York NY 10001,Works at ABC Corp
Jane Smith,jane.smith@company.org,(555) 987-6543,456 Oak Ave Los Angeles CA 90210,SSN: 123-45-6789
Bob Johnson,bob@test.net,555.111.2222,789 Pine Rd Chicago IL 60601,Credit card: 4532-1234-5678-9012
Alice Brown,alice.brown@example.com,555 333 4444,321 Elm St Houston TX 77001,IP: 192.168.1.1"""
    
    sample_file = "/Users/ynaung/MSDS/DE/llm_security/hw4/sample_data.csv"
    with open(sample_file, 'w') as f:
        f.write(sample_data)
    
    return sample_file


if __name__ == "__main__":
    # Test the NER detector
    detector = PIINERDetector()
    
    # Test with sample text
    sample_text = "John Doe's email is john.doe@email.com and his phone is 555-123-4567. His SSN is 123-45-6789."
    entities = detector.detect_entities(sample_text)
    
    print("Sample text analysis:")
    print(f"Text: {sample_text}")
    print(f"Detected entities: {json.dumps(entities, indent=2)}")
    
    # Test with sample CSV
    sample_file = create_sample_data()
    results = detector.process_csv(sample_file)
    summary = detector.get_detection_summary(results)
    
    print(f"\nCSV analysis summary:")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Detection types: {summary['detection_types']}")
    print(f"Columns affected: {summary['columns_affected']}")
