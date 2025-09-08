"""
Proximity Analysis for PII detection using context-based inference.
"""
import re
import json
import logging
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProximityAnalyzer:
    """
    Analyzes proximity of entities to infer additional PII and assess risk levels.
    
    This class implements window-based text scanning to detect:
    - Non-obvious PII that can be inferred from context
    - Risk levels based on entity proximity and combinations
    - Potential re-identification risks
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize the Proximity Analyzer.
        
        Args:
            window_size: Size of the proximity window in characters
        """
        self.window_size = window_size
        self.risk_patterns = self._compile_risk_patterns()
        self.context_keywords = self._load_context_keywords()
    
    def _compile_risk_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for high-risk entity combinations."""
        return {
            'name_with_ssn': re.compile(
                r'([A-Z][a-z]+ [A-Z][a-z]+).*?(\d{3}[-.\s]?\d{2}[-.\s]?\d{4})',
                re.IGNORECASE | re.DOTALL
            ),
            'name_with_phone': re.compile(
                r'([A-Z][a-z]+ [A-Z][a-z]+).*?(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                re.IGNORECASE | re.DOTALL
            ),
            'address_with_name': re.compile(
                r'(\d+\s+[A-Za-z\s]+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive)).*?([A-Z][a-z]+ [A-Z][a-z]+)',
                re.IGNORECASE | re.DOTALL
            ),
            'zip_with_org': re.compile(
                r'(\d{5}(?:-\d{4})?).*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Corp|Corporation|Inc|LLC|Ltd|Company))',
                re.IGNORECASE | re.DOTALL
            )
        }
    
    def _load_context_keywords(self) -> Dict[str, List[str]]:
        """Load context keywords that indicate PII presence."""
        return {
            'personal_identifiers': [
                'ssn', 'social security', 'tax id', 'employee id', 'member id',
                'customer id', 'client id', 'user id', 'account number'
            ],
            'contact_info': [
                'phone', 'mobile', 'cell', 'home', 'work', 'office',
                'email', 'address', 'residence', 'mailing'
            ],
            'financial': [
                'credit card', 'debit card', 'bank account', 'routing number',
                'account balance', 'payment', 'billing'
            ],
            'medical': [
                'patient', 'medical record', 'diagnosis', 'treatment',
                'prescription', 'doctor', 'physician', 'hospital'
            ],
            'employment': [
                'salary', 'wage', 'income', 'employer', 'job title',
                'department', 'manager', 'supervisor'
            ]
        }
    
    def analyze_proximity(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze entity proximity and infer additional PII.
        
        Args:
            text: Input text to analyze
            entities: List of detected entities from NER
            
        Returns:
            Analysis results with inferred PII and risk levels
        """
        if not text or not entities:
            return {
                'inferred_entities': [],
                'risk_assessments': [],
                'context_analysis': {},
                'overall_risk': 'low'
            }
        
        # Find entity proximity relationships
        proximity_relations = self._find_proximity_relations(text, entities)
        
        # Infer additional PII from context
        inferred_entities = self._infer_entities_from_context(text, entities)
        
        # Assess risk levels
        risk_assessments = self._assess_risk_levels(text, entities, proximity_relations)
        
        # Analyze context for sensitive information
        context_analysis = self._analyze_context(text)
        
        # Calculate overall risk
        overall_risk = self._calculate_overall_risk(risk_assessments, context_analysis)
        
        return {
            'inferred_entities': inferred_entities,
            'risk_assessments': risk_assessments,
            'context_analysis': context_analysis,
            'proximity_relations': proximity_relations,
            'overall_risk': overall_risk
        }
    
    def _find_proximity_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find entities that appear within proximity of each other."""
        relations = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                distance = abs(entity1['start'] - entity2['start'])
                
                if distance <= self.window_size:
                    relations.append({
                        'entity1': entity1,
                        'entity2': entity2,
                        'distance': distance,
                        'text_between': text[min(entity1['end'], entity2['end']):max(entity1['start'], entity2['start'])]
                    })
        
        return relations
    
    def _infer_entities_from_context(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Infer additional PII entities from context and patterns."""
        inferred = []
        
        # Check for high-risk patterns
        for pattern_name, pattern in self.risk_patterns.items():
            for match in pattern.finditer(text):
                inferred.append({
                    'text': match.group(),
                    'pattern': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7,  # Lower confidence for inferred entities
                    'method': 'proximity_inference',
                    'type': self._map_pattern_to_type(pattern_name),
                    'risk_level': 'high'
                })
        
        # Check for context-based inferences
        for entity in entities:
            context_window = self._get_context_window(text, entity)
            context_indicators = self._find_context_indicators(context_window)
            
            if context_indicators:
                inferred.append({
                    'text': entity['text'],
                    'context_indicators': context_indicators,
                    'start': entity['start'],
                    'end': entity['end'],
                    'confidence': 0.6,
                    'method': 'context_inference',
                    'type': entity['type'],
                    'risk_level': 'medium'
                })
        
        return inferred
    
    def _get_context_window(self, text: str, entity: Dict[str, Any]) -> str:
        """Get context window around an entity."""
        start = max(0, entity['start'] - self.window_size // 2)
        end = min(len(text), entity['end'] + self.window_size // 2)
        return text[start:end]
    
    def _find_context_indicators(self, context: str) -> List[str]:
        """Find context indicators in the given text."""
        indicators = []
        context_lower = context.lower()
        
        for category, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in context_lower:
                    indicators.append(f"{category}: {keyword}")
        
        return indicators
    
    def _assess_risk_levels(self, text: str, entities: List[Dict[str, Any]], 
                           relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess risk levels for entity combinations."""
        risk_assessments = []
        
        # High-risk combinations
        high_risk_combinations = [
            ('name', 'ssn'), ('name', 'phone'), ('name', 'email'),
            ('address', 'name'), ('zip', 'org'), ('name', 'credit_card')
        ]
        
        for relation in relations:
            try:
                # Validate relation structure
                if not isinstance(relation, dict) or 'entity1' not in relation or 'entity2' not in relation:
                    continue
                
                entity1 = relation['entity1']
                entity2 = relation['entity2']
                
                # Validate entity structure
                if not isinstance(entity1, dict) or not isinstance(entity2, dict):
                    continue
                
                if 'type' not in entity1 or 'type' not in entity2:
                    continue
                
                entity1_type = entity1['type']
                entity2_type = entity2['type']
                
                # Check for high-risk combinations
                if (entity1_type, entity2_type) in high_risk_combinations or \
                   (entity2_type, entity1_type) in high_risk_combinations:
                    risk_level = 'high'
                elif relation.get('distance', 0) < self.window_size // 2:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                risk_assessments.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'distance': relation.get('distance', 0),
                    'risk_level': risk_level,
                    'reason': f"Proximity of {entity1_type} and {entity2_type}"
                })
            except Exception as e:
                logger.warning(f"Error assessing risk level for relation: {e}")
                continue
        
        return risk_assessments
    
    def _analyze_context(self, text: str) -> Dict[str, Any]:
        """Analyze the overall context for sensitive information indicators."""
        context_analysis = {
            'sensitive_categories': [],
            'risk_indicators': [],
            'context_score': 0
        }
        
        text_lower = text.lower()
        
        # Check for sensitive categories
        for category, keywords in self.context_keywords.items():
            category_count = sum(1 for keyword in keywords if keyword in text_lower)
            if category_count > 0:
                context_analysis['sensitive_categories'].append({
                    'category': category,
                    'indicator_count': category_count
                })
        
        # Calculate context score
        total_indicators = sum(cat['indicator_count'] for cat in context_analysis['sensitive_categories'])
        context_analysis['context_score'] = min(total_indicators / 10.0, 1.0)  # Normalize to 0-1
        
        return context_analysis
    
    def _calculate_overall_risk(self, risk_assessments: List[Dict[str, Any]], 
                               context_analysis: Dict[str, Any]) -> str:
        """Calculate overall risk level based on assessments and context."""
        if not risk_assessments:
            return 'low'
        
        high_risk_count = sum(1 for assessment in risk_assessments if assessment['risk_level'] == 'high')
        medium_risk_count = sum(1 for assessment in risk_assessments if assessment['risk_level'] == 'medium')
        
        # Consider context score
        context_score = context_analysis.get('context_score', 0)
        
        if high_risk_count > 0 or context_score > 0.7:
            return 'high'
        elif medium_risk_count > 2 or context_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _map_pattern_to_type(self, pattern_name: str) -> str:
        """Map pattern names to PII types."""
        mapping = {
            'name_with_ssn': 'name_ssn_combination',
            'name_with_phone': 'name_phone_combination',
            'address_with_name': 'address_name_combination',
            'zip_with_org': 'zip_org_combination'
        }
        return mapping.get(pattern_name, 'inferred_combination')
    
    def process_csv_proximity(self, csv_path: str, ner_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process CSV file with proximity analysis.
        
        Args:
            csv_path: Path to the CSV file
            ner_results: Results from NER detector
            
        Returns:
            Proximity analysis results
        """
        try:
            df = pd.read_csv(csv_path)
            
            proximity_results = {
                'file_path': csv_path,
                'total_rows': len(df),
                'proximity_analyses': []
            }
            
            # Group detections by row
            detections_by_row = defaultdict(list)
            for detection in ner_results['detections']:
                detections_by_row[detection['row']].append(detection)
            
            # Analyze each row
            for row_idx, row in df.iterrows():
                if row_idx in detections_by_row:
                    # Get all text from the row
                    row_text = ' '.join(str(row[col]) for col in df.columns if pd.notna(row[col]))
                    
                    # Get entities for this row
                    row_entities = [det['entity'] for det in detections_by_row[row_idx]]
                    
                    # Perform proximity analysis
                    analysis = self.analyze_proximity(row_text, row_entities)
                    analysis['row'] = row_idx
                    
                    proximity_results['proximity_analyses'].append(analysis)
            
            return proximity_results
            
        except Exception as e:
            logger.error(f"Error in proximity analysis for {csv_path}: {e}")
            raise


if __name__ == "__main__":
    # Test the proximity analyzer
    analyzer = ProximityAnalyzer(window_size=50)
    
    # Sample text with entities
    sample_text = "John Doe's email is john.doe@email.com and his phone is 555-123-4567. His SSN is 123-45-6789 and he lives at 123 Main St."
    
    # Mock entities (would normally come from NER detector)
    mock_entities = [
        {'text': 'John Doe', 'type': 'name', 'start': 0, 'end': 8, 'confidence': 0.9},
        {'text': 'john.doe@email.com', 'type': 'email', 'start': 20, 'end': 38, 'confidence': 0.8},
        {'text': '555-123-4567', 'type': 'phone', 'start': 50, 'end': 62, 'confidence': 0.8},
        {'text': '123-45-6789', 'type': 'ssn', 'start': 75, 'end': 86, 'confidence': 0.8},
        {'text': '123 Main St', 'type': 'address', 'start': 100, 'end': 111, 'confidence': 0.7}
    ]
    
    # Perform proximity analysis
    results = analyzer.analyze_proximity(sample_text, mock_entities)
    
    print("Proximity Analysis Results:")
    print(f"Overall Risk: {results['overall_risk']}")
    print(f"Inferred Entities: {len(results['inferred_entities'])}")
    print(f"Risk Assessments: {len(results['risk_assessments'])}")
    print(f"Context Score: {results['context_analysis']['context_score']}")
    
    print("\nDetailed Results:")
    print(json.dumps(results, indent=2, default=str))
