"""
Enhanced Medicine Extraction Module for Prescription OCR
This module provides improved algorithms for detecting and extracting medicine information
from prescription text with better accuracy and handling of various formats.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MedicineInfo:
    """Data class for storing extracted medicine information"""
    name: str
    dosage: str
    frequency: str
    duration: str
    method: str = "oral"  # Default method
    confidence_score: float = 0.0

class EnhancedMedicineExtractor:
    """Enhanced medicine extractor with improved patterns and preprocessing"""
    
    def __init__(self):
        self.medicine_prefixes = [
            # Tablets
            r'(?:Tab|TAB|tab|Tablet|tablet|TABLET)',
            # Capsules  
            r'(?:Cap|CAP|cap|Capsule|capsule|CAPSULE)',
            # Syrups
            r'(?:Syrup|SYRUP|syrup|Syr|SYR|syr|Syp|SYP|syp)',
            # Injections
            r'(?:Inj|INJ|inj|Injection|injection|INJECTION)',
            # Drops
            r'(?:Drop|DROP|drop|Drops|DROPS|drops)',
            # Ointments
            r'(?:Oint|OINT|oint|Ointment|ointment|OINTMENT)',
            # Powders
            r'(?:Powder|POWDER|powder|Pwd|PWD|pwd)',
            # Solutions
            r'(?:Sol|SOL|sol|Solution|solution|SOLUTION)',
            # Suspensions
            r'(?:Susp|SUSP|susp|Suspension|suspension|SUSPENSION)',
        ]
        
        # Comprehensive dosage patterns
        self.dosage_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:mg|MG|Mg)',  # milligrams
            r'(\d+(?:\.\d+)?)\s*(?:g|G|gm|GM|gram|Gram)',  # grams
            r'(\d+(?:\.\d+)?)\s*(?:ml|ML|Ml)',  # milliliters
            r'(\d+(?:\.\d+)?)\s*(?:mcg|MCG|Mcg|μg)',  # micrograms
            r'(\d+(?:\.\d+)?)\s*(?:iu|IU|Iu|units?|Units?)',  # international units
            r'(\d+(?:\.\d+)?)\s*(?:%|percent)',  # percentage
            r'(\d+(?:\.\d+)?)\s*(?:mg/ml|MG/ML)',  # concentration
        ]
        
        # Enhanced frequency patterns
        self.frequency_patterns = [
            # Standard patterns
            r'(\d+)\s*(?:times?\s*(?:a\s*|per\s*)?day|/day|daily)',
            r'(\d+)\s*(?:times?\s*(?:a\s*|per\s*)?week|/week|weekly)',
            r'(\d+)\s*(?:x|X)\s*(?:daily|day|/day)',
            r'(\d+)\s*(?:tid|TID|bd|BD|od|OD|qid|QID)',
            
            # Text-based patterns
            r'(?:once\s*(?:a\s*|per\s*)?day|once\s*daily)',
            r'(?:twice\s*(?:a\s*|per\s*)?day|twice\s*daily)',
            r'(?:thrice\s*(?:a\s*|per\s*)?day|three\s*times\s*(?:a\s*)?day)',
            r'(?:four\s*times\s*(?:a\s*)?day)',
            
            # Medical abbreviations
            r'(?:BID|bid|Bid)',  # twice a day
            r'(?:TID|tid|Tid)',  # three times a day
            r'(?:QID|qid|Qid)',  # four times a day
            r'(?:OD|od|Od)',     # once a day
            r'(?:BD|bd|Bd)',     # twice a day
        ]
        
        # Duration patterns
        self.duration_patterns = [
            r'(?:for\s*)?(\d+)\s*(?:days?|day)',
            r'(?:for\s*)?(\d+)\s*(?:weeks?|week|wks?)',
            r'(?:for\s*)?(\d+)\s*(?:months?|month|mos?)',
            r'(?:continue\s*for\s*)?(\d+)\s*(?:days?|day)',
            r'(\d+)\s*(?:day|days)\s*course',
        ]
        
        # Load common medicine names database
        self.medicine_database = self._load_medicine_database()
    
    def _load_medicine_database(self) -> set:
        """Load comprehensive medicine names from database file"""
        try:
            database_path = os.path.join(os.path.dirname(__file__), 'medicine_database.json')
            if os.path.exists(database_path):
                with open(database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Combine all medicine categories
                all_medicines = set()
                
                # Add medicines from all categories
                for category in data.get('common_medicines', {}).values():
                    all_medicines.update(medicine.lower() for medicine in category)
                
                # Add Indian brand names
                brands = data.get('indian_brands', {}).get('popular_brands', [])
                all_medicines.update(brand.lower() for brand in brands)
                
                # Add generic equivalents
                generics = data.get('indian_brands', {}).get('generic_equivalents', {})
                for brand, generic in generics.items():
                    all_medicines.add(brand.lower())
                    all_medicines.add(generic.lower())
                
                print(f"Loaded {len(all_medicines)} medicines from database")
                return all_medicines
                
        except Exception as e:
            print(f"Error loading medicine database: {e}")
        
        # Fallback to basic medicine set if file loading fails
        basic_medicines = {
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'paracetamol', 'ibuprofen', 
            'diclofenac', 'omeprazole', 'pantoprazole', 'cetrizine', 'loratadine',
            'crocin', 'combiflam', 'dolo', 'disprin', 'voveran'
        }
        
        return basic_medicines
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing to clean OCR errors and improve recognition"""
        if not text:
            return ""
        
        # Preserve line breaks first
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Convert to lowercase for processing
            processed_line = line.lower()
            
            # Fix common OCR errors
            ocr_fixes = {
                # Common character substitutions
                '0': 'o', '5': 's', '8': 'b', '6': 'g',
                'rn': 'm', 'cl': 'd', 'ri': 'n', 'ii': 'll',
                
                # Common word fixes
                'tabl': 'tab', 'capsul': 'cap', 'syru': 'syrup',
                'injecti': 'inj', 'medicin': 'medicine',
                'paracetamo': 'paracetamol', 'ibuprofn': 'ibuprofen',
                'amoxicilli': 'amoxicillin', 'azithromyci': 'azithromycin',
            }
            
            # Apply OCR fixes carefully to avoid breaking numbers
            for wrong, correct in ocr_fixes.items():
                # Only replace if it's a complete word (not part of dosage numbers)
                if len(wrong) > 1:  # Only apply to multi-character fixes
                    processed_line = re.sub(rf'\b{wrong}\b', correct, processed_line)
            
            # Clean up whitespace but preserve structure
            processed_line = re.sub(r'\s+', ' ', processed_line)  # Multiple spaces to single
            processed_line = re.sub(r'[^\w\s\./\-]', ' ', processed_line)  # Remove special chars except . / -
            
            processed_lines.append(processed_line.strip())
        
        # Rejoin with newlines preserved
        return '\n'.join(processed_lines)
    
    def extract_medicines(self, text: str) -> List[MedicineInfo]:
        """Extract medicine information with enhanced accuracy"""
        if not text:
            return []
        
        # Preprocess the text
        cleaned_text = self.preprocess_text(text)
        
        medicines = []
        lines = cleaned_text.split('\n')
        
        # Enhanced processing - look for numbered lists and medicine patterns
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check if this line starts a medicine entry (numbered or with prefix)
            if self._is_medicine_line(line):
                # Extract medicine from this line and possibly next lines
                medicine_info, lines_consumed = self._extract_medicine_block(lines, i)
                if medicine_info:
                    medicines.append(medicine_info)
                i += lines_consumed
            else:
                # Try to extract medicine from current line anyway
                medicine_info = self._extract_from_line(line, lines, i)
                if medicine_info:
                    medicines.append(medicine_info)
                i += 1
        
        # Post-process to remove duplicates and improve accuracy
        return self._post_process_medicines(medicines)
    
    def _is_medicine_line(self, line: str) -> bool:
        """Check if a line likely contains a medicine entry"""
        # Check for numbered list patterns
        if re.match(r'^\d+\.?\s*', line):
            return True
        
        # Check for medicine prefixes
        prefix_pattern = '|'.join(self.medicine_prefixes)
        if re.search(prefix_pattern, line, re.IGNORECASE):
            return True
        
        # Check for medicine names from database
        words = line.lower().split()
        for word in words:
            if word in self.medicine_database:
                return True
        
        return False
    
    def _extract_medicine_block(self, lines: List[str], start_index: int) -> Tuple[Optional[MedicineInfo], int]:
        """Extract medicine information from a block of lines starting at start_index"""
        if start_index >= len(lines):
            return None, 1
        
        # Combine multiple lines for this medicine entry
        medicine_lines = [lines[start_index]]
        lines_consumed = 1
        
        # Look ahead for continuation lines (dosage, frequency, duration info)
        for j in range(start_index + 1, min(start_index + 4, len(lines))):  # Look up to 3 lines ahead
            next_line = lines[j].strip()
            if not next_line:
                break
            
            # Stop if we hit another medicine entry
            if self._is_medicine_line(next_line):
                break
            
            # Include lines that contain dosage, frequency, or duration info
            if self._contains_medicine_info(next_line):
                medicine_lines.append(next_line)
                lines_consumed += 1
            else:
                break
        
        # Extract from combined text
        combined_text = ' '.join(medicine_lines)
        medicine_info = self._extract_from_combined_text(combined_text)
        
        return medicine_info, lines_consumed
    
    def _contains_medicine_info(self, line: str) -> bool:
        """Check if line contains dosage, frequency, or duration information"""
        line_lower = line.lower()
        
        # Check for dosage patterns
        for pattern in self.dosage_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        # Check for frequency patterns
        for pattern in self.frequency_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        # Check for duration patterns
        for pattern in self.duration_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        # Check for instruction words
        instruction_words = ['take', 'tablet', 'capsule', 'times', 'daily', 'day', 'week', 'mg', 'ml']
        return any(word in line_lower for word in instruction_words)
    
    def _extract_from_combined_text(self, text: str) -> Optional[MedicineInfo]:
        """Extract medicine information from combined text"""
        if not text:
            return None
        
        # Extract medicine name
        medicine_name = self._extract_medicine_name(text)
        if not medicine_name:
            return None
        
        # Extract other information
        dosage = self._extract_dosage_from_text(text)
        frequency = self._extract_frequency_from_text(text)
        duration = self._extract_duration_from_text(text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(medicine_name, dosage, frequency, duration)
        
        return MedicineInfo(
            name=medicine_name,
            dosage=dosage or "1 unit",
            frequency=frequency or "1x daily",
            duration=duration or "7 days",
            confidence_score=confidence
        )
    
    def _extract_medicine_name(self, text: str) -> str:
        """Extract medicine name from text"""
        # Try with prefixes first
        prefix_pattern = '|'.join(self.medicine_prefixes)
        medicine_pattern = rf'({prefix_pattern})\s+([a-zA-Z0-9\s\-]+?)(?:\s+\d+|$|\s+for|\s+take|\s+twice|\s+once|\s+thrice)'
        
        match = re.search(medicine_pattern, text, re.IGNORECASE)
        if match and len(match.groups()) >= 2:
            prefix = match.group(1)
            name_part = match.group(2).strip()
            return f"{prefix} {name_part}".strip()
        
        # Try without prefixes using medicine database
        words = text.lower().split()
        for i, word in enumerate(words):
            if word in self.medicine_database:
                # Try to get more context around the medicine name
                start_idx = max(0, i-1)
                end_idx = min(len(words), i+3)
                potential_name = ' '.join(words[start_idx:end_idx])
                
                # Clean up the potential name
                potential_name = re.sub(r'\d+.*$', '', potential_name).strip()
                if potential_name:
                    return potential_name
        
        return ""
    
    def _extract_dosage_from_text(self, text: str) -> str:
        """Extract dosage from text"""
        for pattern in self.dosage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""
    
    def _extract_frequency_from_text(self, text: str) -> str:
        """Extract frequency from text"""
        for pattern in self.frequency_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if match.groups() and match.group(1).isdigit():
                    return f"{match.group(1)}x daily"
                else:
                    return self._normalize_frequency(match.group(0))
        return ""
    
    def _extract_duration_from_text(self, text: str) -> str:
        """Extract duration from text"""
        for pattern in self.duration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""
    
    def _extract_from_line(self, line: str, all_lines: List[str], line_index: int) -> Optional[MedicineInfo]:
        """Extract medicine information from a single line with context"""
        
        # Create combined pattern for medicine prefixes
        prefix_pattern = '|'.join(self.medicine_prefixes)
        
        # Look for medicine name patterns
        medicine_pattern = rf'({prefix_pattern})\s+([a-zA-Z0-9\s\-]+?)(?:\s|$)'
        medicine_match = re.search(medicine_pattern, line, re.IGNORECASE)
        
        if not medicine_match:
            # Try to find medicine names without prefixes using database
            words = line.split()
            for word in words:
                if word.lower() in self.medicine_database:
                    medicine_match = re.search(rf'\b({re.escape(word)})\b', line, re.IGNORECASE)
                    if medicine_match:
                        break
        
        if not medicine_match:
            return None
        
        # Extract medicine name
        if len(medicine_match.groups()) >= 2:
            prefix = medicine_match.group(1)
            name_part = medicine_match.group(2).strip()
            full_name = f"{prefix} {name_part}".strip()
        else:
            full_name = medicine_match.group(1).strip()
        
        # Extract dosage from current line and nearby lines
        dosage = self._extract_dosage(line, all_lines, line_index)
        
        # Extract frequency
        frequency = self._extract_frequency(line, all_lines, line_index)
        
        # Extract duration
        duration = self._extract_duration(line, all_lines, line_index)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(full_name, dosage, frequency, duration)
        
        return MedicineInfo(
            name=full_name,
            dosage=dosage or "1 unit",
            frequency=frequency or "1x daily",
            duration=duration or "7 days",
            confidence_score=confidence
        )
    
    def _extract_dosage(self, line: str, all_lines: List[str], line_index: int) -> str:
        """Extract dosage information from line and context"""
        # Check current line first
        for pattern in self.dosage_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # Check next line if available
        if line_index + 1 < len(all_lines):
            next_line = all_lines[line_index + 1]
            for pattern in self.dosage_patterns:
                match = re.search(pattern, next_line, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        # Look for number patterns as fallback
        number_pattern = r'(\d+(?:\.\d+)?)'
        match = re.search(number_pattern, line)
        if match:
            return f"{match.group(1)}mg"  # Default to mg
        
        return ""
    
    def _extract_frequency(self, line: str, all_lines: List[str], line_index: int) -> str:
        """Extract frequency information"""
        # Check current line and nearby lines
        search_lines = [line]
        if line_index + 1 < len(all_lines):
            search_lines.append(all_lines[line_index + 1])
        if line_index > 0:
            search_lines.append(all_lines[line_index - 1])
        
        for search_line in search_lines:
            for pattern in self.frequency_patterns:
                match = re.search(pattern, search_line, re.IGNORECASE)
                if match:
                    if match.groups():
                        if match.group(1).isdigit():
                            return f"{match.group(1)}x daily"
                        else:
                            return match.group(0)
                    else:
                        return self._normalize_frequency(match.group(0))
        
        return ""
    
    def _extract_duration(self, line: str, all_lines: List[str], line_index: int) -> str:
        """Extract duration information"""
        # Check current line and nearby lines
        search_lines = [line]
        if line_index + 1 < len(all_lines):
            search_lines.append(all_lines[line_index + 1])
        if line_index > 0:
            search_lines.append(all_lines[line_index - 1])
        
        for search_line in search_lines:
            for pattern in self.duration_patterns:
                match = re.search(pattern, search_line, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        return ""
    
    def _normalize_frequency(self, frequency_text: str) -> str:
        """Normalize frequency text to standard format"""
        freq_lower = frequency_text.lower()
        
        if 'once' in freq_lower or 'od' in freq_lower:
            return "1x daily"
        elif 'twice' in freq_lower or 'bid' in freq_lower or 'bd' in freq_lower:
            return "2x daily"
        elif 'thrice' in freq_lower or 'tid' in freq_lower or 'three' in freq_lower:
            return "3x daily"
        elif 'four' in freq_lower or 'qid' in freq_lower:
            return "4x daily"
        
        return frequency_text
    
    def _calculate_confidence(self, name: str, dosage: str, frequency: str, duration: str) -> float:
        """Calculate confidence score for extracted medicine information"""
        score = 0.0
        
        # Name confidence
        if name and len(name) > 2:
            score += 0.4
            # Bonus if name is in database
            if any(med in name.lower() for med in self.medicine_database):
                score += 0.2
        
        # Dosage confidence
        if dosage and any(unit in dosage.lower() for unit in ['mg', 'ml', 'g', 'mcg']):
            score += 0.2
        
        # Frequency confidence
        if frequency and ('x' in frequency or 'daily' in frequency or any(f in frequency.lower() for f in ['bid', 'tid', 'od'])):
            score += 0.1
        
        # Duration confidence
        if duration and any(unit in duration.lower() for unit in ['day', 'week', 'month']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _post_process_medicines(self, medicines: List[MedicineInfo]) -> List[MedicineInfo]:
        """Post-process extracted medicines to remove duplicates and improve quality"""
        if not medicines:
            return []
        
        # Remove duplicates based on medicine name similarity
        unique_medicines = []
        for medicine in medicines:
            is_duplicate = False
            for existing in unique_medicines:
                if self._are_similar_medicines(medicine.name, existing.name):
                    # Keep the one with higher confidence
                    if medicine.confidence_score > existing.confidence_score:
                        unique_medicines.remove(existing)
                        unique_medicines.append(medicine)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_medicines.append(medicine)
        
        # Sort by confidence score
        unique_medicines.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return unique_medicines
    
    def _are_similar_medicines(self, name1: str, name2: str) -> bool:
        """Check if two medicine names are similar (likely the same medicine)"""
        # Simple similarity check - could be enhanced with fuzzy matching
        name1_clean = re.sub(r'[^\w]', '', name1.lower())
        name2_clean = re.sub(r'[^\w]', '', name2.lower())
        
        # Check if one is substring of another
        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True
        
        # Check if they share significant common words
        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())
        
        if words1 and words2:
            common_ratio = len(words1.intersection(words2)) / len(words1.union(words2))
            return common_ratio > 0.6
        
        return False

# Legacy function wrapper for backward compatibility
def extract_medicine_data(text: str) -> List[Dict]:
    """Legacy wrapper function to maintain compatibility with existing code"""
    extractor = EnhancedMedicineExtractor()
    medicine_infos = extractor.extract_medicines(text)
    
    # Convert to legacy format
    return [
        {
            "name": medicine.name,
            "dosage": medicine.dosage,
            "frequency": medicine.frequency,
            "duration": medicine.duration
        }
        for medicine in medicine_infos
    ]