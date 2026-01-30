"""
VTT (WebVTT) Subtitle File Extractor
Processes WebVTT subtitle files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import re

try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class VTTExtractor(BaseExtractor):
    """
    Extractor for WebVTT subtitle files
    
    Features:
    - Extracts timestamps and text
    - Preserves speaker information
    - Handles cues and metadata
    - Formats as readable transcript
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['vtt', 'webvtt']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate VTT file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not a VTT file"
        
        # Check if file starts with WEBVTT
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('WEBVTT'):
                    return False, "Not a valid WebVTT file (missing WEBVTT header)"
        except Exception as e:
            return False, f"Could not read file: {e}"
        
        return True, "Valid VTT file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract VTT subtitle file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting VTT: {file_path}")
            
            # Read file
            content = self._read_vtt_file(file_path)
            
            if not content:
                return "", False, {'error': 'Empty VTT file'}
            
            vtt_stem = Path(file_path).stem
            
            # Parse VTT structure
            cues = self._parse_vtt_cues(content)
            
            if not cues:
                return "", False, {'error': 'No cues found in VTT file'}
            
            # Estimate duration and pages
            last_timestamp = cues[-1]['end_time'] if cues else "00:00:00.000"
            duration_seconds = self._timestamp_to_seconds(last_timestamp)
            estimated_pages = max(1, int(duration_seconds / 300))  # ~5 min per page
            
            # Build markdown
            markdown_content = self._build_markdown(
                vtt_stem, cues, duration_seconds, estimated_pages
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': estimated_pages,
                'total_cues': len(cues),
                'duration_seconds': duration_seconds,
                'duration_formatted': self._seconds_to_timestamp(duration_seconds),
                'extraction_method': 'vtt'
            })
            
            logger.info(f"✓ VTT extraction complete: {len(cues)} cues, {duration_seconds:.1f}s")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"VTT extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _read_vtt_file(self, file_path: str) -> str:
        """Read VTT file with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # Fallback
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _parse_vtt_cues(self, content: str) -> List[Dict]:
        """Parse VTT cues from content"""
        cues = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip WEBVTT header and empty lines
            if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                i += 1
                continue
            
            # Check if this is a timestamp line
            if '-->' in line:
                timestamp_match = re.match(
                    r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})',
                    line
                )
                
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    
                    # Collect text lines until next empty line or timestamp
                    text_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    if text_lines:
                        cues.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'text': ' '.join(text_lines)
                        })
                else:
                    i += 1
            else:
                i += 1
        
        return cues
    
    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert timestamp to seconds"""
        try:
            # Format: HH:MM:SS.mmm
            parts = timestamp.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0.0
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def _build_markdown(
        self,
        vtt_stem: str,
        cues: List[Dict],
        duration_seconds: float,
        total_pages: int
    ) -> str:
        """Build markdown from VTT cues"""
        lines = []
        
        # Header
        lines.append(f"# Transcript: {vtt_stem}")
        lines.append(f"**Format**: WebVTT Subtitles")
        lines.append(f"**Duration**: {self._seconds_to_timestamp(duration_seconds)}")
        lines.append(f"**Total Cues**: {len(cues)}")
        lines.append(f"**Estimated Pages**: {total_pages}\n")
        lines.append("---\n")
        
        # Divide cues into pages based on time
        seconds_per_page = duration_seconds / total_pages if total_pages > 0 else duration_seconds
        
        current_page = 1
        lines.append(f"## Page {current_page} (Transcript)\n")
        
        for i, cue in enumerate(cues):
            # Check if we need a page break
            cue_start_seconds = self._timestamp_to_seconds(cue['start_time'])
            expected_page = int(cue_start_seconds / seconds_per_page) + 1
            
            if expected_page > current_page and current_page < total_pages:
                current_page = expected_page
                lines.append(f"\n## Page {current_page} (Transcript)\n")
            
            # Add cue
            lines.append(f"**[{cue['start_time']} → {cue['end_time']}]**")
            lines.append(f"{cue['text']}\n")
        
        return '\n'.join(lines)
