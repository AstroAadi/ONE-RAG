"""
EML (Email) File Extractor
Processes email files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import email
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import base64


try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class EMLExtractor(BaseExtractor):
    """
    Extractor for EML email files
    
    Features:
    - Extracts headers (From, To, Subject, Date)
    - Extracts body (plain text and HTML)
    - Lists attachments
    - Preserves email thread structure
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['eml', 'email']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate EML file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not an EML file"
        
        # Try to parse as email
        try:
            with open(file_path, 'rb') as f:
                email.message_from_binary_file(f)
            return True, "Valid EML file"
        except Exception as e:
            return False, f"Could not parse as email: {e}"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract EML email file"""
        try:
            is_valid, msg = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg}
            
            logger.info(f"Extracting EML: {file_path}")
            
            # Parse email
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            eml_stem = Path(file_path).stem
            
            # Extract components
            headers = self._extract_headers(msg)
            body_text = self._extract_body(msg)
            attachments = self._extract_attachment_info(msg)
            
            if not body_text and not headers.get('subject'):
                return "", False, {'error': 'Empty email'}
            
            # Build markdown
            markdown_content = self._build_markdown(
                eml_stem, headers, body_text, attachments
            )
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'sender': headers.get('from', 'Unknown'),
                'recipients': headers.get('to', 'Unknown'),
                'subject': headers.get('subject', 'No Subject'),
                'date': headers.get('date', 'Unknown'),
                'attachment_count': len(attachments),
                'extraction_method': 'eml'
            })
            
            logger.info(f"✓ EML extraction complete: {len(attachments)} attachments")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"EML extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_headers(self, msg) -> Dict[str, str]:
        """Extract email headers"""
        headers = {}
        
        headers['from'] = msg.get('From', '')
        headers['to'] = msg.get('To', '')
        headers['cc'] = msg.get('Cc', '')
        headers['bcc'] = msg.get('Bcc', '')
        headers['subject'] = msg.get('Subject', '')
        headers['date'] = msg.get('Date', '')
        headers['message_id'] = msg.get('Message-ID', '')
        headers['in_reply_to'] = msg.get('In-Reply-To', '')
        headers['references'] = msg.get('References', '')
        
        return headers
    
    def _extract_body(self, msg) -> str:
        """Extract email body text"""
        body_parts = []
        
        # Get plain text body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                # Extract text
                if content_type == 'text/plain':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            text = payload.decode(charset, errors='ignore')
                            body_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Could not decode part: {e}")
                
                # Extract HTML and convert to text (basic)
                elif content_type == 'text/html':
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            html = payload.decode(charset, errors='ignore')
                            # Basic HTML stripping
                            text = self._html_to_text(html)
                            body_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Could not decode HTML part: {e}")
        else:
            # Single part message
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or 'utf-8'
                    text = payload.decode(charset, errors='ignore')
                    body_parts.append(text)
            except Exception as e:
                logger.warning(f"Could not decode message: {e}")
        
        return '\n\n'.join(body_parts)
    
    def _html_to_text(self, html: str) -> str:
        """Basic HTML to text conversion"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            text = soup.get_text(separator='\n')
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return '\n'.join(lines)
        except ImportError:
            # Fallback: basic tag removal
            import re
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    
    def _extract_attachment_info(self, msg) -> List[Dict]:
        """Extract attachment information (not content)"""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get('Content-Disposition', ''))
                
                if 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append({
                            'filename': filename,
                            'content_type': part.get_content_type(),
                            'size': len(part.get_payload(decode=True) or b'')
                        })
        
        return attachments
    
    def _build_markdown(
        self,
        eml_stem: str,
        headers: Dict[str, str],
        body_text: str,
        attachments: List[Dict]
    ) -> str:
        """Build markdown from email components"""
        lines = []
        
        # Header
        lines.append(f"# Email: {headers.get('subject', eml_stem)}")
        lines.append(f"**Format**: Email (EML)")
        lines.append("")
        
        # Metadata
        lines.append("## Email Metadata\n")
        lines.append(f"**From**: {headers.get('from', 'Unknown')}")
        lines.append(f"**To**: {headers.get('to', 'Unknown')}")
        
        if headers.get('cc'):
            lines.append(f"**Cc**: {headers['cc']}")
        
        lines.append(f"**Date**: {headers.get('date', 'Unknown')}")
        lines.append(f"**Subject**: {headers.get('subject', 'No Subject')}")
        
        if headers.get('message_id'):
            lines.append(f"**Message-ID**: {headers['message_id']}")
        
        lines.append("")
        lines.append("---\n")
        
        # Body
        lines.append("## Email Body\n")
        if body_text:
            lines.append(body_text)
        else:
            lines.append("*(No text content)*")
        
        lines.append("")
        
        # Attachments
        if attachments:
            lines.append("## Attachments\n")
            for i, att in enumerate(attachments, 1):
                size_kb = att['size'] / 1024
                lines.append(f"{i}. **{att['filename']}**")
                lines.append(f"   - Type: {att['content_type']}")
                lines.append(f"   - Size: {size_kb:.1f} KB\n")
        
        return '\n'.join(lines)
