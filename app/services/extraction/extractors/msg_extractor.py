"""
MSG (Outlook Email) File Extractor
Processes Microsoft Outlook email files for RAG pipeline
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List
import logging
import datetime
try:
    from ..base_extractor import BaseExtractor
except ImportError:
    from base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class MSGExtractor(BaseExtractor):
    """
    Extractor for Microsoft Outlook MSG files
    
    Features:
    - Extracts headers (From, To, Subject, Date)
    - Extracts body text
    - Lists attachments
    - Handles various MSG versions
    
    Requires: msg-parser or extract-msg library
    """
    
    def _get_supported_formats(self) -> List[str]:
        return ['msg']
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Validate MSG file"""
        path = Path(file_path)
        
        if not path.exists():
            return False, "File does not exist"
        
        ext = path.suffix.lower().lstrip('.')
        if ext not in self.supported_formats:
            return False, "Not a MSG file"
        
        # Check file signature (MSG files are OLE/CFB format)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # OLE file signature: D0 CF 11 E0 A1 B1 1A E1
                if header[:4] != b'\xd0\xcf\x11\xe0':
                    return False, "Not a valid MSG file (invalid OLE signature)"
        except Exception as e:
            return False, f"Could not read file: {e}"
        
        return True, "Valid MSG file"
    
    def extract(self, file_path: str) -> Tuple[str, bool, Dict]:
        """Extract MSG email file"""
        try:
            is_valid, msg_text = self.validate_file(file_path)
            if not is_valid:
                return "", False, {'error': msg_text}
            
            logger.info(f"Extracting MSG: {file_path}")
            
            msg_stem = Path(file_path).stem
            
            # Try different MSG parsing libraries
            msg_data = None
            extraction_method = None
            
            # Method 1: msg-parser (preferred for newer Python)
            try:
                import msg_parser
                msg_data = self._extract_with_msg_parser(file_path)
                extraction_method = 'msg-parser'
            except ImportError:
                logger.debug("msg-parser not available, trying extract-msg")
            except Exception as e:
                logger.warning(f"msg-parser failed: {e}")
            
            # Method 2: extract-msg (fallback)
            if not msg_data:
                try:
                    import extract_msg
                    msg_data = self._extract_with_extract_msg(file_path)
                    extraction_method = 'extract-msg'
                except ImportError:
                    logger.debug("extract-msg not available")
                except Exception as e:
                    logger.warning(f"extract-msg failed: {e}")
            
            # Method 3: python-oxmsg (alternative)
            if not msg_data:
                try:
                    from oxmsg import Message
                    msg_data = self._extract_with_oxmsg(file_path)
                    extraction_method = 'python-oxmsg'
                except ImportError:
                    logger.debug("python-oxmsg not available")
                except Exception as e:
                    logger.warning(f"python-oxmsg failed: {e}")
            
            if not msg_data:
                return "", False, {
                    'error': 'No MSG parser library available. Install: msg-parser, extract-msg, or python-oxmsg'
                }
            
            # Build markdown
            markdown_content = self._build_markdown(msg_stem, msg_data)
            
            metadata = self.get_metadata(file_path)
            metadata.update({
                'total_pages': 1,
                'sender': msg_data.get('sender', 'Unknown'),
                'recipients': msg_data.get('to', 'Unknown'),
                'subject': msg_data.get('subject', 'No Subject'),
                'date': msg_data.get('date', 'Unknown'),
                'attachment_count': len(msg_data.get('attachments', [])),
                'extraction_method': f'msg_{extraction_method}'
            })
            
            logger.info(f"✓ MSG extraction complete using {extraction_method}")
            
            return markdown_content, True, metadata
            
        except Exception as e:
            logger.error(f"MSG extraction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", False, {'error': str(e)}
    
    def _extract_with_msg_parser(self, file_path: str) -> Dict:
        """Extract using msg-parser library"""
        import msg_parser
        
        msg_obj = msg_parser.MsOxMessage(file_path)
        
        return {
            'sender': msg_obj.sender or '',
            'to': msg_obj.to or '',
            'cc': msg_obj.cc or '',
            'subject': msg_obj.subject or '',
            'date': str(msg_obj.date) if msg_obj.date else '',
            'body': msg_obj.body or '',
            'attachments': [
                {
                    'filename': att.AttachLongFilename or att.AttachFilename or 'unknown',
                    'size': len(att.data) if hasattr(att, 'data') else 0
                }
                for att in (msg_obj.attachments or [])
            ]
        }
    
    def _extract_with_extract_msg(self, file_path: str) -> Dict:
        """Extract using extract-msg library"""
        import extract_msg
        
        msg = extract_msg.Message(file_path)
        
        # Get attachments info
        attachments = []
        try:
            for att in msg.attachments:
                attachments.append({
                    'filename': att.longFilename or att.shortFilename or 'unknown',
                    'size': len(att.data) if hasattr(att, 'data') else 0
                })
        except:
            pass
        
        return {
            'sender': msg.sender or '',
            'to': msg.to or '',
            'cc': msg.cc or '',
            'subject': msg.subject or '',
            'date': str(msg.date) if msg.date else '',
            'body': msg.body or '',
            'attachments': attachments
        }
    
    def _extract_with_oxmsg(self, file_path: str) -> Dict:
        """Extract using python-oxmsg library"""
        from oxmsg import Message
        
        msg = Message.load(file_path)
        
        return {
            'sender': msg.sender or '',
            'to': ', '.join(msg.to) if isinstance(msg.to, list) else (msg.to or ''),
            'cc': ', '.join(msg.cc) if isinstance(msg.cc, list) else (msg.cc or ''),
            'subject': msg.subject or '',
            'date': str(msg.sent_date) if hasattr(msg, 'sent_date') else '',
            'body': msg.body or '',
            'attachments': [
                {
                    'filename': att.name or 'unknown',
                    'size': len(att.data) if hasattr(att, 'data') else 0
                }
                for att in (msg.attachments or [])
            ]
        }
    
    def _build_markdown(self, msg_stem: str, msg_data: Dict) -> str:
        """Build markdown from MSG data"""
        lines = []
        
        # Header
        subject = msg_data.get('subject', msg_stem)
        lines.append(f"# Email: {subject}")
        lines.append(f"**Format**: Microsoft Outlook Message (MSG)")
        lines.append("")
        
        # Metadata
        lines.append("## Email Metadata\n")
        lines.append(f"**From**: {msg_data.get('sender', 'Unknown')}")
        lines.append(f"**To**: {msg_data.get('to', 'Unknown')}")
        
        if msg_data.get('cc'):
            lines.append(f"**Cc**: {msg_data['cc']}")
        
        lines.append(f"**Date**: {msg_data.get('date', 'Unknown')}")
        lines.append(f"**Subject**: {subject}")
        lines.append("")
        lines.append("---\n")
        
        # Body
        lines.append("## Email Body\n")
        body = msg_data.get('body', '')
        if body:
            lines.append(body)
        else:
            lines.append("*(No text content)*")
        
        lines.append("")
        
        # Attachments
        attachments = msg_data.get('attachments', [])
        if attachments:
            lines.append("## Attachments\n")
            for i, att in enumerate(attachments, 1):
                size_kb = att.get('size', 0) / 1024
                lines.append(f"{i}. **{att.get('filename', 'unknown')}**")
                lines.append(f"   - Size: {size_kb:.1f} KB\n")
        
        return '\n'.join(lines)
