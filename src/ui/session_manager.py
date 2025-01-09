"""
Manages multiple scoring sessions
"""

import os
import logging
from typing import Dict, Optional
import hashlib
from PyQt6.QtCore import QObject, pyqtSignal

from src.audio_processing.scoring_session import ScoringSession

class SessionManager(QObject):
    """Manages multiple scoring sessions"""
    
    # Signals
    session_started = pyqtSignal(str)  # Session ID
    session_ended = pyqtSignal(str)    # Session ID
    
    def __init__(self):
        super().__init__()
        self.active_sessions: Dict[str, ScoringSession] = {}
        
    def _generate_session_id(self, config: Dict) -> str:
        """Generate unique session ID from config
        
        The ID is based on:
        - Reference file path
        - Input source (microphone or file)
        - Input file path (if using file input)
        """
        # Create string to hash
        id_parts = [
            os.path.abspath(config['reference_file']),
            config['input_source']
        ]
        if config['input_source'] == 'file' and config.get('input_file'):
            id_parts.append(os.path.abspath(config['input_file']))
            
        # Create hash
        hasher = hashlib.md5()
        for part in id_parts:
            hasher.update(str(part).encode('utf-8'))
        return hasher.hexdigest()
        
    def start_session(self, config: Dict) -> str:
        """Start a new session or reveal existing one
        
        Args:
            config: Session configuration
            
        Returns:
            Session ID
        """
        try:
            # Generate session ID
            session_id = self._generate_session_id(config)
            
            # Check if session already exists
            if session_id in self.active_sessions:
                # Just show the existing session's window
                session = self.active_sessions[session_id]
                session.visualizer.show()
                session.visualizer.raise_()
                logging.info(f"Revealed existing session {session_id}")
            else:
                # Create new session
                session = ScoringSession(config)
                self.active_sessions[session_id] = session
                self.session_started.emit(session_id)
                logging.info(f"Started new session {session_id}")
            
            return session_id
            
        except Exception as e:
            logging.error(f"Error starting session: {e}")
            logging.exception("Full traceback:")
            raise
            
    def end_session(self, session_id: str):
        """End a session
        
        Args:
            session_id: ID of session to end
        """
        try:
            if session_id in self.active_sessions:
                # Clean up session
                session = self.active_sessions[session_id]
                session.cleanup()
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                self.session_ended.emit(session_id)
                
                logging.info(f"Ended session {session_id}")
                
        except Exception as e:
            logging.error(f"Error ending session: {e}")
            logging.exception("Full traceback:")
            
    def get_session(self, session_id: str) -> Optional[ScoringSession]:
        """Get a session by ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        return self.active_sessions.get(session_id)
        
    def cleanup(self):
        """Clean up all sessions"""
        try:
            # Make copy of keys since we'll be modifying the dict
            session_ids = list(self.active_sessions.keys())
            
            # End each session
            for session_id in session_ids:
                self.end_session(session_id)
                
        except Exception as e:
            logging.error(f"Error cleaning up sessions: {e}")
            logging.exception("Full traceback:")
