#!/usr/bin/env python3
"""
Backend API server for PII Detection Agent.
Integrates the Python PII detection system with the TypeScript frontend.
"""
import os
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our PII detection components
from pii_agent import PIIDetectionAgent
from config import OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PII Detection Agent API",
    description="Advanced PII detection using NER, Proximity Analysis, and Graph Theory",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for storing results
processing_results: Dict[str, Any] = {}

class ProcessingRequest(BaseModel):
    file_path: str

class ReportRequest(BaseModel):
    report_path: str

class DownloadRequest(BaseModel):
    file_path: str

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PII Detection Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "process_csv": "/api/process-csv",
            "get_report": "/api/report",
            "download_file": "/api/download",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test if PII agent can be initialized
        agent = PIIDetectionAgent()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "pii_agent": "ready",
                "spacy_model": "loaded",
                "openai_api": "configured" if OPENAI_API_KEY else "not_configured"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/api/process-csv")
async def process_csv(file: UploadFile = File(...)):
    """
    Process uploaded CSV file for PII detection.
    
    Args:
        file: Uploaded CSV file
        
    Returns:
        Processing results with file paths
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV file")
        
        # Validate file size (100MB limit)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 100MB")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing CSV file: {file.filename}")
        
        # Initialize PII detection agent
        agent = PIIDetectionAgent()
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(temp_file_path), "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the file
        results = agent.process_csv(temp_file_path, output_dir)
        
        # Store results for later retrieval
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        processing_results[session_id] = {
            "results": results,
            "temp_file": temp_file_path,
            "output_dir": output_dir,
            "filename": file.filename
        }
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")
        
        logger.info(f"Processing completed for session: {session_id}")
        
        return {
            "success": results["success"],
            "error": results.get("error"),
            "session_id": session_id,
            "report_path": results.get("report_path"),
            "masked_csv_path": results.get("masked_csv_path"),
            "graph_visualization": results.get("graph_visualization"),
            "graph_data": results.get("graph_data"),
            "messages": results.get("messages", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/report")
async def get_report(request: ReportRequest):
    """
    Get detailed PII detection report.
    
    Args:
        request: Report request with file path
        
    Returns:
        Detailed report data
    """
    try:
        report_path = request.report_path
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Read and parse the report
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        return report_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read report: {str(e)}")

@app.post("/api/download")
async def download_file(request: DownloadRequest):
    """
    Download generated files (masked CSV, graph visualization, etc.).
    
    Args:
        request: Download request with file path
        
    Returns:
        File response
    """
    try:
        file_path = request.file_path
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type based on file extension
        file_ext = Path(file_path).suffix.lower()
        content_types = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.pdf': 'application/pdf'
        }
        
        content_type = content_types.get(file_ext, 'application/octet-stream')
        
        # Generate filename
        filename = Path(file_path).name
        
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@app.get("/api/sessions")
async def list_sessions():
    """List all processing sessions."""
    return {
        "sessions": list(processing_results.keys()),
        "count": len(processing_results)
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a processing session and clean up files."""
    try:
        if session_id not in processing_results:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = processing_results[session_id]
        
        # Clean up output directory
        output_dir = session_data.get("output_dir")
        if output_dir and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        
        # Remove from memory
        del processing_results[session_id]
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in processing_results:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return processing_results[session_id]

if __name__ == "__main__":
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=3001,
        reload=True,
        log_level="info"
    )
