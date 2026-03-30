"""
API Routes - FastAPI endpoints for medical report generation and PDF download
"""
import logging
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model_loader import get_models
from inference import generate_report
from utils import (
    save_uploaded_image,
    cleanup_temp_file,
    generate_pdf_report,
    format_report_text,
    UPLOAD_DIR
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ReportRequest(BaseModel):
    """Request model for direct report generation"""
    image_path: str


class ReportResponse(BaseModel):
    """Response model for report generation"""
    status: str
    report: str
    message: str = ""


class PDFRequest(BaseModel):
    """Request model for PDF generation"""
    report_text: str
    patient_name: str = "Patient"
    patient_age: str = "N/A"


@router.post("/generate-report", response_model=ReportResponse)
async def generate_report_endpoint(
    file: UploadFile = File(...),
    patient_name: str = Form("Patient"),
    patient_age: str = Form("N/A")
) -> ReportResponse:
    """
    Generate medical report from uploaded X-ray image.
    
    Args:
        file: Medical image file (jpg, png)
        patient_name: Patient name for PDF header
        patient_age: Patient age for PDF header
        
    Returns:
        ReportResponse with generated report text
        
    Raises:
        HTTPException: If image invalid, model not loaded, or inference fails
    """
    temp_image_path = None
    
    try:
        logger.info(f"Received report generation request for patient: {patient_name}")
        
        # Validate file was provided
        if not file or not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No image file provided"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Save uploaded image temporarily
        try:
            temp_image_path = save_uploaded_image(file_content, file.filename)
            logger.info(f"Image saved to {temp_image_path}")
        except ValueError as e:
            logger.warning(f"Image validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Load models
        try:
            encoder_model, full_model, tokenizer = get_models()
            logger.info("Models loaded successfully")
        except RuntimeError as e:
            logger.error(f"Model loading failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Models not initialized. Server restart may be needed."
            )
        
        # Generate report
        try:
            logger.info("Starting inference...")
            report_text = generate_report(
                img_path=temp_image_path,
                encoder_model=encoder_model,
                full_model=full_model,
                tokenizer=tokenizer,
                max_len=200
            )
            logger.info("Report generated successfully")
        except ValueError as e:
            logger.error(f"Invalid image for inference: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Image processing error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate report. Please try again."
            )
        
        # Format report
        formatted_report = format_report_text(report_text)
        
        logger.info("Report ready for delivery")
        return ReportResponse(
            status="success",
            report=formatted_report,
            message="Report generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in report generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )
    finally:
        # Cleanup temporary image file
        if temp_image_path:
            cleanup_temp_file(temp_image_path)


@router.post("/download-report")
async def download_report_endpoint(request: PDFRequest):
    """
    Generate and download professional medical PDF report.
    
    Args:
        request: PDFRequest containing report text and patient info
        
    Returns:
        PDF file as downloadable response
        
    Raises:
        HTTPException: If PDF generation fails
    """
    try:
        logger.info(f"Generating PDF for patient: {request.patient_name}")
        
        if not request.report_text or not request.report_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Report text is empty"
            )
        
        # Generate PDF
        try:
            pdf_bytes = generate_pdf_report(
                report_text=request.report_text,
                patient_name=request.patient_name,
                patient_age=request.patient_age
            )
            logger.info("PDF generated successfully")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Failed to generate PDF"
            )
        
        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=medical_report_{request.patient_name.replace(' ', '_')}.pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PDF download: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating PDF"
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    Attempts to load models to ensure they're available.
    """
    try:
        encoder_model, full_model, tokenizer = get_models()
        return {
            "status": "healthy",
            "models_loaded": True,
            "encoder_model": encoder_model.name if hasattr(encoder_model, 'name') else "OK",
            "full_model": full_model.name if hasattr(full_model, 'name') else "OK"
        }
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "models_loaded": False,
            "error": str(e)
        }


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Medical Diagnosis Report API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-report": "Generate medical report from uploaded X-ray",
            "POST /download-report": "Download report as PDF",
            "GET /health": "Health check"
        },
        "documentation": "/docs"
    }
