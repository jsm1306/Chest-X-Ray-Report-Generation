"""
Utility functions for image handling, PDF generation, and validation
"""
import os
import logging
from typing import Tuple
from datetime import datetime
from pathlib import Path
import re

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = "uploads"
ALLOWED_IMAGE_FORMATS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 50  # 50 MB max


def create_upload_dir():
    """Create upload directory if it doesn't exist"""
    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    logger.info(f"Upload directory ready: {UPLOAD_DIR}")


def validate_image_file(filename: str, file_size: int) -> Tuple[bool, str]:
    """
    Validate image file.
    
    Args:
        filename: Name of the file
        file_size: Size of the file in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check extension
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_FORMATS:
        return False, f"Invalid format. Allowed: {', '.join(ALLOWED_IMAGE_FORMATS)}"
    
    # Check file size
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        return False, f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
    
    return True, ""


def save_uploaded_image(file_content: bytes, original_filename: str) -> str:
    """
    Save uploaded image temporarily.
    
    Args:
        file_content: Binary file content
        original_filename: Original filename
        
    Returns:
        Path to saved file
        
    Raises:
        ValueError: If image is invalid
    """
    # Validate image file
    is_valid, error_msg = validate_image_file(original_filename, len(file_content))
    if not is_valid:
        raise ValueError(error_msg)
    
    # Generate safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
    safe_filename = timestamp + re.sub(r'[^a-zA-Z0-9._-]', '_', original_filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # Verify it's a valid image
    try:
        img = Image.open(file_path)
        img.verify()
    except Exception as e:
        os.remove(file_path)
        raise ValueError(f"Invalid image file: {str(e)}")
    
    logger.info(f"Image saved: {file_path}")
    return file_path


def cleanup_temp_file(file_path: str):
    """
    Delete temporary file safely.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {file_path}: {e}")


def generate_pdf_report(
    report_text: str,
    patient_name: str = "Anonymous",
    patient_age: str = "N/A",
    output_path: str = None
) -> bytes:
    """
    Generate professional medical PDF report.
    
    Args:
        report_text: Generated medical report text
        patient_name: Patient name (for header)
        patient_age: Patient age (for header)
        output_path: Path to save PDF (if None, returns bytes)
        
    Returns:
        PDF file bytes if output_path is None, else None
    """
    if output_path is None:
        from io import BytesIO
        output_path = BytesIO()
    
    # Create PDF
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1a5490"),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#2c5aa0"),
        spaceAfter=12,
        spaceBefore=12,
        fontName="Helvetica-Bold",
    )
    
    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
    )
    
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.red,
        alignment=TA_CENTER,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("MEDICAL DIAGNOSTIC REPORT", title_style))
    content.append(Spacer(1, 0.2*inch))
    
    # Patient Information Table
    patient_data = [
        ["Field", "Value"],
        ["Patient Name", patient_name],
        ["Patient Age", patient_age],
        ["Report Date", datetime.now().strftime("%B %d, %Y")],
        ["Report Time", datetime.now().strftime("%I:%M %p")],
    ]
    
    patient_table = Table(
        patient_data,
        colWidths=[2*inch, 4*inch],
    )
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5490")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
    ]))
    
    content.append(patient_table)
    content.append(Spacer(1, 0.3*inch))
    
    # Findings Section
    content.append(Paragraph("FINDINGS", heading_style))
    content.append(Paragraph(report_text, normal_style))
    content.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_text = "⚠ DISCLAIMER: This report is AI-generated and must be verified by a certified medical professional before clinical use."
    content.append(Paragraph(disclaimer_text, disclaimer_style))
    
    content.append(Spacer(1, 0.2*inch))
    
    # Footer line
    footer_text = "For medical inquiries, contact a qualified healthcare professional."
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
    )
    content.append(Paragraph(footer_text, footer_style))
    
    # Build PDF
    doc.build(content)
    
    # If output_path is BytesIO, return bytes
    if hasattr(output_path, "getvalue"):
        return output_path.getvalue()


def format_report_text(raw_report: str) -> str:
    """
    Format and clean generated report text for display.
    
    Args:
        raw_report: Raw report text from model
        
    Returns:
        Formatted report text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', raw_report).strip()
    
    # Ensure proper punctuation
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text
