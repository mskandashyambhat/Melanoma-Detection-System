"""
PDF Report Generator for Melanoma Detection System
Creates comprehensive patient reports with results and recommendations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import os


def generate_patient_report(patient_info, prediction_data, output_path):
    """
    Generate a comprehensive patient report PDF
    
    Args:
        patient_info: Dictionary containing patient information
        prediction_data: Dictionary containing prediction results
        output_path: Path where PDF will be saved
    """
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=0.75*inch
    )
    
    # Container for PDF elements
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1f2937'),
        alignment=TA_LEFT,
        spaceAfter=8
    )
    
    # Title
    title = Paragraph("MELANOMA DETECTION SYSTEM", title_style)
    elements.append(title)
    
    subtitle = Paragraph("Medical Analysis Report", styles['Heading2'])
    elements.append(subtitle)
    elements.append(Spacer(1, 0.3*inch))
    
    # Report Information
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    report_id = f"MDR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    info_data = [
        ['Report ID:', report_id],
        ['Report Date:', report_date],
        ['Status:', 'CONFIDENTIAL']
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e5e7eb')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Patient Information Section
    elements.append(Paragraph("PATIENT INFORMATION", heading_style))
    
    patient_data = [
        ['Full Name:', patient_info.get('name', 'N/A')],
        ['Age:', str(patient_info.get('age', 'N/A'))],
        ['Gender:', patient_info.get('gender', 'N/A')],
        ['Contact:', patient_info.get('phone', 'N/A')],
        ['Email:', patient_info.get('email', 'N/A')],
        ['Medical History:', patient_info.get('medical_history', 'None reported')]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP')
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Uploaded Image Section
    elements.append(Paragraph("ANALYZED SKIN LESION IMAGE", heading_style))
    
    image_path = prediction_data.get('image_path', '')
    if image_path and os.path.exists(image_path):
        try:
            img = RLImage(image_path, width=3*inch, height=3*inch)
            img.hAlign = 'CENTER'
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Image could not be loaded: {str(e)}", normal_style))
    else:
        elements.append(Paragraph("Image not available", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Analysis Results Section
    elements.append(Paragraph("ANALYSIS RESULTS", heading_style))
    
    disease = prediction_data.get('disease', 'Unknown')
    confidence = prediction_data.get('confidence', 0)
    severity = prediction_data.get('severity', 'Unknown')
    
    # Determine color based on severity
    severity_color = colors.green
    if severity == 'Critical' or severity == 'High':
        severity_color = colors.red
    elif severity == 'Medium':
        severity_color = colors.orange
    
    results_data = [
        ['Detected Condition:', disease],
        ['Confidence Level:', f"{confidence}%"],
        ['Severity:', severity]
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 4*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#fef3c7')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('TEXTCOLOR', (1, 2), (1, 2), severity_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Description
    elements.append(Paragraph("CONDITION DESCRIPTION", subheading_style))
    description = prediction_data.get('description', 'No description available')
    elements.append(Paragraph(description, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Recommendations Section
    elements.append(Paragraph("MEDICAL RECOMMENDATIONS", heading_style))
    
    recommendations = prediction_data.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            rec_text = f"{i}. {rec}"
            elements.append(Paragraph(rec_text, normal_style))
    else:
        elements.append(Paragraph("No specific recommendations available.", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Important Notice
    elements.append(Paragraph("IMPORTANT NOTICE", heading_style))
    notice_text = """
    This report is generated by an AI-powered melanoma detection system and should be used 
    for preliminary screening purposes only. It is NOT a substitute for professional medical 
    diagnosis. Please consult with a qualified dermatologist or healthcare provider for 
    proper evaluation, diagnosis, and treatment. Early detection and professional medical 
    advice are crucial for managing skin conditions effectively.
    """
    notice_para = Paragraph(notice_text, ParagraphStyle(
        'Notice',
        parent=normal_style,
        textColor=colors.HexColor('#dc2626'),
        fontSize=10,
        alignment=TA_JUSTIFY
    ))
    elements.append(notice_para)
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer information
    footer_text = """
    <para alignment="center">
    <b>For medical emergencies, call 911 or visit the nearest emergency room.</b><br/>
    For appointments or consultations, use the 'Consult a Doctor' feature in the application.<br/>
    © 2024 Melanoma Detection System. All rights reserved.
    </para>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    print(f"Report generated successfully: {output_path}")


# Test function
if __name__ == "__main__":
    # Sample data for testing
    sample_patient = {
        'name': 'John Doe',
        'age': 45,
        'gender': 'Male',
        'phone': '+1-555-1234',
        'email': 'john.doe@email.com',
        'medical_history': 'No significant medical history. Family history of skin cancer.'
    }
    
    sample_prediction = {
        'disease': 'Melanoma',
        'confidence': 97.5,
        'severity': 'Critical',
        'description': 'A serious form of skin cancer that develops in melanocytes.',
        'recommendations': [
            'Consult a dermatologist immediately',
            'Avoid sun exposure',
            'Get a biopsy for confirmation'
        ],
        'image_path': ''
    }
    
    generate_patient_report(sample_patient, sample_prediction, 'test_report.pdf')
