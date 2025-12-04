"""
FEATURES EXPLANATION PDF GENERATOR
===================================
Creates a PDF explaining how the models classify melanoma and benign lesions.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path

class FeaturesReportGenerator:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / 'Output'
        self.output_dir.mkdir(exist_ok=True)

    def generate_features_report(self):
        """Generate PDF explaining model features and classification"""
        pdf_path = self.output_dir / 'model_features_explanation.pdf'
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )

        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20
        )

        normal_style = styles['Normal']
        normal_style.spaceAfter = 12

        story = []

        # Title
        story.append(Paragraph("Melanoma Detection Models: Features and Classification Explanation", title_style))
        story.append(Spacer(1, 20))

        # Introduction
        intro = """
        This report explains how the three deep learning models (U-Net, ResNet50, and Combined Hybrid) 
        classify skin lesions as melanoma or benign. Each model uses different approaches to extract 
        features from dermoscopic images and make predictions.
        """
        story.append(Paragraph(intro, normal_style))
        story.append(Spacer(1, 20))

        # U-Net Model
        story.append(Paragraph("1. U-Net Model", heading_style))

        unet_desc = """
        <b>Architecture:</b> U-Net is primarily designed for image segmentation but adapted here for classification. 
        It consists of an encoder (downsampling) path and a decoder (upsampling) path with skip connections.

        <b>Feature Extraction:</b>
        - Convolutional layers extract hierarchical features from low-level edges to high-level patterns
        - Skip connections preserve spatial information from encoder to decoder
        - Final global average pooling converts spatial features to classification features

        <b>Classification Approach:</b>
        - Learns to distinguish melanoma from benign by identifying irregular patterns, asymmetry, and color variations
        - Focuses on texture features and boundary irregularities typical of malignant lesions
        - Uses learned filters to detect pigment distribution abnormalities

        <b>Key Features for Melanoma Detection:</b>
        - Asymmetrical shapes and irregular borders
        - Multiple colors and color variations
        - Irregular pigment distribution
        - Texture irregularities and structural disorder
        """
        story.append(Paragraph(unet_desc, normal_style))
        story.append(Spacer(1, 20))

        # ResNet50 Model
        story.append(Paragraph("2. ResNet50 Model", heading_style))

        resnet_desc = """
        <b>Architecture:</b> ResNet50 is a deep convolutional neural network with 50 layers, using residual connections 
        to enable training of very deep networks. Pre-trained on ImageNet for transfer learning.

        <b>Feature Extraction:</b>
        - Multiple convolutional blocks with increasing complexity
        - Residual connections allow learning of residual functions
        - Batch normalization and ReLU activations
        - Fine-tuned for medical image classification

        <b>Classification Approach:</b>
        - Leverages pre-trained features from natural images adapted to medical imaging
        - Learns hierarchical features from general shapes to specific lesion characteristics
        - Uses deep feature representations to capture complex patterns

        <b>Key Features for Melanoma Detection:</b>
        - Complex shape irregularities beyond simple asymmetry
        - Advanced color and texture patterns
        - Structural features learned from large-scale image data
        - Deep semantic understanding of lesion morphology
        """
        story.append(Paragraph(resnet_desc, normal_style))
        story.append(Spacer(1, 20))

        # Combined Hybrid Model
        story.append(Paragraph("3. Combined Hybrid Model (U-Net + ResNet50)", heading_style))

        hybrid_desc = """
        <b>Architecture:</b> Two-stage pipeline where U-Net performs segmentation followed by ResNet50 classification. 
        The U-Net creates a binary mask of the skin lesion, which is then applied to focus ResNet50 on the lesion only.

        <b>Feature Extraction Process:</b>
        1. <b>Segmentation Phase (U-Net):</b> Identifies the lesion boundaries and separates it from surrounding skin
        2. <b>Mask Application:</b> Multiplies the original image with the segmentation mask
        3. <b>Classification Phase (ResNet50):</b> Analyzes only the masked lesion region

        <b>Classification Approach:</b>
        - Combines segmentation accuracy with deep classification features
        - Eliminates background noise and irrelevant skin features
        - Focuses computational resources on the lesion itself
        - Leverages both boundary information and internal lesion features

        <b>Key Features for Melanoma Detection:</b>
        - Precise lesion boundary analysis
        - Internal lesion texture and color patterns
        - Size and shape measurements from segmentation
        - Isolated analysis without background interference
        """
        story.append(Paragraph(hybrid_desc, normal_style))
        story.append(Spacer(1, 20))

        # Common Features
        story.append(Paragraph("4. Common Features Used by All Models", heading_style))

        common_features = """
        <b>Color Features:</b>
        - Multiple color presence (brown, black, blue, red)
        - Color asymmetry and irregular distribution
        - Pigmentation intensity variations

        <b>Shape and Border Features:</b>
        - Asymmetry in shape and color
        - Irregular, scalloped, or poorly defined borders
        - Elongated or irregular shapes

        <b>Texture Features:</b>
        - Irregular dots, globules, or streaks
        - Structural disorder
        - Pigment network irregularities

        <b>Size and Evolution Features:</b>
        - Diameter greater than 6mm
        - Rapid changes in size or appearance
        - Evolution over time (though not available in single images)
        """
        story.append(Paragraph(common_features, normal_style))
        story.append(Spacer(1, 20))

        # Training Data
        story.append(Paragraph("5. Training and Dataset Information", heading_style))

        training_info = """
        <b>Dataset:</b> HAM10000 dataset with binary classification (Melanoma vs Benign)
        - Total images: ~15,000 dermoscopic images
        - Melanoma cases: ~2,400 (15.8%)
        - Benign cases: ~12,800 (84.2%)

        <b>Data Preprocessing:</b>
        - Images resized to 224x224 pixels
        - Normalized pixel values
        - Balanced dataset creation through augmentation
        - Class imbalance handling with focal loss

        <b>Training Techniques:</b>
        - Transfer learning from ImageNet (ResNet50)
        - Focal loss for imbalanced classification
        - Data augmentation (rotation, flip, zoom, color jittering)
        - Early stopping and learning rate scheduling
        """
        story.append(Paragraph(training_info, normal_style))
        story.append(Spacer(1, 20))

        # Conclusion
        story.append(Paragraph("6. Conclusion", heading_style))

        conclusion = """
        The three models use different but complementary approaches to melanoma detection:

        - <b>U-Net:</b> Excels at boundary and texture analysis
        - <b>ResNet50:</b> Provides deep feature understanding and generalization
        - <b>Hybrid:</b> Combines precise segmentation with powerful classification

        All models focus on the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution) 
        and other dermoscopic features established in clinical practice. The hybrid model 
        typically performs best by isolating the lesion for focused analysis.
        """
        story.append(Paragraph(conclusion, normal_style))

        # Build PDF
        doc.build(story)
        print(f"✅ Features explanation PDF saved to: {pdf_path}")

if __name__ == '__main__':
    generator = FeaturesReportGenerator()
    generator.generate_features_report()