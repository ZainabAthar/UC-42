"""
Forensic Report Generator 
Generates comprehensive PDF reports from image authentication analysis results
Supports multiple LLM backends (OpenAI, Anthropic, Local)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from datetime import datetime
import numpy as np
from PIL import Image
import io

# LLM imports (conditional)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ForensicReportGenerator:
    """Generate forensic authentication reports analysis results"""
    
    def __init__(self, api_key: Optional[str] = None, api_provider: str = "local"):
        """
        Initialize the forensic report generator
        
        Args:
            api_key: API key for LLM provider (optional)
            api_provider: LLM provider - "openai", "anthropic", "gemini", "groq", or "local"
        """
        self.api_key = api_key
        self.api_provider = api_provider.lower()
        self.analysis_results = {}
        self.llm_client = None
        self.report_path = None
        self.image_path = None
        
        self._initialize_llm_client()
    
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        if self.api_provider == "openai":
            if not HAS_OPENAI:
                print("  OpenAI library not installed. Falling back to local analysis.")
                self.api_provider = "local"
            elif not self.api_key:
                print("  No API key provided for OpenAI. Falling back to local analysis.")
                self.api_provider = "local"
            else:
                self.llm_client = OpenAI(api_key=self.api_key)
                
        elif self.api_provider == "anthropic":
            if not HAS_ANTHROPIC:
                print("  Anthropic library not installed. Falling back to local analysis.")
                self.api_provider = "local"
            elif not self.api_key:
                print("  No API key provided for Anthropic. Falling back to local analysis.")
                self.api_provider = "local"
            else:
                self.llm_client = Anthropic(api_key=self.api_key)
        
        elif self.api_provider == "gemini":
            if not HAS_GEMINI:
                print("  Google Generative AI library not installed. Falling back to local analysis.")
                self.api_provider = "local"
            elif not self.api_key:
                print("  No API key provided for Gemini. Falling back to local analysis.")
                self.api_provider = "local"
            else:
                genai.configure(api_key=self.api_key)
                self.llm_client = genai
        
        elif self.api_provider == "groq":
            if not HAS_GROQ:
                print("  Groq library not installed. Falling back to local analysis.")
                self.api_provider = "local"
            elif not self.api_key:
                print("  No API key provided for Groq. Falling back to local analysis.")
                self.api_provider = "local"
            else:
                self.llm_client = Groq(api_key=self.api_key)
        
        elif self.api_provider == "local":
            self.llm_client = None
        else:
            print(f"Unknown provider '{self.api_provider}'. Using local analysis.")
            self.api_provider = "local"
    
    def load_analysis_results(self, npz_path: str, image_path: Optional[str] = None) -> bool:
        """
        Load analysis results from .npz file
        
        Args:
            npz_path: Path to the .npz output file from
            image_path: Optional path to the original image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            npz_file = np.load(npz_path)
            
            required_keys = ['map', 'conf', 'score']
            if not all(key in npz_file for key in required_keys):
                print(f" Missing required keys. Found: {list(npz_file.keys())}")
                return False
            
            self.analysis_results = {
                'map': npz_file['map'],
                'conf': npz_file['conf'],
                'score': float(npz_file['score']),
                'imgsize': tuple(npz_file['imgsize']) if 'imgsize' in npz_file else None,
            }
            
            if image_path:
                self.image_path = image_path
            
            print(f" Loaded analysis results from {npz_path}")
            print(f"   Score: {self.analysis_results['score']:.4f}")
            print(f"   Map shape: {self.analysis_results['map'].shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading analysis results: {e}")
            return False
    
    def analyze_with_llm(self, analysis_results: Optional[Dict] = None) -> str:
        """
        Generate forensic analysis text using LLM
        
        Args:
            analysis_results: Optional custom analysis results dict
            
        Returns:
            Analysis text from LLM
        """
        if analysis_results:
            self.analysis_results = analysis_results
        
        if not self.analysis_results:
            return self._generate_local_analysis()
        
        # Prepare analysis prompt
        score = self.analysis_results.get('score', 0)
        map_shape = self.analysis_results['map'].shape if 'map' in self.analysis_results else None
        
        prompt = f"""Analyze this image authentication check result:
- Tamper Detection Score: {score:.4f} (0=authentic, 1=tampered)
- Detection Map Shape: {map_shape}
- Confidence Map available: {'Yes' if 'conf' in self.analysis_results else 'No'}

Provide a brief forensic assessment (2-3 sentences) explaining what this score means for authentication, 
including confidence level and recommendations."""
        
        try:
            if self.api_provider == "openai":
                return self._analyze_openai(prompt)
            elif self.api_provider == "anthropic":
                return self._analyze_anthropic(prompt)
            elif self.api_provider == "gemini":
                return self._analyze_gemini(prompt)
            elif self.api_provider == "groq":
                return self._analyze_groq(prompt)
            else:
                return self._generate_local_analysis()
        except Exception as e:
            print(f"  LLM analysis failed: {e}. Using local analysis.")
            return self._generate_local_analysis()
    
    def _analyze_openai(self, prompt: str) -> str:
        """Get analysis from OpenAI"""
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a forensic image authentication expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI error: {e}")
    
    def _analyze_anthropic(self, prompt: str) -> str:
        """Get analysis from Anthropic"""
        try:
            response = self.llm_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic error: {e}")
    
    def _analyze_gemini(self, prompt: str) -> str:
        """Get analysis from Google Gemini"""
        try:
            model = self.llm_client.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt, generation_config={'max_output_tokens': 300})
            return response.text
        except Exception as e:
            raise Exception(f"Gemini error: {e}")
    
    def _analyze_groq(self, prompt: str) -> str:
        """Get analysis from Groq"""
        try:
            response = self.llm_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq error: {e}")
    
    def _generate_local_analysis(self) -> str:
        """Generate analysis using local rules"""
        score = self.analysis_results.get('score', 0)
        
        if score < 0.2:
            return "Very high confidence that the image is authentic. Detected minimal tampering indicators."
        elif score < 0.4:
            return "High confidence that the image is authentic. Some minor anomalies detected but within expected variation."
        elif score < 0.6:
            return "Moderate confidence. The image shows some suspicious indicators but evidence is inconclusive. Further analysis recommended."
        elif score < 0.8:
            return "Low confidence in authenticity. Multiple tampering indicators detected. Image likely manipulated."
        else:
            return "Very low confidence in authenticity. Strong evidence of tampering detected. Image highly likely to be forged."
    
    def _create_heatmap_image(self, array: np.ndarray, output_path: Optional[str] = None, 
                             cmap: str = 'hot', title: str = 'Heatmap') -> Optional[str]:
        """
        Create a heatmap visualization from numpy array
        
        Args:
            array: 2D numpy array to visualize
            output_path: Optional path to save the heatmap image
            cmap: Matplotlib colormap to use
            title: Title for the heatmap
            
        Returns:
            Path to the generated heatmap image, or None if failed
        """
        if not HAS_MATPLOTLIB:
            print(" Matplotlib not available. Skipping heatmap generation.")
            return None
        
        try:
            # Normalize array to 0-1 range
            arr_min, arr_max = array.min(), array.max()
            if arr_max > arr_min:
                array_normalized = (array - arr_min) / (arr_max - arr_min)
            else:
                array_normalized = np.zeros_like(array)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            im = ax.imshow(array_normalized, cmap=cmap, aspect='auto')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Confidence Score')
            
            # Save to file or buffer
            if output_path is None:
                output_path = "heatmap_temp.png"
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f" Error creating heatmap: {e}")
            return None
    
    def _resize_image_for_pdf(self, image_path: str, max_width: int = 600, max_height: int = 450) -> str:
        """
        Resize image for PDF inclusion
        
        Args:
            image_path: Path to the original image
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            
        Returns:
            Path to the resized image
        """
        try:
            img = Image.open(image_path)
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            output_path = image_path.replace('.', '_resized.')
            img.save(output_path, quality=95)
            
            return output_path
            
        except Exception as e:
            print(f" Error resizing image: {e}")
            return image_path
    
    def _generate_image_explanation(self, score: float) -> str:
        """
        Generate a detailed text explanation of the image analysis
        
        Args:
            score: Tampering score (0-1)
            
        Returns:
            Formatted text explanation
        """
        # Determine characteristics based on score
        if score < 0.2:
            tampering_level = "minimal"
            characteristics = "The detection map shows very few anomalies, with most of the image having consistent properties consistent with an authentic photograph."
            confidence_desc = "very high"
        elif score < 0.4:
            tampering_level = "low"
            characteristics = "The analysis detected some minor anomalies in the image, but they are within acceptable variance for authentic photographs. Natural image compression and processing artifacts may contribute to these detections."
            confidence_desc = "high"
        elif score < 0.6:
            tampering_level = "moderate"
            characteristics = "The image shows moderate levels of tampering indicators. Some regions exhibit properties inconsistent with natural image formation, suggesting possible manipulation or splicing. Further forensic analysis may be warranted."
            confidence_desc = "moderate"
        elif score < 0.8:
            tampering_level = "high"
            characteristics = "Significant tampering indicators are present throughout the image. Multiple regions show inconsistent properties that strongly suggest deliberate manipulation. The localization map identifies specific areas of concern."
            confidence_desc = "high"
        else:
            tampering_level = "very high"
            characteristics = "The image exhibits very strong evidence of tampering. Multiple analysis techniques have detected substantial anomalies that are highly inconsistent with authentic image formation. Professional authentication review is strongly recommended."
            confidence_desc = "very high"
        
        # Calculate tampering percentage
        tampering_percentage = score * 100
        authenticity_percentage = (1 - score) * 100
        
        explanation = f"""
        <b>Detection Overview:</b><br/>
        This image shows <b>{tampering_level}</b> levels of tampering indicators with <b>{confidence_desc}</b> confidence.<br/>
        <br/>
        <b>Analysis Details:</b><br/>
        {characteristics}<br/>
        <br/>
        <b>Numerical Summary:</b><br/>
        • Tampering Score: {score:.4f} ({tampering_percentage:.1f}% weighted towards tampering)<br/>
        • Authenticity Confidence: {authenticity_percentage:.1f}%<br/>
        • Detection Map Coverage: The localization map above shows spatial distribution of detected anomalies, with warmer colors indicating higher tampering confidence.<br/>
        • Confidence Map: The confidence visualization indicates the model's certainty in each pixel-level prediction.
        """
        
        return explanation
    
    def create_pdf_report(self, output_path: str = "forensic_report.pdf") -> bool:
        """
        Create a comprehensive PDF forensic report with all analysis images and metrics
        
        Args:
            output_path: Path where PDF should be saved
            
        Returns:
            True if successful, False otherwise
        """
        if not HAS_REPORTLAB:
            print("❌ ReportLab not installed. Cannot generate PDF.")
            return self._create_text_report(output_path.replace('.pdf', '.txt'))
        
        if not self.analysis_results:
            print("❌ No analysis results loaded. Load results first.")
            return False
        
        try:
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert path to string (use as-is for SimpleDocTemplate)
            output_file = str(output_path)
            
            doc = SimpleDocTemplate(output_file, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
            story = []
            styles = getSampleStyleSheet()
            
            # Define custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=28,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=6,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=13,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=6,
                spaceBefore=10,
                fontName='Helvetica-Bold',
                borderColor=colors.HexColor('#d0d0d0'),
                borderWidth=0,
                borderPadding=0
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubHeading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#2d5a99'),
                spaceAfter=6,
                spaceBefore=8,
                fontName='Helvetica-Bold'
            )
            
            # Title
            story.append(Paragraph("FORENSIC AUTHENTICATION REPORT", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Report info
            report_info = f"""
            <font size="9"><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font>
            """
            story.append(Paragraph(report_info, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Authentication Results Table
            score = self.analysis_results.get('score', 0)
            
            # Determine auth status based on score ranges
            if score < 0.2:
                auth_status = "Minimal tampering indicators detected"
            elif score < 0.4:
                auth_status = "Low tampering indicators detected"
            elif score < 0.6:
                auth_status = "Moderate tampering indicators detected"
            elif score < 0.8:
                auth_status = "High tampering indicators detected"
            else:
                auth_status = "Very high tampering indicators detected"
            
            confidence = abs(1 - score * 2) * 100  # Convert to confidence %
            
            story.append(Paragraph("Authentication Results", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            auth_table_data = [
                ["Metric", "Value"],
                ["Status", auth_status],
                ["Tampering Score", f"{score:.4f}"],
                ["Confidence Level", f"{confidence:.1f}%"],
                ["Detection Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            auth_table = Table(auth_table_data, colWidths=[2.2*inch, 2.8*inch])
            auth_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            
            story.append(auth_table)
            story.append(Spacer(1, 0.25*inch))
            
            # Forensic Analysis
            story.append(Paragraph("Forensic Analysis", heading_style))
            story.append(Spacer(1, 0.08*inch))
            analysis_text = self.analyze_with_llm()
            story.append(Paragraph(analysis_text, styles['BodyText']))
            story.append(Spacer(1, 0.25*inch))
            
            # Page break before images
            story.append(PageBreak())
            
            # Original Image
            if self.image_path and Path(self.image_path).exists():
                story.append(Paragraph("Original Image", heading_style))
                try:
                    orig_img_path = self._resize_image_for_pdf(self.image_path)
                    # Center the image
                    img_table = Table([[RLImage(orig_img_path, width=5*inch, height=3.75*inch)]], colWidths=[5*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                        ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                        ('LEFTPADDING', (0, 0), (0, 0), 0),
                        ('RIGHTPADDING', (0, 0), (0, 0), 0),
                        ('TOPPADDING', (0, 0), (0, 0), 0),
                        ('BOTTOMPADDING', (0, 0), (0, 0), 0),
                    ]))
                    story.append(img_table)
                except Exception as e:
                    story.append(Paragraph(f"<i>Original image could not be displayed: {e}</i>", styles['Normal']))
                
                story.append(Spacer(1, 0.35*inch))
            
            # Localization Map
            story.append(Paragraph("Tampering Localization Map", heading_style))
            story.append(Paragraph(
                "<font size='9'>Red areas indicate detected tampering, blue areas indicate authentic regions.</font>",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.15*inch))
            
            localization_path = self._create_heatmap_image(
                self.analysis_results['map'],
                output_file.replace('.pdf', '_localization.png'),
                cmap='RdBu_r',
                title='Tampering Localization Map'
            )
            
            if localization_path and os.path.exists(localization_path):
                loc_img_table = Table([[RLImage(localization_path, width=5.2*inch, height=4*inch)]], colWidths=[5.2*inch])
                loc_img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (0, 0), 0),
                    ('RIGHTPADDING', (0, 0), (0, 0), 0),
                    ('TOPPADDING', (0, 0), (0, 0), 0),
                    ('BOTTOMPADDING', (0, 0), (0, 0), 0),
                ]))
                story.append(loc_img_table)
            
            story.append(Spacer(1, 0.35*inch))
            
            # Confidence Map
            story.append(Paragraph("Confidence Map", heading_style))
            story.append(Paragraph(
                "<font size='9'>Darker areas indicate higher confidence in the tampering detection.</font>",
                styles['Normal']
            ))
            story.append(Spacer(1, 0.15*inch))
            
            confidence_path = self._create_heatmap_image(
                self.analysis_results['conf'],
                output_file.replace('.pdf', '_confidence.png'),
                cmap='gray',
                title='Confidence Map'
            )
            
            if confidence_path and os.path.exists(confidence_path):
                conf_img_table = Table([[RLImage(confidence_path, width=5.2*inch, height=4*inch)]], colWidths=[5.2*inch])
                conf_img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                    ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                    ('LEFTPADDING', (0, 0), (0, 0), 0),
                    ('RIGHTPADDING', (0, 0), (0, 0), 0),
                    ('TOPPADDING', (0, 0), (0, 0), 0),
                    ('BOTTOMPADDING', (0, 0), (0, 0), 0),
                ]))
                story.append(conf_img_table)
            
            story.append(Spacer(1, 0.35*inch))
            
            # Page break before technical details
            story.append(PageBreak())
            story.append(Paragraph("Technical Details", heading_style))
            
            tech_details = f"""
            <b>Detection Map Shape:</b> {self.analysis_results['map'].shape}<br/>
            <b>Image Size:</b> {self.analysis_results.get('imgsize', 'Unknown')}<br/>
            <b>Confidence Map Available:</b> {'Yes' if 'conf' in self.analysis_results else 'No'}<br/>
            
            <b>Report Generation Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            
            story.append(Paragraph(tech_details, styles['Normal']))
            story.append(Spacer(1, 0.35*inch))
            
            # Image Analysis Explanation
            story.append(Paragraph("Image Analysis Summary", heading_style))
            explanation_text = self._generate_image_explanation(score)
            story.append(Paragraph(explanation_text, styles['Normal']))
            story.append(Spacer(1, 0.35*inch))
            
            # Score Scale Reference
            story.append(Paragraph("Tampering Score Scale Reference", subheading_style))
            score_explanation = f"""
            <b>Score Range Interpretation:</b><br/>
            <br/>
            <b>0.0 - 0.2:</b> Minimal tampering indicators detected<br/>
            <b>0.2 - 0.4:</b> Low tampering indicators detected<br/>
            <b>0.4 - 0.6:</b> Moderate tampering indicators detected<br/>
            <b>0.6 - 0.8:</b> High tampering indicators detected<br/>
            <b>0.8 - 1.0:</b> Very high tampering indicators detected<br/>
            <br/>
            <b>Current Image Score: {score:.4f}</b> (scales from 0 = authentic to 1 = tampered)
            """
            story.append(Paragraph(score_explanation, styles['Normal']))
            
            story.append(Spacer(1, 0.4*inch))
            
            # Professional Footer
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#666666'),
                alignment=TA_CENTER,
                borderColor=colors.HexColor('#cccccc'),
                borderWidth=0.5,
                borderPadding=8,
                borderRadius=2
            )
            
            footer_text = "This report was generated automatically by the Forensic Report Generator. " \
                          "Results should be reviewed by qualified forensic experts. " \
                          
            story.append(Paragraph(footer_text, footer_style))
            
            # Build PDF
            doc.build(story)
            self.report_path = output_file
            print(f" PDF report saved to {output_file}")
            return True
            
        except Exception as e:
            # import traceback
            # print(f"❌ Error generating PDF: {e}")
            # print(f"❌ Error type: {type(e).__name__}")
            # print(f"❌ Traceback: {traceback.format_exc()}")
            # return self._create_text_report(str(Path(output_path)).replace('.pdf', '.txt'))
            raise
    
    def _create_text_report(self, output_path: str = "forensic_report.txt") -> bool:
        """Create a text fallback report"""
        try:
            score = self.analysis_results.get('score', 0)
            
            report_content = f"""
FORENSIC AUTHENTICATION REPORT
==============================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

AUTHENTICATION RESULTS
======================
Status: {'AUTHENTIC' if score < 0.5 else 'TAMPERED'}
Tampering Score: {score:.4f}
Confidence: {abs(1 - score * 2) * 100:.1f}%

FORENSIC ANALYSIS
=================
{self.analyze_with_llm()}

TECHNICAL DETAILS
=================
Map Shape: {self.analysis_results['map'].shape}
Image Size: {self.analysis_results.get('imgsize', 'Unknown')}


DISCLAIMER
==========
This report was automatically generated. Results should be reviewed by qualified experts.
"""
            
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            print(f"Text report saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating text report: {e}")
            return False


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate forensic authentication reports from  analysis"
    )
    parser.add_argument("npz_file", help="Path to  .npz output file")
    parser.add_argument(
        "-o", "--output",
        default="forensic_report.pdf",
        help="Output file path (default: forensic_report.pdf)"
    )
    parser.add_argument(
        "-provider",
        choices=["openai", "anthropic", "local"],
        default="local",
        help="LLM provider for analysis (default: local)"
    )
    parser.add_argument(
        "-key",
        help="API key for LLM provider (or use env variables: OPENAI_API_KEY, ANTHROPIC_API_KEY)"
    )
    
    args = parser.parse_args()
    
    # Get API key from arguments or environment
    api_key = args.key
    if not api_key and args.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif not api_key and args.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Create generator
    generator = ForensicReportGenerator(api_key=api_key, api_provider=args.provider)
    
    # Load and process
    if not generator.load_analysis_results(args.npz_file):
        sys.exit(1)
    
    if not generator.create_pdf_report(args.output):
        print("Report generation partially failed. Check output.")
    
    print(f"\nReport available at: {args.output}")


if __name__ == "__main__":
    main()
