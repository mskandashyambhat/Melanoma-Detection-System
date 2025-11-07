"""
Generate architecture diagrams for U-Net and ResNet50 models
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

# ============================================================================
# 1. U-NET ARCHITECTURE DIAGRAM
# ============================================================================
print("Generating U-Net architecture diagram...")

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(10, 11.5, 'U-Net Architecture for Lesion Segmentation', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# Input
input_box = FancyBboxPatch((1, 9), 1.5, 1.5, boxstyle="round,pad=0.1", 
                           edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(1.75, 9.75, 'Input\n224×224×3', ha='center', va='center', fontsize=8, fontweight='bold')

# Encoder Path (Contracting)
encoder_y = 9
encoder_sizes = [(3, 64), (5, 128), (7, 256), (9, 512)]
encoder_boxes = []

for i, (x, channels) in enumerate(encoder_sizes):
    height = 1.2 - (i * 0.15)
    width = 1.2 - (i * 0.15)
    y_pos = encoder_y - (i * 1.8)
    
    box = FancyBboxPatch((x, y_pos), width, height, boxstyle="round,pad=0.05",
                         edgecolor='darkblue', facecolor='skyblue', linewidth=2)
    ax.add_patch(box)
    encoder_boxes.append((x + width/2, y_pos + height/2))
    
    # Conv + ReLU layers
    size = int(224 / (2**i))
    ax.text(x + width/2, y_pos + height/2, f'Conv+ReLU\n{size}×{size}×{channels}',
            ha='center', va='center', fontsize=7, fontweight='bold')
    
    # MaxPooling arrow (except last)
    if i < len(encoder_sizes) - 1:
        arrow = FancyArrowPatch((x + width/2, y_pos - 0.1), 
                               (encoder_sizes[i+1][0] + (width - i*0.15)/2, y_pos - 1.6),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='red')
        ax.add_patch(arrow)
        ax.text(x + width/2 + 0.5, y_pos - 0.9, 'MaxPool\n2×2', ha='center', 
               fontsize=7, color='red')

# Bottleneck
bottleneck = FancyBboxPatch((10, 2), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=3)
ax.add_patch(bottleneck)
ax.text(10.75, 2.5, 'Bottleneck\n14×14×1024', ha='center', va='center', 
       fontsize=8, fontweight='bold')

# Decoder Path (Expanding)
decoder_y = 9
decoder_sizes = [(13, 512), (15, 256), (17, 128), (19, 64)]
decoder_boxes = []

for i, (x, channels) in enumerate(decoder_sizes):
    height = 0.75 + (i * 0.15)
    width = 0.75 + (i * 0.15)
    y_pos = decoder_y - (i * 1.8)
    
    box = FancyBboxPatch((x, y_pos), width, height, boxstyle="round,pad=0.05",
                         edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box)
    decoder_boxes.append((x + width/2, y_pos + height/2))
    
    # UpConv layers
    size = int(28 * (2**i))
    ax.text(x + width/2, y_pos + height/2, f'UpConv+ReLU\n{size}×{size}×{channels}',
            ha='center', va='center', fontsize=7, fontweight='bold')
    
    # UpSampling arrow (except last)
    if i < len(decoder_sizes) - 1:
        arrow = FancyArrowPatch((x + width/2, y_pos - 0.1),
                               (decoder_sizes[i+1][0] + (width + (i+1)*0.15)/2, y_pos - 1.6),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
        ax.add_patch(arrow)
        ax.text(x + width/2 - 0.5, y_pos - 0.9, 'UpSample\n2×2', ha='center',
               fontsize=7, color='green')

# Skip Connections
for i in range(len(encoder_boxes) - 1):
    enc_x, enc_y = encoder_boxes[i]
    dec_x, dec_y = decoder_boxes[len(decoder_boxes) - 1 - i]
    
    arrow = FancyArrowPatch((enc_x + 0.7, enc_y), (dec_x - 0.7, dec_y),
                           arrowstyle='->', mutation_scale=15, linewidth=1.5,
                           color='orange', linestyle='--')
    ax.add_patch(arrow)

# Add skip connection label
ax.text(10, 10.5, 'Skip Connections (Concatenate)', ha='center', fontsize=9,
       color='orange', style='italic')

# Output
output_box = FancyBboxPatch((18.5, 9), 1.5, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(output_box)
ax.text(19.25, 9.75, 'Output\n224×224×1\n(Mask)', ha='center', va='center',
       fontsize=8, fontweight='bold')

# Legend
legend_y = 0.5
ax.text(1, legend_y, 'Legend:', fontsize=10, fontweight='bold')
ax.text(1, legend_y - 0.3, '→ MaxPooling (Downsampling)', fontsize=8, color='red')
ax.text(1, legend_y - 0.6, '→ UpSampling (Upconversion)', fontsize=8, color='green')
ax.text(1, legend_y - 0.9, '⟶ Skip Connections', fontsize=8, color='orange')

# Model info
ax.text(16, legend_y, 'Total Parameters: 7,759,521', fontsize=9, fontweight='bold')
ax.text(16, legend_y - 0.3, 'Trainable Parameters: 7,759,521', fontsize=9)
ax.text(16, legend_y - 0.6, 'Accuracy: 92%', fontsize=9)
ax.text(16, legend_y - 0.9, 'Dice Coefficient: 0.89', fontsize=9)

plt.tight_layout()
plt.savefig('../reprt/unet_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 2. RESNET50 ARCHITECTURE DIAGRAM
# ============================================================================
print("Generating ResNet50 architecture diagram...")

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(11, 11.5, 'ResNet50 Architecture for Melanoma Classification',
        ha='center', va='top', fontsize=18, fontweight='bold')

# Input
input_box = FancyBboxPatch((0.5, 5), 1.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 6, 'Input\n224×224×3', ha='center', va='center', fontsize=9, fontweight='bold')

# Conv1 + MaxPool
conv1_box = FancyBboxPatch((2.5, 5.2), 1.3, 1.6, boxstyle="round,pad=0.05",
                          edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(conv1_box)
ax.text(3.15, 6, 'Conv1\n7×7, 64\n+MaxPool', ha='center', va='center', fontsize=8)

# Residual Blocks
blocks = [
    ('Conv2_x', 3, 56, 64),
    ('Conv3_x', 4, 28, 128),
    ('Conv4_x', 6, 14, 256),
    ('Conv5_x', 3, 7, 512)
]

x_start = 4.5
for i, (name, num_blocks, size, channels) in enumerate(blocks):
    x_pos = x_start + (i * 3.5)
    
    # Main block
    block_box = FancyBboxPatch((x_pos, 4.5), 2.5, 3, boxstyle="round,pad=0.1",
                              edgecolor='darkblue', facecolor='skyblue', linewidth=2)
    ax.add_patch(block_box)
    ax.text(x_pos + 1.25, 7, name, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Individual residual blocks
    for j in range(num_blocks):
        y_offset = 6.5 - (j * 0.6)
        mini_box = FancyBboxPatch((x_pos + 0.3, y_offset - 0.25), 1.9, 0.4,
                                 boxstyle="round,pad=0.02", edgecolor='navy',
                                 facecolor='lightsteelblue', linewidth=1)
        ax.add_patch(mini_box)
        ax.text(x_pos + 1.25, y_offset - 0.05, f'1×1→3×3→1×1', ha='center',
               va='center', fontsize=6)
    
    # Feature map info
    ax.text(x_pos + 1.25, 4.8, f'{size}×{size}×{channels}', ha='center', va='center',
           fontsize=8, style='italic')
    
    # Connection arrow
    if i < len(blocks) - 1:
        arrow = FancyArrowPatch((x_pos + 2.5, 6), (x_pos + 3.5, 6),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)

# Global Average Pooling
gap_box = FancyBboxPatch((18.5, 5.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                        edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(gap_box)
ax.text(19.25, 6.25, 'Global Avg\nPooling\n1×1×2048', ha='center', va='center',
       fontsize=8, fontweight='bold')

# Fully Connected Layers
fc_box = FancyBboxPatch((20.5, 5.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                       edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(fc_box)
ax.text(21.25, 6.25, 'FC Layers\n+Softmax\n2 classes', ha='center', va='center',
       fontsize=8, fontweight='bold')

# Connections
arrow1 = FancyArrowPatch((17.5, 6), (18.5, 6.25),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((20, 6.25), (20.5, 6.25),
                        arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow2)

# Residual Connection Explanation
explanation_y = 2.5
ax.text(11, explanation_y + 1.5, 'Residual Block Structure:', fontsize=11, fontweight='bold')

# Draw residual block example
res_x, res_y = 6, explanation_y
res_box = FancyBboxPatch((res_x, res_y - 0.5), 4, 1.2, boxstyle="round,pad=0.1",
                        edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(res_box)

# Main path
ax.text(res_x + 0.5, res_y + 0.2, '1×1 Conv', ha='center', fontsize=7,
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(res_x + 1.5, res_y + 0.2, '3×3 Conv', ha='center', fontsize=7,
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(res_x + 2.5, res_y + 0.2, '1×1 Conv', ha='center', fontsize=7,
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Skip connection
skip_arrow = FancyArrowPatch((res_x + 0.2, res_y - 0.3), (res_x + 3.2, res_y - 0.3),
                            arrowstyle='->', mutation_scale=15, linewidth=2,
                            color='orange', linestyle='--')
ax.add_patch(skip_arrow)
ax.text(res_x + 1.7, res_y - 0.6, 'Identity/Skip Connection', ha='center',
       fontsize=7, color='orange', style='italic')

# Add operation
ax.text(res_x + 3.5, res_y + 0.2, '+', ha='center', fontsize=12, fontweight='bold')
ax.text(res_x + 3.7, res_y + 0.2, 'ReLU', ha='center', fontsize=7,
       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Model Statistics
stats_x = 15
stats_y = explanation_y
ax.text(stats_x, stats_y + 0.8, 'Model Statistics:', fontsize=10, fontweight='bold')
ax.text(stats_x, stats_y + 0.5, '• Total Parameters: 23,587,712', fontsize=8)
ax.text(stats_x, stats_y + 0.2, '• Trainable Parameters: 2,622,464', fontsize=8)
ax.text(stats_x, stats_y - 0.1, '• Accuracy: 88.89%', fontsize=8)
ax.text(stats_x, stats_y - 0.4, '• Precision: 50%', fontsize=8)
ax.text(stats_x, stats_y - 0.7, '• Recall: 16.77%', fontsize=8)

# Key Features
features_x = 2
features_y = explanation_y
ax.text(features_x, features_y + 0.8, 'Key Features:', fontsize=10, fontweight='bold')
ax.text(features_x, features_y + 0.5, '✓ 50 layers deep', fontsize=8)
ax.text(features_x, features_y + 0.2, '✓ Residual connections prevent vanishing gradients', fontsize=8)
ax.text(features_x, features_y - 0.1, '✓ Pre-trained on ImageNet (1.2M images)', fontsize=8)
ax.text(features_x, features_y - 0.4, '✓ Fine-tuned on HAM10000 (10K dermoscopic images)', fontsize=8)
ax.text(features_x, features_y - 0.7, '✓ Binary classification: Benign vs Melanoma', fontsize=8)

plt.tight_layout()
plt.savefig('../reprt/resnet50_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# 3. HYBRID MODEL PIPELINE DIAGRAM
# ============================================================================
print("Generating hybrid model pipeline diagram...")

fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(8, 9.5, 'Hybrid U-Net + ResNet50 Pipeline', ha='center', va='top',
       fontsize=20, fontweight='bold')

# Stage 1: Input
stage1_box = FancyBboxPatch((1, 6), 2, 2, boxstyle="round,pad=0.1",
                           edgecolor='blue', facecolor='lightblue', linewidth=3)
ax.add_patch(stage1_box)
ax.text(2, 7, 'Input Image\n224×224×3\n(RGB)', ha='center', va='center',
       fontsize=10, fontweight='bold')

# Stage 2: Preprocessing
stage2_box = FancyBboxPatch((4, 6), 2, 2, boxstyle="round,pad=0.1",
                           edgecolor='green', facecolor='lightgreen', linewidth=3)
ax.add_patch(stage2_box)
ax.text(5, 7, 'Preprocessing\n• Resize\n• Normalize\n• Artifact removal',
       ha='center', va='center', fontsize=9, fontweight='bold')

# Stage 3: U-Net Segmentation
stage3_box = FancyBboxPatch((7, 6), 2.5, 2, boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=3)
ax.add_patch(stage3_box)
ax.text(8.25, 7.5, 'U-Net\nSegmentation', ha='center', va='center',
       fontsize=11, fontweight='bold')
ax.text(8.25, 6.5, '92% Accuracy\n7.8M params', ha='center', va='center',
       fontsize=8)

# Stage 4: Masked Image
stage4_box = FancyBboxPatch((10.5, 6), 2, 2, boxstyle="round,pad=0.1",
                           edgecolor='orange', facecolor='lightyellow', linewidth=3)
ax.add_patch(stage4_box)
ax.text(11.5, 7, 'Segmented\nLesion\n(Masked)', ha='center', va='center',
       fontsize=10, fontweight='bold')

# Stage 5: ResNet50 Classification
stage5_box = FancyBboxPatch((1, 2.5), 2.5, 2, boxstyle="round,pad=0.1",
                           edgecolor='darkblue', facecolor='skyblue', linewidth=3)
ax.add_patch(stage5_box)
ax.text(2.25, 3.75, 'ResNet50\nClassification', ha='center', va='center',
       fontsize=11, fontweight='bold')
ax.text(2.25, 2.9, '88.89% Accuracy\n23.6M params', ha='center', va='center',
       fontsize=8)

# Stage 6: Final Result
stage6_box = FancyBboxPatch((4.5, 2.5), 2.5, 2, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=3)
ax.add_patch(stage6_box)
ax.text(5.75, 3.75, 'Classification\nResult', ha='center', va='center',
       fontsize=11, fontweight='bold')
ax.text(5.75, 3, 'Benign/Melanoma\n+ Confidence', ha='center', va='center',
       fontsize=9)

# Arrows
arrows = [
    ((3, 7), (4, 7)),
    ((6, 7), (7, 7)),
    ((9.5, 7), (10.5, 7)),
    ((11.5, 6), (2.25, 4.5)),
    ((3.5, 3.5), (4.5, 3.5))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=30,
                           linewidth=3, color='black')
    ax.add_patch(arrow)

# Combined Performance Box
perf_box = FancyBboxPatch((9, 1), 6, 3, boxstyle="round,pad=0.15",
                         edgecolor='darkgreen', facecolor='honeydew', linewidth=3)
ax.add_patch(perf_box)
ax.text(12, 3.5, 'Hybrid Model Performance', ha='center', va='top',
       fontsize=13, fontweight='bold', color='darkgreen')
ax.text(12, 3, '• Overall Accuracy: 90.45%', ha='center', fontsize=10)
ax.text(12, 2.6, '• Precision: 71%', ha='center', fontsize=10)
ax.text(12, 2.2, '• Recall: 54%', ha='center', fontsize=10)
ax.text(12, 1.8, '• F1-Score: 61.5%', ha='center', fontsize=10)
ax.text(12, 1.4, '• Total Parameters: 31.3M', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('../reprt/hybrid_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("\n✅ All architecture diagrams generated successfully!")
print("\nFiles saved:")
print("  1. unet_architecture.png")
print("  2. resnet50_architecture.png")
print("  3. hybrid_pipeline.png")
