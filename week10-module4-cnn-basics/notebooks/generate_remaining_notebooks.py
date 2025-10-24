#!/usr/bin/env python3
"""
Generate remaining Jupyter notebooks (2-9) for Week 10 CNN lecture.
"""

import json
import os

def create_notebook_cell(cell_type, source, metadata=None):
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source if isinstance(source, list) else [source]
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def create_notebook_2():
    """Create Notebook 2: 1D Convolution Math & Code."""
    cells = [
        create_notebook_cell("markdown", [
            "# Notebook 2: 1D Convolution Math & Code\n",
            "\n",
            "**Week 10 - Module 4: CNN Basics**  \n",
            "**DO3 (October 27, 2025) - Saturday**  \n",
            "**Duration:** 20-25 minutes\n",
            "\n",
            "---\n",
            "\n",
            "## Learning Objectives\n",
            "\n",
            "By the end of this notebook, you will be able to:\n",
            "\n",
            "1. ✅ **Calculate** 1D convolution operations by hand (step-by-step)\n",
            "2. ✅ **Implement** 1D convolution using NumPy\n",
            "3. ✅ **Understand** the mathematical formula for convolution\n",
            "4. ✅ **Apply** 1D convolution to signal processing problems\n",
            "5. ✅ **Visualize** how convolution transforms signals\n",
            "\n",
            "---\n",
            "\n",
            "## Prerequisites\n",
            "\n",
            "- ✅ Completed Notebook 0 (Setup & Prerequisites)\n",
            "- ✅ Completed Notebook 1 (Convolution Concept & Intuition)\n",
            "- ✅ Understanding of basic multiplication and summation\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("markdown", [
            "## 1. Setup and Imports"
        ]),

        create_notebook_cell("code", [
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy import signal\n",
            "\n",
            "# Set random seed\n",
            "np.random.seed(42)\n",
            "\n",
            "# Configure matplotlib\n",
            "plt.style.use('seaborn-v0_8-darkgrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 11\n",
            "\n",
            "print(\"✅ Setup complete!\")"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 2. The Story: Character: Arjun's ECG Analysis\n",
            "\n",
            "### 📖 Narrative\n",
            "\n",
            "**Character: Arjun**, a biomedical engineering student, receives an ECG (electrocardiogram) signal from **Character: Dr. Priya**'s clinic.\n",
            "\n",
            "**The Problem:**\n",
            "\n",
            "> \"This ECG signal is noisy,\" explains **Character: Dr. Priya**. \"I need to smooth it to identify heart rate patterns. Can you help?\"\n",
            "\n",
            "**Character: Arjun** responds: \"Yes! I'll use 1D convolution with a smoothing filter. Let me show you how it works step-by-step.\"\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("markdown", [
            "## 3. Mathematical Foundation\n",
            "\n",
            "### 📐 The Convolution Formula (1D)\n",
            "\n",
            "For 1D signals, convolution is defined as:\n",
            "\n",
            "$$\n",
            "(f * g)[n] = \\sum_{m=-\\infty}^{\\infty} f[m] \\cdot g[n - m]\n",
            "$$\n",
            "\n",
            "In practice (discrete, finite signals):\n",
            "\n",
            "$$\n",
            "output[n] = \\sum_{k=0}^{K-1} input[n + k] \\cdot kernel[k]\n",
            "$$\n",
            "\n",
            "Where:\n",
            "- `input`: The signal we're analyzing\n",
            "- `kernel`: The filter (pattern detector)\n",
            "- `output`: The convolved result\n",
            "- `K`: Kernel size\n",
            "\n",
            "**In Plain English:**\n",
            "\n",
            "\"At each position `n`, multiply overlapping values and sum them up.\"\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("markdown", [
            "## 4. Hand Calculation Example\n",
            "\n",
            "Let's calculate convolution **by hand** using a simple example.\n",
            "\n",
            "### Example Setup:\n",
            "\n",
            "- **Input**: `[1, 2, 3, 4, 5]`\n",
            "- **Kernel**: `[1, 0, -1]` (edge detector)\n",
            "- **Task**: Calculate the output step-by-step\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Define input and kernel\n",
            "input_signal = np.array([1, 2, 3, 4, 5])\n",
            "kernel = np.array([1, 0, -1])\n",
            "\n",
            "print(\"Input signal:\", input_signal)\n",
            "print(\"Kernel:\", kernel)\n",
            "print(\"\\nWe will calculate the output at each valid position...\\n\")"
        ]),

        create_notebook_cell("markdown", [
            "### Step 1: Position 0\n",
            "\n",
            "```\n",
            "Input:  [1, 2, 3, 4, 5]\n",
            "Kernel:  [1, 0, -1]\n",
            "         ↑  ↑  ↑\n",
            "Position 0-2\n",
            "```\n",
            "\n",
            "**Calculation:**\n",
            "\n",
            "$$\n",
            "output[0] = (1 \\times 1) + (2 \\times 0) + (3 \\times -1)\n",
            "$$\n",
            "\n",
            "$$\n",
            "output[0] = 1 + 0 + (-3) = -2\n",
            "$$\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Manual calculation for position 0\n",
            "pos_0 = (input_signal[0] * kernel[0] + \n",
            "         input_signal[1] * kernel[1] + \n",
            "         input_signal[2] * kernel[2])\n",
            "\n",
            "print(f\"Position 0: {input_signal[0]} × {kernel[0]} + {input_signal[1]} × {kernel[1]} + {input_signal[2]} × {kernel[2]}\")\n",
            "print(f\"Position 0: {input_signal[0] * kernel[0]} + {input_signal[1] * kernel[1]} + {input_signal[2] * kernel[2]}\")\n",
            "print(f\"Position 0: {pos_0}\")\n",
            "print()"
        ]),

        create_notebook_cell("markdown", [
            "### Step 2: Position 1\n",
            "\n",
            "```\n",
            "Input:  [1, 2, 3, 4, 5]\n",
            "Kernel:     [1, 0, -1]\n",
            "            ↑  ↑  ↑\n",
            "Position   1-3\n",
            "```\n",
            "\n",
            "**Calculation:**\n",
            "\n",
            "$$\n",
            "output[1] = (2 \\times 1) + (3 \\times 0) + (4 \\times -1) = -2\n",
            "$$\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Manual calculation for position 1\n",
            "pos_1 = (input_signal[1] * kernel[0] + \n",
            "         input_signal[2] * kernel[1] + \n",
            "         input_signal[3] * kernel[2])\n",
            "\n",
            "print(f\"Position 1: {input_signal[1]} × {kernel[0]} + {input_signal[2]} × {kernel[1]} + {input_signal[3]} × {kernel[2]}\")\n",
            "print(f\"Position 1: {pos_1}\")\n",
            "print()"
        ]),

        create_notebook_cell("markdown", [
            "### Step 3: Position 2\n",
            "\n",
            "```\n",
            "Input:  [1, 2, 3, 4, 5]\n",
            "Kernel:        [1, 0, -1]\n",
            "               ↑  ↑  ↑\n",
            "Position      2-4\n",
            "```\n",
            "\n",
            "**Calculation:**\n",
            "\n",
            "$$\n",
            "output[2] = (3 \\times 1) + (4 \\times 0) + (5 \\times -1) = -2\n",
            "$$\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Manual calculation for position 2\n",
            "pos_2 = (input_signal[2] * kernel[0] + \n",
            "         input_signal[3] * kernel[1] + \n",
            "         input_signal[4] * kernel[2])\n",
            "\n",
            "print(f\"Position 2: {input_signal[2]} × {kernel[0]} + {input_signal[3]} × {kernel[1]} + {input_signal[4]} × {kernel[2]}\")\n",
            "print(f\"Position 2: {pos_2}\")\n",
            "print()\n",
            "\n",
            "# Final output\n",
            "manual_output = np.array([pos_0, pos_1, pos_2])\n",
            "print(\"\\n🎯 Final Output (Manual Calculation):\", manual_output)"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 5. NumPy Implementation\n",
            "\n",
            "Now let's verify our hand calculations using NumPy's `convolve` function.\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Using NumPy's convolve function\n",
            "numpy_output = np.convolve(input_signal, kernel, mode='valid')\n",
            "\n",
            "print(\"NumPy Output:\", numpy_output)\n",
            "print(\"Manual Output:\", manual_output)\n",
            "print(\"\\n✅ Match:\", np.array_equal(numpy_output, manual_output))"
        ]),

        create_notebook_cell("markdown", [
            "### Understanding `mode` Parameter\n",
            "\n",
            "NumPy's `convolve` has three modes:\n",
            "\n",
            "| Mode | Description | Output Size |\n",
            "|------|-------------|-------------|\n",
            "| `'valid'` | Only where input and kernel fully overlap | `N - K + 1` |\n",
            "| `'same'` | Output same size as input (zero-padding) | `N` |\n",
            "| `'full'` | All overlaps (partial too) | `N + K - 1` |\n",
            "\n",
            "Where:\n",
            "- `N` = input size\n",
            "- `K` = kernel size\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Demonstrate different modes\n",
            "valid_output = np.convolve(input_signal, kernel, mode='valid')\n",
            "same_output = np.convolve(input_signal, kernel, mode='same')\n",
            "full_output = np.convolve(input_signal, kernel, mode='full')\n",
            "\n",
            "print(f\"Input size: {len(input_signal)}\")\n",
            "print(f\"Kernel size: {len(kernel)}\")\n",
            "print(f\"\\nValid mode output ({len(valid_output)}): {valid_output}\")\n",
            "print(f\"Same mode output ({len(same_output)}): {same_output}\")\n",
            "print(f\"Full mode output ({len(full_output)}): {full_output}\")"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 6. Implementing Custom 1D Convolution\n",
            "\n",
            "Let's write our own convolution function from scratch to deeply understand the process.\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "def conv1d_manual(input_arr, kernel_arr):\n",
            "    \"\"\"\n",
            "    Manual implementation of 1D convolution (valid mode).\n",
            "    \n",
            "    Parameters:\n",
            "    -----------\n",
            "    input_arr : np.ndarray\n",
            "        Input 1D signal\n",
            "    kernel_arr : np.ndarray\n",
            "        Convolution kernel\n",
            "    \n",
            "    Returns:\n",
            "    --------\n",
            "    output : np.ndarray\n",
            "        Convolved output\n",
            "    \"\"\"\n",
            "    n = len(input_arr)\n",
            "    k = len(kernel_arr)\n",
            "    output_size = n - k + 1\n",
            "    \n",
            "    output = np.zeros(output_size)\n",
            "    \n",
            "    for i in range(output_size):\n",
            "        # Extract window\n",
            "        window = input_arr[i:i+k]\n",
            "        # Element-wise multiply and sum\n",
            "        output[i] = np.sum(window * kernel_arr)\n",
            "    \n",
            "    return output\n",
            "\n",
            "# Test our implementation\n",
            "custom_output = conv1d_manual(input_signal, kernel)\n",
            "\n",
            "print(\"Custom Implementation:\", custom_output)\n",
            "print(\"NumPy Implementation:\", numpy_output)\n",
            "print(\"\\n✅ Match:\", np.array_equal(custom_output, numpy_output))"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 7. Real-World Application: ECG Signal Smoothing\n",
            "\n",
            "Let's help **Character: Arjun** smooth the noisy ECG signal.\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Generate synthetic noisy ECG signal\n",
            "t = np.linspace(0, 2, 200)\n",
            "clean_ecg = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)\n",
            "noise = 0.3 * np.random.randn(len(t))\n",
            "noisy_ecg = clean_ecg + noise\n",
            "\n",
            "# Create smoothing filter (moving average)\n",
            "smoothing_kernel = np.ones(5) / 5  # Average of 5 points\n",
            "\n",
            "# Apply convolution\n",
            "smoothed_ecg = np.convolve(noisy_ecg, smoothing_kernel, mode='same')\n",
            "\n",
            "# Visualize\n",
            "fig, axes = plt.subplots(3, 1, figsize=(14, 10))\n",
            "\n",
            "# Noisy signal\n",
            "axes[0].plot(t, noisy_ecg, color='red', alpha=0.7, linewidth=0.8)\n",
            "axes[0].set_title('Noisy ECG Signal', fontsize=14, fontweight='bold')\n",
            "axes[0].set_ylabel('Amplitude')\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Smoothing kernel\n",
            "axes[1].stem(smoothing_kernel, basefmt=' ')\n",
            "axes[1].set_title('Smoothing Kernel (Moving Average)', fontsize=14, fontweight='bold')\n",
            "axes[1].set_ylabel('Weight')\n",
            "axes[1].grid(True, alpha=0.3)\n",
            "\n",
            "# Smoothed signal\n",
            "axes[2].plot(t, noisy_ecg, color='red', alpha=0.3, linewidth=0.8, label='Noisy')\n",
            "axes[2].plot(t, smoothed_ecg, color='green', linewidth=2, label='Smoothed')\n",
            "axes[2].set_title('Smoothed ECG Signal', fontsize=14, fontweight='bold')\n",
            "axes[2].set_xlabel('Time (s)')\n",
            "axes[2].set_ylabel('Amplitude')\n",
            "axes[2].legend()\n",
            "axes[2].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"✅ Character: Arjun successfully smoothed the ECG signal!\")\n",
            "print(f\"   Noise reduction: {np.std(noisy_ecg) / np.std(smoothed_ecg):.2f}x\")"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 8. Different Types of 1D Filters\n",
            "\n",
            "Let's explore various 1D filters and their effects.\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("code", [
            "# Create test signal\n",
            "test_signal = np.array([0, 0, 0, 5, 5, 5, 0, 0, 0])\n",
            "\n",
            "# Define different filters\n",
            "filters = {\n",
            "    'Edge Detector': np.array([1, 0, -1]),\n",
            "    'Smoothing': np.array([1, 1, 1]) / 3,\n",
            "    'Sharpening': np.array([-1, 3, -1]),\n",
            "    'Identity': np.array([0, 1, 0])\n",
            "}\n",
            "\n",
            "# Apply each filter\n",
            "fig, axes = plt.subplots(len(filters) + 1, 1, figsize=(12, 12))\n",
            "\n",
            "# Original signal\n",
            "axes[0].stem(test_signal, basefmt=' ')\n",
            "axes[0].set_title('Original Signal', fontsize=13, fontweight='bold')\n",
            "axes[0].set_ylabel('Value')\n",
            "axes[0].grid(True, alpha=0.3)\n",
            "\n",
            "# Apply filters\n",
            "for idx, (name, filt) in enumerate(filters.items(), 1):\n",
            "    output = np.convolve(test_signal, filt, mode='same')\n",
            "    axes[idx].stem(output, basefmt=' ', linefmt='r-', markerfmt='ro')\n",
            "    axes[idx].set_title(f'{name} Filter', fontsize=13, fontweight='bold')\n",
            "    axes[idx].set_ylabel('Value')\n",
            "    axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=0.8)\n",
            "    axes[idx].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"🎯 Filter Effects:\")\n",
            "print(\"  • Edge Detector: Highlights changes\")\n",
            "print(\"  • Smoothing: Reduces noise, blurs\")\n",
            "print(\"  • Sharpening: Enhances edges\")\n",
            "print(\"  • Identity: No change (passes through)\")"
        ]),

        create_notebook_cell("markdown", [
            "---\n",
            "\n",
            "## 9. Summary and Key Takeaways\n",
            "\n",
            "### 🎯 What We Learned\n",
            "\n",
            "1. **Mathematical Formula**\n",
            "   - Convolution = multiply overlapping values, then sum\n",
            "   - Output size: `N - K + 1` (valid mode)\n",
            "\n",
            "2. **Hand Calculation**\n",
            "   - Slide kernel across input\n",
            "   - At each position: multiply and sum\n",
            "   - Result shows pattern detection\n",
            "\n",
            "3. **NumPy Implementation**\n",
            "   - `np.convolve(input, kernel, mode='valid')`\n",
            "   - Three modes: valid, same, full\n",
            "\n",
            "4. **Real-World Applications**\n",
            "   - Signal smoothing (ECG, audio)\n",
            "   - Edge detection\n",
            "   - Noise reduction\n",
            "\n",
            "### 🔮 What's Next?\n",
            "\n",
            "In **Notebook 3**, we'll extend to **2D Convolution for Images**:\n",
            "- 2D convolution mathematics\n",
            "- Image filtering (blur, edge detection)\n",
            "- Feature map visualization\n",
            "\n",
            "---"
        ]),

        create_notebook_cell("markdown", [
            "## 10. Practice Exercises\n",
            "\n",
            "### Exercise 1: Hand Calculation\n",
            "Calculate convolution manually:\n",
            "- Input: `[2, 4, 6, 8]`\n",
            "- Kernel: `[1, 2]`\n",
            "\n",
            "### Exercise 2: Custom Filter Design\n",
            "Design a kernel that:\n",
            "- Detects rising edges (increasing values)\n",
            "- Test on: `[1, 1, 1, 5, 5, 5]`\n",
            "\n",
            "### Exercise 3: Mode Comparison\n",
            "Apply convolution with all three modes:\n",
            "- Input: `[1, 2, 3, 4, 5, 6]`\n",
            "- Kernel: `[1, 1, 1]`\n",
            "- Compare output sizes\n",
            "\n",
            "---\n",
            "\n",
            "**Next Notebook:** [Notebook 3: 2D Convolution for Images](03_2d_convolution_images.ipynb)\n",
            "\n",
            "---\n",
            "\n",
            "*Week 10 - Deep Neural Network Architectures (21CSE558T)*  \n",
            "*SRM University - M.Tech Program*"
        ])
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

# Save notebook 2
notebook_2 = create_notebook_2()
with open('02_1d_convolution_math_code.ipynb', 'w') as f:
    json.dump(notebook_2, f, indent=1)

print("✅ Generated Notebook 2: 1D Convolution Math & Code")
