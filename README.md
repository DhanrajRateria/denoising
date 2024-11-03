# Image Processing Project: Denoising and Sharpening Techniques

This project implements various image denoising and sharpening techniques, including both classical methods and deep learning approaches.

## Features

- Classical denoising methods:
  - Gaussian filter
  - Median filter
  - Bilateral filter
- Deep learning denoising:
  - U-Net implementation
- Image sharpening:
  - Unsharp masking
  - Laplacian edge enhancement
- Quality metrics:
  - Signal-to-Noise Ratio (SNR)
  - Edge strength analysis
- Comprehensive visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DhanrajRateria/denoising.git
cd denoising
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

1. Place your input images in the `data/input` directory.

2. Run the main script:
```bash
python src/main.py
```

3. Check the results in the `data/output` directory.

## Project Structure

```
image_processing/
├── data/
│   ├── input/
│   ├── output/
│   └── models/
├── src/
│   ├── denoising/
│   ├── sharpening/
│   └── utils/
├── tests/
└── notebooks/
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.