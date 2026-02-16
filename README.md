# sensPy

A high-performance Python port of the R sensR package for sensory discrimination analysis.

## Guiding Philosophy

sensPy is built on the principle of strict numerical parity with the R sensR package, ensuring sensory scientists can transition to Python without losing statistical rigor. It prioritizes performance through Numba-accelerated Thurstonian modeling and robust validation against established golden data fixtures.

## Core System

The architecture is centered around specialized statistical modules for sensory protocols such as Triangle, Duo-Trio, and 2-AFC. It leverages SciPy for optimization and Numba for performance-critical calculations, providing a multi-layered approach from raw data processing to complex power simulations.

## Features

- Thurstonian Modeling: Estimate sensitivity indices like d-prime and probabilities of discrimination across various protocols.
- Comprehensive Protocol Support: Built-in implementations for Triangle, Duo-Trio, 2-AFC, Same-Different, and A-Not-A tests.
- Advanced Statistical Engines: Specialized modules for ANOVA, Beta-Binomial models for overdispersion, and d-prime comparisons.
- Numerical Parity: Strict alignment with R sensR outputs, validated through an extensive testing suite and rpy2 integration.
- High-Performance Execution: Performance-critical calculations accelerated via Numba for large-scale sensory data analysis.
- Interactive Visualizations: Native support for Plotly-based interactive plots and optional Matplotlib static exports.

## Tech Stack

Python 3.10+, NumPy, SciPy, Pandas, Numba, Plotly, and Poetry/uv for dependency management.

## Getting Started

```bash
npm install
npm run dev
```

To get started, install the dependencies using npm: "npm install". You can then run the analysis scripts or tests using "npm run test" or "npm run start".
