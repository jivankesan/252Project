# 252Project

## Overview

This repository contains the code and resources for the 252 Project.

## Installation

To install the project, follow these steps:

1. Clone the repository:

2. Navigate to the project directory:
   ```sh
   cd 252Project
   ```
3. Install the dependencies:

   ```sh
   # Ensure you have the docker daemon running, or use a venv
   docker build -t mte252-audio-project .
   docker run -it --rm --name audio-container -v "$(pwd)":/app -w /app mte252-audio-project
   ```

## Usage

To use this project, follow these steps:

1. [Step 1: Run the Processor]
   ```sh
   # Process the raw audio files, it will place your output files and graphs in the output folder
   python Processor.py
   ```
2. [Step 2: Run the similarity test]
   ```sh
   # run the stoi test based on ground truth audio, output will be in stoi_metrics.txt
   python similarity_test.py
   ```
