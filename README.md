# LeafGuard AI — Plant Leaf Disease Detection

An AI-powered web application that uses deep learning to identify potential plant leaf diseases from uploaded images and provide disease-related guidance.

## Features

* **AI Disease Prediction:** Analyze an uploaded plant leaf image using a trained MobileNetV2 model.
* **Prediction Confidence:** View the model's confidence score for its prediction.
* **Visual Explanation:** Generate a Grad-CAM heatmap to highlight image regions that influenced the prediction.
* **Disease Advisory:** Display guidance related to the predicted disease.
* **Prediction History:** Save and review previous predictions.
* **Search and Filtering:** Find records and filter prediction history.
* **CSV Export:** Export prediction history for further analysis.
* **PDF Reports:** Generate downloadable reports containing prediction details and visual explanations.
* **Dashboard Statistics:** View a summary of prediction activity.

## Technology Stack

| Component         | Technologies                   |
| ----------------- | ------------------------------ |
| Frontend          | React, Vite, JavaScript        |
| Backend           | Python, FastAPI                |
| Machine Learning  | TensorFlow, Keras, MobileNetV2 |
| Image Processing  | Pillow, NumPy                  |
| Database          | SQLite                         |
| Data Export       | CSV, ReportLab PDF             |
| Development Tools | Git, GitHub, VS Code           |

## Project Structure

```text
LeafGuard-AI/
├── backend/          # FastAPI backend and API endpoints
├── frontend/         # React and Vite web application
├── ml/
│   └── models/       # Trained model and class labels
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

* Python 3.12
* Node.js and npm
* Git

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd LeafGuard-AI
```
 `https://github.com/vikashmauryadev/LeafGuard-AI` 

### 2. Set up the Python environment

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies after the project's dependency file has been prepared:

```bash
pip install -r backend/requirements.txt
```

### 3. Start the backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API should be available at:

* API: http://127.0.0.1:8000
* Interactive API documentation: http://127.0.0.1:8000/docs

### 4. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

## Machine Learning Model

LeafGuard AI uses a trained MobileNetV2-based image classification model to predict a plant leaf disease class.

The model and class-label mapping are stored in `ml/models/`.

Prediction results depend on the model's training data, image quality, and the similarity between uploaded images and the training dataset.

## Important Disclaimer

LeafGuard AI is an experimental decision-support tool. Predictions and recommendations may be incorrect and should not be treated as a definitive agricultural diagnosis. Verify important plant-health decisions with a qualified agricultural expert.

## Current Status

The application has a working local prediction workflow, visual explanations, prediction history, CSV export, and PDF reporting. Deployment and future enhancements can be added as the project evolves.

## Future Improvements

* Deploy the frontend and backend for public access.
* Add user authentication and individual prediction histories.
* Improve model evaluation using independent test data.
* Expand support for additional crops and diseases.
* Improve advisory content and multilingual accessibility.

## License

@Vikash Maurya 
