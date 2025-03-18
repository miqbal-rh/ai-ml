PSP Defect Classification and Suggestion System
Overview
This project is an AI-driven defect classification and suggestion system designed for analyzing software defect logs. The system processes log files from multiple developers, extracts meaningful insights using machine learning techniques, and provides automated suggestions for resolving common errors.

Features
Automated Log Processing: Reads log files from developer-specific directories.
Text Feature Extraction: Uses TF-IDF vectorization to convert logs into numerical representations.
K-Means Clustering: Groups logs into clusters to identify similar errors.
Dimensionality Reduction: Applies UMAP for 2D visualization.
Error Suggestions: Leverages Sentence Transformers and FAISS to provide solutions for common errors.
Interactive Visualization: Uses Plotly to create a 3D scatter plot of clustered logs.
Real-Time Monitoring: Monitors new log files and updates analysis periodically.
Installation
Prerequisites
Ensure you have the following dependencies installed:

bash
Copy
Edit
pip install numpy pandas plotly umap-learn faiss-cpu tqdm scikit-learn sentence-transformers joblib
Usage
Running the Log Processor
To start monitoring and processing logs, run:

bash
Copy
Edit
python log_analysis_offline.py
The script will continuously process logs and update the analysis every 10 seconds.

Log Directory Structure
Place log files under a directory structured as follows:

matlab
Copy
Edit
BASE_DIR/
├── Developer1/
│   ├── log1.log
│   ├── log2.log
├── Developer2/
│   ├── log1.log
│   ├── log2.log
Update the BASE_DIR constant in log_analysis_offline.py with the actual directory path.

How It Works
Loading Logs: Extracts log data from each developer’s directory.
Feature Extraction: Applies TF-IDF vectorization.
Clustering: Groups similar logs using K-Means.
Dimensionality Reduction: Uses UMAP to reduce feature space for visualization.
Error Suggestion: Uses FAISS and Sentence Transformers to find relevant solutions.
Visualization: Generates a 3D scatter plot.
Real-time Monitoring: Continuously monitors new logs and updates analysis.
Example Output
Clusters logs into categories like NullPointerException, SyntaxError, etc.
Provides suggestions for resolving issues.
Displays a 3D scatter plot for interactive exploration.
Example 3D Plot
Here’s an example of what the 3D plot might look like when visualizing the clustered logs:

![3D Plot Example](defect_classification_view.png)


License
This project is open-source under the MIT License.

Contributions
Feel free to submit issues or pull requests to improve the system.

