# Intent Detection Tool

A POC for processing customer support messages, classifying their intent, and routing them into your workflow. Classify messages via k-nearest neighbors over SBERT embeddings of labeled examples.

No labeled data? Use 🧪 Intent Lab to discover & name intents from historical messages, review them, and one-click generate a training set.

Auto-route: After prediction, creates a Plain ticket, and if the intent is a bug or feature, also creates a Shortcut story.

Features a Streamlit-based UI for setup, classification, and training data creation.

## Features

- **AI-Powered Classification**: Uses OpenAI's GPT-4 to classify message intent and generate summaries
- **Intent Lab**: Discover & defined intent list from historical messages and one-click generate a training set
- **Ticket Management**: Creates tickets in Plain
- **Bug/Feature Escalation**: Automatically creates issues in Shortcut for bugs and features
- **Streamlit UI**: User-friendly interface for manual message processing and configuration

## Prerequisites

- Python 3.8+
- OpenAI API Key (for message classification)
- Plain API Key (for ticket management)
- Shortcut API Key (for bug/feature tracking)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sashurova00/Galileo_AI_Interview.git
   cd intent-detection-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. In the sidebar, enter your API keys and click "Save Configuration"

4. Use the "Process Message" tab to manually process support messages or generate intents and training data from historical messages via "Intent Lab"

## Architecture

The application follows a simple, modular architecture:

- `app.py`: Main Streamlit application with UI components
- `SupportProcessor` class: Core logic for message processing
- Environment variables: For secure API key management

## Future Work
- Human‑in‑the‑loop review of training data. 
- Live connection to Slack for real-time message processing.
- Expand data to include action paths for intents i.e. routed to XYZ engineering team.
- Enable review of existing features/bugs in Shortcut to avoid duplicate creation.
- Intent drift detection and catalog versioning.
- Introduce self service with intent catalog management for users.

## License

MIT
