# Basketball Feedback Generator

A real-time basketball shot analysis tool that uses computer vision and AI to provide instant feedback on shooting form.


## How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohtumm/Basketball-Feedback-Generator.git
   cd Basketball-Feedback-Generator
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

1. **Activate the virtual environment**
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the application**
   ```bash
   python feedback_generator.py
   ```

3. **Use the application**
   - Take a shot!
   - The system will detect the motion and provide AI-generated feedback
   - Press 'q' to quit