# 🚀 Gemini Code Analysis

A powerful code analysis tool powered by Google's Gemini 2.5 Pro AI model. Upload your code files and interact with Gemini to analyze, understand, and improve your codebase.

## ✨ Features

- 📁 Support for multiple file formats (ZIP archives and individual text files)
- 💬 Interactive chat interface with Gemini 2.5 Pro
- 🔍 Deep code analysis capabilities
- 🌊 Stream-based responses for better user experience
- 🔄 Session management for continuous context
- 🎯 Support for 25+ programming languages and text formats

## 🛠️ Prerequisites

- Python 3.8 or higher
- Google AI Platform API key
- Internet connection

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Gemini-2.5-Pro-Coding-App.git
cd Gemini-2.5-Pro-Coding-App
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## 🚀 Usage

1. Start the application:

```bash
python main.py
```

2. Open your browser and navigate to:

```
http://localhost:8001
```

3. Upload your code files:

   - Click the "Upload" button
   - Select individual files or ZIP archives
   - Supported file types include: .py, .js, .java, .cpp, and many more

4. Start analyzing:
   - Ask questions about your code
   - Request code reviews
   - Get explanations and suggestions

## 💡 Example Queries

- "Analyze the code structure and suggest improvements"
- "Find potential security vulnerabilities in this code"
- "Explain how this function works"
- "Suggest ways to optimize this implementation"

## 🔧 Supported File Types

```
.py, .js, .java, .cpp, .h, .cs, .php, .rb, .go, .rs
.html, .css, .tsx, .jsx, .ts
.json, .yaml, .yml, .toml, .xml
.txt, .md, .ini, .conf, .cfg, .sh, .bat
```

## ⚠️ Important Notes

- Keep your API key secure and never commit it to version control
- The application runs locally on port 8001 by default
- Large files may take longer to process

## 📄 License

MIT License - feel free to use and modify for your needs.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
