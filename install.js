module.exports = {
    run: [
        {
            method: "shell.run",
            params: {
                message: "python -m venv venv"
            }
        },
        {
            method: "shell.run",
            params: {
                venv: "venv",
                message: "pip install uv"
            }
        },
        {
            method: "shell.run",
            params: {
                venv: "venv",
                message: "uv pip install -r requirements.txt"
            }
        },
        {
            method: "shell.run",
            params: {
                venv: "venv",
                message: "python apply_twikit_patches.py"
            }
        },
        {
            method: "shell.run",
            params: {
                venv: "venv",
                message: "python -m app.config --ensure"
            }
        },
        {
            method: "fs.make",
            params: {
                path: "output"
            }
        },
        {
            method: "fs.make",
            params: {
                path: "browser_session"
            }
        },
        {
            method: "notify",
            params: {
                html: "✅ Installation complete! Next step: Login to X"
            }
        }
    ]
}
