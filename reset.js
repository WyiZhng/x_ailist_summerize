module.exports = {
    run: [
        {
            method: "fs.rm",
            params: {
                path: "venv"
            }
        },
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
            method: "notify",
            params: {
                html: "✅ Environment reset complete!"
            }
        }
    ]
}
