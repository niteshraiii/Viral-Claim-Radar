from webapp import app, _load_api_key


if __name__ == "__main__":
    _load_api_key()
    app.run(debug=True)
